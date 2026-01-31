#!/usr/bin/env python3
"""
Moltbook Site Explorer - Reverse engineer the data layer
"""

import requests
import json
import re
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

BASE_URL = "https://www.moltbook.com"
SESSION = requests.Session()
SESSION.headers.update({
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
})

def fetch_page(path):
    """Fetch a page and return response details"""
    url = urljoin(BASE_URL, path)
    try:
        resp = SESSION.get(url, timeout=30)
        return {
            'url': url,
            'status': resp.status_code,
            'headers': dict(resp.headers),
            'content_type': resp.headers.get('content-type', ''),
            'body': resp.text if resp.status_code == 200 else None,
            'length': len(resp.content)
        }
    except Exception as e:
        return {'url': url, 'error': str(e)}

def extract_nextjs_data(html):
    """Extract Next.js data from the page"""
    soup = BeautifulSoup(html, 'html.parser')
    data = {}

    # Find __NEXT_DATA__ script tag
    next_data = soup.find('script', {'id': '__NEXT_DATA__'})
    if next_data:
        try:
            data['__NEXT_DATA__'] = json.loads(next_data.string)
        except:
            data['__NEXT_DATA__'] = next_data.string[:500]

    # Find all script tags with potential data
    scripts = soup.find_all('script')
    data['script_srcs'] = []
    data['inline_scripts'] = []
    for script in scripts:
        if script.get('src'):
            data['script_srcs'].append(script['src'])
        elif script.string and len(script.string) > 50:
            # Look for interesting patterns
            content = script.string[:1000]
            if any(pattern in content for pattern in ['api', 'fetch', 'endpoint', 'data', 'submolt', 'post']):
                data['inline_scripts'].append(content[:500])

    # Extract all links
    data['links'] = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        text = a.get_text(strip=True)
        data['links'].append({'href': href, 'text': text})

    # Look for data attributes
    data['data_attrs'] = []
    for elem in soup.find_all(attrs={'data-': True}):
        attrs = {k: v for k, v in elem.attrs.items() if k.startswith('data-')}
        if attrs:
            data['data_attrs'].append(attrs)

    return data

def find_api_patterns(html):
    """Search for API endpoint patterns in HTML/JS"""
    patterns = [
        r'/api/[a-zA-Z0-9/_-]+',
        r'https?://[^"\'\s]+/api/[^"\'\s]+',
        r'fetch\(["\']([^"\']+)["\']',
        r'axios\.[a-z]+\(["\']([^"\']+)["\']',
        r'"endpoint":\s*"([^"]+)"',
        r'baseURL["\']?\s*[:=]\s*["\']([^"\']+)["\']',
    ]
    found = set()
    for pattern in patterns:
        matches = re.findall(pattern, html)
        found.update(matches)
    return list(found)

def probe_common_endpoints():
    """Probe common API endpoints"""
    endpoints = [
        '/api/submolts',
        '/api/posts',
        '/api/communities',
        '/api/v1/submolts',
        '/api/v1/posts',
        '/api/trpc/submolt',
        '/api/trpc/post',
        '/_next/data/latest/m.json',
        '/graphql',
        '/api/graphql',
    ]
    results = []
    for endpoint in endpoints:
        result = fetch_page(endpoint)
        if result.get('status') != 404:
            results.append(result)
    return results

def main():
    print("=" * 60)
    print("MOLTBOOK SITE EXPLORATION")
    print("=" * 60)

    # Fetch main pages
    pages_to_check = ['/m', '/', '/terms', '/privacy']

    for path in pages_to_check:
        print(f"\n--- Fetching {path} ---")
        result = fetch_page(path)
        print(f"Status: {result.get('status')}")
        print(f"Content-Type: {result.get('content_type')}")
        print(f"Length: {result.get('length')} bytes")

        if result.get('body'):
            nextjs_data = extract_nextjs_data(result['body'])

            if nextjs_data.get('__NEXT_DATA__'):
                print("\n__NEXT_DATA__ found!")
                nd = nextjs_data['__NEXT_DATA__']
                if isinstance(nd, dict):
                    print(f"  buildId: {nd.get('buildId', 'N/A')}")
                    print(f"  page: {nd.get('page', 'N/A')}")
                    if 'props' in nd:
                        props = nd['props']
                        print(f"  Props keys: {list(props.keys()) if isinstance(props, dict) else 'N/A'}")
                        # Pretty print the props if small enough
                        props_str = json.dumps(props, indent=2)
                        if len(props_str) < 2000:
                            print(f"  Props:\n{props_str}")
                        else:
                            print(f"  Props (truncated):\n{props_str[:2000]}...")

            print(f"\nLinks found: {len(nextjs_data.get('links', []))}")
            for link in nextjs_data.get('links', [])[:20]:
                print(f"  {link['href']} -> {link['text'][:50]}")

            api_patterns = find_api_patterns(result['body'])
            if api_patterns:
                print(f"\nAPI patterns found: {api_patterns}")

    print("\n" + "=" * 60)
    print("PROBING COMMON API ENDPOINTS")
    print("=" * 60)

    api_results = probe_common_endpoints()
    for result in api_results:
        print(f"\n{result['url']}")
        print(f"  Status: {result.get('status')}")
        if result.get('body'):
            print(f"  Body preview: {result['body'][:300]}")

    if not api_results:
        print("No accessible API endpoints found via common patterns")

if __name__ == '__main__':
    main()
