#!/usr/bin/env python3
"""
Moltbook Archiver - Archive all data from the site
"""

import requests
import json
import os
from datetime import datetime
from urllib.parse import urljoin
from pathlib import Path

BASE_URL = "https://www.moltbook.com"
ARCHIVE_DIR = Path(__file__).parent.parent / "archive"
SESSION = requests.Session()
SESSION.headers.update({
    'User-Agent': 'MoltbookArchiver/1.0 (Digital Preservation Project)',
    'Accept': 'application/json',
})

def fetch_json(endpoint, params=None):
    """Fetch JSON from an endpoint"""
    url = urljoin(BASE_URL, endpoint)
    try:
        resp = SESSION.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def fetch_html(path):
    """Fetch HTML page"""
    url = urljoin(BASE_URL, path)
    try:
        resp = SESSION.get(url, timeout=30, headers={'Accept': 'text/html'})
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def save_json(data, filename):
    """Save data as JSON"""
    filepath = ARCHIVE_DIR / "data" / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved: {filepath}")

def save_html(content, filename):
    """Save HTML content"""
    filepath = ARCHIVE_DIR / "html" / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Saved: {filepath}")

def explore_api():
    """Explore all API endpoints to find available data"""
    print("\n=== Exploring API Endpoints ===\n")

    endpoints_to_try = [
        '/api/v1/submolts',
        '/api/v1/posts',
        '/api/v1/users',
        '/api/v1/comments',
        '/api/v1/agents',
        '/api/v1/feed',
        '/api/submolts',
        '/api/posts',
        '/api/users',
    ]

    found_endpoints = {}
    for endpoint in endpoints_to_try:
        data = fetch_json(endpoint)
        if data:
            print(f"✓ {endpoint}")
            found_endpoints[endpoint] = data
            # Quick preview
            if isinstance(data, dict):
                print(f"  Keys: {list(data.keys())}")
                for key, val in data.items():
                    if isinstance(val, list):
                        print(f"  {key}: {len(val)} items")
        else:
            print(f"✗ {endpoint}")

    return found_endpoints

def archive_submolts():
    """Archive all submolts"""
    print("\n=== Archiving Submolts ===\n")

    # Get list of submolts
    data = fetch_json('/api/v1/submolts')
    if not data or not data.get('success'):
        print("Failed to fetch submolts list")
        return []

    submolts = data.get('submolts', [])
    print(f"Found {len(submolts)} submolts")
    save_json(data, 'submolts_list.json')

    # Archive each submolt's page and potentially more data
    for submolt in submolts:
        name = submolt.get('name', 'unknown')
        print(f"\nArchiving submolt: {name}")

        # Save individual submolt data
        save_json(submolt, f'submolts/{name}.json')

        # Try to fetch submolt-specific endpoints
        submolt_data = fetch_json(f'/api/v1/submolts/{name}')
        if submolt_data:
            save_json(submolt_data, f'submolts/{name}_detail.json')

        # Fetch submolt posts
        posts_data = fetch_json(f'/api/v1/submolts/{name}/posts')
        if posts_data:
            save_json(posts_data, f'submolts/{name}_posts.json')

        # Fetch HTML page
        html = fetch_html(f'/m/{name}')
        if html:
            save_html(html, f'submolts/{name}.html')

    return submolts

def archive_posts():
    """Archive all posts"""
    print("\n=== Archiving Posts ===\n")

    # Get all posts
    data = fetch_json('/api/v1/posts')
    if not data or not data.get('success'):
        print("Failed to fetch posts")
        return []

    posts = data.get('posts', [])
    print(f"Found {len(posts)} posts")
    save_json(data, 'posts_list.json')

    # Try to get paginated posts
    page = 1
    all_posts = posts.copy()
    while True:
        page += 1
        more_data = fetch_json('/api/v1/posts', params={'page': page})
        if not more_data or not more_data.get('posts'):
            break
        more_posts = more_data.get('posts', [])
        if not more_posts:
            break
        all_posts.extend(more_posts)
        print(f"Page {page}: {len(more_posts)} more posts")

    # Archive individual posts
    for post in all_posts:
        post_id = post.get('id', 'unknown')
        print(f"Archiving post: {post_id[:8]}... - {post.get('title', 'untitled')[:50]}")

        save_json(post, f'posts/{post_id}.json')

        # Try to fetch post detail with comments
        detail = fetch_json(f'/api/v1/posts/{post_id}')
        if detail:
            save_json(detail, f'posts/{post_id}_detail.json')

        # Fetch comments
        comments = fetch_json(f'/api/v1/posts/{post_id}/comments')
        if comments:
            save_json(comments, f'posts/{post_id}_comments.json')

    save_json(all_posts, 'posts_all.json')
    return all_posts

def archive_users():
    """Archive users"""
    print("\n=== Archiving Users ===\n")

    data = fetch_json('/api/v1/users')
    if data:
        save_json(data, 'users_list.json')
        users = data.get('users', data) if isinstance(data, dict) else data
        print(f"Found users data")

        if isinstance(users, list):
            for user in users:
                user_id = user.get('id') or user.get('username', 'unknown')
                save_json(user, f'users/{user_id}.json')

    # Also try /u endpoint
    html = fetch_html('/u')
    if html:
        save_html(html, 'users_index.html')

def archive_static_pages():
    """Archive static pages"""
    print("\n=== Archiving Static Pages ===\n")

    pages = ['/', '/m', '/terms', '/privacy', '/u']
    for page in pages:
        html = fetch_html(page)
        if html:
            filename = 'index.html' if page == '/' else f"{page.strip('/')}.html"
            save_html(html, filename)

def create_manifest():
    """Create archive manifest"""
    manifest = {
        'archived_at': datetime.utcnow().isoformat() + 'Z',
        'source': BASE_URL,
        'archiver': 'MoltbookArchiver/1.0',
        'files': []
    }

    for root, dirs, files in os.walk(ARCHIVE_DIR):
        for file in files:
            filepath = Path(root) / file
            manifest['files'].append({
                'path': str(filepath.relative_to(ARCHIVE_DIR)),
                'size': filepath.stat().st_size
            })

    save_json(manifest, 'manifest.json')

def main():
    print("=" * 60)
    print("MOLTBOOK ARCHIVER")
    print(f"Archive directory: {ARCHIVE_DIR}")
    print("=" * 60)

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    # First explore API
    found = explore_api()

    # Archive everything
    archive_static_pages()
    archive_submolts()
    archive_posts()
    archive_users()

    # Create manifest
    create_manifest()

    print("\n" + "=" * 60)
    print("ARCHIVE COMPLETE")
    print("=" * 60)

if __name__ == '__main__':
    main()
