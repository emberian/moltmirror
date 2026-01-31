#!/usr/bin/env python3
"""
Comprehensive Moltbook Archiver
Archives all data from the moltbook API including submolts, posts, comments, and users.
"""

import requests
import json
import os
import time
from datetime import datetime
from urllib.parse import urljoin
from pathlib import Path
from collections import defaultdict

BASE_URL = "https://www.moltbook.com"
ARCHIVE_DIR = Path(__file__).parent.parent / "archive"
SESSION = requests.Session()
SESSION.headers.update({
    'User-Agent': 'MoltbookArchiver/2.0 (Digital Preservation Project)',
    'Accept': 'application/json',
})

# Track unique data for deduplication
seen_users = {}
seen_posts = {}
all_comments = []

def fetch_json(endpoint, params=None, retries=3):
    """Fetch JSON from an endpoint with retries"""
    url = urljoin(BASE_URL, endpoint)
    for attempt in range(retries):
        try:
            resp = SESSION.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 404:
                return None
            else:
                print(f"  HTTP {resp.status_code} for {url}")
        except Exception as e:
            print(f"  Error fetching {url}: {e}")
            if attempt < retries - 1:
                time.sleep(1)
    return None

def fetch_html(path):
    """Fetch HTML page"""
    url = urljoin(BASE_URL, path)
    try:
        resp = SESSION.get(url, timeout=30, headers={'Accept': 'text/html'})
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        return None

def save_json(data, filename):
    """Save data as JSON"""
    filepath = ARCHIVE_DIR / "data" / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return filepath

def save_html(content, filename):
    """Save HTML content"""
    filepath = ARCHIVE_DIR / "html" / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(content)
    return filepath

def track_user(user_data):
    """Track unique user data"""
    if not user_data:
        return
    user_id = user_data.get('id')
    if user_id and user_id not in seen_users:
        seen_users[user_id] = user_data
    elif user_id:
        # Merge additional data
        existing = seen_users[user_id]
        for key, val in user_data.items():
            if val and not existing.get(key):
                existing[key] = val

def explore_comment_endpoints(post_id):
    """Try different comment endpoint patterns"""
    patterns = [
        f'/api/v1/posts/{post_id}/comments',
        f'/api/v1/comments?post_id={post_id}',
        f'/api/posts/{post_id}/comments',
    ]
    for pattern in patterns:
        data = fetch_json(pattern)
        if data:
            return data
    return None

def archive_post_page(post_id, submolt_name):
    """Archive individual post page which may have comments rendered"""
    html = fetch_html(f'/m/{submolt_name}/post/{post_id}')
    if html:
        save_html(html, f'posts/{post_id}.html')
        return html
    return None

def archive_submolts():
    """Archive all submolts with full detail"""
    print("\n" + "="*60)
    print("ARCHIVING SUBMOLTS")
    print("="*60)

    # Get submolt list
    data = fetch_json('/api/v1/submolts')
    if not data or not data.get('success'):
        print("Failed to fetch submolts!")
        return []

    submolts = data.get('submolts', [])
    print(f"Found {len(submolts)} submolts")
    save_json(data, 'submolts_list.json')

    all_posts = []

    for i, submolt in enumerate(submolts):
        name = submolt.get('name', 'unknown')
        display_name = submolt.get('display_name', name)
        print(f"\n[{i+1}/{len(submolts)}] {display_name} (m/{name})")
        print(f"  Subscribers: {submolt.get('subscriber_count', 0)}")

        # Save basic submolt data
        save_json(submolt, f'submolts/{name}.json')

        # Get submolt detail with posts
        detail = fetch_json(f'/api/v1/submolts/{name}')
        if detail and detail.get('success'):
            save_json(detail, f'submolts/{name}_detail.json')

            posts = detail.get('posts', [])
            print(f"  Posts: {len(posts)}")

            for post in posts:
                post_id = post.get('id')
                if post_id:
                    seen_posts[post_id] = post
                    all_posts.append(post)

                    # Track author
                    author = post.get('author')
                    if author:
                        track_user(author)

            # Track moderators
            submolt_data = detail.get('submolt', {})
            for mod in submolt_data.get('moderators', []):
                track_user(mod)
            if submolt_data.get('created_by'):
                track_user(submolt_data['created_by'])

        # Get submolt HTML
        html = fetch_html(f'/m/{name}')
        if html:
            save_html(html, f'submolts/{name}.html')

        # Small delay to be nice
        time.sleep(0.1)

    return all_posts

def archive_posts():
    """Archive posts from main feed and individual pages"""
    print("\n" + "="*60)
    print("ARCHIVING POSTS FROM MAIN FEED")
    print("="*60)

    # Get main posts feed
    data = fetch_json('/api/v1/posts')
    if not data or not data.get('success'):
        print("Failed to fetch main posts feed")
        return

    posts = data.get('posts', [])
    print(f"Found {len(posts)} posts in main feed")
    save_json(data, 'posts_feed.json')

    for post in posts:
        post_id = post.get('id')
        if post_id:
            seen_posts[post_id] = post
            author = post.get('author')
            if author:
                track_user(author)

    # Try pagination
    for page in range(2, 100):
        more = fetch_json('/api/v1/posts', params={'page': page})
        if not more or not more.get('posts'):
            break
        new_posts = more.get('posts', [])
        if not new_posts:
            break
        print(f"Page {page}: {len(new_posts)} more posts")
        for post in new_posts:
            post_id = post.get('id')
            if post_id:
                seen_posts[post_id] = post

    # Also try offset-based pagination
    for offset in range(50, 10000, 50):
        more = fetch_json('/api/v1/posts', params={'offset': offset})
        if not more or not more.get('posts'):
            break
        new_posts = more.get('posts', [])
        if not new_posts:
            break
        print(f"Offset {offset}: {len(new_posts)} more posts")
        for post in new_posts:
            post_id = post.get('id')
            if post_id:
                seen_posts[post_id] = post

def archive_individual_posts():
    """Try to get detailed info for each post including comments"""
    print("\n" + "="*60)
    print("ARCHIVING INDIVIDUAL POSTS")
    print("="*60)

    print(f"Processing {len(seen_posts)} unique posts")

    for i, (post_id, post) in enumerate(seen_posts.items()):
        if i % 50 == 0:
            print(f"Progress: {i}/{len(seen_posts)}")

        title = post.get('title', 'untitled')[:50]
        submolt = post.get('submolt', {})
        submolt_name = submolt.get('name', 'general')

        # Save post data
        save_json(post, f'posts/{post_id}.json')

        # Try to get post detail
        detail = fetch_json(f'/api/v1/posts/{post_id}')
        if detail:
            save_json(detail, f'posts/{post_id}_detail.json')
            # Check for comments in detail
            if detail.get('comments'):
                all_comments.extend(detail.get('comments', []))

        # Try comment endpoints
        comments = explore_comment_endpoints(post_id)
        if comments:
            save_json(comments, f'posts/{post_id}_comments.json')
            if isinstance(comments, dict):
                all_comments.extend(comments.get('comments', []))
            elif isinstance(comments, list):
                all_comments.extend(comments)

        time.sleep(0.05)

def archive_users():
    """Archive all discovered users"""
    print("\n" + "="*60)
    print("ARCHIVING USERS")
    print("="*60)

    print(f"Discovered {len(seen_users)} unique users")

    # Save all users
    save_json(list(seen_users.values()), 'users_all.json')

    # Try to get more user data
    for user_id, user in seen_users.items():
        username = user.get('name', 'unknown')
        save_json(user, f'users/{username}.json')

        # Try user profile endpoint
        profile = fetch_json(f'/api/v1/users/{user_id}')
        if profile:
            save_json(profile, f'users/{username}_profile.json')

        # Try by username
        profile2 = fetch_json(f'/api/v1/users/{username}')
        if profile2:
            save_json(profile2, f'users/{username}_by_name.json')

    # Try users listing endpoints
    for endpoint in ['/api/v1/users', '/api/v1/agents', '/api/v1/users/top']:
        data = fetch_json(endpoint)
        if data:
            name = endpoint.split('/')[-1]
            save_json(data, f'users_{name}_endpoint.json')

def archive_static_pages():
    """Archive static pages"""
    print("\n" + "="*60)
    print("ARCHIVING STATIC PAGES")
    print("="*60)

    pages = ['/', '/m', '/u', '/terms', '/privacy']
    for page in pages:
        print(f"Fetching {page}")
        html = fetch_html(page)
        if html:
            filename = 'index.html' if page == '/' else f"{page.strip('/').replace('/', '_')}.html"
            save_html(html, filename)

def create_comprehensive_manifest():
    """Create detailed archive manifest"""
    print("\n" + "="*60)
    print("CREATING MANIFEST")
    print("="*60)

    manifest = {
        'archived_at': datetime.utcnow().isoformat() + 'Z',
        'source': BASE_URL,
        'archiver': 'MoltbookArchiver/2.0',
        'stats': {
            'submolts': 0,
            'posts': len(seen_posts),
            'users': len(seen_users),
            'comments': len(all_comments),
        },
        'files': []
    }

    # Count files
    for root, dirs, files in os.walk(ARCHIVE_DIR):
        for file in files:
            filepath = Path(root) / file
            rel_path = str(filepath.relative_to(ARCHIVE_DIR))
            manifest['files'].append({
                'path': rel_path,
                'size': filepath.stat().st_size
            })
            if 'submolts/' in rel_path and rel_path.endswith('.json') and '_' not in Path(rel_path).stem:
                manifest['stats']['submolts'] += 1

    save_json(manifest, 'manifest.json')

    # Save comments separately
    if all_comments:
        save_json(all_comments, 'comments_all.json')

    # Create a combined dataset
    combined = {
        'archived_at': manifest['archived_at'],
        'submolts': [],
        'posts': list(seen_posts.values()),
        'users': list(seen_users.values()),
        'comments': all_comments,
    }

    # Load submolt data
    submolts_path = ARCHIVE_DIR / 'data' / 'submolts_list.json'
    if submolts_path.exists():
        with open(submolts_path) as f:
            data = json.load(f)
            combined['submolts'] = data.get('submolts', [])

    save_json(combined, 'moltbook_complete.json')

    print(f"\nArchive Statistics:")
    print(f"  Submolts: {manifest['stats']['submolts']}")
    print(f"  Posts: {manifest['stats']['posts']}")
    print(f"  Users: {manifest['stats']['users']}")
    print(f"  Comments: {manifest['stats']['comments']}")
    print(f"  Total files: {len(manifest['files'])}")

def main():
    print("="*60)
    print("COMPREHENSIVE MOLTBOOK ARCHIVER")
    print(f"Started at: {datetime.now().isoformat()}")
    print(f"Archive directory: {ARCHIVE_DIR}")
    print("="*60)

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    # Archive in order
    archive_static_pages()
    all_posts = archive_submolts()
    archive_posts()
    archive_individual_posts()
    archive_users()
    create_comprehensive_manifest()

    print("\n" + "="*60)
    print("ARCHIVE COMPLETE")
    print(f"Finished at: {datetime.now().isoformat()}")
    print("="*60)

if __name__ == '__main__':
    main()
