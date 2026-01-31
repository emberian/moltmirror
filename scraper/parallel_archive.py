#!/usr/bin/env python3
"""
Parallel Moltbook Archiver using asyncio
Fetches all data concurrently for speed.
"""

import asyncio
import aiohttp
import aiofiles
import json
import os
from datetime import datetime
from pathlib import Path
from collections import defaultdict

BASE_URL = "https://www.moltbook.com"
ARCHIVE_DIR = Path(__file__).parent.parent / "archive"
CONCURRENCY = 20  # Max concurrent requests

# Collected data
seen_users = {}
seen_posts = {}
all_comments = []

async def fetch_json(session, endpoint, semaphore):
    """Fetch JSON with concurrency control"""
    url = f"{BASE_URL}{endpoint}"
    async with semaphore:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    return await resp.json()
                return None
        except Exception as e:
            print(f"  Error: {endpoint} - {e}")
            return None

async def fetch_html(session, path, semaphore):
    """Fetch HTML with concurrency control"""
    url = f"{BASE_URL}{path}"
    async with semaphore:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    return await resp.text()
                return None
        except Exception as e:
            return None

async def save_json(data, filename):
    """Save data as JSON async"""
    filepath = ARCHIVE_DIR / "data" / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(filepath, 'w') as f:
        await f.write(json.dumps(data, indent=2, ensure_ascii=False))

async def save_html(content, filename):
    """Save HTML async"""
    filepath = ARCHIVE_DIR / "html" / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(filepath, 'w') as f:
        await f.write(content)

def track_user(user_data):
    """Track unique user"""
    if not user_data:
        return
    user_id = user_data.get('id')
    if user_id:
        if user_id not in seen_users:
            seen_users[user_id] = user_data
        else:
            for key, val in user_data.items():
                if val and not seen_users[user_id].get(key):
                    seen_users[user_id][key] = val

async def archive_submolt(session, semaphore, submolt, index, total):
    """Archive a single submolt"""
    name = submolt.get('name', 'unknown')
    display_name = submolt.get('display_name', name)

    # Get detail with posts
    detail = await fetch_json(session, f'/api/v1/submolts/{name}', semaphore)

    posts = []
    if detail and detail.get('success'):
        await save_json(detail, f'submolts/{name}_detail.json')
        posts = detail.get('posts', [])

        # Track users
        submolt_data = detail.get('submolt', {})
        for mod in submolt_data.get('moderators', []):
            track_user(mod)
        track_user(submolt_data.get('created_by'))

        for post in posts:
            post_id = post.get('id')
            if post_id:
                seen_posts[post_id] = post
            track_user(post.get('author'))

    # Get HTML
    html = await fetch_html(session, f'/m/{name}', semaphore)
    if html:
        await save_html(html, f'submolts/{name}.html')

    print(f"[{index+1}/{total}] m/{name}: {len(posts)} posts")
    return posts

async def archive_post_detail(session, semaphore, post_id):
    """Try to get post detail and comments"""
    detail = await fetch_json(session, f'/api/v1/posts/{post_id}', semaphore)
    if detail:
        await save_json(detail, f'posts/{post_id}_detail.json')
        if detail.get('comments'):
            return detail.get('comments', [])
    return []

async def main():
    print("="*60)
    print("PARALLEL MOLTBOOK ARCHIVER")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Concurrency: {CONCURRENCY}")
    print("="*60)

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(CONCURRENCY)

    headers = {
        'User-Agent': 'MoltbookArchiver/3.0 (Digital Preservation)',
        'Accept': 'application/json',
    }

    async with aiohttp.ClientSession(headers=headers) as session:
        # 1. Get submolt list
        print("\n[1/5] Fetching submolt list...")
        submolts_data = await fetch_json(session, '/api/v1/submolts', semaphore)
        if not submolts_data or not submolts_data.get('success'):
            print("Failed to get submolts!")
            return

        submolts = submolts_data.get('submolts', [])
        print(f"Found {len(submolts)} submolts")
        await save_json(submolts_data, 'submolts_list.json')

        # 2. Archive all submolts in parallel
        print("\n[2/5] Archiving submolts in parallel...")
        tasks = [
            archive_submolt(session, semaphore, s, i, len(submolts))
            for i, s in enumerate(submolts)
        ]
        await asyncio.gather(*tasks)

        # 3. Get main posts feed
        print("\n[3/5] Fetching main posts feed...")
        posts_data = await fetch_json(session, '/api/v1/posts', semaphore)
        if posts_data and posts_data.get('success'):
            await save_json(posts_data, 'posts_feed.json')
            for post in posts_data.get('posts', []):
                post_id = post.get('id')
                if post_id:
                    seen_posts[post_id] = post
                track_user(post.get('author'))

        print(f"Total unique posts: {len(seen_posts)}")

        # 4. Try to get post details (might have comments)
        print("\n[4/5] Fetching post details...")
        post_ids = list(seen_posts.keys())
        detail_tasks = [
            archive_post_detail(session, semaphore, pid)
            for pid in post_ids
        ]
        results = await asyncio.gather(*detail_tasks)
        for comments in results:
            all_comments.extend(comments)

        # Save individual posts
        for post_id, post in seen_posts.items():
            await save_json(post, f'posts/{post_id}.json')

        # 5. Archive static pages
        print("\n[5/5] Archiving static pages...")
        pages = ['/', '/m', '/u', '/terms', '/privacy']
        html_tasks = [fetch_html(session, p, semaphore) for p in pages]
        html_results = await asyncio.gather(*html_tasks)
        for page, html in zip(pages, html_results):
            if html:
                filename = 'index.html' if page == '/' else f"{page.strip('/').replace('/', '_')}.html"
                await save_html(html, filename)

    # Save collected data
    print("\n[FINAL] Saving collected data...")
    await save_json(list(seen_users.values()), 'users_all.json')
    await save_json(list(seen_posts.values()), 'posts_all.json')
    if all_comments:
        await save_json(all_comments, 'comments_all.json')

    # Save users individually
    for user_id, user in seen_users.items():
        username = user.get('name', user_id)
        await save_json(user, f'users/{username}.json')

    # Create combined archive
    combined = {
        'archived_at': datetime.utcnow().isoformat() + 'Z',
        'source': BASE_URL,
        'stats': {
            'submolts': len(submolts),
            'posts': len(seen_posts),
            'users': len(seen_users),
            'comments': len(all_comments),
        },
        'submolts': submolts,
        'posts': list(seen_posts.values()),
        'users': list(seen_users.values()),
        'comments': all_comments,
    }
    await save_json(combined, 'moltbook_complete.json')

    # Stats
    print("\n" + "="*60)
    print("ARCHIVE COMPLETE")
    print("="*60)
    print(f"Submolts: {len(submolts)}")
    print(f"Posts: {len(seen_posts)}")
    print(f"Users: {len(seen_users)}")
    print(f"Comments: {len(all_comments)}")

    # Count files
    file_count = sum(1 for _ in ARCHIVE_DIR.rglob('*') if _.is_file())
    total_size = sum(f.stat().st_size for f in ARCHIVE_DIR.rglob('*') if f.is_file())
    print(f"Total files: {file_count}")
    print(f"Total size: {total_size / 1024 / 1024:.2f} MB")
    print(f"Finished: {datetime.now().isoformat()}")

if __name__ == '__main__':
    asyncio.run(main())
