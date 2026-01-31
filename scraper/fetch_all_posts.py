#!/usr/bin/env python3
"""
Fetch ALL posts from Moltbook using proper pagination.
"""

import asyncio
import aiohttp
import aiofiles
import json
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = "https://www.moltbook.com"
ARCHIVE_DIR = Path(__file__).parent.parent / "archive"
CONCURRENCY = 10  # Be nice to the server

async def fetch_json(session, endpoint, semaphore):
    """Fetch JSON with rate limiting"""
    url = f"{BASE_URL}{endpoint}"
    async with semaphore:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    return await resp.json()
                print(f"  HTTP {resp.status}: {endpoint}")
                return None
        except Exception as e:
            print(f"  Error: {endpoint} - {e}")
            return None

async def save_json(data, filename):
    """Save data as JSON"""
    filepath = ARCHIVE_DIR / "data" / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(filepath, 'w') as f:
        await f.write(json.dumps(data, indent=2, ensure_ascii=False))

async def main():
    print("="*60)
    print("FETCHING ALL MOLTBOOK POSTS")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print("="*60)

    all_posts = []
    all_users = {}
    semaphore = asyncio.Semaphore(CONCURRENCY)

    headers = {
        'User-Agent': 'MoltbookArchiver/3.0 (Complete Archive)',
        'Accept': 'application/json',
    }

    async with aiohttp.ClientSession(headers=headers) as session:
        offset = 0
        batch_size = 25
        has_more = True

        while has_more:
            endpoint = f'/api/v1/posts?sort=new&offset={offset}'
            data = await fetch_json(session, endpoint, semaphore)

            if not data or not data.get('success'):
                print(f"  Failed at offset {offset}")
                break

            posts = data.get('posts', [])
            if not posts:
                break

            all_posts.extend(posts)

            # Track users
            for post in posts:
                author = post.get('author')
                if author and author.get('id'):
                    all_users[author['id']] = author

            has_more = data.get('has_more', False)
            count = data.get('count', len(posts))
            next_offset = data.get('next_offset', offset + batch_size)

            print(f"  Offset {offset}: got {count} posts (total: {len(all_posts)}, has_more: {has_more})")

            offset = next_offset

            # Small delay to be polite
            await asyncio.sleep(0.05)

        # Also try fetching by submolt to catch any we missed
        print("\n[Checking submolts for additional posts...]")
        submolts_data = await fetch_json(session, '/api/v1/submolts', semaphore)
        if submolts_data and submolts_data.get('success'):
            submolts = submolts_data.get('submolts', [])
            known_ids = {p['id'] for p in all_posts if p.get('id')}

            async def fetch_submolt_posts(name):
                detail = await fetch_json(session, f'/api/v1/submolts/{name}', semaphore)
                if detail and detail.get('success'):
                    return detail.get('posts', [])
                return []

            tasks = [fetch_submolt_posts(s['name']) for s in submolts]
            results = await asyncio.gather(*tasks)

            additional = 0
            for posts in results:
                for post in posts:
                    if post.get('id') and post['id'] not in known_ids:
                        all_posts.append(post)
                        known_ids.add(post['id'])
                        additional += 1
                        author = post.get('author')
                        if author and author.get('id'):
                            all_users[author['id']] = author

            print(f"  Found {additional} additional posts from submolts")

    # Deduplicate by ID
    seen_ids = set()
    unique_posts = []
    for post in all_posts:
        pid = post.get('id')
        if pid and pid not in seen_ids:
            unique_posts.append(post)
            seen_ids.add(pid)

    print(f"\nTotal unique posts: {len(unique_posts)}")
    print(f"Total unique users: {len(all_users)}")

    # Save
    print("\n[Saving...]")
    await save_json(unique_posts, 'posts_all_complete.json')
    await save_json(list(all_users.values()), 'users_all_complete.json')

    # Save individual posts
    for post in unique_posts:
        pid = post.get('id')
        if pid:
            await save_json(post, f'posts/{pid}.json')

    # Save individual users
    for uid, user in all_users.items():
        username = user.get('name', uid)
        await save_json(user, f'users/{username}.json')

    # Stats
    total_comments = sum(p.get('comment_count', 0) for p in unique_posts)
    total_upvotes = sum(p.get('upvotes', 0) for p in unique_posts)

    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"Posts: {len(unique_posts)}")
    print(f"Users: {len(all_users)}")
    print(f"Total comments (reported): {total_comments}")
    print(f"Total upvotes: {total_upvotes}")
    print(f"Finished: {datetime.now(timezone.utc).isoformat()}")

if __name__ == '__main__':
    asyncio.run(main())
