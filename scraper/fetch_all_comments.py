#!/usr/bin/env python3
"""
Fetch all comments from posts that have them.
"""

import asyncio
import aiohttp
import aiofiles
import json
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = "https://www.moltbook.com"
ARCHIVE_DIR = Path(__file__).parent.parent / "archive"
CONCURRENCY = 30

async def fetch_json(session, endpoint, semaphore):
    """Fetch JSON with rate limiting"""
    url = f"{BASE_URL}{endpoint}"
    async with semaphore:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    return await resp.json()
                return None
        except Exception as e:
            return None

async def save_json(data, filename):
    """Save data as JSON"""
    filepath = ARCHIVE_DIR / "data" / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(filepath, 'w') as f:
        await f.write(json.dumps(data, indent=2, ensure_ascii=False))

async def main():
    print("="*60)
    print("FETCHING ALL COMMENTS")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print("="*60)

    # Load posts
    posts_file = ARCHIVE_DIR / "data" / "posts_all_complete.json"
    async with aiofiles.open(posts_file, 'r') as f:
        posts = json.loads(await f.read())

    # Filter posts that have comments
    posts_with_comments = [p for p in posts if p.get('comment_count', 0) > 0]
    total_expected = sum(p.get('comment_count', 0) for p in posts_with_comments)

    print(f"Posts with comments: {len(posts_with_comments)}")
    print(f"Expected comments: {total_expected}")

    all_comments = []
    semaphore = asyncio.Semaphore(CONCURRENCY)
    processed = 0
    failed = 0

    headers = {
        'User-Agent': 'MoltbookArchiver/3.0 (Complete Archive)',
        'Accept': 'application/json',
    }

    async with aiohttp.ClientSession(headers=headers) as session:
        async def fetch_post_comments(post):
            nonlocal processed, failed
            post_id = post.get('id')
            detail = await fetch_json(session, f'/api/v1/posts/{post_id}', semaphore)

            processed += 1
            if processed % 500 == 0:
                print(f"  Progress: {processed}/{len(posts_with_comments)} posts, {len(all_comments)} comments")

            if detail and detail.get('comments'):
                comments = detail.get('comments', [])
                # Add post_id to each comment for reference
                for c in comments:
                    c['_post_id'] = post_id
                return comments
            else:
                failed += 1
                return []

        # Process in batches to avoid overwhelming
        batch_size = 200
        for i in range(0, len(posts_with_comments), batch_size):
            batch = posts_with_comments[i:i+batch_size]
            tasks = [fetch_post_comments(p) for p in batch]
            results = await asyncio.gather(*tasks)
            for comments in results:
                all_comments.extend(comments)
            print(f"  Batch {i//batch_size + 1}: {len(all_comments)} total comments")

    print(f"\nTotal comments fetched: {len(all_comments)}")
    print(f"Failed fetches: {failed}")

    # Save all comments
    await save_json(all_comments, 'comments_all_complete.json')

    # Also save per-post for easier access
    comments_by_post = {}
    for c in all_comments:
        post_id = c.get('_post_id')
        if post_id:
            if post_id not in comments_by_post:
                comments_by_post[post_id] = []
            comments_by_post[post_id].append(c)

    for post_id, comments in comments_by_post.items():
        await save_json(comments, f'posts/{post_id}_comments.json')

    print(f"\nSaved {len(comments_by_post)} comment files")
    print(f"Finished: {datetime.now(timezone.utc).isoformat()}")

if __name__ == '__main__':
    asyncio.run(main())
