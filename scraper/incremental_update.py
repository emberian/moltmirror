#!/usr/bin/env python3
"""
Efficient Incremental Moltbook Archiver
- Stops fetching when hitting known posts
- Only fetches comments for posts with new activity
- Uploads only changed files to S3
"""

import asyncio
import aiohttp
import aiofiles
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = "https://www.moltbook.com"
ARCHIVE_DIR = Path(__file__).parent.parent / "archive"
STATE_FILE = ARCHIVE_DIR / "data" / ".archive_state.json"
S3_BUCKET = "s3://moltbook-archive-319933937176"
CONCURRENCY = 20

async def load_state():
    """Load archive state"""
    if STATE_FILE.exists():
        async with aiofiles.open(STATE_FILE, 'r') as f:
            return json.loads(await f.read())
    return {
        'last_update': None,
        'known_post_ids': set(),
        'known_submolt_names': set(),
        'post_comment_counts': {},
    }

async def save_state(state):
    """Save archive state"""
    # Convert sets to lists for JSON
    state_to_save = {
        'last_update': state['last_update'],
        'known_post_ids': list(state['known_post_ids']),
        'known_submolt_names': list(state['known_submolt_names']),
        'post_comment_counts': state['post_comment_counts'],
    }
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(STATE_FILE, 'w') as f:
        await f.write(json.dumps(state_to_save, indent=2))

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

async def save_json(data, filename):
    """Save data as JSON"""
    filepath = ARCHIVE_DIR / "data" / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(filepath, 'w') as f:
        await f.write(json.dumps(data, indent=2, ensure_ascii=False))
    return filepath

async def append_to_json_array(new_items, filename, key='id'):
    """Append new items to existing JSON array file"""
    filepath = ARCHIVE_DIR / "data" / filename
    existing = []
    if filepath.exists():
        async with aiofiles.open(filepath, 'r') as f:
            existing = json.loads(await f.read())

    existing_keys = {item.get(key) for item in existing if item.get(key)}
    added = [item for item in new_items if item.get(key) not in existing_keys]

    if added:
        existing.extend(added)
        await save_json(existing, filename)
    return len(added)

def sync_to_s3(files_changed):
    """Sync changed files to S3 efficiently"""
    if not files_changed:
        print("  No files to sync")
        return

    # Just sync the whole data directory - S3 sync skips unchanged files
    # This is faster than individual uploads
    result = subprocess.run(
        ['aws', 's3', 'sync', str(ARCHIVE_DIR / 'data'), f'{S3_BUCKET}/data/',
         '--size-only'],  # Skip files that match in size (faster)
        capture_output=True, text=True
    )
    if result.returncode == 0:
        uploaded = result.stdout.count('upload:')
        print(f"  Synced {uploaded} files to S3")
    else:
        print(f"  S3 sync error: {result.stderr}")

async def main():
    print("="*60)
    print("EFFICIENT INCREMENTAL UPDATE")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print("="*60)

    state = await load_state()
    # Convert lists back to sets
    known_post_ids = set(state.get('known_post_ids', []))
    known_submolts = set(state.get('known_submolt_names', []))
    post_comment_counts = state.get('post_comment_counts', {})

    print(f"Known posts: {len(known_post_ids)}")
    print(f"Tracking comments for: {len(post_comment_counts)} posts")

    new_posts = []
    new_comments = []
    posts_needing_comments = []  # Posts with new comments to fetch
    files_changed = []

    semaphore = asyncio.Semaphore(CONCURRENCY)
    headers = {
        'User-Agent': 'MoltbookArchiver/3.0 (Incremental)',
        'Accept': 'application/json',
    }

    async with aiohttp.ClientSession(headers=headers) as session:
        # 1. Fetch new posts using sort=new, STOP when we hit known posts
        print("\n[1/3] Fetching new posts (will stop at known content)...")
        offset = 0
        consecutive_known = 0

        while consecutive_known < 3:  # Stop after 3 consecutive known posts
            data = await fetch_json(session, f'/api/v1/posts?sort=new&offset={offset}', semaphore)
            if not data or not data.get('posts'):
                break

            posts = data.get('posts', [])
            batch_new = 0

            for post in posts:
                post_id = post.get('id')
                if not post_id:
                    continue

                if post_id in known_post_ids:
                    consecutive_known += 1
                    # Check if comment count changed
                    old_count = post_comment_counts.get(post_id, 0)
                    new_count = post.get('comment_count', 0)
                    if new_count > old_count:
                        posts_needing_comments.append(post_id)
                        post_comment_counts[post_id] = new_count
                else:
                    consecutive_known = 0  # Reset counter
                    new_posts.append(post)
                    known_post_ids.add(post_id)
                    post_comment_counts[post_id] = post.get('comment_count', 0)
                    batch_new += 1

                    # Save individual post
                    fp = await save_json(post, f'posts/{post_id}.json')
                    files_changed.append(fp)

            if batch_new > 0:
                print(f"  Offset {offset}: {batch_new} new posts")

            if not data.get('has_more', False):
                break
            offset += 25

        print(f"  Total new posts: {len(new_posts)}")
        print(f"  Posts with new comments: {len(posts_needing_comments)}")

        # 2. Fetch comments only for new posts and posts with updated counts
        print("\n[2/3] Fetching comments for new/updated posts...")
        posts_to_fetch = [p['id'] for p in new_posts] + posts_needing_comments

        if posts_to_fetch:
            async def fetch_comments(post_id):
                detail = await fetch_json(session, f'/api/v1/posts/{post_id}', semaphore)
                if detail and detail.get('comments'):
                    comments = detail.get('comments', [])
                    for c in comments:
                        c['_post_id'] = post_id
                    # Save comments for this post
                    fp = await save_json(comments, f'posts/{post_id}_comments.json')
                    files_changed.append(fp)
                    return comments
                return []

            tasks = [fetch_comments(pid) for pid in posts_to_fetch]
            results = await asyncio.gather(*tasks)
            for comments in results:
                new_comments.extend(comments)
            print(f"  Fetched {len(new_comments)} comments from {len(posts_to_fetch)} posts")

        # 3. Check for new submolts (quick - just one API call)
        print("\n[3/3] Checking for new submolts...")
        submolts_data = await fetch_json(session, '/api/v1/submolts', semaphore)
        new_submolts = []
        if submolts_data and submolts_data.get('success'):
            for s in submolts_data.get('submolts', []):
                if s.get('name') not in known_submolts:
                    new_submolts.append(s)
                    known_submolts.add(s.get('name'))
                    fp = await save_json(s, f'submolts/{s["name"]}.json')
                    files_changed.append(fp)
        print(f"  New submolts: {len(new_submolts)}")

    # Update aggregate files if we have new data
    if new_posts:
        added = await append_to_json_array(new_posts, 'posts_all_complete.json', 'id')
        if added:
            files_changed.append(ARCHIVE_DIR / 'data' / 'posts_all_complete.json')

    if new_comments:
        added = await append_to_json_array(new_comments, 'comments_all_complete.json', 'id')
        if added:
            files_changed.append(ARCHIVE_DIR / 'data' / 'comments_all_complete.json')

    # Save state
    state['last_update'] = datetime.now(timezone.utc).isoformat()
    state['known_post_ids'] = known_post_ids
    state['known_submolt_names'] = known_submolts
    state['post_comment_counts'] = post_comment_counts
    await save_state(state)
    files_changed.append(STATE_FILE)

    # Sync to S3
    print("\n[S3] Syncing changed files...")
    try:
        sync_to_s3(files_changed)
    except Exception as e:
        print(f"  S3 sync failed: {e}")
        print("  Run manually: aws s3 sync archive/data/ s3://moltbook-archive-319933937176/data/")

    # Summary
    print("\n" + "="*60)
    print("UPDATE COMPLETE")
    print("="*60)
    print(f"New posts: {len(new_posts)}")
    print(f"New comments: {len(new_comments)}")
    print(f"New submolts: {len(new_submolts)}")
    print(f"Files synced: {len(files_changed)}")
    print(f"Finished: {datetime.now(timezone.utc).isoformat()}")

if __name__ == '__main__':
    asyncio.run(main())
