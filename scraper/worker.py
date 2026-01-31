#!/usr/bin/env python3
"""
Simple distributed worker for Moltbook archiving.
Run with: python worker.py <worker_id> <total_workers> <task_type>

task_type: posts | comments
"""

import sys
import json
import time
import requests
import boto3
from datetime import datetime, timezone

BASE_URL = "https://www.moltbook.com"
BUCKET = "moltbook-archive-319933937176"

s3 = boto3.client('s3')
session = requests.Session()
session.headers.update({
    'User-Agent': 'MoltbookArchiver/3.0 (Distributed Worker)',
    'Accept': 'application/json',
})

def fetch_json(endpoint, retries=3):
    """Fetch with retries"""
    for attempt in range(retries):
        try:
            resp = session.get(f"{BASE_URL}{endpoint}", timeout=30)
            if resp.status_code == 200:
                return resp.json()
            time.sleep(0.5)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
    return None

def upload_to_s3(data, key):
    """Upload JSON to S3"""
    s3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=json.dumps(data, ensure_ascii=False),
        ContentType='application/json'
    )

def fetch_posts_chunk(worker_id, total_workers):
    """Fetch a chunk of posts based on worker ID"""
    print(f"Worker {worker_id}/{total_workers}: Fetching posts...")

    # First, get total count by fetching first page
    first = fetch_json('/api/v1/posts?sort=new&offset=0')
    if not first:
        print("Failed to get first page")
        return

    # Estimate total (we'll keep going until has_more=False)
    # Each worker handles offsets where offset % total_workers == worker_id

    all_posts = []
    offset = worker_id * 25  # Start at our chunk
    step = total_workers * 25  # Skip other workers' chunks

    while True:
        data = fetch_json(f'/api/v1/posts?sort=new&offset={offset}')
        if not data or not data.get('posts'):
            break

        posts = data.get('posts', [])
        all_posts.extend(posts)

        if len(all_posts) % 500 == 0:
            print(f"  Worker {worker_id}: {len(all_posts)} posts at offset {offset}")

        if not data.get('has_more', False):
            # Check if we've gone past the end
            break

        offset += step
        time.sleep(0.05)  # Be nice

    print(f"Worker {worker_id}: Fetched {len(all_posts)} posts, uploading...")
    upload_to_s3(all_posts, f'posts/worker_{worker_id}.json')
    print(f"Worker {worker_id}: Done!")
    return len(all_posts)

def fetch_comments_for_posts(worker_id, total_workers, post_ids):
    """Fetch comments for assigned posts"""
    print(f"Worker {worker_id}/{total_workers}: Fetching comments for {len(post_ids)} posts...")

    # Worker handles posts where index % total_workers == worker_id
    my_posts = [pid for i, pid in enumerate(post_ids) if i % total_workers == worker_id]

    all_comments = []
    for i, post_id in enumerate(my_posts):
        detail = fetch_json(f'/api/v1/posts/{post_id}')
        if detail and detail.get('comments'):
            comments = detail.get('comments', [])
            for c in comments:
                c['_post_id'] = post_id
            all_comments.extend(comments)

        if (i + 1) % 100 == 0:
            print(f"  Worker {worker_id}: {i+1}/{len(my_posts)} posts, {len(all_comments)} comments")

        time.sleep(0.03)

    print(f"Worker {worker_id}: Fetched {len(all_comments)} comments, uploading...")
    upload_to_s3(all_comments, f'comments/worker_{worker_id}.json')
    print(f"Worker {worker_id}: Done!")
    return len(all_comments)

def main():
    if len(sys.argv) < 4:
        print("Usage: python worker.py <worker_id> <total_workers> <task_type> [post_ids_file]")
        sys.exit(1)

    worker_id = int(sys.argv[1])
    total_workers = int(sys.argv[2])
    task_type = sys.argv[3]

    print(f"Starting worker {worker_id}/{total_workers} for task: {task_type}")
    print(f"Time: {datetime.now(timezone.utc).isoformat()}")

    if task_type == 'posts':
        count = fetch_posts_chunk(worker_id, total_workers)
    elif task_type == 'comments':
        # Download post IDs from S3
        try:
            obj = s3.get_object(Bucket=BUCKET, Key='post_ids.json')
            post_ids = json.loads(obj['Body'].read())
            count = fetch_comments_for_posts(worker_id, total_workers, post_ids)
        except Exception as e:
            print(f"Error getting post IDs: {e}")
            sys.exit(1)
    else:
        print(f"Unknown task type: {task_type}")
        sys.exit(1)

    # Write completion marker
    upload_to_s3({
        'worker_id': worker_id,
        'task_type': task_type,
        'count': count,
        'completed_at': datetime.now(timezone.utc).isoformat()
    }, f'status/worker_{worker_id}_{task_type}_done.json')

if __name__ == '__main__':
    main()
