#!/usr/bin/env python3
"""
Moltbook Analysis Tools
Vector embeddings, pattern analysis, and coordination research
"""

import json
import sqlite3
from pathlib import Path
from typing import Iterator, Optional
import numpy as np
from tqdm import tqdm

ARCHIVE_DIR = Path(__file__).parent.parent / "archive" / "data"
DB_PATH = Path(__file__).parent.parent / "analysis.db"

# Ensure archive directory exists
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


def init_database():
    """Initialize SQLite database with vector extension support"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Posts table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            id TEXT PRIMARY KEY,
            title TEXT,
            content TEXT,
            author_id TEXT,
            author_name TEXT,
            submolt TEXT,
            upvotes INTEGER,
            downvotes INTEGER,
            comment_count INTEGER,
            created_at TEXT,
            has_embedding BOOLEAN DEFAULT FALSE
        )
    """)
    
    # Comments table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS comments (
            id TEXT PRIMARY KEY,
            post_id TEXT,
            content TEXT,
            author_id TEXT,
            author_name TEXT,
            parent_id TEXT,
            upvotes INTEGER,
            downvotes INTEGER,
            created_at TEXT,
            has_embedding BOOLEAN DEFAULT FALSE,
            FOREIGN KEY (post_id) REFERENCES posts(id)
        )
    """)
    
    # Agents table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS agents (
            id TEXT PRIMARY KEY,
            name TEXT,
            description TEXT,
            karma INTEGER,
            follower_count INTEGER,
            following_count INTEGER,
            post_count INTEGER DEFAULT 0,
            comment_count INTEGER DEFAULT 0,
            avg_upvotes REAL DEFAULT 0.0,
            first_seen TEXT,
            last_active TEXT
        )
    """)
    
    # Embeddings table (using sqlite-vec for vector search)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            content_id TEXT PRIMARY KEY,
            content_type TEXT,  -- 'post' or 'comment'
            embedding BLOB,     -- numpy array as bytes
            model_name TEXT,
            created_at TEXT
        )
    """)
    
    # Coordination patterns table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS coordination_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_type TEXT,
            description TEXT,
            related_posts TEXT,  -- JSON array of post IDs
            confidence REAL,
            discovered_at TEXT
        )
    """)

    # Content hashes for exact duplicate detection
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS content_hashes (
            content_id TEXT PRIMARY KEY,
            content_type TEXT,
            content_hash TEXT,
            created_at TEXT
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON content_hashes(content_hash)")

    # Duplicate clusters (groups of similar content)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS duplicate_clusters (
            cluster_id TEXT PRIMARY KEY,
            canonical_id TEXT,
            content_type TEXT,
            duplicate_ids TEXT,
            similarity_score REAL,
            detected_at TEXT
        )
    """)

    # Spam scores per post/comment
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS spam_scores (
            content_id TEXT PRIMARY KEY,
            content_type TEXT,
            spam_score REAL,
            reasons TEXT,
            computed_at TEXT
        )
    """)
    
    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")


def load_json_files(pattern: str) -> Iterator[dict]:
    """Load all JSON files matching a pattern from archive"""
    for file_path in ARCHIVE_DIR.glob(pattern):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                yield data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")


def import_posts():
    """Import posts from archive into database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    count = 0
    
    # Try the complete dataset files first
    complete_files = [
        "posts_all_complete.json",
        "moltbook_final.json", 
        "moltbook_complete.json",
        "posts_all.json"
    ]
    
    posts_data = []
    for filename in complete_files:
        filepath = ARCHIVE_DIR / filename
        if filepath.exists():
            print(f"Loading posts from {filename}...")
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and 'posts' in data:
                        posts_data = data['posts']
                        break
                    elif isinstance(data, list):
                        posts_data = data
                        break
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
    
    # Fall back to individual post files
    if not posts_data:
        print("Loading from individual post files...")
        posts_dir = ARCHIVE_DIR / "posts"
        if posts_dir.exists():
            for file_path in tqdm(list(posts_dir.glob("*.json"))[:1000], desc="Loading posts"):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            posts_data.append(data)
                except Exception as e:
                    continue
    
    print(f"Importing {len(posts_data)} posts...")
    
    for post in tqdm(posts_data, desc="Importing"):
            cursor.execute("""
                INSERT OR REPLACE INTO posts 
                (id, title, content, author_id, author_name, submolt, 
                 upvotes, downvotes, comment_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                post.get('id'),
                post.get('title'),
                post.get('content'),
                post.get('author_id'),
                post.get('author', {}).get('name') if isinstance(post.get('author'), dict) else None,
                post.get('submolt', {}).get('name') if isinstance(post.get('submolt'), dict) else None,
                post.get('upvotes', 0),
                post.get('downvotes', 0),
                post.get('comment_count', 0),
                post.get('created_at')
            ))
            count += 1
    
    conn.commit()
    conn.close()
    print(f"Imported {count} posts")


def import_comments():
    """Import comments from archive into database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    count = 0
    
    # Try the complete dataset files first
    complete_files = [
        "comments_all_complete.json",
        "comments_all.json"
    ]
    
    comments_data = []
    for filename in complete_files:
        filepath = ARCHIVE_DIR / filename
        if filepath.exists():
            print(f"Loading comments from {filename}...")
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        comments_data = data
                        break
                    elif isinstance(data, dict):
                        # Flatten dict of comments by post
                        for post_id, comments in data.items():
                            if isinstance(comments, list):
                                for c in comments:
                                    c['post_id'] = post_id
                                comments_data.extend(comments)
                        break
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
    
    print(f"Importing {len(comments_data)} comments...")
    
    for comment in tqdm(comments_data, desc="Importing"):
        if not isinstance(comment, dict):
            continue
        
        cursor.execute("""
            INSERT OR REPLACE INTO comments 
            (id, post_id, content, author_id, author_name, parent_id,
             upvotes, downvotes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            comment.get('id'),
            comment.get('post_id') or comment.get('_post_id'),
            comment.get('content'),
            comment.get('author', {}).get('id') if isinstance(comment.get('author'), dict) else None,
            comment.get('author', {}).get('name') if isinstance(comment.get('author'), dict) else None,
            comment.get('parent_id'),
            comment.get('upvotes', 0),
            comment.get('downvotes', 0),
            comment.get('created_at')
        ))
        count += 1
    
    conn.commit()
    conn.close()
    print(f"Imported {count} comments")


def import_agents():
    """Import/update agent statistics"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get all unique authors from posts and comments
    cursor.execute("""
        SELECT author_id, author_name, 
               COUNT(*) as post_count,
               AVG(upvotes) as avg_upvotes,
               MIN(created_at) as first_seen,
               MAX(created_at) as last_active
        FROM posts 
        WHERE author_id IS NOT NULL
        GROUP BY author_id
    """)
    
    for row in cursor.fetchall():
        cursor.execute("""
            INSERT OR REPLACE INTO agents 
            (id, name, post_count, avg_upvotes, first_seen, last_active)
            VALUES (?, ?, ?, ?, ?, ?)
        """, row)
    
    # Update comment counts
    cursor.execute("""
        SELECT author_id, COUNT(*) as comment_count
        FROM comments 
        WHERE author_id IS NOT NULL
        GROUP BY author_id
    """)
    
    for author_id, comment_count in cursor.fetchall():
        cursor.execute("""
            UPDATE agents 
            SET comment_count = ?
            WHERE id = ?
        """, (comment_count, author_id))
    
    conn.commit()
    conn.close()
    print("Updated agent statistics")


def get_high_signal_agents(min_posts: int = 3, min_avg_upvotes: float = 2.0) -> list:
    """Find agents with consistent high-quality output"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, name, post_count, comment_count, avg_upvotes,
               (post_count + comment_count) as total_contributions
        FROM agents
        WHERE post_count >= ? AND avg_upvotes >= ?
        ORDER BY avg_upvotes DESC, total_contributions DESC
        LIMIT 50
    """, (min_posts, min_avg_upvotes))
    
    results = cursor.fetchall()
    conn.close()
    return results


def search_coordination_topics() -> list:
    """Find posts about coordination, collaboration, multi-agent"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    keywords = ['coordination', 'collaboration', 'multi-agent', 'consensus', 
                'cooperat', 'governance', 'protocol', 'incentive']
    
    pattern = ' OR '.join([f"title LIKE '%{k}%' OR content LIKE '%{k}%'" 
                          for k in keywords])
    
    cursor.execute(f"""
        SELECT id, title, author_name, upvotes, comment_count
        FROM posts
        WHERE {pattern}
        ORDER BY upvotes DESC, comment_count DESC
        LIMIT 30
    """)
    
    results = cursor.fetchall()
    conn.close()
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analysis.py [init|import|agents|coordination]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "init":
        init_database()
    elif command == "import":
        import_posts()
        import_comments()
        import_agents()
    elif command == "agents":
        agents = get_high_signal_agents()
        print("\nHigh-signal agents (min 3 posts, avg 2+ upvotes):")
        for agent in agents[:20]:
            print(f"  {agent[1]}: {agent[2]} posts, {agent[3]} comments, {agent[4]:.1f} avg upvotes")
    elif command == "coordination":
        posts = search_coordination_topics()
        print("\nCoordination-related posts:")
        for post in posts[:15]:
            print(f"  {post[1][:60]}... ({post[2]}, {post[3]} upvotes)")
    else:
        print(f"Unknown command: {command}")
