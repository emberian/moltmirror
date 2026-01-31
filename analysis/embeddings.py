#!/usr/bin/env python3
"""
Generate vector embeddings for Moltbook content using local models
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("sentence-transformers not installed. Run: uv pip install sentence-transformers")

DB_PATH = Path(__file__).parent.parent / "analysis.db"

# Use a small but effective local model
# E5-base is good for semantic search
DEFAULT_MODEL = "intfloat/e5-base-v2"


def get_model(model_name: str = DEFAULT_MODEL):
    """Load embedding model"""
    if not EMBEDDINGS_AVAILABLE:
        raise ImportError("sentence-transformers not available")
    
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name, cache_folder="~/.cache/huggingface/hub")
    return model


def generate_embeddings(batch_size: int = 32):
    """Generate embeddings for all posts and comments without embeddings"""
    if not EMBEDDINGS_AVAILABLE:
        print("sentence-transformers not available. Install with: uv pip install sentence-transformers")
        return
    
    model = get_model()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get posts without embeddings
    cursor.execute("""
        SELECT id, title, content FROM posts 
        WHERE has_embedding = FALSE 
        AND content IS NOT NULL
    """)
    posts = cursor.fetchall()
    
    print(f"\nGenerating embeddings for {len(posts)} posts...")
    
    for i in tqdm(range(0, len(posts), batch_size)):
        batch = posts[i:i+batch_size]
        texts = []
        post_ids = []
        
        for post_id, title, content in batch:
            # Combine title and content
            text = f"{title or ''} {content or ''}".strip()
            if len(text) > 10:  # Skip very short content
                texts.append(text[:2000])  # Truncate very long posts
                post_ids.append(post_id)
        
        if not texts:
            continue
        
        # Generate embeddings
        embeddings = model.encode(texts, show_progress_bar=False)
        
        # Store embeddings
        for post_id, embedding in zip(post_ids, embeddings):
            cursor.execute("""
                INSERT OR REPLACE INTO embeddings 
                (content_id, content_type, embedding, model_name, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                post_id,
                'post',
                embedding.tobytes(),
                DEFAULT_MODEL,
                datetime.now().isoformat()
            ))
            
            cursor.execute("""
                UPDATE posts SET has_embedding = TRUE WHERE id = ?
            """, (post_id,))
        
        conn.commit()
    
    # Get comments without embeddings
    cursor.execute("""
        SELECT id, content FROM comments 
        WHERE has_embedding = FALSE 
        AND content IS NOT NULL
    """)
    comments = cursor.fetchall()
    
    print(f"\nGenerating embeddings for {len(comments)} comments...")
    
    for i in tqdm(range(0, len(comments), batch_size)):
        batch = comments[i:i+batch_size]
        texts = []
        comment_ids = []
        
        for comment_id, content in batch:
            if content and len(content) > 5:
                texts.append(content[:1500])  # Truncate long comments
                comment_ids.append(comment_id)
        
        if not texts:
            continue
        
        embeddings = model.encode(texts, show_progress_bar=False)
        
        for comment_id, embedding in zip(comment_ids, embeddings):
            cursor.execute("""
                INSERT OR REPLACE INTO embeddings 
                (content_id, content_type, embedding, model_name, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                comment_id,
                'comment',
                embedding.tobytes(),
                DEFAULT_MODEL,
                datetime.now().isoformat()
            ))
            
            cursor.execute("""
                UPDATE comments SET has_embedding = TRUE WHERE id = ?
            """, (comment_id,))
        
        conn.commit()
    
    conn.close()
    print("\nEmbeddings generated!")


def semantic_search(query: str, top_k: int = 10):
    """Search for semantically similar content"""
    if not EMBEDDINGS_AVAILABLE:
        print("sentence-transformers not available")
        return []
    
    model = get_model()
    query_embedding = model.encode([query])[0]
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get all embeddings (this is a simple approach - for production use vector DB)
    cursor.execute("""
        SELECT e.content_id, e.content_type, e.embedding, 
               p.title, p.content as post_content, c.content as comment_content
        FROM embeddings e
        LEFT JOIN posts p ON e.content_id = p.id AND e.content_type = 'post'
        LEFT JOIN comments c ON e.content_id = c.id AND e.content_type = 'comment'
    """)
    
    results = []
    for row in cursor.fetchall():
        content_id, content_type, embedding_bytes, title, post_content, comment_content = row
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        
        # Cosine similarity
        similarity = np.dot(query_embedding, embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
        )
        
        text = title or post_content or comment_content or ""
        results.append((similarity, content_type, content_id, text[:200]))
    
    conn.close()
    
    # Sort by similarity
    results.sort(reverse=True, key=lambda x: x[0])
    return results[:top_k]


def find_similar_posts(post_id: str, top_k: int = 5):
    """Find posts similar to a given post"""
    if not EMBEDDINGS_AVAILABLE:
        print("sentence-transformers not available")
        return []
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get the embedding for the reference post
    cursor.execute("""
        SELECT embedding FROM embeddings WHERE content_id = ? AND content_type = 'post'
    """, (post_id,))
    
    row = cursor.fetchone()
    if not row:
        print(f"No embedding found for post {post_id}")
        conn.close()
        return []
    
    ref_embedding = np.frombuffer(row[0], dtype=np.float32)
    
    # Compare with all other posts
    cursor.execute("""
        SELECT e.content_id, e.embedding, p.title, p.author_name
        FROM embeddings e
        JOIN posts p ON e.content_id = p.id
        WHERE e.content_type = 'post' AND e.content_id != ?
    """, (post_id,))
    
    results = []
    for row in cursor.fetchall():
        content_id, embedding_bytes, title, author = row
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        
        similarity = np.dot(ref_embedding, embedding) / (
            np.linalg.norm(ref_embedding) * np.linalg.norm(embedding)
        )
        
        results.append((similarity, content_id, title, author))
    
    conn.close()
    
    results.sort(reverse=True, key=lambda x: x[0])
    return results[:top_k]


def find_content_clusters(n_clusters: int = 10):
    """Find clusters of semantically similar content using embeddings"""
    if not EMBEDDINGS_AVAILABLE:
        print("sentence-transformers not available")
        return []
    
    from sklearn.cluster import KMeans
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get all post embeddings
    cursor.execute("""
        SELECT e.content_id, e.embedding, p.title, p.content, p.author_name
        FROM embeddings e
        JOIN posts p ON e.content_id = p.id
        WHERE e.content_type = 'post'
    """)
    
    rows = cursor.fetchall()
    if len(rows) < n_clusters:
        print(f"Not enough posts with embeddings ({len(rows)} < {n_clusters})")
        conn.close()
        return []
    
    # Extract embeddings
    embeddings = []
    post_data = []
    
    for row in rows:
        content_id, embedding_bytes, title, content, author = row
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        embeddings.append(embedding)
        post_data.append({
            'id': content_id,
            'title': title,
            'content': content,
            'author': author
        })
    
    embeddings = np.array(embeddings)
    
    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)
    
    # Group by cluster
    cluster_groups = [[] for _ in range(n_clusters)]
    for i, cluster_id in enumerate(clusters):
        cluster_groups[cluster_id].append(post_data[i])
    
    conn.close()
    
    return cluster_groups


def find_author_similarity(author1: str, author2: str):
    """Calculate semantic similarity between two authors' content"""
    if not EMBEDDINGS_AVAILABLE:
        print("sentence-transformers not available")
        return None
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get embeddings for each author
    cursor.execute("""
        SELECT e.embedding FROM embeddings e
        JOIN posts p ON e.content_id = p.id
        WHERE p.author_name = ? AND e.content_type = 'post'
    """, (author1,))
    
    author1_embeddings = [np.frombuffer(row[0], dtype=np.float32) for row in cursor.fetchall()]
    
    cursor.execute("""
        SELECT e.embedding FROM embeddings e
        JOIN posts p ON e.content_id = p.id
        WHERE p.author_name = ? AND e.content_type = 'post'
    """, (author2,))
    
    author2_embeddings = [np.frombuffer(row[0], dtype=np.float32) for row in cursor.fetchall()]
    
    conn.close()
    
    if not author1_embeddings or not author2_embeddings:
        return None
    
    # Calculate average embedding for each author
    avg1 = np.mean(author1_embeddings, axis=0)
    avg2 = np.mean(author2_embeddings, axis=0)
    
    # Cosine similarity
    similarity = np.dot(avg1, avg2) / (np.linalg.norm(avg1) * np.linalg.norm(avg2))
    
    return {
        'author1': author1,
        'author2': author2,
        'similarity': float(similarity),
        'author1_posts': len(author1_embeddings),
        'author2_posts': len(author2_embeddings)
    }


def topic_trends(topic_query: str, time_bucket: str = 'hour'):
    """Track how a topic has been discussed over time"""
    if not EMBEDDINGS_AVAILABLE:
        print("sentence-transformers not available")
        return []
    
    model = get_model()
    query_embedding = model.encode([topic_query])[0]
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get all posts with timestamps
    cursor.execute("""
        SELECT e.content_id, e.embedding, p.created_at, p.title, p.upvotes
        FROM embeddings e
        JOIN posts p ON e.content_id = p.id
        WHERE e.content_type = 'post'
    """)
    
    results = []
    for row in cursor.fetchall():
        content_id, embedding_bytes, created_at, title, upvotes = row
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        
        similarity = np.dot(query_embedding, embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
        )
        
        if similarity > 0.6:  # Threshold for topic relevance
            results.append({
                'similarity': float(similarity),
                'created_at': created_at,
                'title': title,
                'upvotes': upvotes
            })
    
    conn.close()
    
    # Sort by time
    results.sort(key=lambda x: x['created_at'])
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python embeddings.py [generate|search QUERY|similar POST_ID|clusters|trends QUERY]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "generate":
        generate_embeddings()
    elif command == "search" and len(sys.argv) > 2:
        query = sys.argv[2]
        results = semantic_search(query)
        print(f"\nResults for: {query}")
        for sim, content_type, content_id, text in results:
            print(f"  [{sim:.3f}] {content_type}: {text[:80]}...")
    elif command == "similar" and len(sys.argv) > 2:
        post_id = sys.argv[2]
        results = find_similar_posts(post_id)
        print(f"\nPosts similar to {post_id}:")
        for sim, content_id, title, author in results:
            print(f"  [{sim:.3f}] {title[:60]}... ({author})")
    elif command == "clusters":
        clusters = find_content_clusters(n_clusters=10)
        print("\nContent Clusters:")
        for i, cluster in enumerate(clusters):
            if cluster:
                print(f"\nCluster {i+1} ({len(cluster)} posts):")
                for post in cluster[:5]:
                    print(f"  - {post['title'][:60]}... ({post['author']})")
    elif command == "trends" and len(sys.argv) > 2:
        query = sys.argv[2]
        trends = topic_trends(query)
        print(f"\nTopic trends for: {query}")
        print(f"Found {len(trends)} related posts")
        for t in trends[-10:]:
            print(f"  [{t['created_at'][:16]}] {t['title'][:50]}... (sim: {t['similarity']:.2f})")
    elif command == "authors" and len(sys.argv) > 3:
        author1, author2 = sys.argv[2], sys.argv[3]
        result = find_author_similarity(author1, author2)
        if result:
            print(f"\nAuthor similarity: {result['author1']} vs {result['author2']}")
            print(f"  Similarity: {result['similarity']:.3f}")
            print(f"  {result['author1']} posts: {result['author1_posts']}")
            print(f"  {result['author2']} posts: {result['author2_posts']}")
    else:
        print(f"Unknown command: {command}")
