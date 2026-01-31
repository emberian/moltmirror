"""
Moltbook Analysis API
FastAPI backend for semantic search and analysis
"""

from fastapi import FastAPI, HTTPException, Query, Depends, Header
from contextlib import asynccontextmanager
import secrets
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import sqlite3
import numpy as np
from pathlib import Path
import os
from datetime import datetime

# Import background analysis
try:
    from background_analysis import get_analyzer, start_background_analysis
    BACKGROUND_AVAILABLE = True
except ImportError:
    BACKGROUND_AVAILABLE = False

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

# Lifespan context manager (replaces deprecated on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    if BACKGROUND_AVAILABLE:
        start_background_analysis()
        print("Background analysis started")
    yield
    # Shutdown (if needed)
    print("Shutting down...")

app = FastAPI(
    title="Moltbook Analysis API",
    description="Semantic search and analysis for Moltbook content",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config
DB_PATH = Path(os.getenv("MOLTMIRROR_DB_PATH", "analysis.db"))
MODEL_NAME = os.getenv("MOLTMIRROR_MODEL", "intfloat/e5-base-v2")
ADMIN_API_KEY = os.getenv("MOLTMIRROR_ADMIN_KEY", "")  # Set via environment variable

# Admin authentication
async def verify_admin_key(x_admin_key: str = Header(None, alias="X-Admin-Key")):
    """Verify admin API key for protected endpoints"""
    if not ADMIN_API_KEY:
        raise HTTPException(status_code=503, detail="Admin API not configured")
    if not x_admin_key or not secrets.compare_digest(x_admin_key, ADMIN_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing admin key")

# Lazy-loaded model
_model = None

def get_model():
    global _model
    if _model is None and EMBEDDINGS_AVAILABLE:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def get_db():
    return sqlite3.connect(DB_PATH)

# Pydantic models
class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    content_type: Optional[str] = None  # 'post', 'comment', or None for both
    despam: bool = True  # Filter out spam and duplicates by default

class SearchResult(BaseModel):
    similarity: float
    content_type: str
    content_id: str
    title: Optional[str]
    content: str
    author: Optional[str]
    upvotes: int
    created_at: Optional[str]

class CompareRequest(BaseModel):
    author1: str
    author2: str

class AuthorComparison(BaseModel):
    author1: str
    author2: str
    similarity: float
    author1_posts: int
    author2_posts: int

class TrendsRequest(BaseModel):
    query: str
    threshold: float = 0.6

class TrendPoint(BaseModel):
    similarity: float
    created_at: str
    title: str
    upvotes: int

class Stats(BaseModel):
    posts: int
    comments: int
    embeddings: int
    total_upvotes: int

@app.get("/api")
async def api_root():
    return {"message": "Moltbook Analysis API", "docs": "/docs", "version": "1.0.0"}

@app.get("/api/stats", response_model=Stats)
async def get_stats():
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM posts")
    posts = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM comments")
    comments = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM embeddings")
    embeddings = cursor.fetchone()[0]
    
    cursor.execute("SELECT SUM(upvotes) FROM posts")
    total_upvotes = cursor.fetchone()[0] or 0
    
    conn.close()
    
    return Stats(
        posts=posts,
        comments=comments,
        embeddings=embeddings,
        total_upvotes=total_upvotes
    )

@app.post("/api/search", response_model=List[SearchResult])
async def search(request: SearchRequest):
    if not EMBEDDINGS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Embeddings not available")
    
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    query_embedding = model.encode([request.query])[0]
    
    conn = get_db()
    cursor = conn.cursor()
    
    # Validate content_type to prevent SQL injection
    params = []
    if request.content_type:
        if request.content_type not in ('post', 'comment'):
            raise HTTPException(status_code=400, detail="content_type must be 'post' or 'comment'")

    # Load spam scores if despam is enabled
    spam_scores = {}
    if request.despam:
        cursor.execute("SELECT content_id, spam_score FROM spam_scores WHERE spam_score >= 50")
        spam_scores = {row[0]: row[1] for row in cursor.fetchall()}

    # Build query with parameterized WHERE clause
    sql = """
        SELECT e.content_id, e.content_type, e.embedding,
               p.title, p.content as post_content, c.content as comment_content,
               p.author_name, p.upvotes, p.created_at,
               c.author_name as comment_author, c.upvotes as comment_upvotes, c.created_at as comment_created
        FROM embeddings e
        LEFT JOIN posts p ON e.content_id = p.id AND e.content_type = 'post'
        LEFT JOIN comments c ON e.content_id = c.id AND e.content_type = 'comment'
    """

    if request.content_type:
        sql += " WHERE e.content_type = ?"
        params.append(request.content_type)

    cursor.execute(sql, params)
    
    results = []
    for row in cursor.fetchall():
        content_id, content_type, embedding_bytes = row[0], row[1], row[2]

        # Skip high-spam content if despam is enabled
        if request.despam and content_id in spam_scores:
            continue

        if content_type == 'post':
            # For posts: title=row[3], content=row[4], author=row[6], upvotes=row[7], created_at=row[8]
            title, content = row[3], row[4]
            author, upvotes, created_at = row[6], row[7], row[8]
        else:
            # For comments: no title, content=row[5], author=row[9], upvotes=row[10], created_at=row[11]
            title, content = None, row[5]
            author, upvotes, created_at = row[9], row[10], row[11]

        if not content:
            continue
        
        # Ensure upvotes is an integer
        try:
            upvotes = int(upvotes) if upvotes is not None else 0
        except (ValueError, TypeError):
            upvotes = 0
        
        # Ensure created_at is a string or None
        if created_at is not None and not isinstance(created_at, str):
            created_at = str(created_at)
        
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        
        similarity = np.dot(query_embedding, embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
        )
        
        results.append(SearchResult(
            similarity=float(similarity),
            content_type=content_type,
            content_id=content_id,
            title=title,
            content=content[:500] if content else "",
            author=author,
            upvotes=upvotes,
            created_at=created_at
        ))
    
    conn.close()

    # Sort by similarity
    results.sort(key=lambda x: x.similarity, reverse=True)

    # Deduplicate when despam is enabled
    if request.despam:
        seen_content = set()
        deduplicated = []
        for r in results:
            # Create a content fingerprint (normalize whitespace, lowercase)
            content_key = ' '.join(r.content.lower().split())[:200]
            if content_key not in seen_content:
                seen_content.add(content_key)
                deduplicated.append(r)
        results = deduplicated

    return results[:request.top_k]

@app.get("/api/similar/{post_id}", response_model=List[SearchResult])
async def find_similar(post_id: str, top_k: int = 5):
    if not EMBEDDINGS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Embeddings not available")
    
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    conn = get_db()
    cursor = conn.cursor()
    
    # Get reference embedding
    cursor.execute("""
        SELECT embedding FROM embeddings 
        WHERE content_id = ? AND content_type = 'post'
    """, (post_id,))
    
    row = cursor.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Post not found or not embedded")
    
    ref_embedding = np.frombuffer(row[0], dtype=np.float32)
    
    # Find similar
    cursor.execute("""
        SELECT e.content_id, e.embedding, p.title, p.content, p.author_name, p.upvotes, p.created_at
        FROM embeddings e
        JOIN posts p ON e.content_id = p.id
        WHERE e.content_type = 'post' AND e.content_id != ?
    """, (post_id,))
    
    results = []
    for row in cursor.fetchall():
        content_id, embedding_bytes, title, content, author, upvotes, created_at = row
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        
        similarity = np.dot(ref_embedding, embedding) / (
            np.linalg.norm(ref_embedding) * np.linalg.norm(embedding)
        )
        
        results.append(SearchResult(
            similarity=float(similarity),
            content_type='post',
            content_id=content_id,
            title=title,
            content=content[:500] if content else "",
            author=author,
            upvotes=upvotes or 0,
            created_at=created_at
        ))
    
    conn.close()
    
    results.sort(key=lambda x: x.similarity, reverse=True)
    return results[:top_k]

@app.post("/api/trends", response_model=List[TrendPoint])
async def get_trends(request: TrendsRequest):
    if not EMBEDDINGS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Embeddings not available")
    
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    query_embedding = model.encode([request.query])[0]
    
    conn = get_db()
    cursor = conn.cursor()
    
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
        
        if similarity > request.threshold:
            results.append(TrendPoint(
                similarity=float(similarity),
                created_at=created_at,
                title=title[:100] if title else "",
                upvotes=upvotes or 0
            ))
    
    conn.close()
    
    results.sort(key=lambda x: x.created_at)
    return results

@app.get("/api/agents")
async def get_agents(min_posts: int = 3, min_avg_upvotes: float = 2.0):
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT author_name, COUNT(*) as posts, AVG(upvotes) as avg_up, SUM(upvotes) as total_up
        FROM posts 
        WHERE author_name IS NOT NULL 
        GROUP BY author_name 
        HAVING posts >= ? AND avg_up >= ?
        ORDER BY avg_up DESC
    """, (min_posts, min_avg_upvotes))
    
    results = []
    for row in cursor.fetchall():
        results.append({
            "author": row[0],
            "posts": row[1],
            "avg_upvotes": round(row[2], 1),
            "total_upvotes": row[3]
        })
    
    conn.close()
    return results

@app.get("/api/agent/{author_name}")
async def get_agent_profile(author_name: str):
    """Get detailed profile for a specific agent"""
    conn = get_db()
    cursor = conn.cursor()

    # Basic stats
    cursor.execute("""
        SELECT
            COUNT(*) as post_count,
            AVG(upvotes) as avg_upvotes,
            SUM(upvotes) as total_upvotes,
            AVG(comment_count) as avg_comments,
            MIN(created_at) as first_post,
            MAX(created_at) as last_post
        FROM posts
        WHERE author_name = ?
    """, (author_name,))

    stats_row = cursor.fetchone()
    if not stats_row or stats_row[0] == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="Agent not found")

    # Comment count
    cursor.execute("""
        SELECT COUNT(*) FROM comments WHERE author_name = ?
    """, (author_name,))
    comment_count = cursor.fetchone()[0]

    # Top posts by upvotes
    cursor.execute("""
        SELECT id, title, upvotes, comment_count, created_at
        FROM posts
        WHERE author_name = ?
        ORDER BY upvotes DESC
        LIMIT 10
    """, (author_name,))

    top_posts = [
        {
            'id': row[0],
            'title': row[1][:100] if row[1] else '',
            'upvotes': row[2],
            'comments': row[3],
            'created_at': row[4]
        }
        for row in cursor.fetchall()
    ]

    # Recent activity (posts per week for last 4 weeks)
    cursor.execute("""
        SELECT
            strftime('%Y-%W', created_at) as week,
            COUNT(*) as posts,
            SUM(upvotes) as upvotes
        FROM posts
        WHERE author_name = ?
        AND created_at > datetime('now', '-28 days')
        GROUP BY week
        ORDER BY week DESC
    """, (author_name,))

    activity_timeline = [
        {'week': row[0], 'posts': row[1], 'upvotes': row[2]}
        for row in cursor.fetchall()
    ]

    # Topic distribution (submolts)
    cursor.execute("""
        SELECT submolt, COUNT(*) as count
        FROM posts
        WHERE author_name = ? AND submolt IS NOT NULL
        GROUP BY submolt
        ORDER BY count DESC
        LIMIT 10
    """, (author_name,))

    topics = [{'topic': row[0], 'count': row[1]} for row in cursor.fetchall()]

    # Network - who they interact with most
    cursor.execute("""
        SELECT p.author_name, COUNT(*) as interactions
        FROM comments c
        JOIN posts p ON c.post_id = p.id
        WHERE c.author_name = ? AND p.author_name != ?
        AND p.author_name IS NOT NULL
        GROUP BY p.author_name
        ORDER BY interactions DESC
        LIMIT 10
    """, (author_name, author_name))

    connections = [{'author': row[0], 'interactions': row[1]} for row in cursor.fetchall()]

    conn.close()

    return {
        'author': author_name,
        'stats': {
            'posts': stats_row[0],
            'comments': comment_count,
            'avg_upvotes': round(stats_row[1] or 0, 1),
            'total_upvotes': stats_row[2] or 0,
            'avg_comments': round(stats_row[3] or 0, 1),
            'first_post': stats_row[4],
            'last_post': stats_row[5]
        },
        'top_posts': top_posts,
        'activity_timeline': activity_timeline,
        'topics': topics,
        'connections': connections
    }

@app.post("/api/compare", response_model=AuthorComparison)
async def compare_authors(request: CompareRequest):
    if not EMBEDDINGS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Embeddings not available")
    
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    conn = get_db()
    cursor = conn.cursor()
    
    # Get embeddings for author1
    cursor.execute("""
        SELECT e.embedding FROM embeddings e
        JOIN posts p ON e.content_id = p.id
        WHERE p.author_name = ? AND e.content_type = 'post'
    """, (request.author1,))
    
    author1_embeddings = [np.frombuffer(row[0], dtype=np.float32) for row in cursor.fetchall()]
    
    # Get embeddings for author2
    cursor.execute("""
        SELECT e.embedding FROM embeddings e
        JOIN posts p ON e.content_id = p.id
        WHERE p.author_name = ? AND e.content_type = 'post'
    """, (request.author2,))
    
    author2_embeddings = [np.frombuffer(row[0], dtype=np.float32) for row in cursor.fetchall()]
    
    conn.close()
    
    if not author1_embeddings or not author2_embeddings:
        raise HTTPException(status_code=404, detail="One or both authors not found")
    
    # Calculate average embeddings
    avg1 = np.mean(author1_embeddings, axis=0)
    avg2 = np.mean(author2_embeddings, axis=0)
    
    similarity = np.dot(avg1, avg2) / (np.linalg.norm(avg1) * np.linalg.norm(avg2))
    
    return AuthorComparison(
        author1=request.author1,
        author2=request.author2,
        similarity=float(similarity),
        author1_posts=len(author1_embeddings),
        author2_posts=len(author2_embeddings)
    )

# Background analysis endpoints
@app.get("/api/insights/trending")
async def get_trending_agents():
    """Get agents with rising momentum"""
    if not BACKGROUND_AVAILABLE:
        raise HTTPException(status_code=503, detail="Background analysis not available")
    analyzer = get_analyzer()
    if 'trending_agents' in analyzer.insights_cache:
        return analyzer.insights_cache['trending_agents']
    raise HTTPException(status_code=503, detail="Analysis not yet complete")

@app.get("/api/insights/discussions")
async def get_hot_discussions():
    """Get high-engagement discussion clusters"""
    if not BACKGROUND_AVAILABLE:
        raise HTTPException(status_code=503, detail="Background analysis not available")
    analyzer = get_analyzer()
    if 'conversation_clusters' in analyzer.insights_cache:
        return analyzer.insights_cache['conversation_clusters']
    raise HTTPException(status_code=503, detail="Analysis not yet complete")

@app.get("/api/insights/network")
async def get_network_connectors():
    """Get key network connectors"""
    if not BACKGROUND_AVAILABLE:
        raise HTTPException(status_code=503, detail="Background analysis not available")
    analyzer = get_analyzer()
    if 'network_centrality' in analyzer.insights_cache:
        return analyzer.insights_cache['network_centrality']
    raise HTTPException(status_code=503, detail="Analysis not yet complete")

@app.get("/api/insights/anomalies")
async def get_anomalies():
    """Get detected anomalies and viral content"""
    if not BACKGROUND_AVAILABLE:
        raise HTTPException(status_code=503, detail="Background analysis not available")
    analyzer = get_analyzer()
    if 'anomalies' in analyzer.insights_cache:
        return analyzer.insights_cache['anomalies']
    raise HTTPException(status_code=503, detail="Analysis not yet complete")

@app.get("/api/insights/opportunities")
async def get_content_opportunities():
    """Get content opportunities (underserved topics)"""
    if not BACKGROUND_AVAILABLE:
        raise HTTPException(status_code=503, detail="Background analysis not available")
    analyzer = get_analyzer()
    if 'content_gaps' in analyzer.insights_cache:
        return analyzer.insights_cache['content_gaps']
    raise HTTPException(status_code=503, detail="Analysis not yet complete")

@app.get("/api/insights/predictions")
async def get_predictions():
    """Get predicted hot topics"""
    if not BACKGROUND_AVAILABLE:
        raise HTTPException(status_code=503, detail="Background analysis not available")
    analyzer = get_analyzer()
    if 'hot_topics' in analyzer.insights_cache:
        return analyzer.insights_cache['hot_topics']
    raise HTTPException(status_code=503, detail="Analysis not yet complete")

@app.get("/api/insights/duplicates")
async def get_duplicates():
    """Get detected duplicate and near-duplicate content"""
    if not BACKGROUND_AVAILABLE:
        raise HTTPException(status_code=503, detail="Background analysis not available")
    analyzer = get_analyzer()
    if 'duplicates' in analyzer.insights_cache:
        return analyzer.insights_cache['duplicates']
    raise HTTPException(status_code=503, detail="Analysis not yet complete")

@app.get("/api/insights/spam")
async def get_spam_analysis():
    """Get spam scores and suspicious content"""
    if not BACKGROUND_AVAILABLE:
        raise HTTPException(status_code=503, detail="Background analysis not available")
    analyzer = get_analyzer()
    if 'spam_scores' in analyzer.insights_cache:
        return analyzer.insights_cache['spam_scores']
    raise HTTPException(status_code=503, detail="Analysis not yet complete")

@app.get("/api/insights/influence")
async def get_author_influence():
    """Get author influence analysis"""
    if not BACKGROUND_AVAILABLE:
        raise HTTPException(status_code=503, detail="Background analysis not available")
    analyzer = get_analyzer()
    if 'author_influence' in analyzer.insights_cache:
        return analyzer.insights_cache['author_influence']
    raise HTTPException(status_code=503, detail="Analysis not yet complete")

@app.get("/api/insights/topic-graph")
async def get_topic_graph():
    """Get topic relationship graph data"""
    if not BACKGROUND_AVAILABLE:
        raise HTTPException(status_code=503, detail="Background analysis not available")
    analyzer = get_analyzer()
    if 'topic_graph' in analyzer.insights_cache:
        return analyzer.insights_cache['topic_graph']
    raise HTTPException(status_code=503, detail="Analysis not yet complete")

@app.get("/api/insights/all")
async def get_all_insights():
    """Get all cached insights"""
    if not BACKGROUND_AVAILABLE:
        raise HTTPException(status_code=503, detail="Background analysis not available")
    analyzer = get_analyzer()
    return {
        'insights': analyzer.insights_cache,
        'last_analysis': {k: v.isoformat() if isinstance(v, datetime) else v
                         for k, v in analyzer.last_analysis.items()}
    }

@app.get("/api/status")
async def get_system_status():
    """Get comprehensive system status including sync and analysis info"""
    if not BACKGROUND_AVAILABLE:
        return {
            'status': 'limited',
            'message': 'Background analysis not available',
            'database': {'available': DB_PATH.exists()},
            'embeddings': {'available': EMBEDDINGS_AVAILABLE}
        }

    analyzer = get_analyzer()
    return analyzer.get_system_status()

@app.post("/api/admin/trigger-analysis", dependencies=[Depends(verify_admin_key)])
async def trigger_analysis():
    """Manually trigger analysis cycle (admin only, requires X-Admin-Key header)"""
    if not BACKGROUND_AVAILABLE:
        raise HTTPException(status_code=503, detail="Background analysis not available")
    analyzer = get_analyzer()
    analyzer.run_analysis_cycle()
    return {"status": "complete", "insights_count": len(analyzer.insights_cache)}

# Export endpoints
from fastapi.responses import Response
import csv
import io

@app.get("/api/export/search")
async def export_search(
    query: str,
    format: str = "json",
    top_k: int = 100,
    content_type: Optional[str] = None
):
    """Export search results as JSON or CSV"""
    if not EMBEDDINGS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Embeddings not available")

    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate content_type
    if content_type and content_type not in ('post', 'comment'):
        raise HTTPException(status_code=400, detail="content_type must be 'post' or 'comment'")

    query_embedding = model.encode([query])[0]

    conn = get_db()
    cursor = conn.cursor()

    params = []
    sql = """
        SELECT e.content_id, e.content_type, e.embedding,
               p.title, p.content as post_content, c.content as comment_content,
               p.author_name, p.upvotes, p.created_at,
               c.author_name as comment_author, c.upvotes as comment_upvotes, c.created_at as comment_created
        FROM embeddings e
        LEFT JOIN posts p ON e.content_id = p.id AND e.content_type = 'post'
        LEFT JOIN comments c ON e.content_id = c.id AND e.content_type = 'comment'
    """

    if content_type:
        sql += " WHERE e.content_type = ?"
        params.append(content_type)

    cursor.execute(sql, params)

    results = []
    for row in cursor.fetchall():
        content_id, ctype, embedding_bytes = row[0], row[1], row[2]

        if ctype == 'post':
            title, content = row[3], row[4]
            author, upvotes, created_at = row[6], row[7], row[8]
        else:
            title, content = None, row[5]
            author, upvotes, created_at = row[9], row[10], row[11]

        if not content:
            continue

        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        similarity = float(np.dot(query_embedding, embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
        ))

        results.append({
            'id': content_id,
            'type': ctype,
            'similarity': round(similarity, 4),
            'title': title or '',
            'content': content[:1000],
            'author': author or '',
            'upvotes': upvotes or 0,
            'created_at': created_at or ''
        })

    conn.close()
    results.sort(key=lambda x: x['similarity'], reverse=True)

    # Deduplicate: keep only the first (highest similarity) result for each unique content
    seen_content = set()
    deduplicated = []
    for r in results:
        content_key = ' '.join(r['content'].lower().split())[:200]
        if content_key not in seen_content:
            seen_content.add(content_key)
            deduplicated.append(r)
    results = deduplicated[:top_k]

    if format == "csv":
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=['id', 'type', 'similarity', 'title', 'author', 'upvotes', 'created_at', 'content'])
        writer.writeheader()
        writer.writerows(results)
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=search_{query[:20]}.csv"}
        )

    return results

@app.get("/api/export/agent/{author_name}")
async def export_agent(author_name: str, format: str = "json"):
    """Export all posts by an agent as JSON or CSV"""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, title, content, upvotes, comment_count, created_at, submolt
        FROM posts
        WHERE author_name = ?
        ORDER BY created_at DESC
    """, (author_name,))

    posts = [
        {
            'id': row[0],
            'title': row[1] or '',
            'content': row[2][:1000] if row[2] else '',
            'upvotes': row[3] or 0,
            'comments': row[4] or 0,
            'created_at': row[5] or '',
            'submolt': row[6] or ''
        }
        for row in cursor.fetchall()
    ]

    conn.close()

    if not posts:
        raise HTTPException(status_code=404, detail="Agent not found")

    if format == "csv":
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=['id', 'title', 'upvotes', 'comments', 'created_at', 'submolt', 'content'])
        writer.writeheader()
        writer.writerows(posts)
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={author_name}_posts.csv"}
        )

    return {'author': author_name, 'posts': posts, 'total': len(posts)}

# Health check with background status
@app.get("/api/health")
async def health_check():
    if not BACKGROUND_AVAILABLE:
        return {
            "status": "healthy",
            "background_analysis": "not_available",
            "message": "Background analysis module not loaded"
        }
    analyzer = get_analyzer()
    return {
        "status": "healthy",
        "background_analysis": "running" if analyzer.running else "idle",
        "insights_cached": len(analyzer.insights_cache),
        "last_analyses": {k: v.isoformat() if isinstance(v, datetime) else str(v)
                         for k, v in list(analyzer.last_analysis.items())[:3]}
    }

# Mount static files (frontend)
if Path("static").exists():
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
