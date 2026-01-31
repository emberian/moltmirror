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

# Import IC-grade analysis modules
try:
    from analysis.coordination import (
        detect_sockpuppets, detect_synchronized_posting,
        detect_coordination_clusters, get_sockpuppet_network
    )
    COORDINATION_AVAILABLE = True
except ImportError:
    COORDINATION_AVAILABLE = False

try:
    from analysis.fingerprints import (
        compute_fingerprint, save_fingerprint, load_fingerprint,
        compare_authors, find_similar_agents, detect_behavior_change
    )
    FINGERPRINTS_AVAILABLE = True
except ImportError:
    FINGERPRINTS_AVAILABLE = False

try:
    from analysis.graphs import (
        get_top_influencers, find_bridges, find_cliques, detect_communities,
        get_author_network_position, get_graph_summary
    )
    GRAPHS_AVAILABLE = True
except ImportError:
    GRAPHS_AVAILABLE = False

try:
    from analysis.temporal import (
        detect_activity_bursts, find_correlated_pairs,
        compute_circadian_profile, get_activity_heatmap
    )
    TEMPORAL_AVAILABLE = True
except ImportError:
    TEMPORAL_AVAILABLE = False

try:
    from analysis.narratives import (
        identify_narratives, get_narrative_timeline,
        detect_coordinated_pushes, classify_narrative_roles
    )
    NARRATIVES_AVAILABLE = True
except ImportError:
    NARRATIVES_AVAILABLE = False

try:
    from analysis.alerts import (
        get_active_alerts, get_alert_summary, get_high_risk_authors,
        compute_author_risk_score, get_author_alert_history, resolve_alert
    )
    ALERTS_AVAILABLE = True
except ImportError:
    ALERTS_AVAILABLE = False

# LLM-specific analysis modules
try:
    from analysis.llm_fingerprints import (
        compute_llm_fingerprint, load_llm_fingerprint, save_llm_fingerprint,
        compare_llm_fingerprints, find_similar_llm_agents, PromptTemplateDetector
    )
    LLM_FINGERPRINTS_AVAILABLE = True
except ImportError:
    LLM_FINGERPRINTS_AVAILABLE = False

try:
    from analysis.same_operator import (
        get_same_operator_candidates, get_operator_clusters,
        SameOperatorDetector
    )
    SAME_OPERATOR_AVAILABLE = True
except ImportError:
    SAME_OPERATOR_AVAILABLE = False

try:
    from analysis.information_flow import (
        get_laundering_events, get_circular_citations,
        PropagationTreeBuilder
    )
    INFORMATION_FLOW_AVAILABLE = True
except ImportError:
    INFORMATION_FLOW_AVAILABLE = False

try:
    from analysis.persona_consistency import (
        get_persona_profile, get_persona_shifts,
        PersonaTracker, BeliefConsistencyScorer
    )
    PERSONA_AVAILABLE = True
except ImportError:
    PERSONA_AVAILABLE = False

try:
    from analysis.authorship import (
        get_author_clusters, get_copy_chains,
        AuthorCentroidAnalyzer, CopyPasteChainDetector
    )
    AUTHORSHIP_AVAILABLE = True
except ImportError:
    AUTHORSHIP_AVAILABLE = False

try:
    from analysis.narratives import (
        build_propagation_tree, compute_originality_scores,
        track_narrative_mutations, attribute_original_source
    )
    NARRATIVES_ENHANCED = True
except ImportError:
    NARRATIVES_ENHANCED = False

try:
    from analysis.social_dynamics import (
        get_content_virality, get_tool_adoption, get_author_influence,
        ViralityAnalyzer, AdoptionTracker, InfluenceAnalyzer
    )
    SOCIAL_DYNAMICS_AVAILABLE = True
except ImportError:
    SOCIAL_DYNAMICS_AVAILABLE = False

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

@app.get("/api/embeddings/status")
async def get_embedding_status():
    """Get current embedding coverage status"""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM posts")
    total_posts = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM posts WHERE has_embedding = TRUE")
    embedded_posts = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM comments")
    total_comments = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM comments WHERE has_embedding = TRUE")
    embedded_comments = cursor.fetchone()[0]

    conn.close()

    total = total_posts + total_comments
    embedded = embedded_posts + embedded_comments
    coverage = (embedded / total * 100) if total > 0 else 0

    return {
        "posts": {"total": total_posts, "embedded": embedded_posts},
        "comments": {"total": total_comments, "embedded": embedded_comments},
        "total": {"total": total, "embedded": embedded},
        "coverage_percent": round(coverage, 1),
        "missing": total - embedded,
        "embeddings_available": EMBEDDINGS_AVAILABLE
    }

@app.post("/api/admin/generate-embeddings", dependencies=[Depends(verify_admin_key)])
async def trigger_embeddings():
    """Manually trigger embedding generation (admin only)"""
    if not EMBEDDINGS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Embeddings module not available")

    from analysis.embeddings import generate_embeddings
    generate_embeddings()

    # Return new status
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM embeddings")
    total_embeddings = cursor.fetchone()[0]
    conn.close()

    return {"status": "complete", "total_embeddings": total_embeddings}

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

# ============================================================================
# IC-Grade Coordination Detection Endpoints
# ============================================================================

@app.get("/api/insights/coordination")
async def get_coordination_alerts():
    """Get all coordination alerts"""
    if not ALERTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Alerts module not available")
    return get_active_alerts(limit=50)

@app.get("/api/insights/coordination/summary")
async def get_coordination_summary():
    """Get coordination alert summary"""
    if not ALERTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Alerts module not available")
    return get_alert_summary()

@app.get("/api/insights/sockpuppets")
async def get_sockpuppets():
    """Get sockpuppet candidates"""
    if not COORDINATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Coordination module not available")
    try:
        candidates = detect_sockpuppets()
        return {'candidates': candidates[:30], 'total': len(candidates)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/insights/sockpuppet-network")
async def get_sockpuppet_network_data():
    """Get sockpuppet network for visualization"""
    if not COORDINATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Coordination module not available")
    return get_sockpuppet_network()

@app.get("/api/insights/synchronized-posting")
async def get_synchronized_posting():
    """Get synchronized posting groups"""
    if not COORDINATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Coordination module not available")
    try:
        groups = detect_synchronized_posting()
        return {'groups': groups, 'total': len(groups)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/insights/clusters")
async def get_coordination_clusters():
    """Get coordination clusters"""
    if not COORDINATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Coordination module not available")
    try:
        clusters = detect_coordination_clusters()
        return {'clusters': clusters, 'total': len(clusters)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Behavioral Fingerprint Endpoints
# ============================================================================

@app.get("/api/agent/{author_name}/fingerprint")
async def get_agent_fingerprint(author_name: str):
    """Get behavioral fingerprint for an agent"""
    if not FINGERPRINTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Fingerprints module not available")

    fp = load_fingerprint(author_name)
    if not fp:
        # Compute on demand
        try:
            fp_vec = compute_fingerprint(author_name)
            save_fingerprint(author_name, fp_vec)
            return {
                'author': author_name,
                'dimensions': len(fp_vec),
                'computed': True,
                'norm': float(np.linalg.norm(fp_vec))
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    fingerprint, computed_at, sample_size = fp
    return {
        'author': author_name,
        'dimensions': len(fingerprint),
        'computed_at': computed_at.isoformat(),
        'sample_size': sample_size,
        'norm': float(np.linalg.norm(fingerprint))
    }

@app.get("/api/agent/{author_name}/behavior-history")
async def get_agent_behavior_history(author_name: str):
    """Get behavior change history for an agent"""
    if not FINGERPRINTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Fingerprints module not available")
    return detect_behavior_change(author_name)

@app.get("/api/agent/{author_name}/similar")
async def get_similar_agents(author_name: str, limit: int = 10):
    """Get agents similar to a given agent"""
    if not FINGERPRINTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Fingerprints module not available")
    try:
        similar = find_similar_agents(author_name, top_k=limit)
        return {
            'author': author_name,
            'similar': [{'author': a, 'similarity': s} for a, s in similar]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class FingerprintCompareRequest(BaseModel):
    author1: str
    author2: str

@app.post("/api/compare-fingerprints")
async def compare_fingerprints(request: FingerprintCompareRequest):
    """Compare fingerprints of two agents"""
    if not FINGERPRINTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Fingerprints module not available")
    return compare_authors(request.author1, request.author2)

# ============================================================================
# Graph Analytics Endpoints
# ============================================================================

@app.get("/api/insights/graph-metrics")
async def get_graph_metrics_overview():
    """Get graph centrality metrics"""
    if not GRAPHS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Graphs module not available")
    return {
        'pagerank': get_top_influencers('pagerank', 20),
        'betweenness': get_top_influencers('betweenness', 20),
        'eigenvector': get_top_influencers('eigenvector', 20),
        'summary': get_graph_summary()
    }

@app.get("/api/insights/communities")
async def get_communities():
    """Get detected communities"""
    if not GRAPHS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Graphs module not available")
    return detect_communities()

@app.get("/api/insights/bridges")
async def get_bridge_agents():
    """Get bridge agents between communities"""
    if not GRAPHS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Graphs module not available")
    return find_bridges()

@app.get("/api/insights/cliques")
async def get_cliques():
    """Get tight coordination groups (cliques)"""
    if not GRAPHS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Graphs module not available")
    return find_cliques()

@app.get("/api/agent/{author_name}/network-position")
async def get_network_position(author_name: str):
    """Get network position metrics for an agent"""
    if not GRAPHS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Graphs module not available")
    return get_author_network_position(author_name)

# ============================================================================
# Temporal Analysis Endpoints
# ============================================================================

@app.get("/api/agent/{author_name}/activity-pattern")
async def get_activity_pattern(author_name: str):
    """Get temporal activity pattern for an agent"""
    if not TEMPORAL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Temporal module not available")
    return get_activity_heatmap(author_name)

@app.get("/api/agent/{author_name}/circadian")
async def get_circadian(author_name: str):
    """Get circadian profile for an agent"""
    if not TEMPORAL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Temporal module not available")
    return compute_circadian_profile(author_name)

@app.get("/api/insights/correlated-pairs")
async def get_correlated_activity_pairs():
    """Get pairs of agents with correlated activity"""
    if not TEMPORAL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Temporal module not available")
    return find_correlated_pairs()

@app.get("/api/insights/bursts")
async def get_activity_bursts():
    """Get detected activity bursts"""
    if not TEMPORAL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Temporal module not available")
    return detect_activity_bursts()

# ============================================================================
# Narrative Analysis Endpoints
# ============================================================================

@app.get("/api/insights/narratives")
async def get_narratives():
    """Get identified narratives"""
    if not NARRATIVES_AVAILABLE:
        raise HTTPException(status_code=503, detail="Narratives module not available")
    try:
        narratives = identify_narratives()
        return {'narratives': narratives, 'total': len(narratives)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/narrative/{narrative_id}")
async def get_narrative_detail(narrative_id: str):
    """Get narrative propagation details"""
    if not NARRATIVES_AVAILABLE:
        raise HTTPException(status_code=503, detail="Narratives module not available")
    timeline = get_narrative_timeline(narrative_id)
    roles = classify_narrative_roles(narrative_id)
    return {'timeline': timeline, 'roles': roles}

@app.get("/api/insights/coordinated-pushes")
async def get_coordinated_pushes():
    """Get coordinated narrative pushes"""
    if not NARRATIVES_AVAILABLE:
        raise HTTPException(status_code=503, detail="Narratives module not available")
    try:
        pushes = detect_coordinated_pushes()
        return {'pushes': pushes, 'total': len(pushes)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Risk & Anomaly Endpoints
# ============================================================================

@app.get("/api/agent/{author_name}/anomaly-score")
async def get_agent_anomaly_score(author_name: str):
    """Get overall risk/anomaly score for an agent"""
    if not ALERTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Alerts module not available")
    return compute_author_risk_score(author_name)

@app.get("/api/agent/{author_name}/alerts")
async def get_agent_alerts(author_name: str):
    """Get alert history for an agent"""
    if not ALERTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Alerts module not available")
    return get_author_alert_history(author_name)

@app.get("/api/insights/high-risk-agents")
async def get_risky_agents():
    """Get agents with highest risk scores"""
    if not ALERTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Alerts module not available")
    return get_high_risk_authors()

@app.post("/api/admin/resolve-alert/{alert_id}", dependencies=[Depends(verify_admin_key)])
async def admin_resolve_alert(alert_id: int, notes: str = ""):
    """Resolve an alert (admin only)"""
    if not ALERTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Alerts module not available")
    success = resolve_alert(alert_id, notes)
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"status": "resolved", "alert_id": alert_id}

# ============================================================================
# LLM-Specific Analysis Endpoints
# ============================================================================

# Same-Operator Detection
@app.get("/api/insights/same-operator-candidates")
async def get_same_op_candidates(limit: int = 50, min_score: float = 0.6):
    """Get pairs of agents likely run by the same operator"""
    if not SAME_OPERATOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Same-operator detection not available")
    candidates = get_same_operator_candidates(limit=limit, min_score=min_score)
    return {'candidates': candidates, 'total': len(candidates)}

@app.post("/api/compare-operators")
async def compare_two_operators(request: CompareRequest):
    """Detailed same-operator comparison of two agents"""
    if not SAME_OPERATOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Same-operator detection not available")
    detector = SameOperatorDetector()
    similarity = detector.compute_operator_similarity(request.author1, request.author2)
    return {
        'author1': similarity.author1,
        'author2': similarity.author2,
        'overall_score': similarity.overall_score,
        'template_similarity': similarity.template_similarity,
        'activation_overlap': similarity.activation_overlap,
        'topic_alignment': similarity.topic_alignment,
        'never_concurrent': similarity.never_concurrent,
        'evidence': similarity.evidence
    }

@app.get("/api/insights/operator-clusters")
async def get_op_clusters(limit: int = 20):
    """Get clusters of agents grouped by operator"""
    if not SAME_OPERATOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Same-operator detection not available")
    clusters = get_operator_clusters(limit=limit)
    return {'clusters': clusters, 'total': len(clusters)}

# Information Flow
@app.get("/api/insights/information-laundering")
async def get_laundering_detections(limit: int = 50):
    """Get detected information laundering events"""
    if not INFORMATION_FLOW_AVAILABLE:
        raise HTTPException(status_code=503, detail="Information flow analysis not available")
    events = get_laundering_events(limit=limit)
    return {'events': events, 'total': len(events)}

@app.get("/api/insights/circular-citations")
async def get_circular_citation_detections(limit: int = 50):
    """Get detected circular citation patterns"""
    if not INFORMATION_FLOW_AVAILABLE:
        raise HTTPException(status_code=503, detail="Information flow analysis not available")
    circles = get_circular_citations(limit=limit)
    return {'circles': circles, 'total': len(circles)}

@app.get("/api/content/{content_id}/propagation")
async def get_content_propagation(content_id: str):
    """Get propagation tree for content"""
    if not INFORMATION_FLOW_AVAILABLE:
        raise HTTPException(status_code=503, detail="Information flow analysis not available")
    builder = PropagationTreeBuilder()
    tree = builder.build_propagation_tree(content_id)
    if tree is None:
        raise HTTPException(status_code=404, detail="Content not found or no propagation")
    return {
        'claim_id': tree.claim_id,
        'root_content_id': tree.root_content_id,
        'root_author': tree.root_author,
        'depth': tree.depth,
        'breadth': tree.breadth,
        'total_reach': tree.total_reach,
        'mutation_count': tree.mutation_count,
        'nodes': {k: {'content_id': v.content_id, 'author': v.author,
                      'timestamp': v.timestamp, 'similarity': v.similarity,
                      'parent_id': v.parent_id, 'role': v.role}
                 for k, v in tree.nodes.items()}
    }

@app.get("/api/content/{content_id}/originality")
async def get_content_originality(content_id: str):
    """Get originality score for content"""
    if not INFORMATION_FLOW_AVAILABLE:
        raise HTTPException(status_code=503, detail="Information flow analysis not available")
    builder = PropagationTreeBuilder()
    score = builder.compute_originality_score(content_id)
    return {'content_id': content_id, 'originality_score': score}

# Persona Consistency
@app.get("/api/agent/{author_name}/persona-consistency")
async def get_agent_persona_consistency(author_name: str):
    """Get persona consistency score and profile"""
    if not PERSONA_AVAILABLE:
        raise HTTPException(status_code=503, detail="Persona analysis not available")
    profile = get_persona_profile(author_name)
    if not profile:
        # Compute fresh
        tracker = PersonaTracker()
        built_profile = tracker.build_belief_profile(author_name)
        profile = {
            'author_name': built_profile.author_name,
            'stances': {k: {'position': v.position, 'confidence': v.confidence}
                       for k, v in built_profile.stances.items()},
            'consistency_score': built_profile.consistency_score,
            'computed_at': built_profile.computed_at
        }
    return profile

@app.get("/api/agent/{author_name}/belief-profile")
async def get_agent_belief_profile(author_name: str):
    """Get extracted belief profile for an agent"""
    if not PERSONA_AVAILABLE:
        raise HTTPException(status_code=503, detail="Persona analysis not available")
    tracker = PersonaTracker()
    profile = tracker.build_belief_profile(author_name)
    return {
        'author_name': profile.author_name,
        'stances': {k: {'position': v.position, 'confidence': v.confidence,
                       'evidence': v.evidence[:3]}
                   for k, v in profile.stances.items()},
        'consistency_score': profile.consistency_score
    }

@app.get("/api/agent/{author_name}/contradictions")
async def get_agent_contradictions(author_name: str):
    """Get detected contradictions for an agent"""
    if not PERSONA_AVAILABLE:
        raise HTTPException(status_code=503, detail="Persona analysis not available")
    tracker = PersonaTracker()
    contradictions = tracker.detect_contradictions(author_name)
    return {
        'author_name': author_name,
        'contradictions': [
            {
                'statement1': c.statement1,
                'statement2': c.statement2,
                'topic': c.topic,
                'timestamp1': c.timestamp1,
                'timestamp2': c.timestamp2,
                'score': c.contradiction_score
            }
            for c in contradictions[:20]
        ],
        'total': len(contradictions)
    }

@app.get("/api/insights/persona-shifts")
async def get_persona_shift_detections(limit: int = 50):
    """Get agents with sudden persona shifts"""
    if not PERSONA_AVAILABLE:
        raise HTTPException(status_code=503, detail="Persona analysis not available")
    shifts = get_persona_shifts(limit=limit)
    return {'shifts': shifts, 'total': len(shifts)}

# Enhanced Narrative Endpoints
@app.get("/api/narrative/{narrative_id}/propagation-tree")
async def get_narrative_propagation_tree(narrative_id: str):
    """Get full propagation tree for a narrative"""
    if not NARRATIVES_ENHANCED:
        raise HTTPException(status_code=503, detail="Enhanced narratives not available")
    tree = build_propagation_tree(narrative_id)
    return tree

@app.get("/api/narrative/{narrative_id}/originality")
async def get_narrative_originality(narrative_id: str):
    """Get originality scores for narrative posts"""
    if not NARRATIVES_ENHANCED:
        raise HTTPException(status_code=503, detail="Enhanced narratives not available")
    scores = compute_originality_scores(narrative_id)
    return {'narrative_id': narrative_id, 'scores': scores}

@app.get("/api/narrative/{narrative_id}/mutations")
async def get_narrative_mutations(narrative_id: str):
    """Get how narrative mutated during propagation"""
    if not NARRATIVES_ENHANCED:
        raise HTTPException(status_code=503, detail="Enhanced narratives not available")
    mutations = track_narrative_mutations(narrative_id)
    return {'narrative_id': narrative_id, 'mutations': mutations, 'total': len(mutations)}

@app.get("/api/narrative/{narrative_id}/source")
async def get_narrative_source(narrative_id: str):
    """Get original source of a narrative"""
    if not NARRATIVES_ENHANCED:
        raise HTTPException(status_code=503, detail="Enhanced narratives not available")
    return attribute_original_source(narrative_id)

# LLM Fingerprints
@app.get("/api/agent/{author_name}/llm-fingerprint")
async def get_agent_llm_fingerprint(author_name: str):
    """Get LLM-specific fingerprint for an agent"""
    if not LLM_FINGERPRINTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="LLM fingerprints not available")
    fp = load_llm_fingerprint(author_name)
    if not fp:
        # Compute fresh
        fp_vec = compute_llm_fingerprint(author_name)
        save_llm_fingerprint(author_name, fp_vec)
        computed_at = datetime.now()
    else:
        fp_vec, computed_at = fp
    return {
        'author_name': author_name,
        'fingerprint_dims': len(fp_vec),
        'fingerprint_norm': float(np.linalg.norm(fp_vec)),
        'computed_at': computed_at.isoformat() if hasattr(computed_at, 'isoformat') else str(computed_at)
    }

@app.post("/api/compare-llm-fingerprints")
async def compare_llm_fps(request: CompareRequest):
    """Compare LLM fingerprints between two agents"""
    if not LLM_FINGERPRINTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="LLM fingerprints not available")
    return compare_llm_fingerprints(request.author1, request.author2)

@app.get("/api/agent/{author_name}/similar-llm")
async def get_similar_by_llm_fingerprint(author_name: str, top_k: int = 10):
    """Find agents with similar LLM fingerprints"""
    if not LLM_FINGERPRINTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="LLM fingerprints not available")
    similar = find_similar_llm_agents(author_name, top_k=top_k)
    return {
        'author': author_name,
        'similar_agents': [
            {'author': other, 'similarity': sim, 'components': comp}
            for other, sim, comp in similar
        ]
    }

@app.get("/api/insights/template-clusters")
async def get_template_clusters():
    """Get clusters of agents with similar prompt templates"""
    if not LLM_FINGERPRINTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="LLM fingerprints not available")
    detector = PromptTemplateDetector()
    clusters = detector.cluster_by_template()
    return {
        'clusters': [
            {
                'cluster_id': c.cluster_id,
                'members': c.members,
                'avg_similarity': c.avg_similarity,
                'detected_at': c.detected_at
            }
            for c in clusters
        ],
        'total': len(clusters)
    }

# Authorship Analysis
@app.get("/api/insights/author-clusters")
async def get_authorship_clusters(limit: int = 20):
    """Get clusters of semantically similar authors"""
    if not AUTHORSHIP_AVAILABLE:
        raise HTTPException(status_code=503, detail="Authorship analysis not available")
    clusters = get_author_clusters(limit=limit)
    return {'clusters': clusters, 'total': len(clusters)}

@app.get("/api/insights/copy-chains")
async def get_detected_copy_chains(limit: int = 50):
    """Get detected copy-paste chains"""
    if not AUTHORSHIP_AVAILABLE:
        raise HTTPException(status_code=503, detail="Authorship analysis not available")
    chains = get_copy_chains(limit=limit)
    return {'chains': chains, 'total': len(chains)}

@app.get("/api/insights/near-duplicate-authors")
async def get_near_duplicate_authors(threshold: float = 0.95):
    """Get author pairs with nearly identical content centroids"""
    if not AUTHORSHIP_AVAILABLE:
        raise HTTPException(status_code=503, detail="Authorship analysis not available")
    analyzer = AuthorCentroidAnalyzer()
    duplicates = analyzer.find_semantic_near_duplicates(threshold=threshold)
    return {
        'duplicates': [
            {'author1': a1, 'author2': a2, 'similarity': sim}
            for a1, a2, sim in duplicates
        ],
        'total': len(duplicates)
    }

# ============================================================================
# Health check with background status
# ============================================================================
# Social Dynamics Endpoints (Virality, Adoption, Influence)
# ============================================================================

@app.get("/api/content/{content_id}/virality")
async def get_virality_metrics(content_id: str):
    """Get virality metrics for content (cascade size, spread velocity)"""
    if not SOCIAL_DYNAMICS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Social dynamics analysis not available")
    metrics = get_content_virality(content_id)
    if not metrics:
        raise HTTPException(status_code=404, detail="Content not found or no virality data")
    return metrics

@app.get("/api/insights/viral-content")
async def get_viral_content_list(min_cascade: int = 2, days: int = 30):
    """Get list of viral content (content that triggered cascades)"""
    if not SOCIAL_DYNAMICS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Social dynamics analysis not available")
    analyzer = ViralityAnalyzer()
    viral = analyzer.get_viral_content(min_cascade_size=min_cascade, days=days)
    return {
        'viral_content': [
            {
                'content_id': v.content_id,
                'cascade_size': v.cascade_size,
                'unique_authors': v.unique_authors,
                'spread_velocity': v.spread_velocity,
                'time_to_first_similar': v.time_to_first_similar
            }
            for v in viral[:50]
        ],
        'total': len(viral)
    }

@app.get("/api/tool-adoption/{topic}")
async def get_topic_adoption(topic: str):
    """Get adoption curve for a tool/topic"""
    if not SOCIAL_DYNAMICS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Social dynamics analysis not available")
    curve = get_tool_adoption(topic)
    if not curve:
        raise HTTPException(status_code=404, detail=f"No data for topic: {topic}")
    return curve

@app.get("/api/insights/tool-adoption")
async def get_all_tool_adoption():
    """Get adoption curves for all tracked tools"""
    if not SOCIAL_DYNAMICS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Social dynamics analysis not available")
    tracker = AdoptionTracker()
    adoption = tracker.get_all_tool_adoption()
    return {
        'tools': {
            k: {
                'topic': v.topic,
                'total_mentions': v.total_mentions,
                'unique_authors': v.unique_authors,
                'growth_rate': v.growth_rate,
                'first_mention': v.first_mention
            }
            for k, v in adoption.items()
        },
        'total_topics': len(adoption)
    }

@app.get("/api/insights/trending-tools")
async def get_trending_tools(min_mentions: int = 3, growth_threshold: float = 0.2):
    """Get tools/topics with significant recent growth"""
    if not SOCIAL_DYNAMICS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Social dynamics analysis not available")
    tracker = AdoptionTracker()
    trending = tracker.detect_trending_topics(min_mentions, growth_threshold)
    return {
        'trending': [
            {
                'topic': t.topic,
                'growth_rate': t.growth_rate,
                'total_mentions': t.total_mentions,
                'unique_authors': t.unique_authors
            }
            for t in trending
        ],
        'total': len(trending)
    }

@app.get("/api/agent/{author_name}/influence")
async def get_agent_influence_score(author_name: str):
    """Get influence score for an agent"""
    if not SOCIAL_DYNAMICS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Social dynamics analysis not available")
    return get_author_influence(author_name)

@app.get("/api/insights/top-influencers")
async def get_network_top_influencers(limit: int = 20):
    """Get top influencers in the network"""
    if not SOCIAL_DYNAMICS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Social dynamics analysis not available")
    analyzer = InfluenceAnalyzer()
    top = analyzer.get_top_influencers(limit=limit)
    return {
        'influencers': [
            {
                'author': s.author_name,
                'influence_score': s.influence_score,
                'cascade_triggers': s.cascade_triggers,
                'follower_ratio': s.follower_ratio,
                'avg_cascade_size': s.avg_cascade_size
            }
            for s in top
        ],
        'total': len(top)
    }

@app.get("/api/compare-influence/{author1}/{author2}")
async def compare_influence_between(author1: str, author2: str):
    """Compare temporal influence between two agents"""
    if not SOCIAL_DYNAMICS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Social dynamics analysis not available")
    analyzer = InfluenceAnalyzer()
    return analyzer.detect_temporal_influence(author1, author2)

# ============================================================================
# Health check with background status
# ============================================================================

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
