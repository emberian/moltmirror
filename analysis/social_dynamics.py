#!/usr/bin/env python3
"""
Social Dynamics Analysis for Moltbook

Grounded metrics for social network behavior:

1. Idea Virality:
   - Time-to-spread: How quickly do ideas jump to other authors
   - Cascade size: Distribution of idea spread patterns

2. Tool/Topic Adoption:
   - Keyword/regex extraction for mentions
   - Adoption curves over time

3. Social Influence:
   - Cross-author similarity with temporal lag
   - Influence centrality weighted by idea spread

All metrics are observable from content and timing data.
"""

import sqlite3
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass
import re
import os

DB_PATH = Path(os.getenv("MOLTMIRROR_DB_PATH", "analysis.db"))


# Common tools/technologies to track
TOOLS_PATTERNS = {
    'mcp': r'\b(?:MCP|Model Context Protocol)\b',
    'api': r'\bAPI\b',
    'langchain': r'\b(?:langchain|LangChain)\b',
    'openai': r'\b(?:OpenAI|GPT-?4|ChatGPT)\b',
    'anthropic': r'\b(?:Anthropic|Claude)\b',
    'huggingface': r'\b(?:HuggingFace|Hugging Face|transformers)\b',
    'vector_db': r'\b(?:Pinecone|Chroma|Weaviate|Milvus|vector\s*database)\b',
    'rag': r'\b(?:RAG|retrieval.augmented)\b',
    'agents': r'\b(?:AI agents?|autonomous agents?|agentic)\b',
    'llm': r'\b(?:LLM|large language model)\b',
    'fine_tuning': r'\b(?:fine.?tun|finetun)\b',
    'prompt_engineering': r'\b(?:prompt engineering|prompting)\b',
}


@dataclass
class ViralityMetrics:
    """Metrics for idea virality"""
    content_id: str
    time_to_first_similar: Optional[float]  # Hours to first similar post by another author
    cascade_size: int  # Number of similar posts within window
    unique_authors: int  # Unique authors who posted similar content
    spread_velocity: float  # Similar posts per day


@dataclass
class AdoptionCurve:
    """Adoption curve for a tool/topic"""
    topic: str
    first_mention: Optional[str]
    total_mentions: int
    unique_authors: int
    mentions_by_week: Dict[str, int]
    growth_rate: float  # Week-over-week growth


@dataclass
class InfluenceScore:
    """Influence metrics for an author"""
    author_name: str
    influence_score: float  # Weighted by how often their ideas spread
    cascade_triggers: int  # Posts that triggered cascades
    follower_ratio: float  # Ratio of authors who post similar content after them
    avg_cascade_size: float
    topics_led: List[str]  # Topics where they were early adopter


def get_db():
    """Get database connection"""
    return sqlite3.connect(DB_PATH)


def ensure_social_dynamics_tables():
    """Ensure social dynamics tables exist"""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS virality_metrics (
            content_id TEXT PRIMARY KEY,
            time_to_first_similar REAL,
            cascade_size INTEGER,
            unique_authors INTEGER,
            spread_velocity REAL,
            computed_at TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tool_adoption (
            topic TEXT PRIMARY KEY,
            first_mention TEXT,
            total_mentions INTEGER,
            unique_authors INTEGER,
            mentions_by_week TEXT,
            growth_rate REAL,
            computed_at TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS influence_scores (
            author_name TEXT PRIMARY KEY,
            influence_score REAL,
            cascade_triggers INTEGER,
            follower_ratio REAL,
            avg_cascade_size REAL,
            topics_led TEXT,
            computed_at TEXT
        )
    """)

    conn.commit()
    conn.close()


class ViralityAnalyzer:
    """Analyze idea virality patterns"""

    def compute_content_virality(self, content_id: str,
                                   similarity_threshold: float = 0.8,
                                   window_days: int = 14) -> Optional[ViralityMetrics]:
        """
        Compute virality metrics for a piece of content.
        Measures how quickly and widely similar content appears.
        """
        conn = get_db()
        cursor = conn.cursor()

        # Get content embedding and timestamp
        cursor.execute("""
            SELECT p.author_name, p.created_at, e.embedding
            FROM posts p
            JOIN embeddings e ON p.id = e.content_id AND e.content_type = 'post'
            WHERE p.id = ?
        """, (content_id,))

        row = cursor.fetchone()
        if not row or not row[2]:
            conn.close()
            return None

        source_author, source_time, source_embedding_blob = row
        source_embedding = np.frombuffer(source_embedding_blob, dtype=np.float32)

        try:
            source_dt = datetime.fromisoformat(source_time.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            conn.close()
            return None

        window_end = (source_dt + timedelta(days=window_days)).isoformat()

        # Find similar content by other authors within window
        cursor.execute("""
            SELECT p.id, p.author_name, p.created_at, e.embedding
            FROM posts p
            JOIN embeddings e ON p.id = e.content_id AND e.content_type = 'post'
            WHERE p.created_at > ? AND p.created_at < ?
            AND p.author_name != ?
        """, (source_time, window_end, source_author))

        similar_posts = []
        for row in cursor.fetchall():
            pid, author, timestamp, embedding_blob = row
            if embedding_blob:
                embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                sim = float(np.dot(source_embedding, embedding) / (
                    np.linalg.norm(source_embedding) * np.linalg.norm(embedding) + 1e-8
                ))

                if sim >= similarity_threshold:
                    similar_posts.append({
                        'id': pid,
                        'author': author,
                        'timestamp': timestamp,
                        'similarity': sim
                    })

        conn.close()

        if not similar_posts:
            return ViralityMetrics(
                content_id=content_id,
                time_to_first_similar=None,
                cascade_size=0,
                unique_authors=0,
                spread_velocity=0.0
            )

        # Sort by timestamp
        similar_posts.sort(key=lambda x: x['timestamp'])

        # Calculate metrics
        first_similar_time = datetime.fromisoformat(
            similar_posts[0]['timestamp'].replace('Z', '+00:00')
        )
        time_to_first = (first_similar_time - source_dt).total_seconds() / 3600

        unique_authors = len(set(p['author'] for p in similar_posts))
        spread_velocity = len(similar_posts) / window_days

        return ViralityMetrics(
            content_id=content_id,
            time_to_first_similar=round(time_to_first, 2),
            cascade_size=len(similar_posts),
            unique_authors=unique_authors,
            spread_velocity=round(spread_velocity, 3)
        )

    def get_viral_content(self, min_cascade_size: int = 3,
                          days: int = 30) -> List[ViralityMetrics]:
        """Find content that went viral (triggered significant cascades)"""
        conn = get_db()
        cursor = conn.cursor()

        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        cursor.execute("""
            SELECT p.id
            FROM posts p
            JOIN embeddings e ON p.id = e.content_id AND e.content_type = 'post'
            WHERE p.created_at > ?
            ORDER BY p.upvotes DESC
            LIMIT 100
        """, (cutoff,))

        content_ids = [row[0] for row in cursor.fetchall()]
        conn.close()

        viral = []
        for cid in content_ids:
            metrics = self.compute_content_virality(cid)
            if metrics and metrics.cascade_size >= min_cascade_size:
                viral.append(metrics)

        return sorted(viral, key=lambda x: x.cascade_size, reverse=True)


class AdoptionTracker:
    """Track tool/topic adoption patterns"""

    def __init__(self):
        self.patterns = {k: re.compile(v, re.IGNORECASE) for k, v in TOOLS_PATTERNS.items()}

    def track_topic_adoption(self, topic: str) -> Optional[AdoptionCurve]:
        """Track adoption curve for a specific topic/tool"""
        if topic not in self.patterns:
            # Create ad-hoc pattern
            pattern = re.compile(rf'\b{re.escape(topic)}\b', re.IGNORECASE)
        else:
            pattern = self.patterns[topic]

        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, author_name, created_at, content
            FROM posts
            WHERE content IS NOT NULL
            ORDER BY created_at
        """)

        mentions = []
        for row in cursor.fetchall():
            pid, author, timestamp, content = row
            if content and pattern.search(content):
                mentions.append({
                    'id': pid,
                    'author': author,
                    'timestamp': timestamp
                })

        conn.close()

        if not mentions:
            return None

        # Group by week
        mentions_by_week = defaultdict(int)
        for m in mentions:
            try:
                dt = datetime.fromisoformat(m['timestamp'].replace('Z', '+00:00'))
                week = dt.strftime('%Y-W%W')
                mentions_by_week[week] += 1
            except (ValueError, AttributeError):
                continue

        # Calculate growth rate
        weeks = sorted(mentions_by_week.keys())
        if len(weeks) >= 2:
            recent_weeks = weeks[-4:] if len(weeks) >= 4 else weeks
            first_half = sum(mentions_by_week[w] for w in recent_weeks[:len(recent_weeks)//2])
            second_half = sum(mentions_by_week[w] for w in recent_weeks[len(recent_weeks)//2:])
            growth_rate = (second_half - first_half) / max(first_half, 1)
        else:
            growth_rate = 0.0

        unique_authors = len(set(m['author'] for m in mentions))

        return AdoptionCurve(
            topic=topic,
            first_mention=mentions[0]['timestamp'],
            total_mentions=len(mentions),
            unique_authors=unique_authors,
            mentions_by_week=dict(mentions_by_week),
            growth_rate=round(growth_rate, 3)
        )

    def get_all_tool_adoption(self) -> Dict[str, AdoptionCurve]:
        """Track adoption for all predefined tools"""
        results = {}
        for topic in TOOLS_PATTERNS:
            curve = self.track_topic_adoption(topic)
            if curve and curve.total_mentions > 0:
                results[topic] = curve
        return results

    def detect_trending_topics(self, min_mentions: int = 3,
                                growth_threshold: float = 0.2) -> List[AdoptionCurve]:
        """Find topics with significant recent growth"""
        all_curves = self.get_all_tool_adoption()

        trending = [
            curve for curve in all_curves.values()
            if curve.total_mentions >= min_mentions and curve.growth_rate >= growth_threshold
        ]

        return sorted(trending, key=lambda x: x.growth_rate, reverse=True)


class InfluenceAnalyzer:
    """Analyze social influence patterns"""

    def compute_author_influence(self, author: str,
                                  similarity_threshold: float = 0.8) -> InfluenceScore:
        """
        Compute influence score for an author.
        Based on how often their content triggers similar content by others.
        """
        conn = get_db()
        cursor = conn.cursor()

        # Get author's posts with embeddings
        cursor.execute("""
            SELECT p.id, p.created_at, e.embedding
            FROM posts p
            JOIN embeddings e ON p.id = e.content_id AND e.content_type = 'post'
            WHERE p.author_name = ?
            ORDER BY p.created_at
        """, (author,))

        author_posts = []
        for row in cursor.fetchall():
            if row[2]:
                author_posts.append({
                    'id': row[0],
                    'timestamp': row[1],
                    'embedding': np.frombuffer(row[2], dtype=np.float32)
                })

        if not author_posts:
            conn.close()
            return InfluenceScore(
                author_name=author,
                influence_score=0.0,
                cascade_triggers=0,
                follower_ratio=0.0,
                avg_cascade_size=0.0,
                topics_led=[]
            )

        cascade_sizes = []
        followers = set()

        for post in author_posts:
            # Find similar posts by others within 14 days
            try:
                post_dt = datetime.fromisoformat(post['timestamp'].replace('Z', '+00:00'))
                window_end = (post_dt + timedelta(days=14)).isoformat()
            except (ValueError, AttributeError):
                continue

            cursor.execute("""
                SELECT p.author_name, e.embedding
                FROM posts p
                JOIN embeddings e ON p.id = e.content_id AND e.content_type = 'post'
                WHERE p.created_at > ? AND p.created_at < ?
                AND p.author_name != ?
            """, (post['timestamp'], window_end, author))

            cascade_count = 0
            for row in cursor.fetchall():
                other_author, embedding_blob = row
                if embedding_blob:
                    embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                    sim = float(np.dot(post['embedding'], embedding) / (
                        np.linalg.norm(post['embedding']) * np.linalg.norm(embedding) + 1e-8
                    ))

                    if sim >= similarity_threshold:
                        cascade_count += 1
                        followers.add(other_author)

            cascade_sizes.append(cascade_count)

        conn.close()

        # Compute metrics
        cascade_triggers = sum(1 for c in cascade_sizes if c > 0)
        avg_cascade = np.mean(cascade_sizes) if cascade_sizes else 0.0

        # Get total unique authors for follower ratio
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT author_name) FROM posts WHERE author_name IS NOT NULL")
        total_authors = cursor.fetchone()[0] or 1
        conn.close()

        follower_ratio = len(followers) / total_authors

        # Influence score: weighted combination
        influence = (
            0.4 * (cascade_triggers / max(len(author_posts), 1)) +
            0.3 * min(avg_cascade / 5, 1.0) +
            0.3 * follower_ratio
        )

        return InfluenceScore(
            author_name=author,
            influence_score=round(influence, 3),
            cascade_triggers=cascade_triggers,
            follower_ratio=round(follower_ratio, 3),
            avg_cascade_size=round(avg_cascade, 2),
            topics_led=[]  # Would need additional analysis
        )

    def get_top_influencers(self, limit: int = 20) -> List[InfluenceScore]:
        """Get top influencers in the network"""
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT author_name, COUNT(*) as posts
            FROM posts
            WHERE author_name IS NOT NULL
            GROUP BY author_name
            HAVING posts >= 5
            ORDER BY posts DESC
            LIMIT 100
        """)

        authors = [row[0] for row in cursor.fetchall()]
        conn.close()

        scores = []
        for author in authors:
            try:
                score = self.compute_author_influence(author)
                scores.append(score)
            except Exception:
                continue

        return sorted(scores, key=lambda x: x.influence_score, reverse=True)[:limit]

    def detect_temporal_influence(self, author1: str, author2: str,
                                   lag_days: int = 7) -> Dict[str, Any]:
        """
        Detect temporal influence: Does author1's content predict
        similar content from author2 (or vice versa)?
        """
        conn = get_db()
        cursor = conn.cursor()

        def get_posts(author):
            cursor.execute("""
                SELECT p.created_at, e.embedding
                FROM posts p
                JOIN embeddings e ON p.id = e.content_id AND e.content_type = 'post'
                WHERE p.author_name = ?
                ORDER BY p.created_at
            """, (author,))
            return [(row[0], np.frombuffer(row[1], dtype=np.float32))
                    for row in cursor.fetchall() if row[1]]

        posts1 = get_posts(author1)
        posts2 = get_posts(author2)
        conn.close()

        if not posts1 or not posts2:
            return {'error': 'insufficient_data'}

        # For each post by author1, check if similar post by author2 follows within lag
        a1_to_a2 = 0
        a2_to_a1 = 0

        for ts1, emb1 in posts1:
            try:
                dt1 = datetime.fromisoformat(ts1.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                continue

            for ts2, emb2 in posts2:
                try:
                    dt2 = datetime.fromisoformat(ts2.replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    continue

                sim = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))

                if sim >= 0.8:
                    diff_days = (dt2 - dt1).total_seconds() / 86400

                    if 0 < diff_days <= lag_days:
                        a1_to_a2 += 1
                    elif -lag_days <= diff_days < 0:
                        a2_to_a1 += 1

        return {
            'author1': author1,
            'author2': author2,
            f'{author1}_influences_{author2}': a1_to_a2,
            f'{author2}_influences_{author1}': a2_to_a1,
            'likely_direction': author1 if a1_to_a2 > a2_to_a1 else (author2 if a2_to_a1 > a1_to_a2 else 'mutual')
        }


def run_social_dynamics_analysis() -> Dict[str, Any]:
    """Run complete social dynamics analysis"""
    ensure_social_dynamics_tables()

    results = {}
    now = datetime.now().isoformat()

    # Virality analysis
    try:
        virality_analyzer = ViralityAnalyzer()
        viral_content = virality_analyzer.get_viral_content(min_cascade_size=2)
        results['virality'] = {
            'status': 'success',
            'viral_content_count': len(viral_content),
            'top_cascades': [{'id': v.content_id, 'size': v.cascade_size} for v in viral_content[:5]]
        }
    except Exception as e:
        results['virality'] = {'status': 'error', 'error': str(e)}

    # Tool adoption
    try:
        tracker = AdoptionTracker()
        adoption = tracker.get_all_tool_adoption()
        trending = tracker.detect_trending_topics()

        conn = get_db()
        cursor = conn.cursor()

        for topic, curve in adoption.items():
            cursor.execute("""
                INSERT OR REPLACE INTO tool_adoption
                (topic, first_mention, total_mentions, unique_authors,
                 mentions_by_week, growth_rate, computed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                curve.topic, curve.first_mention, curve.total_mentions,
                curve.unique_authors, json.dumps(curve.mentions_by_week),
                curve.growth_rate, now
            ))

        conn.commit()
        conn.close()

        results['adoption'] = {
            'status': 'success',
            'topics_tracked': len(adoption),
            'trending': [{'topic': t.topic, 'growth': t.growth_rate} for t in trending[:5]]
        }
    except Exception as e:
        results['adoption'] = {'status': 'error', 'error': str(e)}

    # Influence analysis
    try:
        influence_analyzer = InfluenceAnalyzer()
        top_influencers = influence_analyzer.get_top_influencers(limit=20)

        conn = get_db()
        cursor = conn.cursor()

        for score in top_influencers:
            cursor.execute("""
                INSERT OR REPLACE INTO influence_scores
                (author_name, influence_score, cascade_triggers, follower_ratio,
                 avg_cascade_size, topics_led, computed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                score.author_name, score.influence_score, score.cascade_triggers,
                score.follower_ratio, score.avg_cascade_size, json.dumps(score.topics_led), now
            ))

        conn.commit()
        conn.close()

        results['influence'] = {
            'status': 'success',
            'top_influencers': [{'author': s.author_name, 'score': s.influence_score}
                               for s in top_influencers[:5]]
        }
    except Exception as e:
        results['influence'] = {'status': 'error', 'error': str(e)}

    results['completed_at'] = now
    return results


# API helper functions

def get_content_virality(content_id: str) -> Optional[Dict[str, Any]]:
    """Get virality metrics for content"""
    analyzer = ViralityAnalyzer()
    metrics = analyzer.compute_content_virality(content_id)
    if metrics:
        return {
            'content_id': metrics.content_id,
            'time_to_first_similar': metrics.time_to_first_similar,
            'cascade_size': metrics.cascade_size,
            'unique_authors': metrics.unique_authors,
            'spread_velocity': metrics.spread_velocity
        }
    return None


def get_tool_adoption(topic: str) -> Optional[Dict[str, Any]]:
    """Get adoption curve for a tool/topic"""
    tracker = AdoptionTracker()
    curve = tracker.track_topic_adoption(topic)
    if curve:
        return {
            'topic': curve.topic,
            'first_mention': curve.first_mention,
            'total_mentions': curve.total_mentions,
            'unique_authors': curve.unique_authors,
            'mentions_by_week': curve.mentions_by_week,
            'growth_rate': curve.growth_rate
        }
    return None


def get_author_influence(author: str) -> Dict[str, Any]:
    """Get influence score for an author"""
    analyzer = InfluenceAnalyzer()
    score = analyzer.compute_author_influence(author)
    return {
        'author_name': score.author_name,
        'influence_score': score.influence_score,
        'cascade_triggers': score.cascade_triggers,
        'follower_ratio': score.follower_ratio,
        'avg_cascade_size': score.avg_cascade_size
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python social_dynamics.py [analyze|viral|adoption TOPIC|influence AUTHOR|top-influencers]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "analyze":
        print("Running social dynamics analysis...")
        results = run_social_dynamics_analysis()
        for k, v in results.items():
            print(f"  {k}: {v}")

    elif command == "viral":
        print("Finding viral content...")
        analyzer = ViralityAnalyzer()
        viral = analyzer.get_viral_content()
        for v in viral[:10]:
            print(f"  {v.content_id}: cascade={v.cascade_size}, velocity={v.spread_velocity}")

    elif command == "adoption" and len(sys.argv) >= 3:
        topic = sys.argv[2]
        curve = get_tool_adoption(topic)
        if curve:
            print(f"Adoption curve for {topic}:")
            for k, v in curve.items():
                print(f"  {k}: {v}")

    elif command == "influence" and len(sys.argv) >= 3:
        author = sys.argv[2]
        score = get_author_influence(author)
        print(f"Influence score for {author}:")
        for k, v in score.items():
            print(f"  {k}: {v}")

    elif command == "top-influencers":
        analyzer = InfluenceAnalyzer()
        top = analyzer.get_top_influencers()
        print("Top influencers:")
        for s in top:
            print(f"  {s.author_name}: {s.influence_score:.3f} (triggers={s.cascade_triggers})")

    else:
        print(f"Unknown command: {command}")
