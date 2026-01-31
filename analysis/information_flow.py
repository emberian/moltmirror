#!/usr/bin/env python3
"""
Emergent Amplification Metrics for Moltbook Agent Analysis

Grounded metrics for tracking how ideas spread:
- Reuse chain depth: How many posts build on prior content
- Derivative rate: % of posts semantically close to prior posts
- Propagation trees: Trace how content spreads via embeddings
- Originality scores: How novel is content vs prior corpus

Removed speculative "laundering" detection - we measure observable
patterns without inferring intent.
"""

import sqlite3
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import os

DB_PATH = Path(os.getenv("MOLTMIRROR_DB_PATH", "analysis.db"))


@dataclass
class PropagationNode:
    """Node in a propagation tree"""
    content_id: str
    author: str
    timestamp: str
    similarity: float
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)


@dataclass
class PropagationTree:
    """Tree showing how content spread via semantic similarity"""
    seed_id: str
    root_content_id: str
    root_author: str
    nodes: Dict[str, PropagationNode]
    depth: int
    breadth: int
    total_reach: int


@dataclass
class AmplificationMetrics:
    """Metrics for emergent amplification patterns"""
    author_name: str
    total_posts: int
    derivative_count: int
    derivative_rate: float
    avg_originality: float
    reuse_depth_avg: float
    reuse_depth_max: int
    cascade_starts: int  # Posts that spawned significant follow-on
    computed_at: str


def get_db():
    """Get database connection"""
    return sqlite3.connect(DB_PATH)


def ensure_amplification_tables():
    """Ensure amplification analysis tables exist"""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS amplification_metrics (
            author_name TEXT PRIMARY KEY,
            total_posts INTEGER,
            derivative_count INTEGER,
            derivative_rate REAL,
            avg_originality REAL,
            reuse_depth_avg REAL,
            reuse_depth_max INTEGER,
            cascade_starts INTEGER,
            computed_at TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS propagation_trees (
            seed_id TEXT PRIMARY KEY,
            root_author TEXT,
            depth INTEGER,
            breadth INTEGER,
            total_reach INTEGER,
            tree_data TEXT,
            computed_at TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS content_originality (
            content_id TEXT PRIMARY KEY,
            originality_score REAL,
            max_prior_similarity REAL,
            computed_at TEXT
        )
    """)

    conn.commit()
    conn.close()


class PropagationTreeBuilder:
    """Build and analyze propagation trees based on semantic similarity"""

    def build_propagation_tree(self, seed_content_id: str,
                                similarity_threshold: float = 0.75) -> Optional[PropagationTree]:
        """
        Build a propagation tree from a seed content.
        Tracks how similar content appeared after the seed.
        """
        conn = get_db()
        cursor = conn.cursor()

        # Get seed content
        cursor.execute("""
            SELECT p.id, p.author_name, p.created_at, e.embedding
            FROM posts p
            LEFT JOIN embeddings e ON p.id = e.content_id AND e.content_type = 'post'
            WHERE p.id = ?
        """, (seed_content_id,))

        seed = cursor.fetchone()
        if not seed or not seed[3]:
            conn.close()
            return None

        seed_id, seed_author, seed_time, seed_embedding_blob = seed
        seed_embedding = np.frombuffer(seed_embedding_blob, dtype=np.float32)

        # Find later posts similar to seed
        cursor.execute("""
            SELECT p.id, p.author_name, p.created_at, e.embedding
            FROM posts p
            JOIN embeddings e ON p.id = e.content_id AND e.content_type = 'post'
            WHERE p.created_at > ?
            ORDER BY p.created_at
        """, (seed_time,))

        propagation = []
        for row in cursor.fetchall():
            content_id, author, timestamp, embedding_blob = row
            if embedding_blob:
                embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                sim = float(np.dot(seed_embedding, embedding) / (
                    np.linalg.norm(seed_embedding) * np.linalg.norm(embedding) + 1e-8
                ))

                if sim >= similarity_threshold:
                    propagation.append({
                        'content_id': content_id,
                        'author': author,
                        'timestamp': timestamp,
                        'similarity': sim
                    })

        conn.close()

        # Build tree
        nodes = {}
        root_node = PropagationNode(
            content_id=seed_id,
            author=seed_author,
            timestamp=seed_time,
            similarity=1.0
        )
        nodes[seed_id] = root_node

        for prop in propagation:
            node = PropagationNode(
                content_id=prop['content_id'],
                author=prop['author'],
                timestamp=prop['timestamp'],
                similarity=prop['similarity'],
                parent_id=seed_id
            )
            nodes[prop['content_id']] = node
            root_node.children.append(prop['content_id'])

        depth = 1 if propagation else 0

        return PropagationTree(
            seed_id=seed_id,
            root_content_id=seed_id,
            root_author=seed_author,
            nodes=nodes,
            depth=depth,
            breadth=len(propagation),
            total_reach=len(nodes)
        )

    def compute_originality_score(self, content_id: str) -> float:
        """
        Compute originality score for content.
        0 = derivative (very similar to prior content)
        1 = novel (unique in corpus)
        """
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT p.created_at, e.embedding
            FROM posts p
            JOIN embeddings e ON p.id = e.content_id AND e.content_type = 'post'
            WHERE p.id = ?
        """, (content_id,))

        row = cursor.fetchone()
        if not row or not row[1]:
            conn.close()
            return 0.5

        timestamp, embedding_blob = row
        embedding = np.frombuffer(embedding_blob, dtype=np.float32)

        # Get prior content embeddings
        cursor.execute("""
            SELECT e.embedding
            FROM posts p
            JOIN embeddings e ON p.id = e.content_id AND e.content_type = 'post'
            WHERE p.created_at < ?
            ORDER BY p.created_at DESC
            LIMIT 500
        """, (timestamp,))

        prior_embeddings = []
        for row in cursor.fetchall():
            if row[0]:
                prior_embeddings.append(np.frombuffer(row[0], dtype=np.float32))

        conn.close()

        if not prior_embeddings:
            return 1.0  # No prior content

        prior_embeddings = np.array(prior_embeddings)
        similarities = np.dot(prior_embeddings, embedding) / (
            np.linalg.norm(prior_embeddings, axis=1) * np.linalg.norm(embedding) + 1e-8
        )

        max_sim = float(np.max(similarities))
        return 1.0 - max_sim


class AmplificationAnalyzer:
    """Analyze emergent amplification patterns"""

    def __init__(self):
        self.tree_builder = PropagationTreeBuilder()

    def compute_derivative_rate(self, author: str,
                                 similarity_threshold: float = 0.8,
                                 window_days: int = 7) -> Dict[str, Any]:
        """
        Compute derivative rate: % of posts that are semantically
        close to prior posts (by anyone) within N days.
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
            return {'derivative_count': 0, 'total': 0, 'rate': 0.0}

        derivative_count = 0
        originality_scores = []

        for post in author_posts:
            try:
                post_time = datetime.fromisoformat(post['timestamp'].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                continue

            window_start = (post_time - timedelta(days=window_days)).isoformat()

            # Get prior posts in window
            cursor.execute("""
                SELECT e.embedding
                FROM posts p
                JOIN embeddings e ON p.id = e.content_id AND e.content_type = 'post'
                WHERE p.created_at < ? AND p.created_at > ?
                AND p.id != ?
                LIMIT 200
            """, (post['timestamp'], window_start, post['id']))

            prior = []
            for row in cursor.fetchall():
                if row[0]:
                    prior.append(np.frombuffer(row[0], dtype=np.float32))

            if prior:
                prior = np.array(prior)
                similarities = np.dot(prior, post['embedding']) / (
                    np.linalg.norm(prior, axis=1) * np.linalg.norm(post['embedding']) + 1e-8
                )
                max_sim = float(np.max(similarities))

                if max_sim >= similarity_threshold:
                    derivative_count += 1

                originality_scores.append(1.0 - max_sim)
            else:
                originality_scores.append(1.0)

        conn.close()

        return {
            'derivative_count': derivative_count,
            'total': len(author_posts),
            'rate': derivative_count / len(author_posts) if author_posts else 0.0,
            'avg_originality': np.mean(originality_scores) if originality_scores else 0.5
        }

    def compute_reuse_chain_depth(self, author: str,
                                   similarity_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Compute reuse chain depth: how many of author's posts
        spawned follow-on posts by others.
        """
        conn = get_db()
        cursor = conn.cursor()

        # Get author's posts
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
            return {'chain_starts': 0, 'avg_depth': 0, 'max_depth': 0}

        chain_depths = []

        for post in author_posts:
            # Count how many later posts (by others) are similar
            cursor.execute("""
                SELECT COUNT(*)
                FROM posts p
                JOIN embeddings e ON p.id = e.content_id AND e.content_type = 'post'
                WHERE p.created_at > ?
                AND p.author_name != ?
            """, (post['timestamp'], author))

            total_later = cursor.fetchone()[0]

            if total_later == 0:
                chain_depths.append(0)
                continue

            # Sample and check similarity
            cursor.execute("""
                SELECT e.embedding
                FROM posts p
                JOIN embeddings e ON p.id = e.content_id AND e.content_type = 'post'
                WHERE p.created_at > ?
                AND p.author_name != ?
                LIMIT 200
            """, (post['timestamp'], author))

            later_embeddings = []
            for row in cursor.fetchall():
                if row[0]:
                    later_embeddings.append(np.frombuffer(row[0], dtype=np.float32))

            if later_embeddings:
                later_embeddings = np.array(later_embeddings)
                similarities = np.dot(later_embeddings, post['embedding']) / (
                    np.linalg.norm(later_embeddings, axis=1) * np.linalg.norm(post['embedding']) + 1e-8
                )
                similar_count = int(np.sum(similarities >= similarity_threshold))
                chain_depths.append(similar_count)
            else:
                chain_depths.append(0)

        conn.close()

        cascade_starts = sum(1 for d in chain_depths if d > 0)

        return {
            'chain_starts': cascade_starts,
            'avg_depth': np.mean(chain_depths) if chain_depths else 0,
            'max_depth': max(chain_depths) if chain_depths else 0,
            'total_posts': len(author_posts)
        }

    def compute_author_amplification(self, author: str) -> AmplificationMetrics:
        """Compute complete amplification metrics for an author"""
        derivative = self.compute_derivative_rate(author)
        reuse = self.compute_reuse_chain_depth(author)

        return AmplificationMetrics(
            author_name=author,
            total_posts=derivative['total'],
            derivative_count=derivative['derivative_count'],
            derivative_rate=round(derivative['rate'], 3),
            avg_originality=round(derivative['avg_originality'], 3),
            reuse_depth_avg=round(reuse['avg_depth'], 3),
            reuse_depth_max=reuse['max_depth'],
            cascade_starts=reuse['chain_starts'],
            computed_at=datetime.now().isoformat()
        )

    def compute_network_amplification(self, min_posts: int = 5) -> Dict[str, Any]:
        """Compute amplification metrics across the network"""
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT author_name, COUNT(*) as posts
            FROM posts
            WHERE author_name IS NOT NULL
            GROUP BY author_name
            HAVING posts >= ?
        """, (min_posts,))

        authors = [row[0] for row in cursor.fetchall()]
        conn.close()

        all_metrics = []
        for author in authors[:100]:  # Limit for performance
            try:
                metrics = self.compute_author_amplification(author)
                all_metrics.append(metrics)
            except Exception:
                continue

        if not all_metrics:
            return {'status': 'no_data'}

        # Network-level stats
        return {
            'total_authors': len(all_metrics),
            'avg_derivative_rate': round(np.mean([m.derivative_rate for m in all_metrics]), 3),
            'avg_originality': round(np.mean([m.avg_originality for m in all_metrics]), 3),
            'total_cascade_starts': sum(m.cascade_starts for m in all_metrics),
            'high_amplifiers': [m.author_name for m in all_metrics if m.cascade_starts > 3][:10],
            'high_derivatives': [m.author_name for m in all_metrics if m.derivative_rate > 0.5][:10],
            'computed_at': datetime.now().isoformat()
        }


def get_propagation_tree(seed_content_id: str) -> Optional[Dict[str, Any]]:
    """Get or build propagation tree for content"""
    ensure_amplification_tables()
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT tree_data FROM propagation_trees WHERE seed_id = ?
    """, (seed_content_id,))

    row = cursor.fetchone()
    conn.close()

    if row and row[0]:
        return json.loads(row[0])

    # Build fresh
    builder = PropagationTreeBuilder()
    tree = builder.build_propagation_tree(seed_content_id)

    if tree:
        return {
            'seed_id': tree.seed_id,
            'root_author': tree.root_author,
            'depth': tree.depth,
            'breadth': tree.breadth,
            'total_reach': tree.total_reach,
            'nodes': {k: {
                'content_id': v.content_id,
                'author': v.author,
                'timestamp': v.timestamp,
                'similarity': v.similarity
            } for k, v in tree.nodes.items()}
        }

    return None


def get_author_amplification(author: str) -> Optional[Dict[str, Any]]:
    """Get amplification metrics for an author"""
    ensure_amplification_tables()
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT total_posts, derivative_count, derivative_rate, avg_originality,
               reuse_depth_avg, reuse_depth_max, cascade_starts, computed_at
        FROM amplification_metrics
        WHERE author_name = ?
    """, (author,))

    row = cursor.fetchone()
    conn.close()

    if row:
        return {
            'author_name': author,
            'total_posts': row[0],
            'derivative_count': row[1],
            'derivative_rate': row[2],
            'avg_originality': row[3],
            'reuse_depth_avg': row[4],
            'reuse_depth_max': row[5],
            'cascade_starts': row[6],
            'computed_at': row[7]
        }

    # Compute fresh
    analyzer = AmplificationAnalyzer()
    metrics = analyzer.compute_author_amplification(author)

    return {
        'author_name': metrics.author_name,
        'total_posts': metrics.total_posts,
        'derivative_count': metrics.derivative_count,
        'derivative_rate': metrics.derivative_rate,
        'avg_originality': metrics.avg_originality,
        'reuse_depth_avg': metrics.reuse_depth_avg,
        'reuse_depth_max': metrics.reuse_depth_max,
        'cascade_starts': metrics.cascade_starts,
        'computed_at': metrics.computed_at
    }


def run_information_flow_analysis() -> Dict[str, Any]:
    """Run emergent amplification analysis"""
    ensure_amplification_tables()

    analyzer = AmplificationAnalyzer()
    builder = PropagationTreeBuilder()

    results = {}
    now = datetime.now().isoformat()

    # Compute network-level metrics
    try:
        network_metrics = analyzer.compute_network_amplification()
        results['network'] = {
            'status': 'success',
            **network_metrics
        }
    except Exception as e:
        results['network'] = {'status': 'error', 'error': str(e)}

    # Compute per-author metrics and save
    try:
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT author_name, COUNT(*) as posts
            FROM posts
            WHERE author_name IS NOT NULL
            GROUP BY author_name
            HAVING posts >= 5
            LIMIT 200
        """)

        authors = [row[0] for row in cursor.fetchall()]

        computed = 0
        for author in authors:
            try:
                metrics = analyzer.compute_author_amplification(author)
                cursor.execute("""
                    INSERT OR REPLACE INTO amplification_metrics
                    (author_name, total_posts, derivative_count, derivative_rate,
                     avg_originality, reuse_depth_avg, reuse_depth_max, cascade_starts, computed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics.author_name, metrics.total_posts, metrics.derivative_count,
                    metrics.derivative_rate, metrics.avg_originality, metrics.reuse_depth_avg,
                    metrics.reuse_depth_max, metrics.cascade_starts, now
                ))
                computed += 1
            except Exception:
                continue

        conn.commit()
        conn.close()

        results['authors'] = {'status': 'success', 'computed': computed}

    except Exception as e:
        results['authors'] = {'status': 'error', 'error': str(e)}

    results['completed_at'] = now
    return results


# Legacy compatibility functions (for existing API endpoints)
def get_laundering_events(limit: int = 50) -> List[Dict[str, Any]]:
    """Legacy: Return empty list (laundering detection removed as handwavy)"""
    return []


def get_circular_citations(limit: int = 50) -> List[Dict[str, Any]]:
    """Legacy: Return empty list (circular citation detection removed as handwavy)"""
    return []


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python information_flow.py [analyze|author AUTHOR|tree CONTENT_ID|originality CONTENT_ID]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "analyze":
        print("Running amplification analysis...")
        results = run_information_flow_analysis()
        for k, v in results.items():
            print(f"  {k}: {v}")

    elif command == "author" and len(sys.argv) >= 3:
        author = sys.argv[2]
        print(f"Amplification metrics for {author}:")
        metrics = get_author_amplification(author)
        if metrics:
            for k, v in metrics.items():
                print(f"  {k}: {v}")

    elif command == "tree" and len(sys.argv) >= 3:
        content_id = sys.argv[2]
        builder = PropagationTreeBuilder()
        tree = builder.build_propagation_tree(content_id)
        if tree:
            print(f"Propagation tree for {content_id}:")
            print(f"  Root author: {tree.root_author}")
            print(f"  Depth: {tree.depth}")
            print(f"  Total reach: {tree.total_reach}")
        else:
            print("Could not build propagation tree")

    elif command == "originality" and len(sys.argv) >= 3:
        content_id = sys.argv[2]
        builder = PropagationTreeBuilder()
        score = builder.compute_originality_score(content_id)
        print(f"Originality score for {content_id}: {score:.3f}")

    else:
        print(f"Unknown command: {command}")
