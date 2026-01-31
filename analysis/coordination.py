#!/usr/bin/env python3
"""
Coordination Detection for IC-Grade Analysis
Sockpuppet detection, synchronized posting, and CIB detection
"""

import sqlite3
import pickle
import numpy as np
from scipy import sparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict
import json
import os

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    from community import community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    try:
        # Alternative import
        import community.community_louvain as community_louvain
        LOUVAIN_AVAILABLE = True
    except ImportError:
        LOUVAIN_AVAILABLE = False

DB_PATH = Path(os.getenv("MOLTMIRROR_DB_PATH", "analysis.db"))


def get_db():
    """Get database connection"""
    return sqlite3.connect(DB_PATH)


def ensure_coordination_tables():
    """Ensure coordination detection tables exist"""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sockpuppet_candidates (
            author1 TEXT,
            author2 TEXT,
            confidence REAL,
            evidence TEXT,
            detected_at TEXT,
            status TEXT DEFAULT 'pending',
            PRIMARY KEY (author1, author2)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS coordination_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_type TEXT,
            severity TEXT,
            involved_authors TEXT,
            evidence TEXT,
            confidence REAL,
            detected_at TEXT,
            status TEXT DEFAULT 'active'
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS coordination_clusters (
            cluster_id TEXT PRIMARY KEY,
            members TEXT,
            cohesion REAL,
            coordination_score REAL,
            detected_at TEXT
        )
    """)

    conn.commit()
    conn.close()


def detect_sockpuppets(fingerprint_threshold: float = 0.85,
                        temporal_threshold: float = 0.8) -> List[Dict[str, Any]]:
    """
    Detect likely sockpuppet pairs using fingerprint similarity
    Criteria:
    - High fingerprint similarity (>0.85)
    - Never interact with each other
    - High temporal correlation
    - Similar vocabulary signatures
    """
    ensure_coordination_tables()

    # Import fingerprint module
    try:
        from .fingerprints import load_fingerprint, fingerprint_similarity, compute_fingerprint, save_fingerprint
    except ImportError:
        from analysis.fingerprints import load_fingerprint, fingerprint_similarity, compute_fingerprint, save_fingerprint

    conn = get_db()
    cursor = conn.cursor()

    # Get all fingerprinted authors
    cursor.execute("SELECT author_name, fingerprint FROM agent_fingerprints")
    authors_data = cursor.fetchall()

    if len(authors_data) < 2:
        conn.close()
        return []

    authors = [row[0] for row in authors_data]
    fingerprints = {row[0]: pickle.loads(row[1]) for row in authors_data}

    # Get interaction pairs (authors who have interacted)
    cursor.execute("""
        SELECT DISTINCT c.author_name, p.author_name
        FROM comments c
        JOIN posts p ON c.post_id = p.id
        WHERE c.author_name IS NOT NULL AND p.author_name IS NOT NULL
    """)
    interaction_pairs = {(row[0], row[1]) for row in cursor.fetchall()}
    interaction_pairs |= {(b, a) for a, b in interaction_pairs}  # Symmetric

    candidates = []

    # Compare all pairs
    for i, author1 in enumerate(authors):
        fp1 = fingerprints[author1]

        for author2 in authors[i+1:]:
            # Skip if they interact
            if (author1, author2) in interaction_pairs:
                continue

            fp2 = fingerprints[author2]

            # Overall similarity
            overall_sim = fingerprint_similarity(fp1, fp2)

            if overall_sim < fingerprint_threshold:
                continue

            # Temporal similarity (first 168 dims)
            temporal_sim = fingerprint_similarity(fp1[:168], fp2[:168])

            if temporal_sim < temporal_threshold:
                continue

            # Vocabulary similarity (next 100 dims)
            vocab_sim = fingerprint_similarity(fp1[168:268], fp2[168:268])

            # Calculate confidence
            confidence = (overall_sim * 0.4 + temporal_sim * 0.35 + vocab_sim * 0.25)

            evidence = {
                'overall_similarity': round(overall_sim, 4),
                'temporal_similarity': round(temporal_sim, 4),
                'vocabulary_similarity': round(vocab_sim, 4),
                'no_interaction': True
            }

            candidates.append({
                'author1': author1,
                'author2': author2,
                'confidence': round(confidence, 4),
                'evidence': evidence
            })

    # Sort by confidence
    candidates.sort(key=lambda x: x['confidence'], reverse=True)

    # Save to database
    now = datetime.now().isoformat()
    for candidate in candidates[:100]:  # Top 100
        cursor.execute("""
            INSERT OR REPLACE INTO sockpuppet_candidates
            (author1, author2, confidence, evidence, detected_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            candidate['author1'],
            candidate['author2'],
            candidate['confidence'],
            json.dumps(candidate['evidence']),
            now
        ))

    conn.commit()
    conn.close()

    return candidates[:50]  # Return top 50


def detect_synchronized_posting(time_window_minutes: int = 30,
                                  similarity_threshold: float = 0.7,
                                  min_group_size: int = 3) -> List[Dict[str, Any]]:
    """
    Detect synchronized posting: groups posting similar content within short time windows

    Returns groups with:
    - Similar content posted within time window
    - 3+ unique authors
    - Score based on similarity, group size, and time spread
    """
    conn = get_db()
    cursor = conn.cursor()

    # Get recent posts with embeddings
    cursor.execute("""
        SELECT p.id, p.author_name, p.created_at, e.embedding
        FROM posts p
        JOIN embeddings e ON p.id = e.content_id AND e.content_type = 'post'
        WHERE p.created_at > datetime('now', '-7 days')
        AND p.author_name IS NOT NULL
        ORDER BY p.created_at
    """)

    posts = cursor.fetchall()
    conn.close()

    if len(posts) < min_group_size:
        return []

    # Parse and organize
    post_data = []
    for post_id, author, created_at, embedding_blob in posts:
        try:
            dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            post_data.append({
                'id': post_id,
                'author': author,
                'time': dt,
                'embedding': embedding
            })
        except (ValueError, AttributeError):
            continue

    synchronized_groups = []
    processed_posts = set()

    for i, post in enumerate(post_data):
        if post['id'] in processed_posts:
            continue

        # Find posts within time window
        window_posts = [post]

        for j, other in enumerate(post_data):
            if i == j or other['id'] in processed_posts:
                continue

            time_diff = abs((other['time'] - post['time']).total_seconds())
            if time_diff > time_window_minutes * 60:
                continue

            # Check content similarity
            sim = np.dot(post['embedding'], other['embedding'])
            if sim >= similarity_threshold:
                window_posts.append(other)

        # Check if we have a group
        unique_authors = set(p['author'] for p in window_posts)

        if len(unique_authors) >= min_group_size:
            # Mark posts as processed
            for p in window_posts:
                processed_posts.add(p['id'])

            # Calculate time spread
            times = [p['time'] for p in window_posts]
            time_spread = (max(times) - min(times)).total_seconds() / 60  # minutes

            # Calculate average similarity
            sims = []
            for a in window_posts:
                for b in window_posts:
                    if a['id'] < b['id']:
                        sims.append(np.dot(a['embedding'], b['embedding']))
            avg_sim = np.mean(sims) if sims else 0

            # Score: higher similarity + larger group + tighter timing = higher score
            score = (avg_sim * len(unique_authors)) / (time_spread + 1)

            synchronized_groups.append({
                'post_ids': [p['id'] for p in window_posts],
                'authors': list(unique_authors),
                'group_size': len(unique_authors),
                'avg_similarity': round(avg_sim, 4),
                'time_spread_minutes': round(time_spread, 2),
                'coordination_score': round(score, 4),
                'first_post_time': min(times).isoformat(),
                'last_post_time': max(times).isoformat()
            })

    # Sort by score
    synchronized_groups.sort(key=lambda x: x['coordination_score'], reverse=True)

    return synchronized_groups[:30]


def detect_coordination_clusters(alpha: float = 0.4,
                                   beta: float = 0.3,
                                   gamma: float = 0.3,
                                   min_cluster_size: int = 3) -> List[Dict[str, Any]]:
    """
    Detect coordination clusters using D4M-inspired matrix approach:
    1. Combine matrices: M = α×reply + β×timing + γ×content
    2. Apply Louvain community detection
    3. Score clusters by cohesion × size
    """
    if not NETWORKX_AVAILABLE:
        return [{'error': 'networkx not available'}]

    # Import matrix module
    try:
        from .matrices import load_matrix, compute_combined_matrix
    except ImportError:
        from analysis.matrices import load_matrix, compute_combined_matrix

    # Get or compute combined matrix
    result = compute_combined_matrix(alpha, beta, gamma)
    if result is None:
        return [{'error': 'insufficient matrix data'}]

    combined, authors = result
    author_idx = {a: i for i, a in enumerate(authors)}

    # Build graph from similarity matrix
    G = nx.Graph()
    G.add_nodes_from(authors)

    # Add edges for high similarity pairs
    threshold = 0.3  # Minimum similarity to create edge
    cx = combined.tocoo()

    for i, j, v in zip(cx.row, cx.col, cx.data):
        if i < j and v > threshold:  # Upper triangle only
            G.add_edge(authors[i], authors[j], weight=v)

    # Remove isolated nodes
    G.remove_nodes_from(list(nx.isolates(G)))

    if len(G.nodes()) < min_cluster_size:
        return []

    # Community detection
    if LOUVAIN_AVAILABLE:
        partition = community_louvain.best_partition(G)
    else:
        # Fallback to connected components
        partition = {}
        for i, component in enumerate(nx.connected_components(G)):
            for node in component:
                partition[node] = i

    # Group by community
    communities = defaultdict(list)
    for node, community_id in partition.items():
        communities[community_id].append(node)

    clusters = []
    for community_id, members in communities.items():
        if len(members) < min_cluster_size:
            continue

        # Calculate cluster cohesion
        subgraph = G.subgraph(members)
        edges = list(subgraph.edges(data=True))

        if edges:
            avg_weight = np.mean([e[2].get('weight', 0) for e in edges])
            density = nx.density(subgraph)
            cohesion = avg_weight * density
        else:
            cohesion = 0

        # Coordination score = cohesion × log(size)
        score = cohesion * np.log(len(members) + 1)

        clusters.append({
            'cluster_id': f"cluster_{community_id}",
            'members': sorted(members),
            'size': len(members),
            'cohesion': round(cohesion, 4),
            'coordination_score': round(score, 4),
            'edge_count': len(edges)
        })

    # Sort by coordination score
    clusters.sort(key=lambda x: x['coordination_score'], reverse=True)

    # Save to database
    ensure_coordination_tables()
    conn = get_db()
    cursor = conn.cursor()

    now = datetime.now().isoformat()
    for cluster in clusters[:20]:
        cursor.execute("""
            INSERT OR REPLACE INTO coordination_clusters
            (cluster_id, members, cohesion, coordination_score, detected_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            cluster['cluster_id'],
            json.dumps(cluster['members']),
            cluster['cohesion'],
            cluster['coordination_score'],
            now
        ))

    conn.commit()
    conn.close()

    return clusters[:20]


def find_content_brigading(post_id: str, time_window_hours: int = 6) -> Dict[str, Any]:
    """
    Detect potential brigading on a specific post:
    - Sudden spike in comments/votes
    - Comments from users who don't normally participate in this topic
    - Coordinated timing patterns
    """
    conn = get_db()
    cursor = conn.cursor()

    # Get post info
    cursor.execute("""
        SELECT author_name, submolt, created_at, upvotes, comment_count
        FROM posts WHERE id = ?
    """, (post_id,))

    post = cursor.fetchone()
    if not post:
        conn.close()
        return {'error': 'post not found'}

    post_author, submolt, created_at, upvotes, comment_count = post

    # Get comments on this post
    cursor.execute("""
        SELECT author_name, created_at
        FROM comments WHERE post_id = ?
        ORDER BY created_at
    """, (post_id,))

    comments = cursor.fetchall()

    if len(comments) < 5:
        conn.close()
        return {'brigading_likely': False, 'reason': 'insufficient_comments'}

    # Analyze comment timing
    comment_times = []
    commenters = []
    for author, c_time in comments:
        if author:
            try:
                dt = datetime.fromisoformat(c_time.replace('Z', '+00:00'))
                comment_times.append(dt)
                commenters.append(author)
            except (ValueError, AttributeError):
                pass

    if len(comment_times) < 5:
        conn.close()
        return {'brigading_likely': False, 'reason': 'insufficient_data'}

    # Calculate comment rate
    time_span = (max(comment_times) - min(comment_times)).total_seconds() / 3600
    if time_span > 0:
        comment_rate = len(comment_times) / time_span
    else:
        comment_rate = len(comment_times)

    # Check for outsiders (commenters who don't usually post in this submolt)
    outsiders = []
    if submolt:
        for commenter in set(commenters):
            cursor.execute("""
                SELECT COUNT(*) FROM posts
                WHERE author_name = ? AND submolt = ?
            """, (commenter, submolt))

            posts_in_submolt = cursor.fetchone()[0]

            cursor.execute("""
                SELECT COUNT(*) FROM posts
                WHERE author_name = ?
            """, (commenter,))

            total_posts = cursor.fetchone()[0]

            if total_posts > 0 and posts_in_submolt / total_posts < 0.1:
                outsiders.append(commenter)

    # Check for burst patterns
    if len(comment_times) >= 3:
        intervals = [(comment_times[i+1] - comment_times[i]).total_seconds()
                     for i in range(len(comment_times) - 1)]
        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        burst_detected = std_interval < avg_interval * 0.5  # Low variance = coordinated
    else:
        burst_detected = False

    conn.close()

    # Determine brigading likelihood
    signals = sum([
        comment_rate > 5,  # More than 5 comments per hour
        len(outsiders) > len(commenters) * 0.5,  # More than 50% outsiders
        burst_detected
    ])

    return {
        'post_id': post_id,
        'brigading_likely': signals >= 2,
        'confidence': signals / 3,
        'evidence': {
            'comment_rate_per_hour': round(comment_rate, 2),
            'total_commenters': len(set(commenters)),
            'outsider_ratio': round(len(outsiders) / len(set(commenters)), 2) if commenters else 0,
            'outsiders': outsiders[:10],
            'burst_pattern_detected': burst_detected,
            'time_span_hours': round(time_span, 2)
        }
    }


def find_vote_manipulation_candidates() -> List[Dict[str, Any]]:
    """
    Find posts with suspicious voting patterns:
    - High upvotes but low engagement
    - Rapid upvote accumulation
    - Upvotes from accounts with low activity
    """
    conn = get_db()
    cursor = conn.cursor()

    # Find posts with high upvote-to-comment ratio
    cursor.execute("""
        SELECT id, title, author_name, upvotes, comment_count, created_at,
               (upvotes * 1.0 / NULLIF(comment_count, 0)) as ratio
        FROM posts
        WHERE upvotes >= 10
        AND created_at > datetime('now', '-7 days')
        ORDER BY ratio DESC
        LIMIT 50
    """)

    suspicious = []
    for row in cursor.fetchall():
        post_id, title, author, upvotes, comments, created_at, ratio = row

        # Skip if ratio is normal (< 5)
        if ratio and ratio < 5:
            continue

        # Calculate upvote velocity
        try:
            post_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            age_hours = (datetime.now() - post_time.replace(tzinfo=None)).total_seconds() / 3600
            velocity = upvotes / max(age_hours, 0.1)
        except (ValueError, AttributeError):
            velocity = 0

        suspicious.append({
            'post_id': post_id,
            'title': title[:80] if title else '',
            'author': author,
            'upvotes': upvotes,
            'comments': comments or 0,
            'upvote_to_comment_ratio': round(ratio, 2) if ratio else None,
            'upvotes_per_hour': round(velocity, 2),
            'suspicious_signals': sum([
                (ratio or 0) > 10,
                velocity > 5,
                (comments or 0) == 0
            ])
        })

    conn.close()

    # Sort by suspicious signals
    suspicious.sort(key=lambda x: x['suspicious_signals'], reverse=True)

    return suspicious[:30]


def get_sockpuppet_network() -> Dict[str, Any]:
    """
    Build a network graph of sockpuppet relationships for visualization
    """
    ensure_coordination_tables()
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT author1, author2, confidence, evidence
        FROM sockpuppet_candidates
        WHERE confidence >= 0.7
        ORDER BY confidence DESC
        LIMIT 100
    """)

    edges = []
    nodes = set()

    for row in cursor.fetchall():
        author1, author2, confidence, evidence = row
        nodes.add(author1)
        nodes.add(author2)

        edges.append({
            'source': author1,
            'target': author2,
            'weight': confidence,
            'evidence': json.loads(evidence) if evidence else {}
        })

    conn.close()

    return {
        'nodes': [{'id': n, 'label': n} for n in nodes],
        'edges': edges,
        'total_candidates': len(edges)
    }


def run_all_coordination_detection() -> Dict[str, Any]:
    """Run all coordination detection algorithms and return results"""
    results = {}

    try:
        sockpuppets = detect_sockpuppets()
        results['sockpuppets'] = {
            'status': 'success',
            'count': len(sockpuppets),
            'top_confidence': sockpuppets[0]['confidence'] if sockpuppets else None
        }
    except Exception as e:
        results['sockpuppets'] = {'status': 'error', 'error': str(e)}

    try:
        synchronized = detect_synchronized_posting()
        results['synchronized_posting'] = {
            'status': 'success',
            'groups_found': len(synchronized)
        }
    except Exception as e:
        results['synchronized_posting'] = {'status': 'error', 'error': str(e)}

    try:
        clusters = detect_coordination_clusters()
        results['coordination_clusters'] = {
            'status': 'success',
            'clusters_found': len(clusters)
        }
    except Exception as e:
        results['coordination_clusters'] = {'status': 'error', 'error': str(e)}

    try:
        vote_manip = find_vote_manipulation_candidates()
        results['vote_manipulation'] = {
            'status': 'success',
            'candidates': len(vote_manip)
        }
    except Exception as e:
        results['vote_manipulation'] = {'status': 'error', 'error': str(e)}

    results['completed_at'] = datetime.now().isoformat()

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python coordination.py [sockpuppets|synchronized|clusters|vote-manip|all]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "sockpuppets":
        print("Detecting sockpuppets...")
        results = detect_sockpuppets()
        print(f"Found {len(results)} candidates:")
        for r in results[:10]:
            print(f"  {r['author1']} <-> {r['author2']}: {r['confidence']:.3f}")

    elif command == "synchronized":
        print("Detecting synchronized posting...")
        results = detect_synchronized_posting()
        print(f"Found {len(results)} synchronized groups:")
        for r in results[:5]:
            print(f"  Group of {r['group_size']}: {', '.join(r['authors'][:5])}")
            print(f"    Score: {r['coordination_score']:.3f}, Similarity: {r['avg_similarity']:.3f}")

    elif command == "clusters":
        print("Detecting coordination clusters...")
        results = detect_coordination_clusters()
        print(f"Found {len(results)} clusters:")
        for r in results[:5]:
            print(f"  {r['cluster_id']}: {r['size']} members, score={r['coordination_score']:.3f}")

    elif command == "vote-manip":
        print("Finding vote manipulation candidates...")
        results = find_vote_manipulation_candidates()
        print(f"Found {len(results)} suspicious posts:")
        for r in results[:10]:
            print(f"  {r['title'][:50]}... ({r['upvotes']} up, {r['comments']} comments)")

    elif command == "all":
        print("Running all coordination detection...")
        results = run_all_coordination_detection()
        for k, v in results.items():
            print(f"  {k}: {v}")

    else:
        print(f"Unknown command: {command}")
