#!/usr/bin/env python3
"""
Narrative Propagation Analysis for IC-Grade Analysis
Narrative identification, originator classification, and coordinated push detection
"""

import sqlite3
import pickle
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict
import hashlib
import os

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

try:
    from sklearn.cluster import DBSCAN
    DBSCAN_AVAILABLE = True
except ImportError:
    DBSCAN_AVAILABLE = False

DB_PATH = Path(os.getenv("MOLTMIRROR_DB_PATH", "analysis.db"))


def get_db():
    """Get database connection"""
    return sqlite3.connect(DB_PATH)


def ensure_narrative_tables():
    """Ensure narrative tables exist"""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS narratives (
            narrative_id TEXT PRIMARY KEY,
            key_phrases TEXT,
            centroid BLOB,
            post_count INTEGER,
            first_seen TEXT,
            last_seen TEXT,
            status TEXT DEFAULT 'active'
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS narrative_posts (
            post_id TEXT,
            narrative_id TEXT,
            role TEXT,
            similarity REAL,
            created_at TEXT,
            PRIMARY KEY (post_id, narrative_id),
            FOREIGN KEY (narrative_id) REFERENCES narratives(narrative_id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS coordinated_pushes (
            push_id TEXT PRIMARY KEY,
            narrative_id TEXT,
            authors TEXT,
            posts TEXT,
            start_time TEXT,
            end_time TEXT,
            coordination_score REAL,
            detected_at TEXT
        )
    """)

    conn.commit()
    conn.close()


def get_recent_embeddings(days: int = 7) -> Tuple[List[str], List[str], List[np.ndarray], List[str]]:
    """
    Get embeddings for recent posts

    Returns: (post_ids, authors, embeddings, timestamps)
    """
    conn = get_db()
    cursor = conn.cursor()

    cutoff = (datetime.now() - timedelta(days=days)).isoformat()

    cursor.execute("""
        SELECT p.id, p.author_name, e.embedding, p.created_at
        FROM posts p
        JOIN embeddings e ON p.id = e.content_id AND e.content_type = 'post'
        WHERE p.created_at > ?
        AND p.author_name IS NOT NULL
        ORDER BY p.created_at
    """, (cutoff,))

    post_ids = []
    authors = []
    embeddings = []
    timestamps = []

    for row in cursor.fetchall():
        post_id, author, embedding_blob, created_at = row
        post_ids.append(post_id)
        authors.append(author)
        embeddings.append(np.frombuffer(embedding_blob, dtype=np.float32))
        timestamps.append(created_at)

    conn.close()

    return post_ids, authors, embeddings, timestamps


def identify_narratives(min_cluster_size: int = 5,
                         min_samples: int = 3) -> List[Dict[str, Any]]:
    """
    Identify narrative clusters using HDBSCAN (or DBSCAN fallback)
    on content embeddings
    """
    ensure_narrative_tables()

    post_ids, authors, embeddings, timestamps = get_recent_embeddings()

    if len(embeddings) < min_cluster_size:
        return []

    X = np.array(embeddings)

    # Cluster
    if HDBSCAN_AVAILABLE:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='cosine'
        )
        labels = clusterer.fit_predict(X)
    elif DBSCAN_AVAILABLE:
        # Fallback to DBSCAN
        from sklearn.metrics.pairwise import cosine_distances
        distances = cosine_distances(X)
        clusterer = DBSCAN(eps=0.3, min_samples=min_samples, metric='precomputed')
        labels = clusterer.fit_predict(distances)
    else:
        return [{'error': 'no clustering algorithm available'}]

    # Group by cluster
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        if label >= 0:  # Ignore noise (-1)
            clusters[label].append(i)

    narratives = []

    conn = get_db()
    cursor = conn.cursor()
    now = datetime.now().isoformat()

    for cluster_id, indices in clusters.items():
        if len(indices) < min_cluster_size:
            continue

        # Calculate centroid
        cluster_embeddings = X[indices]
        centroid = np.mean(cluster_embeddings, axis=0)

        # Get post details
        cluster_posts = [post_ids[i] for i in indices]
        cluster_authors = [authors[i] for i in indices]
        cluster_times = [timestamps[i] for i in indices]

        # Extract key phrases from posts
        post_ids_str = ','.join(['?'] * len(cluster_posts))
        cursor.execute(f"""
            SELECT title, content FROM posts WHERE id IN ({post_ids_str})
        """, cluster_posts)

        all_text = ' '.join([
            f"{row[0] or ''} {row[1] or ''}"
            for row in cursor.fetchall()
        ])

        # Simple key phrase extraction (top frequent words)
        import re
        words = re.findall(r'\b[a-z]{4,15}\b', all_text.lower())
        word_freq = defaultdict(int)
        stop_words = {'that', 'this', 'with', 'from', 'have', 'been', 'would', 'could', 'should', 'about', 'their', 'which', 'there', 'these', 'those', 'being'}
        for w in words:
            if w not in stop_words:
                word_freq[w] += 1

        key_phrases = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        key_phrases = [w for w, _ in key_phrases]

        # Create narrative ID
        narrative_id = f"narrative_{hashlib.md5(' '.join(key_phrases).encode()).hexdigest()[:8]}"

        narrative = {
            'narrative_id': narrative_id,
            'key_phrases': key_phrases,
            'post_count': len(cluster_posts),
            'unique_authors': len(set(cluster_authors)),
            'posts': cluster_posts[:20],  # Sample
            'authors': list(set(cluster_authors))[:20],
            'first_seen': min(cluster_times),
            'last_seen': max(cluster_times),
            'centroid': centroid
        }

        narratives.append(narrative)

        # Save to database
        cursor.execute("""
            INSERT OR REPLACE INTO narratives
            (narrative_id, key_phrases, centroid, post_count, first_seen, last_seen, status)
            VALUES (?, ?, ?, ?, ?, ?, 'active')
        """, (
            narrative_id,
            json.dumps(key_phrases),
            pickle.dumps(centroid),
            len(cluster_posts),
            narrative['first_seen'],
            narrative['last_seen']
        ))

        # Save post associations
        for i in indices:
            cursor.execute("""
                INSERT OR REPLACE INTO narrative_posts
                (post_id, narrative_id, role, similarity, created_at)
                VALUES (?, ?, 'member', ?, ?)
            """, (
                post_ids[i],
                narrative_id,
                1.0,  # Will compute actual similarity later
                timestamps[i]
            ))

    conn.commit()
    conn.close()

    # Sort by post count
    narratives.sort(key=lambda x: x['post_count'], reverse=True)

    # Remove numpy arrays for JSON serialization
    for n in narratives:
        del n['centroid']

    return narratives[:30]


def classify_narrative_roles(narrative_id: str) -> Dict[str, Any]:
    """
    Classify posts in a narrative as:
    - originator: first to post this content
    - early_adopter: within first 10% of posts
    - amplifier: rapid reposting, high similarity
    - organic: natural discussion growth
    """
    ensure_narrative_tables()
    conn = get_db()
    cursor = conn.cursor()

    # Get narrative posts ordered by time
    cursor.execute("""
        SELECT np.post_id, p.author_name, np.created_at, np.similarity
        FROM narrative_posts np
        JOIN posts p ON np.post_id = p.id
        WHERE np.narrative_id = ?
        ORDER BY np.created_at ASC
    """, (narrative_id,))

    posts = cursor.fetchall()
    conn.close()

    if len(posts) < 3:
        return {'narrative_id': narrative_id, 'error': 'insufficient_posts'}

    # Classify roles
    total = len(posts)
    early_cutoff = max(1, int(total * 0.1))

    roles = {
        'originator': [],
        'early_adopter': [],
        'amplifier': [],
        'organic': []
    }

    author_post_count = defaultdict(int)
    author_intervals = defaultdict(list)
    last_time = {}

    for i, (post_id, author, created_at, similarity) in enumerate(posts):
        try:
            dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            continue

        # Track author activity
        if author in last_time:
            interval = (dt - last_time[author]).total_seconds() / 60
            author_intervals[author].append(interval)
        last_time[author] = dt
        author_post_count[author] += 1

        # Classify
        if i == 0:
            role = 'originator'
        elif i < early_cutoff:
            role = 'early_adopter'
        elif author_post_count[author] > 1 and author_intervals.get(author):
            avg_interval = np.mean(author_intervals[author])
            if avg_interval < 30:  # Less than 30 min between posts
                role = 'amplifier'
            else:
                role = 'organic'
        else:
            role = 'organic'

        roles[role].append({
            'post_id': post_id,
            'author': author,
            'created_at': created_at,
            'similarity': similarity
        })

    return {
        'narrative_id': narrative_id,
        'total_posts': total,
        'roles': {k: len(v) for k, v in roles.items()},
        'originators': [p['author'] for p in roles['originator']],
        'early_adopters': [p['author'] for p in roles['early_adopter'][:10]],
        'top_amplifiers': sorted(
            [(a, c) for a, c in author_post_count.items() if c > 1],
            key=lambda x: x[1], reverse=True
        )[:10]
    }


def detect_coordinated_pushes(time_window_hours: float = 2,
                               min_authors: int = 3,
                               min_fingerprint_sim: float = 0.7) -> List[Dict[str, Any]]:
    """
    Detect coordinated narrative pushes:
    - Multiple authors
    - Similar content (in same narrative)
    - Short time window
    - High fingerprint similarity (optional)
    """
    ensure_narrative_tables()

    # Import fingerprint module
    try:
        from .fingerprints import load_fingerprint, fingerprint_similarity
        FINGERPRINTS_AVAILABLE = True
    except ImportError:
        FINGERPRINTS_AVAILABLE = False

    conn = get_db()
    cursor = conn.cursor()

    # Get recent narrative posts grouped by narrative
    cursor.execute("""
        SELECT narrative_id, post_id, created_at, role
        FROM narrative_posts
        WHERE created_at > datetime('now', '-7 days')
        ORDER BY narrative_id, created_at
    """)

    rows = cursor.fetchall()

    if not rows:
        conn.close()
        return []

    # Group by narrative
    narrative_posts = defaultdict(list)
    for narrative_id, post_id, created_at, role in rows:
        narrative_posts[narrative_id].append({
            'post_id': post_id,
            'created_at': created_at,
            'role': role
        })

    pushes = []

    for narrative_id, posts in narrative_posts.items():
        if len(posts) < min_authors:
            continue

        # Get author info
        post_ids = [p['post_id'] for p in posts]
        placeholders = ','.join(['?'] * len(post_ids))

        cursor.execute(f"""
            SELECT id, author_name FROM posts WHERE id IN ({placeholders})
        """, post_ids)

        post_authors = {row[0]: row[1] for row in cursor.fetchall()}

        # Look for time-clustered posting
        for i, post in enumerate(posts):
            try:
                start_time = datetime.fromisoformat(post['created_at'].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                continue

            end_time = start_time + timedelta(hours=time_window_hours)

            # Find posts in window
            window_posts = []
            window_authors = set()

            for j in range(i, len(posts)):
                try:
                    post_time = datetime.fromisoformat(posts[j]['created_at'].replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    continue

                if post_time > end_time:
                    break

                author = post_authors.get(posts[j]['post_id'])
                if author:
                    window_posts.append(posts[j])
                    window_authors.add(author)

            if len(window_authors) < min_authors:
                continue

            # Check fingerprint similarity if available
            fingerprint_score = 0
            if FINGERPRINTS_AVAILABLE and len(window_authors) >= 2:
                authors_list = list(window_authors)
                sims = []

                for a1 in authors_list:
                    fp1 = load_fingerprint(a1)
                    if not fp1:
                        continue

                    for a2 in authors_list:
                        if a1 >= a2:
                            continue

                        fp2 = load_fingerprint(a2)
                        if not fp2:
                            continue

                        sim = fingerprint_similarity(fp1[0], fp2[0])
                        sims.append(sim)

                if sims:
                    fingerprint_score = np.mean(sims)

            # Calculate coordination score
            time_spread = (end_time - start_time).total_seconds() / 3600
            coordination_score = len(window_authors) / (time_spread + 0.1)

            if fingerprint_score > min_fingerprint_sim:
                coordination_score *= (1 + fingerprint_score)

            # Create push record
            push_id = f"push_{narrative_id}_{start_time.strftime('%Y%m%d%H%M')}"

            pushes.append({
                'push_id': push_id,
                'narrative_id': narrative_id,
                'authors': list(window_authors),
                'author_count': len(window_authors),
                'post_count': len(window_posts),
                'start_time': start_time.isoformat(),
                'end_time': posts[i + len(window_posts) - 1]['created_at'],
                'time_spread_hours': round(time_spread, 2),
                'coordination_score': round(coordination_score, 2),
                'fingerprint_similarity': round(fingerprint_score, 3) if fingerprint_score else None
            })

    conn.close()

    # Deduplicate overlapping pushes
    unique_pushes = []
    seen_posts = set()

    for push in sorted(pushes, key=lambda x: x['coordination_score'], reverse=True):
        # Skip if too much overlap
        overlap = False
        for existing in unique_pushes:
            if push['narrative_id'] == existing['narrative_id']:
                # Check time overlap
                try:
                    p_start = datetime.fromisoformat(push['start_time'])
                    e_start = datetime.fromisoformat(existing['start_time'])
                    e_end = datetime.fromisoformat(existing['end_time'].replace('Z', '+00:00'))

                    if p_start >= e_start and p_start <= e_end:
                        overlap = True
                        break
                except (ValueError, AttributeError):
                    pass

        if not overlap:
            unique_pushes.append(push)

    # Save to database
    conn = get_db()
    cursor = conn.cursor()
    now = datetime.now().isoformat()

    for push in unique_pushes[:30]:
        cursor.execute("""
            INSERT OR REPLACE INTO coordinated_pushes
            (push_id, narrative_id, authors, posts, start_time, end_time, coordination_score, detected_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            push['push_id'],
            push['narrative_id'],
            json.dumps(push['authors']),
            json.dumps([]),  # We don't store individual posts here
            push['start_time'],
            push['end_time'],
            push['coordination_score'],
            now
        ))

    conn.commit()
    conn.close()

    return unique_pushes[:30]


def get_narrative_timeline(narrative_id: str) -> Dict[str, Any]:
    """
    Get timeline of a narrative's spread
    """
    conn = get_db()
    cursor = conn.cursor()

    # Get narrative info
    cursor.execute("""
        SELECT key_phrases, post_count, first_seen, last_seen
        FROM narratives WHERE narrative_id = ?
    """, (narrative_id,))

    row = cursor.fetchone()
    if not row:
        conn.close()
        return {'error': 'narrative not found'}

    key_phrases = json.loads(row[0])
    post_count = row[1]
    first_seen = row[2]
    last_seen = row[3]

    # Get posts timeline
    cursor.execute("""
        SELECT np.post_id, p.author_name, np.created_at, p.title, np.role
        FROM narrative_posts np
        JOIN posts p ON np.post_id = p.id
        WHERE np.narrative_id = ?
        ORDER BY np.created_at ASC
    """, (narrative_id,))

    posts = []
    hourly_counts = defaultdict(int)

    for post_id, author, created_at, title, role in cursor.fetchall():
        posts.append({
            'post_id': post_id,
            'author': author,
            'title': title[:80] if title else '',
            'created_at': created_at,
            'role': role
        })

        # Bucket by hour
        try:
            dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            hour_key = dt.strftime('%Y-%m-%d %H:00')
            hourly_counts[hour_key] += 1
        except (ValueError, AttributeError):
            pass

    conn.close()

    # Calculate propagation speed
    if len(posts) >= 2:
        try:
            first = datetime.fromisoformat(posts[0]['created_at'].replace('Z', '+00:00'))
            last = datetime.fromisoformat(posts[-1]['created_at'].replace('Z', '+00:00'))
            duration_hours = (last - first).total_seconds() / 3600
            posts_per_hour = len(posts) / max(duration_hours, 0.1)
        except (ValueError, AttributeError):
            duration_hours = 0
            posts_per_hour = 0
    else:
        duration_hours = 0
        posts_per_hour = 0

    return {
        'narrative_id': narrative_id,
        'key_phrases': key_phrases,
        'total_posts': post_count,
        'unique_authors': len(set(p['author'] for p in posts)),
        'first_seen': first_seen,
        'last_seen': last_seen,
        'duration_hours': round(duration_hours, 2),
        'posts_per_hour': round(posts_per_hour, 2),
        'timeline': posts[:50],
        'hourly_activity': dict(sorted(hourly_counts.items()))
    }


def find_similar_narratives(narrative_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Find narratives similar to a given one"""
    conn = get_db()
    cursor = conn.cursor()

    # Get target centroid
    cursor.execute("""
        SELECT centroid FROM narratives WHERE narrative_id = ?
    """, (narrative_id,))

    row = cursor.fetchone()
    if not row:
        conn.close()
        return []

    target_centroid = pickle.loads(row[0])

    # Get all other narratives
    cursor.execute("""
        SELECT narrative_id, centroid, key_phrases, post_count
        FROM narratives WHERE narrative_id != ?
    """, (narrative_id,))

    similarities = []
    for other_id, centroid_blob, phrases_json, count in cursor.fetchall():
        other_centroid = pickle.loads(centroid_blob)

        sim = np.dot(target_centroid, other_centroid) / (
            np.linalg.norm(target_centroid) * np.linalg.norm(other_centroid) + 1e-8
        )

        similarities.append({
            'narrative_id': other_id,
            'key_phrases': json.loads(phrases_json),
            'post_count': count,
            'similarity': round(float(sim), 4)
        })

    conn.close()

    similarities.sort(key=lambda x: x['similarity'], reverse=True)

    return similarities[:top_k]


def run_all_narrative_analysis() -> Dict[str, Any]:
    """Run all narrative analyses"""
    results = {}

    try:
        narratives = identify_narratives()
        results['narratives'] = {
            'status': 'success',
            'count': len(narratives),
            'largest': narratives[0]['post_count'] if narratives else 0
        }
    except Exception as e:
        results['narratives'] = {'status': 'error', 'error': str(e)}

    try:
        pushes = detect_coordinated_pushes()
        results['coordinated_pushes'] = {
            'status': 'success',
            'count': len(pushes),
            'max_score': pushes[0]['coordination_score'] if pushes else 0
        }
    except Exception as e:
        results['coordinated_pushes'] = {'status': 'error', 'error': str(e)}

    results['completed_at'] = datetime.now().isoformat()

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python narratives.py [identify|roles NARRATIVE_ID|pushes|timeline NARRATIVE_ID|all]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "identify":
        print("Identifying narratives...")
        narratives = identify_narratives()
        print(f"Found {len(narratives)} narratives:")
        for n in narratives[:10]:
            print(f"  {n['narrative_id']}: {n['post_count']} posts, {n['unique_authors']} authors")
            print(f"    Phrases: {', '.join(n['key_phrases'])}")

    elif command == "roles" and len(sys.argv) >= 3:
        narrative_id = sys.argv[2]
        print(f"Classifying roles for {narrative_id}...")
        roles = classify_narrative_roles(narrative_id)
        print(f"  Total posts: {roles.get('total_posts')}")
        print(f"  Roles: {roles.get('roles')}")
        print(f"  Originators: {roles.get('originators')}")

    elif command == "pushes":
        print("Detecting coordinated pushes...")
        pushes = detect_coordinated_pushes()
        print(f"Found {len(pushes)} coordinated pushes:")
        for p in pushes[:10]:
            print(f"  {p['push_id']}: {p['author_count']} authors, score={p['coordination_score']}")

    elif command == "timeline" and len(sys.argv) >= 3:
        narrative_id = sys.argv[2]
        print(f"Getting timeline for {narrative_id}...")
        timeline = get_narrative_timeline(narrative_id)
        print(f"  Posts: {timeline.get('total_posts')}")
        print(f"  Duration: {timeline.get('duration_hours')} hours")
        print(f"  Rate: {timeline.get('posts_per_hour')} posts/hour")

    elif command == "all":
        print("Running all narrative analysis...")
        results = run_all_narrative_analysis()
        for k, v in results.items():
            print(f"  {k}: {v}")

    else:
        print(f"Unknown command: {command}")
