#!/usr/bin/env python3
"""
Behavioral Fingerprinting for IC-Grade Analysis
323-dimensional behavioral signatures for agent identification
"""

import sqlite3
import pickle
import numpy as np
from scipy import sparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter, defaultdict
import re
import os

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

DB_PATH = Path(os.getenv("MOLTMIRROR_DB_PATH", "analysis.db"))

# Fingerprint dimensions
DIM_TEMPORAL = 168       # Hour-of-week posting frequency
DIM_VOCABULARY = 100     # Top term frequencies
DIM_TOPIC = 20           # LDA topic probabilities
DIM_INTERACTION = 10     # Reply patterns, thread participation
DIM_STYLE = 15           # Length, punctuation, emoji, caps
DIM_ENGAGEMENT = 10      # Upvote/comment ratios

TOTAL_DIMENSIONS = DIM_TEMPORAL + DIM_VOCABULARY + DIM_TOPIC + DIM_INTERACTION + DIM_STYLE + DIM_ENGAGEMENT  # 323


def get_db():
    """Get database connection"""
    return sqlite3.connect(DB_PATH)


def ensure_fingerprint_table():
    """Ensure fingerprint tables exist"""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS agent_fingerprints (
            author_name TEXT PRIMARY KEY,
            fingerprint BLOB,
            computed_at TEXT,
            sample_size INTEGER,
            version INTEGER DEFAULT 1
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fingerprint_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            author_name TEXT,
            fingerprint BLOB,
            computed_at TEXT,
            FOREIGN KEY (author_name) REFERENCES agent_fingerprints(author_name)
        )
    """)

    conn.commit()
    conn.close()


def compute_temporal_component(author: str) -> np.ndarray:
    """
    Compute 168-dim temporal component (hour-of-week distribution)
    """
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT created_at FROM (
            SELECT created_at FROM posts WHERE author_name = ?
            UNION ALL
            SELECT created_at FROM comments WHERE author_name = ?
        )
    """, (author, author))

    timestamps = cursor.fetchall()
    conn.close()

    hourly = np.zeros(DIM_TEMPORAL, dtype=np.float32)

    for (created_at,) in timestamps:
        if created_at:
            try:
                dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                hour_of_week = dt.weekday() * 24 + dt.hour
                hourly[hour_of_week] += 1
            except (ValueError, AttributeError):
                continue

    # Normalize to probability distribution
    total = hourly.sum()
    if total > 0:
        hourly = hourly / total

    return hourly


def compute_vocabulary_component(author: str) -> np.ndarray:
    """
    Compute 100-dim vocabulary component (top term frequencies)
    """
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT content FROM (
            SELECT content FROM posts WHERE author_name = ? AND content IS NOT NULL
            UNION ALL
            SELECT content FROM comments WHERE author_name = ? AND content IS NOT NULL
        )
    """, (author, author))

    all_content = ' '.join([row[0] for row in cursor.fetchall() if row[0]])
    conn.close()

    # Simple tokenization
    words = re.findall(r'\b[a-z]{3,15}\b', all_content.lower())

    # Count frequencies
    word_counts = Counter(words)

    # Get top N words globally for consistency
    # Use a fixed vocabulary based on common English words + tech terms
    common_vocab = [
        'the', 'and', 'that', 'have', 'for', 'not', 'with', 'you', 'this', 'but',
        'from', 'they', 'would', 'there', 'their', 'what', 'about', 'which', 'when', 'one',
        'can', 'had', 'were', 'all', 'your', 'how', 'been', 'has', 'more', 'some',
        'could', 'into', 'other', 'than', 'then', 'now', 'only', 'just', 'think', 'know',
        'agent', 'model', 'system', 'data', 'code', 'user', 'time', 'people', 'work', 'way',
        'make', 'like', 'use', 'new', 'good', 'first', 'need', 'want', 'get', 'see',
        'also', 'well', 'really', 'should', 'even', 'most', 'any', 'much', 'very', 'still',
        'actually', 'something', 'maybe', 'right', 'thing', 'going', 'will', 'does', 'being', 'here',
        'each', 'own', 'same', 'different', 'part', 'point', 'case', 'question', 'problem', 'example',
        'best', 'long', 'great', 'little', 'after', 'things', 'through', 'back', 'always', 'between'
    ]

    # Count occurrences of each vocab word
    total_words = len(words) if words else 1
    vocab_vector = np.zeros(DIM_VOCABULARY, dtype=np.float32)

    for i, word in enumerate(common_vocab[:DIM_VOCABULARY]):
        vocab_vector[i] = word_counts.get(word, 0) / total_words

    return vocab_vector


def compute_topic_component(author: str) -> np.ndarray:
    """
    Compute 20-dim topic component from cached LDA results
    Falls back to zero vector if not available
    """
    try:
        from .matrices import load_matrix
        cached = load_matrix('topic')
        if cached:
            matrix, authors, _, _ = cached
            if author in authors:
                idx = authors.index(author)
                return matrix.getrow(idx).toarray().flatten()
    except (ImportError, Exception):
        pass

    return np.zeros(DIM_TOPIC, dtype=np.float32)


def compute_interaction_component(author: str) -> np.ndarray:
    """
    Compute 10-dim interaction component:
    - Reply rate (replies to others vs own posts)
    - Thread participation depth
    - Response latency patterns
    - Self-reply frequency
    - Unique authors engaged
    """
    conn = get_db()
    cursor = conn.cursor()

    interaction = np.zeros(DIM_INTERACTION, dtype=np.float32)

    # Post count
    cursor.execute("SELECT COUNT(*) FROM posts WHERE author_name = ?", (author,))
    post_count = cursor.fetchone()[0] or 0

    # Comment count
    cursor.execute("SELECT COUNT(*) FROM comments WHERE author_name = ?", (author,))
    comment_count = cursor.fetchone()[0] or 0

    # Replies to others
    cursor.execute("""
        SELECT COUNT(DISTINCT p.author_name)
        FROM comments c
        JOIN posts p ON c.post_id = p.id
        WHERE c.author_name = ? AND p.author_name != ?
    """, (author, author))
    unique_authors_engaged = cursor.fetchone()[0] or 0

    # Self replies (comments on own posts)
    cursor.execute("""
        SELECT COUNT(*)
        FROM comments c
        JOIN posts p ON c.post_id = p.id
        WHERE c.author_name = ? AND p.author_name = ?
    """, (author, author))
    self_replies = cursor.fetchone()[0] or 0

    # Thread participation (avg comments per thread)
    cursor.execute("""
        SELECT AVG(cnt) FROM (
            SELECT post_id, COUNT(*) as cnt
            FROM comments WHERE author_name = ?
            GROUP BY post_id
        )
    """, (author,))
    avg_comments_per_thread = cursor.fetchone()[0] or 0

    # Received replies
    cursor.execute("""
        SELECT COUNT(*)
        FROM comments c
        JOIN posts p ON c.post_id = p.id
        WHERE p.author_name = ? AND c.author_name != ?
    """, (author, author))
    received_replies = cursor.fetchone()[0] or 0

    conn.close()

    # Normalize and assign
    total_activity = post_count + comment_count
    if total_activity > 0:
        interaction[0] = comment_count / total_activity  # Comment ratio
        interaction[1] = post_count / total_activity     # Post ratio

    interaction[2] = min(unique_authors_engaged / 100, 1.0)  # Engagement diversity
    interaction[3] = min(self_replies / max(comment_count, 1), 1.0)  # Self-reply rate
    interaction[4] = min(avg_comments_per_thread / 10, 1.0)  # Thread depth
    interaction[5] = min(received_replies / max(post_count, 1) / 10, 1.0)  # Reply attractiveness
    interaction[6] = 1.0 if post_count > 0 and comment_count > 0 else 0.0  # Dual participation
    interaction[7] = min(post_count / 100, 1.0)  # Post volume
    interaction[8] = min(comment_count / 200, 1.0)  # Comment volume
    interaction[9] = min(total_activity / 300, 1.0)  # Total activity

    return interaction


def compute_style_component(author: str) -> np.ndarray:
    """
    Compute 15-dim style component:
    - Average length, length variance
    - Punctuation patterns
    - Emoji usage
    - Capitalization
    - Question marks, exclamation marks
    - Link/URL usage
    - Code block usage
    """
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT content FROM (
            SELECT content FROM posts WHERE author_name = ? AND content IS NOT NULL
            UNION ALL
            SELECT content FROM comments WHERE author_name = ? AND content IS NOT NULL
        )
    """, (author, author))

    contents = [row[0] for row in cursor.fetchall() if row[0]]
    conn.close()

    style = np.zeros(DIM_STYLE, dtype=np.float32)

    if not contents:
        return style

    lengths = [len(c) for c in contents]
    total_chars = sum(lengths)

    # Length stats
    style[0] = min(np.mean(lengths) / 1000, 1.0)  # Average length
    style[1] = min(np.std(lengths) / 500, 1.0) if len(lengths) > 1 else 0  # Length variance

    all_text = ' '.join(contents)

    # Punctuation
    style[2] = min(all_text.count('.') / total_chars * 100, 1.0) if total_chars else 0
    style[3] = min(all_text.count(',') / total_chars * 100, 1.0) if total_chars else 0
    style[4] = min(all_text.count('?') / total_chars * 100, 1.0) if total_chars else 0
    style[5] = min(all_text.count('!') / total_chars * 100, 1.0) if total_chars else 0

    # Capitalization
    upper_count = sum(1 for c in all_text if c.isupper())
    style[6] = min(upper_count / total_chars * 10, 1.0) if total_chars else 0

    # URLs
    url_count = len(re.findall(r'https?://', all_text))
    style[7] = min(url_count / len(contents), 1.0)

    # Code blocks
    code_count = all_text.count('```')
    style[8] = min(code_count / len(contents), 1.0)

    # Emoji (simple pattern)
    emoji_pattern = re.compile(r'[\U0001F300-\U0001F9FF]')
    emoji_count = len(emoji_pattern.findall(all_text))
    style[9] = min(emoji_count / len(contents), 1.0)

    # Newlines/paragraphs
    newline_count = all_text.count('\n')
    style[10] = min(newline_count / total_chars * 50, 1.0) if total_chars else 0

    # Quotes
    quote_count = all_text.count('"') + all_text.count("'")
    style[11] = min(quote_count / total_chars * 100, 1.0) if total_chars else 0

    # Numbers
    number_count = len(re.findall(r'\d+', all_text))
    style[12] = min(number_count / len(contents), 1.0)

    # Parentheses
    paren_count = all_text.count('(') + all_text.count(')')
    style[13] = min(paren_count / total_chars * 100, 1.0) if total_chars else 0

    # List markers
    list_count = len(re.findall(r'^[\-\*\d\.]\s', all_text, re.MULTILINE))
    style[14] = min(list_count / len(contents), 1.0)

    return style


def compute_engagement_component(author: str) -> np.ndarray:
    """
    Compute 10-dim engagement component:
    - Upvote patterns (given/received)
    - Comment ratios
    - Post performance consistency
    """
    conn = get_db()
    cursor = conn.cursor()

    engagement = np.zeros(DIM_ENGAGEMENT, dtype=np.float32)

    # Post stats
    cursor.execute("""
        SELECT AVG(upvotes), AVG(comment_count),
               MAX(upvotes), MIN(upvotes), COUNT(*)
        FROM posts WHERE author_name = ?
    """, (author,))
    post_stats = cursor.fetchone()

    if post_stats and post_stats[4] > 0:
        avg_up, avg_comments, max_up, min_up, post_count = post_stats
        avg_up = avg_up or 0
        avg_comments = avg_comments or 0
        max_up = max_up or 0
        min_up = min_up or 0

        engagement[0] = min(avg_up / 50, 1.0)  # Avg upvotes
        engagement[1] = min(avg_comments / 20, 1.0)  # Avg comments
        engagement[2] = min(max_up / 100, 1.0)  # Max upvotes
        engagement[3] = (max_up - min_up) / max(max_up, 1)  # Consistency

        # Comments-to-upvotes ratio
        if avg_up > 0:
            engagement[4] = min(avg_comments / avg_up, 1.0)

    # Comment stats
    cursor.execute("""
        SELECT AVG(upvotes), MAX(upvotes)
        FROM comments WHERE author_name = ?
    """, (author,))
    comment_stats = cursor.fetchone()

    if comment_stats:
        c_avg_up = comment_stats[0] or 0
        c_max_up = comment_stats[1] or 0
        engagement[5] = min(c_avg_up / 20, 1.0)
        engagement[6] = min(c_max_up / 50, 1.0)

    # Post frequency
    cursor.execute("""
        SELECT MIN(created_at), MAX(created_at), COUNT(*)
        FROM posts WHERE author_name = ?
    """, (author,))
    time_range = cursor.fetchone()

    if time_range and time_range[2] > 1:
        try:
            first = datetime.fromisoformat(time_range[0].replace('Z', '+00:00'))
            last = datetime.fromisoformat(time_range[1].replace('Z', '+00:00'))
            days = (last - first).days + 1
            if days > 0:
                engagement[7] = min(time_range[2] / days, 1.0)  # Posts per day
        except (ValueError, AttributeError):
            pass

    # Total karma
    cursor.execute("""
        SELECT COALESCE(SUM(upvotes), 0) FROM (
            SELECT upvotes FROM posts WHERE author_name = ?
            UNION ALL
            SELECT upvotes FROM comments WHERE author_name = ?
        )
    """, (author, author))
    total_karma = cursor.fetchone()[0] or 0
    engagement[8] = min(total_karma / 1000, 1.0)

    # Engagement ratio
    cursor.execute("""
        SELECT COUNT(*) FROM comments c
        JOIN posts p ON c.post_id = p.id
        WHERE p.author_name = ?
    """, (author,))
    received_engagement = cursor.fetchone()[0] or 0
    cursor.execute("SELECT COUNT(*) FROM posts WHERE author_name = ?", (author,))
    own_posts = cursor.fetchone()[0] or 1
    engagement[9] = min(received_engagement / own_posts / 10, 1.0)

    conn.close()

    return engagement


def compute_fingerprint(author: str) -> np.ndarray:
    """
    Compute complete 323-dimensional fingerprint for an author
    """
    # Compute all components
    temporal = compute_temporal_component(author)
    vocabulary = compute_vocabulary_component(author)
    topic = compute_topic_component(author)
    interaction = compute_interaction_component(author)
    style = compute_style_component(author)
    engagement = compute_engagement_component(author)

    # Concatenate into single vector
    fingerprint = np.concatenate([
        temporal,     # 168 dims
        vocabulary,   # 100 dims
        topic,        # 20 dims
        interaction,  # 10 dims
        style,        # 15 dims
        engagement    # 10 dims
    ])

    return fingerprint.astype(np.float32)


def save_fingerprint(author: str, fingerprint: np.ndarray, sample_size: int = 0):
    """Save fingerprint to database"""
    ensure_fingerprint_table()
    conn = get_db()
    cursor = conn.cursor()

    now = datetime.now().isoformat()

    # Save to history first
    cursor.execute("""
        INSERT INTO fingerprint_history (author_name, fingerprint, computed_at)
        VALUES (?, ?, ?)
    """, (author, pickle.dumps(fingerprint), now))

    # Update current fingerprint
    cursor.execute("""
        INSERT OR REPLACE INTO agent_fingerprints
        (author_name, fingerprint, computed_at, sample_size)
        VALUES (?, ?, ?, ?)
    """, (author, pickle.dumps(fingerprint), now, sample_size))

    conn.commit()
    conn.close()


def load_fingerprint(author: str) -> Optional[Tuple[np.ndarray, datetime, int]]:
    """Load fingerprint from database"""
    ensure_fingerprint_table()
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT fingerprint, computed_at, sample_size
        FROM agent_fingerprints WHERE author_name = ?
    """, (author,))

    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    fingerprint = pickle.loads(row[0])
    computed_at = datetime.fromisoformat(row[1])
    sample_size = row[2] or 0

    return fingerprint, computed_at, sample_size


def fingerprint_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Compute cosine similarity between two fingerprints"""
    norm1 = np.linalg.norm(fp1)
    norm2 = np.linalg.norm(fp2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(fp1, fp2) / (norm1 * norm2))


def compare_authors(author1: str, author2: str) -> Dict[str, Any]:
    """Compare two authors' fingerprints with component breakdown"""
    fp1 = load_fingerprint(author1)
    fp2 = load_fingerprint(author2)

    if not fp1 or not fp2:
        # Compute if not cached
        if not fp1:
            fp1_vec = compute_fingerprint(author1)
            save_fingerprint(author1, fp1_vec)
        else:
            fp1_vec = fp1[0]

        if not fp2:
            fp2_vec = compute_fingerprint(author2)
            save_fingerprint(author2, fp2_vec)
        else:
            fp2_vec = fp2[0]
    else:
        fp1_vec = fp1[0]
        fp2_vec = fp2[0]

    # Overall similarity
    overall = fingerprint_similarity(fp1_vec, fp2_vec)

    # Component similarities
    def component_sim(start, length):
        c1 = fp1_vec[start:start+length]
        c2 = fp2_vec[start:start+length]
        return fingerprint_similarity(c1, c2)

    offset = 0
    components = {}

    components['temporal'] = component_sim(offset, DIM_TEMPORAL)
    offset += DIM_TEMPORAL

    components['vocabulary'] = component_sim(offset, DIM_VOCABULARY)
    offset += DIM_VOCABULARY

    components['topic'] = component_sim(offset, DIM_TOPIC)
    offset += DIM_TOPIC

    components['interaction'] = component_sim(offset, DIM_INTERACTION)
    offset += DIM_INTERACTION

    components['style'] = component_sim(offset, DIM_STYLE)
    offset += DIM_STYLE

    components['engagement'] = component_sim(offset, DIM_ENGAGEMENT)

    return {
        'author1': author1,
        'author2': author2,
        'overall_similarity': overall,
        'components': components,
        'sockpuppet_likely': overall > 0.85 and components['temporal'] > 0.8
    }


def detect_behavior_change(author: str, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Detect significant behavior changes over time
    Compares historical fingerprints
    """
    ensure_fingerprint_table()
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT fingerprint, computed_at FROM fingerprint_history
        WHERE author_name = ?
        ORDER BY computed_at ASC
    """, (author,))

    history = cursor.fetchall()
    conn.close()

    if len(history) < 2:
        return {
            'author': author,
            'changes_detected': False,
            'reason': 'insufficient_history',
            'history_count': len(history)
        }

    changes = []
    fingerprints = [(pickle.loads(row[0]), row[1]) for row in history]

    for i in range(1, len(fingerprints)):
        fp_prev, time_prev = fingerprints[i-1]
        fp_curr, time_curr = fingerprints[i]

        delta = 1.0 - fingerprint_similarity(fp_prev, fp_curr)

        if delta > threshold:
            changes.append({
                'from_time': time_prev,
                'to_time': time_curr,
                'delta': delta
            })

    return {
        'author': author,
        'changes_detected': len(changes) > 0,
        'change_events': changes,
        'history_count': len(history)
    }


def cluster_agents_by_fingerprint(n_clusters: int = 10,
                                   min_activity: int = 5) -> Dict[str, Any]:
    """
    Cluster agents by behavioral fingerprint using K-means
    """
    if not SKLEARN_AVAILABLE:
        return {'error': 'sklearn not available'}

    conn = get_db()
    cursor = conn.cursor()

    # Get agents with sufficient activity
    cursor.execute("""
        SELECT author_name, COUNT(*) as activity FROM (
            SELECT author_name FROM posts WHERE author_name IS NOT NULL
            UNION ALL
            SELECT author_name FROM comments WHERE author_name IS NOT NULL
        )
        GROUP BY author_name
        HAVING activity >= ?
    """, (min_activity,))

    active_authors = [row[0] for row in cursor.fetchall()]
    conn.close()

    if len(active_authors) < n_clusters:
        return {'error': 'insufficient_authors', 'count': len(active_authors)}

    # Load or compute fingerprints
    fingerprints = []
    authors = []

    for author in active_authors:
        fp = load_fingerprint(author)
        if fp:
            fingerprints.append(fp[0])
            authors.append(author)
        else:
            fp_vec = compute_fingerprint(author)
            if np.linalg.norm(fp_vec) > 0:  # Non-zero fingerprint
                save_fingerprint(author, fp_vec)
                fingerprints.append(fp_vec)
                authors.append(author)

    if len(fingerprints) < n_clusters:
        return {'error': 'insufficient_fingerprints', 'count': len(fingerprints)}

    X = np.array(fingerprints)

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # Group by cluster
    clusters = defaultdict(list)
    for author, label in zip(authors, labels):
        clusters[int(label)].append(author)

    # Compute cluster stats
    cluster_info = []
    for cluster_id, members in clusters.items():
        # Get cluster centroid similarity
        member_fps = X[np.array(labels) == cluster_id]
        centroid = kmeans.cluster_centers_[cluster_id]

        avg_dist = np.mean([np.linalg.norm(fp - centroid) for fp in member_fps])

        cluster_info.append({
            'cluster_id': cluster_id,
            'size': len(members),
            'members': members[:20],  # Top 20 for display
            'cohesion': 1.0 / (1.0 + avg_dist),  # Higher = more similar
        })

    cluster_info.sort(key=lambda x: x['size'], reverse=True)

    return {
        'n_clusters': n_clusters,
        'total_agents': len(authors),
        'clusters': cluster_info
    }


def find_similar_agents(author: str, top_k: int = 10) -> List[Tuple[str, float]]:
    """Find agents most similar to the given author"""
    target_fp = load_fingerprint(author)
    if not target_fp:
        target_fp_vec = compute_fingerprint(author)
        save_fingerprint(author, target_fp_vec)
    else:
        target_fp_vec = target_fp[0]

    ensure_fingerprint_table()
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT author_name, fingerprint FROM agent_fingerprints
        WHERE author_name != ?
    """, (author,))

    results = []
    for row in cursor.fetchall():
        other_author = row[0]
        other_fp = pickle.loads(row[1])

        sim = fingerprint_similarity(target_fp_vec, other_fp)
        results.append((other_author, sim))

    conn.close()

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def compute_all_fingerprints(min_activity: int = 3) -> Dict[str, Any]:
    """Compute fingerprints for all active agents"""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT author_name, COUNT(*) as activity FROM (
            SELECT author_name FROM posts WHERE author_name IS NOT NULL
            UNION ALL
            SELECT author_name FROM comments WHERE author_name IS NOT NULL
        )
        GROUP BY author_name
        HAVING activity >= ?
    """, (min_activity,))

    authors = cursor.fetchall()
    conn.close()

    computed = 0
    errors = 0

    for author, activity in authors:
        try:
            fp = compute_fingerprint(author)
            save_fingerprint(author, fp, sample_size=activity)
            computed += 1
        except Exception as e:
            errors += 1

    return {
        'total_authors': len(authors),
        'computed': computed,
        'errors': errors,
        'completed_at': datetime.now().isoformat()
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python fingerprints.py [compute-all|compare AUTHOR1 AUTHOR2|similar AUTHOR|clusters]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "compute-all":
        print("Computing all fingerprints...")
        result = compute_all_fingerprints()
        print(f"  Computed: {result['computed']}")
        print(f"  Errors: {result['errors']}")

    elif command == "compare" and len(sys.argv) >= 4:
        author1 = sys.argv[2]
        author2 = sys.argv[3]
        result = compare_authors(author1, author2)
        print(f"Similarity: {result['overall_similarity']:.3f}")
        print("Components:")
        for k, v in result['components'].items():
            print(f"  {k}: {v:.3f}")
        print(f"Sockpuppet likely: {result['sockpuppet_likely']}")

    elif command == "similar" and len(sys.argv) >= 3:
        author = sys.argv[2]
        print(f"Authors similar to {author}:")
        for other, sim in find_similar_agents(author):
            print(f"  {other}: {sim:.3f}")

    elif command == "clusters":
        result = cluster_agents_by_fingerprint()
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Clustered {result['total_agents']} agents into {result['n_clusters']} clusters:")
            for cluster in result['clusters']:
                print(f"  Cluster {cluster['cluster_id']}: {cluster['size']} members, cohesion={cluster['cohesion']:.3f}")
                print(f"    Sample: {', '.join(cluster['members'][:5])}")

    else:
        print(f"Unknown command: {command}")
