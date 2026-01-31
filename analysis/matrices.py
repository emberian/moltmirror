#!/usr/bin/env python3
"""
Sparse Matrix Infrastructure for IC-Grade Analysis
D4M-inspired sparse matrix representations for coordination detection
"""

import sqlite3
import pickle
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, lil_matrix
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import os

try:
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer
    LDA_AVAILABLE = True
except ImportError:
    LDA_AVAILABLE = False

DB_PATH = Path(os.getenv("MOLTMIRROR_DB_PATH", "analysis.db"))

# Constants
NUM_TOPICS = 20  # LDA topics
HOURS_PER_WEEK = 168  # 7 * 24 for temporal matrix
SIMILARITY_THRESHOLD = 0.7  # For content similarity matrix


def get_db():
    """Get database connection"""
    return sqlite3.connect(DB_PATH)


def ensure_matrix_table():
    """Ensure matrix_cache table exists"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS matrix_cache (
            matrix_name TEXT PRIMARY KEY,
            data BLOB,
            shape TEXT,
            nnz INTEGER,
            row_labels BLOB,
            col_labels BLOB,
            computed_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_matrix(name: str, matrix: csr_matrix,
                row_labels: Optional[List[str]] = None,
                col_labels: Optional[List[str]] = None):
    """Save a sparse matrix to the cache"""
    ensure_matrix_table()
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO matrix_cache
        (matrix_name, data, shape, nnz, row_labels, col_labels, computed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        name,
        pickle.dumps(matrix),
        f"{matrix.shape[0]}x{matrix.shape[1]}",
        matrix.nnz,
        pickle.dumps(row_labels) if row_labels else None,
        pickle.dumps(col_labels) if col_labels else None,
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()


def load_matrix(name: str) -> Optional[Tuple[csr_matrix, List[str], List[str], datetime]]:
    """Load a sparse matrix from the cache"""
    ensure_matrix_table()
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT data, row_labels, col_labels, computed_at FROM matrix_cache
        WHERE matrix_name = ?
    """, (name,))

    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    data_blob, row_labels_blob, col_labels_blob, computed_at = row

    matrix = pickle.loads(data_blob)
    row_labels = pickle.loads(row_labels_blob) if row_labels_blob else None
    col_labels = pickle.loads(col_labels_blob) if col_labels_blob else None
    computed_dt = datetime.fromisoformat(computed_at)

    return matrix, row_labels, col_labels, computed_dt


def get_matrix_info(name: str) -> Optional[Dict]:
    """Get metadata about a cached matrix without loading it"""
    ensure_matrix_table()
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT shape, nnz, computed_at FROM matrix_cache
        WHERE matrix_name = ?
    """, (name,))

    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        'name': name,
        'shape': row[0],
        'nnz': row[1],
        'computed_at': row[2]
    }


def build_interaction_matrix() -> Tuple[csr_matrix, List[str]]:
    """
    Build Author×Author interaction matrix (who replies to whom)
    Returns sparse matrix and list of author names (row/col labels)
    """
    conn = get_db()
    cursor = conn.cursor()

    # Get all reply interactions
    cursor.execute("""
        SELECT c.author_name, p.author_name, COUNT(*) as interactions
        FROM comments c
        JOIN posts p ON c.post_id = p.id
        WHERE c.author_name IS NOT NULL
        AND p.author_name IS NOT NULL
        AND c.author_name != p.author_name
        GROUP BY c.author_name, p.author_name
    """)

    interactions = cursor.fetchall()
    conn.close()

    # Get unique authors
    authors = set()
    for commenter, poster, _ in interactions:
        authors.add(commenter)
        authors.add(poster)

    author_list = sorted(list(authors))
    author_idx = {author: i for i, author in enumerate(author_list)}
    n = len(author_list)

    # Build sparse matrix
    matrix = lil_matrix((n, n), dtype=np.float32)

    for commenter, poster, count in interactions:
        i = author_idx[commenter]
        j = author_idx[poster]
        matrix[i, j] = count

    csr = matrix.tocsr()

    # Save to cache
    save_matrix('interaction', csr, author_list, author_list)

    return csr, author_list


def build_topic_matrix(n_topics: int = NUM_TOPICS) -> Tuple[csr_matrix, List[str], List[str]]:
    """
    Build Author×Topic matrix using LDA topic modeling
    Returns sparse matrix, author list, and topic labels
    """
    if not LDA_AVAILABLE:
        raise ImportError("scikit-learn required for LDA topic modeling")

    conn = get_db()
    cursor = conn.cursor()

    # Get all content per author
    cursor.execute("""
        SELECT author_name, GROUP_CONCAT(content, ' ') as all_content
        FROM (
            SELECT author_name, content FROM posts
            WHERE author_name IS NOT NULL AND content IS NOT NULL
            UNION ALL
            SELECT author_name, content FROM comments
            WHERE author_name IS NOT NULL AND content IS NOT NULL
        )
        GROUP BY author_name
    """)

    author_content = cursor.fetchall()
    conn.close()

    if not author_content:
        return csr_matrix((0, n_topics)), [], []

    authors = [row[0] for row in author_content]
    documents = [row[1] or '' for row in author_content]

    # Vectorize documents
    vectorizer = CountVectorizer(
        max_features=5000,
        stop_words='english',
        min_df=2,
        max_df=0.95
    )

    try:
        doc_term_matrix = vectorizer.fit_transform(documents)
    except ValueError:
        # Not enough documents
        return csr_matrix((len(authors), n_topics)), authors, [f"topic_{i}" for i in range(n_topics)]

    # Fit LDA
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=10,
        learning_method='online'
    )

    topic_matrix = lda.fit_transform(doc_term_matrix)

    # Get top words for each topic (for labels)
    feature_names = vectorizer.get_feature_names_out()
    topic_labels = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-4:-1]]
        topic_labels.append(f"t{topic_idx}:" + ','.join(top_words))

    csr = csr_matrix(topic_matrix.astype(np.float32))

    # Save to cache
    save_matrix('topic', csr, authors, topic_labels)

    return csr, authors, topic_labels


def build_temporal_matrix() -> Tuple[csr_matrix, List[str]]:
    """
    Build Author×TimeSlot matrix (168 hours = 7 days × 24 hours)
    Shows posting cadence per hour-of-week
    """
    conn = get_db()
    cursor = conn.cursor()

    # Get all posts/comments with timestamps
    cursor.execute("""
        SELECT author_name, created_at
        FROM (
            SELECT author_name, created_at FROM posts
            WHERE author_name IS NOT NULL AND created_at IS NOT NULL
            UNION ALL
            SELECT author_name, created_at FROM comments
            WHERE author_name IS NOT NULL AND created_at IS NOT NULL
        )
    """)

    timestamps = cursor.fetchall()
    conn.close()

    # Count activity per hour-of-week per author
    author_hourly = defaultdict(lambda: np.zeros(HOURS_PER_WEEK, dtype=np.float32))

    for author, created_at in timestamps:
        try:
            dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            # Hour of week: day_of_week * 24 + hour
            hour_of_week = dt.weekday() * 24 + dt.hour
            author_hourly[author][hour_of_week] += 1
        except (ValueError, AttributeError):
            continue

    authors = sorted(list(author_hourly.keys()))
    author_idx = {author: i for i, author in enumerate(authors)}
    n = len(authors)

    # Build matrix
    matrix = lil_matrix((n, HOURS_PER_WEEK), dtype=np.float32)

    for author, counts in author_hourly.items():
        i = author_idx[author]
        # Normalize to probability distribution
        total = counts.sum()
        if total > 0:
            matrix[i, :] = counts / total

    csr = matrix.tocsr()

    # Hour labels
    hour_labels = [f"{d}:{h:02d}" for d in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                   for h in range(24)]

    # Save to cache
    save_matrix('temporal', csr, authors, hour_labels)

    return csr, authors


def build_content_similarity_matrix(threshold: float = SIMILARITY_THRESHOLD,
                                     max_items: int = 5000) -> Tuple[csr_matrix, List[str]]:
    """
    Build Content×Content similarity matrix using embeddings
    Only stores similarities above threshold (sparse)
    """
    conn = get_db()
    cursor = conn.cursor()

    # Get embeddings
    cursor.execute("""
        SELECT content_id, embedding FROM embeddings
        WHERE content_type = 'post'
        ORDER BY content_id
        LIMIT ?
    """, (max_items,))

    embeddings_data = cursor.fetchall()
    conn.close()

    if len(embeddings_data) < 2:
        return csr_matrix((0, 0)), []

    content_ids = [row[0] for row in embeddings_data]
    embeddings = np.array([np.frombuffer(row[1], dtype=np.float32)
                          for row in embeddings_data])

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings_normalized = embeddings / norms

    n = len(content_ids)

    # Build sparse similarity matrix (only values above threshold)
    # Process in batches to manage memory
    batch_size = 500
    rows = []
    cols = []
    data = []

    for start_i in range(0, n, batch_size):
        end_i = min(start_i + batch_size, n)
        batch = embeddings_normalized[start_i:end_i]

        # Compute similarities with all other items
        similarities = batch @ embeddings_normalized.T

        # Find pairs above threshold
        for local_i, global_i in enumerate(range(start_i, end_i)):
            for j in range(global_i + 1, n):  # Upper triangle only
                sim = similarities[local_i, j]
                if sim >= threshold:
                    rows.append(global_i)
                    cols.append(j)
                    data.append(sim)
                    # Make symmetric
                    rows.append(j)
                    cols.append(global_i)
                    data.append(sim)

    # Create sparse matrix
    if data:
        matrix = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    else:
        matrix = csr_matrix((n, n), dtype=np.float32)

    # Save to cache
    save_matrix('content_similarity', matrix, content_ids, content_ids)

    return matrix, content_ids


def incremental_update_interaction(new_interactions: List[Tuple[str, str, int]]):
    """
    Incrementally update interaction matrix with new data
    new_interactions: list of (commenter, poster, count) tuples
    """
    cached = load_matrix('interaction')

    if cached is None:
        # No existing matrix, build from scratch
        build_interaction_matrix()
        return

    matrix, authors, _, _ = cached
    author_idx = {author: i for i, author in enumerate(authors)}

    # Check for new authors
    new_authors = set()
    for commenter, poster, _ in new_interactions:
        if commenter not in author_idx:
            new_authors.add(commenter)
        if poster not in author_idx:
            new_authors.add(poster)

    if new_authors:
        # Need to expand matrix
        old_n = len(authors)
        new_n = old_n + len(new_authors)

        # Create new expanded matrix
        expanded = lil_matrix((new_n, new_n), dtype=np.float32)
        expanded[:old_n, :old_n] = matrix.tolil()

        # Add new authors
        for new_author in sorted(new_authors):
            author_idx[new_author] = len(authors)
            authors.append(new_author)

        matrix = expanded
    else:
        matrix = matrix.tolil()

    # Add new interactions
    for commenter, poster, count in new_interactions:
        i = author_idx[commenter]
        j = author_idx[poster]
        matrix[i, j] += count

    csr = matrix.tocsr()
    save_matrix('interaction', csr, authors, authors)


def get_author_interaction_vector(author: str) -> Optional[np.ndarray]:
    """Get interaction vector for a specific author"""
    cached = load_matrix('interaction')
    if cached is None:
        return None

    matrix, authors, _, _ = cached

    if author not in authors:
        return None

    idx = authors.index(author)

    # Return both outgoing and incoming as combined vector
    outgoing = matrix.getrow(idx).toarray().flatten()
    incoming = matrix.getcol(idx).toarray().flatten()

    return np.concatenate([outgoing, incoming])


def get_author_topic_distribution(author: str) -> Optional[np.ndarray]:
    """Get topic distribution for a specific author"""
    cached = load_matrix('topic')
    if cached is None:
        return None

    matrix, authors, _, _ = cached

    if author not in authors:
        return None

    idx = authors.index(author)
    return matrix.getrow(idx).toarray().flatten()


def get_author_temporal_pattern(author: str) -> Optional[np.ndarray]:
    """Get temporal activity pattern for a specific author"""
    cached = load_matrix('temporal')
    if cached is None:
        return None

    matrix, authors, _, _ = cached

    if author not in authors:
        return None

    idx = authors.index(author)
    return matrix.getrow(idx).toarray().flatten()


def find_similar_content(content_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
    """Find content similar to given content_id"""
    cached = load_matrix('content_similarity')
    if cached is None:
        return []

    matrix, content_ids, _, _ = cached

    if content_id not in content_ids:
        return []

    idx = content_ids.index(content_id)
    row = matrix.getrow(idx).toarray().flatten()

    # Get top similar items
    top_indices = np.argsort(row)[-top_k-1:-1][::-1]  # Exclude self

    results = []
    for i in top_indices:
        if row[i] > 0:
            results.append((content_ids[i], float(row[i])))

    return results


def compute_combined_matrix(alpha: float = 0.4, beta: float = 0.3, gamma: float = 0.3) -> Optional[Tuple[csr_matrix, List[str]]]:
    """
    Compute combined similarity matrix: M = α×reply + β×timing + γ×content
    Used for coordination detection
    """
    # Load all matrices
    interaction_cached = load_matrix('interaction')
    temporal_cached = load_matrix('temporal')
    topic_cached = load_matrix('topic')

    if not all([interaction_cached, temporal_cached, topic_cached]):
        return None

    interaction, int_authors, _, _ = interaction_cached
    temporal, temp_authors, _, _ = temporal_cached
    topic, topic_authors, _, _ = topic_cached

    # Find common authors
    common_authors = set(int_authors) & set(temp_authors) & set(topic_authors)
    common_authors = sorted(list(common_authors))

    if len(common_authors) < 2:
        return None

    n = len(common_authors)
    author_idx = {author: i for i, author in enumerate(common_authors)}

    # Extract submatrices for common authors
    def extract_submatrix(matrix, authors, common):
        idx_map = {a: authors.index(a) for a in common if a in authors}
        new_matrix = lil_matrix((len(common), matrix.shape[1]), dtype=np.float32)
        for new_i, author in enumerate(common):
            if author in idx_map:
                old_i = idx_map[author]
                new_matrix[new_i, :] = matrix.getrow(old_i).toarray()
        return new_matrix.tocsr()

    int_sub = extract_submatrix(interaction, int_authors, common_authors)
    temp_sub = extract_submatrix(temporal, temp_authors, common_authors)
    topic_sub = extract_submatrix(topic, topic_authors, common_authors)

    # Compute pairwise similarities for each feature
    # Normalize rows
    def normalize_rows(m):
        norms = sparse.linalg.norm(m, axis=1)
        norms[norms == 0] = 1
        return m.multiply(1 / norms.reshape(-1, 1))

    int_norm = normalize_rows(int_sub[:, :n] if int_sub.shape[1] >= n else int_sub)
    temp_norm = normalize_rows(temp_sub)
    topic_norm = normalize_rows(topic_sub)

    # Compute similarity matrices (cosine similarity)
    int_sim = int_norm @ int_norm.T
    temp_sim = temp_norm @ temp_norm.T
    topic_sim = topic_norm @ topic_norm.T

    # Combined matrix
    combined = alpha * int_sim + beta * temp_sim + gamma * topic_sim

    # Convert to CSR
    if sparse.issparse(combined):
        combined_csr = combined.tocsr()
    else:
        combined_csr = csr_matrix(combined)

    # Save to cache
    save_matrix('combined', combined_csr, common_authors, common_authors)

    return combined_csr, common_authors


def build_all_matrices() -> Dict[str, Any]:
    """Build all matrices and return status"""
    results = {}

    try:
        matrix, authors = build_interaction_matrix()
        results['interaction'] = {
            'status': 'success',
            'shape': matrix.shape,
            'nnz': matrix.nnz,
            'authors': len(authors)
        }
    except Exception as e:
        results['interaction'] = {'status': 'error', 'error': str(e)}

    try:
        matrix, authors, topics = build_topic_matrix()
        results['topic'] = {
            'status': 'success',
            'shape': matrix.shape,
            'authors': len(authors),
            'topics': len(topics)
        }
    except Exception as e:
        results['topic'] = {'status': 'error', 'error': str(e)}

    try:
        matrix, authors = build_temporal_matrix()
        results['temporal'] = {
            'status': 'success',
            'shape': matrix.shape,
            'authors': len(authors)
        }
    except Exception as e:
        results['temporal'] = {'status': 'error', 'error': str(e)}

    try:
        matrix, content_ids = build_content_similarity_matrix()
        results['content_similarity'] = {
            'status': 'success',
            'shape': matrix.shape,
            'nnz': matrix.nnz,
            'items': len(content_ids)
        }
    except Exception as e:
        results['content_similarity'] = {'status': 'error', 'error': str(e)}

    try:
        result = compute_combined_matrix()
        if result:
            matrix, authors = result
            results['combined'] = {
                'status': 'success',
                'shape': matrix.shape,
                'authors': len(authors)
            }
        else:
            results['combined'] = {'status': 'skipped', 'reason': 'insufficient data'}
    except Exception as e:
        results['combined'] = {'status': 'error', 'error': str(e)}

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python matrices.py [build|info|test]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "build":
        print("Building all matrices...")
        results = build_all_matrices()
        for name, status in results.items():
            print(f"  {name}: {status}")

    elif command == "info":
        print("Matrix cache info:")
        for name in ['interaction', 'topic', 'temporal', 'content_similarity', 'combined']:
            info = get_matrix_info(name)
            if info:
                print(f"  {name}: {info['shape']}, nnz={info['nnz']}, computed={info['computed_at']}")
            else:
                print(f"  {name}: not cached")

    elif command == "test":
        # Test with a sample author
        print("Testing matrix operations...")

        cached = load_matrix('temporal')
        if cached:
            matrix, authors, _, _ = cached
            if authors:
                test_author = authors[0]
                pattern = get_author_temporal_pattern(test_author)
                print(f"  Temporal pattern for {test_author}: max={pattern.max():.3f}")

    else:
        print(f"Unknown command: {command}")
