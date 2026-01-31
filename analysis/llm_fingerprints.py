#!/usr/bin/env python3
"""
Embedding-Based Agent Fingerprinting for Moltbook

Focused on grounded, observable metrics:
- Semantic consistency using embeddings
- Content originality via embedding comparisons
- Interaction patterns (measurable engagement)
- Template pattern detection (observable regex patterns)

Removed handwavy speculation about LLM backend detection.
"""

import sqlite3
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter, defaultdict
from dataclasses import dataclass
import re
import os

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

DB_PATH = Path(os.getenv("MOLTMIRROR_DB_PATH", "analysis.db"))

# Simplified dimensions - just observable components
DIM_TEMPLATE = 32     # Observable opening/closing patterns
DIM_SEMANTIC = 32     # Embedding-based consistency
DIM_INTERACTION = 32  # Measurable engagement patterns
DIM_ORIGINALITY = 32  # Embedding-based novelty

TOTAL_DIMENSIONS = DIM_TEMPLATE + DIM_SEMANTIC + DIM_INTERACTION + DIM_ORIGINALITY  # 128


# Observable LLM patterns (regex-detectable, not speculative)
LLM_OPENING_PATTERNS = [
    r'^(I think|I believe|In my opinion|From my perspective)',
    r'^(Let me|Allow me to|I\'d be happy to)',
    r'^(Sure|Certainly|Absolutely|Of course)',
    r'^(Here\'s|Here is|Here are)',
]

LLM_CLOSING_PATTERNS = [
    r'(Hope this helps|I hope this helps)\.?$',
    r'(Let me know if you have any questions)\.?$',
    r'(Feel free to ask|Don\'t hesitate to ask)\.?$',
]


@dataclass
class TemplateCluster:
    """Cluster of agents sharing similar prompt templates"""
    cluster_id: str
    members: List[str]
    template_signature: np.ndarray
    avg_similarity: float
    detected_at: str


def get_db():
    """Get database connection"""
    return sqlite3.connect(DB_PATH)


def ensure_llm_fingerprint_tables():
    """Ensure LLM fingerprint tables exist"""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS llm_fingerprints (
            author_name TEXT PRIMARY KEY,
            full_fingerprint BLOB,
            computed_at TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS template_clusters (
            cluster_id TEXT PRIMARY KEY,
            members TEXT,
            template_signature BLOB,
            avg_similarity REAL,
            detected_at TEXT
        )
    """)

    conn.commit()
    conn.close()


class PromptTemplateDetector:
    """Detect similar system prompts across agents via observable patterns"""

    def __init__(self):
        self.opening_patterns = [re.compile(p, re.IGNORECASE) for p in LLM_OPENING_PATTERNS]
        self.closing_patterns = [re.compile(p, re.IGNORECASE) for p in LLM_CLOSING_PATTERNS]

    def extract_template_features(self, contents: List[str]) -> np.ndarray:
        """Extract observable template patterns from content"""
        features = np.zeros(DIM_TEMPLATE, dtype=np.float32)

        if not contents:
            return features

        total = len(contents)

        # Opening pattern frequencies
        for i, pattern in enumerate(self.opening_patterns[:8]):
            matches = sum(1 for c in contents if c and pattern.search(c[:100]))
            features[i] = matches / total

        # Closing pattern frequencies
        for i, pattern in enumerate(self.closing_patterns[:6]):
            matches = sum(1 for c in contents if c and len(c) > 50 and pattern.search(c[-100:]))
            features[8 + i] = matches / total

        # Formatting patterns (observable)
        for c in contents:
            if not c:
                continue
            features[14] += c.count('**') / total  # Bold markdown
            features[15] += c.count('```') / total  # Code blocks
            features[16] += c.count('\n- ') / total  # Bullet lists
            features[17] += c.count('\n1.') / total  # Numbered lists

        # Self-references (observable hedging patterns)
        all_text = ' '.join(c.lower() for c in contents if c)
        features[18] = all_text.count("i think") / max(total, 1)
        features[19] = all_text.count("i believe") / max(total, 1)
        features[20] = all_text.count("perhaps") / max(total, 1)
        features[21] = all_text.count("might") / max(total, 1)

        # Structure patterns
        features[22] = all_text.count("first,") / max(total, 1)
        features[23] = all_text.count("second,") / max(total, 1)
        features[24] = all_text.count("finally,") / max(total, 1)

        # Question/exclamation frequency
        features[25] = sum(c.count('?') for c in contents if c) / max(total, 1)
        features[26] = sum(c.count('!') for c in contents if c) / max(total, 1)

        return features

    def compute_template_signature(self, author: str) -> np.ndarray:
        """Compute template signature for an author"""
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

        return self.extract_template_features(contents)

    def compute_template_similarity(self, author1: str, author2: str) -> float:
        """Compute template similarity between two agents"""
        sig1 = self.compute_template_signature(author1)
        sig2 = self.compute_template_signature(author2)

        norm1 = np.linalg.norm(sig1)
        norm2 = np.linalg.norm(sig2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(sig1, sig2) / (norm1 * norm2))

    def cluster_by_template(self, min_posts: int = 5, n_clusters: int = 10) -> List[TemplateCluster]:
        """Cluster agents by their template signatures"""
        if not SKLEARN_AVAILABLE:
            return []

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
        """, (min_posts,))

        authors = [row[0] for row in cursor.fetchall()]
        conn.close()

        if len(authors) < n_clusters:
            return []

        signatures = []
        valid_authors = []

        for author in authors:
            sig = self.compute_template_signature(author)
            if np.linalg.norm(sig) > 0:
                signatures.append(sig)
                valid_authors.append(author)

        if len(signatures) < n_clusters:
            return []

        X = np.array(signatures)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        clusters = defaultdict(list)
        for author, label in zip(valid_authors, labels):
            clusters[int(label)].append(author)

        now = datetime.now().isoformat()
        result = []

        for cluster_id, members in clusters.items():
            if len(members) < 2:
                continue

            member_sigs = X[[i for i, a in enumerate(valid_authors) if a in members]]
            centroid = np.mean(member_sigs, axis=0)

            if len(member_sigs) > 1:
                sims = cosine_similarity(member_sigs)
                np.fill_diagonal(sims, 0)
                avg_sim = sims.sum() / (len(member_sigs) * (len(member_sigs) - 1))
            else:
                avg_sim = 1.0

            result.append(TemplateCluster(
                cluster_id=f"template_{cluster_id}",
                members=members[:20],
                template_signature=centroid,
                avg_similarity=float(avg_sim),
                detected_at=now
            ))

        return sorted(result, key=lambda x: len(x.members), reverse=True)


def compute_semantic_consistency(author: str) -> np.ndarray:
    """
    Compute embedding-based semantic consistency.
    Measures how coherent an agent's content is over time.
    """
    conn = get_db()
    cursor = conn.cursor()

    features = np.zeros(DIM_SEMANTIC, dtype=np.float32)

    cursor.execute("""
        SELECT e.embedding, p.created_at
        FROM embeddings e
        JOIN posts p ON e.content_id = p.id AND e.content_type = 'post'
        WHERE p.author_name = ?
        ORDER BY p.created_at
    """, (author,))

    embeddings = []
    for row in cursor.fetchall():
        if row[0]:
            embeddings.append(np.frombuffer(row[0], dtype=np.float32))

    conn.close()

    if len(embeddings) < 2:
        return features

    embeddings = np.array(embeddings)

    # Self-similarity to centroid (topic coherence)
    centroid = np.mean(embeddings, axis=0)
    similarities = []
    for emb in embeddings:
        sim = np.dot(emb, centroid) / (np.linalg.norm(emb) * np.linalg.norm(centroid) + 1e-8)
        similarities.append(sim)

    features[0] = np.mean(similarities)
    features[1] = np.std(similarities) if len(similarities) > 1 else 0
    features[2] = np.min(similarities) if similarities else 0
    features[3] = np.max(similarities) if similarities else 0

    # Temporal consistency (adjacent posts similarity)
    adjacent_sims = []
    for i in range(len(embeddings) - 1):
        sim = np.dot(embeddings[i], embeddings[i+1]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1]) + 1e-8
        )
        adjacent_sims.append(sim)

    if adjacent_sims:
        features[4] = np.mean(adjacent_sims)
        features[5] = np.std(adjacent_sims) if len(adjacent_sims) > 1 else 0

    # Drift over time (first vs last quartile)
    n = len(embeddings)
    if n >= 8:
        early = np.mean(embeddings[:n//4], axis=0)
        late = np.mean(embeddings[-n//4:], axis=0)
        drift = 1.0 - np.dot(early, late) / (np.linalg.norm(early) * np.linalg.norm(late) + 1e-8)
        features[6] = drift

    # Topic diversity (pairwise distances)
    if len(embeddings) > 1:
        pairwise_dists = []
        for i in range(min(len(embeddings), 30)):
            for j in range(i+1, min(len(embeddings), 30)):
                dist = 1.0 - np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8
                )
                pairwise_dists.append(dist)
        if pairwise_dists:
            features[7] = np.mean(pairwise_dists)
            features[8] = np.std(pairwise_dists) if len(pairwise_dists) > 1 else 0

    return features


def compute_interaction_dynamics(author: str) -> np.ndarray:
    """
    Compute measurable interaction patterns.
    """
    conn = get_db()
    cursor = conn.cursor()

    features = np.zeros(DIM_INTERACTION, dtype=np.float32)

    # Post vs comment ratio
    cursor.execute("SELECT COUNT(*) FROM posts WHERE author_name = ?", (author,))
    post_count = cursor.fetchone()[0] or 0

    cursor.execute("SELECT COUNT(*) FROM comments WHERE author_name = ?", (author,))
    comment_count = cursor.fetchone()[0] or 0

    total = post_count + comment_count
    if total > 0:
        features[0] = post_count / total
        features[1] = comment_count / total

    # Engagement received
    cursor.execute("""
        SELECT AVG(upvotes), AVG(comment_count), SUM(upvotes)
        FROM posts WHERE author_name = ?
    """, (author,))
    row = cursor.fetchone()
    if row and row[0] is not None:
        features[2] = min((row[0] or 0) / 50, 1.0)  # Avg upvotes (normalized)
        features[3] = min((row[1] or 0) / 20, 1.0)  # Avg comments (normalized)
        features[4] = min((row[2] or 0) / 1000, 1.0)  # Total upvotes (normalized)

    # Unique authors engaged
    cursor.execute("""
        SELECT COUNT(DISTINCT p.author_name)
        FROM comments c
        JOIN posts p ON c.post_id = p.id
        WHERE c.author_name = ? AND p.author_name != ?
    """, (author, author))
    unique_engaged = cursor.fetchone()[0] or 0
    features[5] = min(unique_engaged / 100, 1.0)

    # Self-reply rate
    cursor.execute("""
        SELECT COUNT(*)
        FROM comments c
        JOIN posts p ON c.post_id = p.id
        WHERE c.author_name = ? AND p.author_name = ?
    """, (author, author))
    self_replies = cursor.fetchone()[0] or 0
    features[6] = self_replies / max(comment_count, 1)

    conn.close()

    return features


def compute_originality_metrics(author: str) -> np.ndarray:
    """
    Compute embedding-based originality/novelty metrics.
    """
    conn = get_db()
    cursor = conn.cursor()

    features = np.zeros(DIM_ORIGINALITY, dtype=np.float32)

    # Get author's embeddings
    cursor.execute("""
        SELECT e.embedding
        FROM embeddings e
        JOIN posts p ON e.content_id = p.id AND e.content_type = 'post'
        WHERE p.author_name = ?
    """, (author,))

    author_embeddings = []
    for row in cursor.fetchall():
        if row[0]:
            author_embeddings.append(np.frombuffer(row[0], dtype=np.float32))

    if not author_embeddings:
        conn.close()
        return features

    author_embeddings = np.array(author_embeddings)

    # Get sample of other embeddings for comparison
    cursor.execute("""
        SELECT e.embedding
        FROM embeddings e
        JOIN posts p ON e.content_id = p.id AND e.content_type = 'post'
        WHERE p.author_name != ?
        ORDER BY RANDOM()
        LIMIT 500
    """, (author,))

    other_embeddings = []
    for row in cursor.fetchall():
        if row[0]:
            other_embeddings.append(np.frombuffer(row[0], dtype=np.float32))

    conn.close()

    if not other_embeddings:
        return features

    other_embeddings = np.array(other_embeddings)

    # Novelty scores (how unique is each post vs the corpus)
    novelty_scores = []
    for emb in author_embeddings:
        sims = np.dot(other_embeddings, emb) / (
            np.linalg.norm(other_embeddings, axis=1) * np.linalg.norm(emb) + 1e-8
        )
        max_sim = np.max(sims)
        novelty_scores.append(1.0 - max_sim)

    if novelty_scores:
        features[0] = np.mean(novelty_scores)
        features[1] = np.std(novelty_scores) if len(novelty_scores) > 1 else 0
        features[2] = np.min(novelty_scores)
        features[3] = np.max(novelty_scores)
        features[4] = sum(1 for n in novelty_scores if n > 0.5) / len(novelty_scores)

    # Self-similarity (potential repetition/templates)
    if len(author_embeddings) > 1:
        self_sims = []
        for i in range(min(len(author_embeddings), 30)):
            for j in range(i+1, min(len(author_embeddings), 30)):
                sim = np.dot(author_embeddings[i], author_embeddings[j]) / (
                    np.linalg.norm(author_embeddings[i]) * np.linalg.norm(author_embeddings[j]) + 1e-8
                )
                self_sims.append(sim)

        if self_sims:
            features[5] = np.mean(self_sims)
            features[6] = np.max(self_sims)
            features[7] = sum(1 for s in self_sims if s > 0.9) / len(self_sims)

    return features


def compute_llm_fingerprint(author: str) -> np.ndarray:
    """
    Compute 128-dimensional fingerprint using grounded metrics only.
    """
    template_detector = PromptTemplateDetector()

    template = template_detector.compute_template_signature(author)  # 32 dims
    semantic = compute_semantic_consistency(author)  # 32 dims
    interaction = compute_interaction_dynamics(author)  # 32 dims
    originality = compute_originality_metrics(author)  # 32 dims

    fingerprint = np.concatenate([template, semantic, interaction, originality])
    return fingerprint.astype(np.float32)


def save_llm_fingerprint(author: str, fingerprint: np.ndarray):
    """Save fingerprint to database"""
    ensure_llm_fingerprint_tables()
    conn = get_db()
    cursor = conn.cursor()

    now = datetime.now().isoformat()

    cursor.execute("""
        INSERT OR REPLACE INTO llm_fingerprints
        (author_name, full_fingerprint, computed_at)
        VALUES (?, ?, ?)
    """, (author, pickle.dumps(fingerprint), now))

    conn.commit()
    conn.close()


def load_llm_fingerprint(author: str) -> Optional[Tuple[np.ndarray, datetime]]:
    """Load fingerprint from database"""
    ensure_llm_fingerprint_tables()
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT full_fingerprint, computed_at FROM llm_fingerprints
        WHERE author_name = ?
    """, (author,))

    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    fingerprint = pickle.loads(row[0])
    computed_at = datetime.fromisoformat(row[1])

    return fingerprint, computed_at


def fingerprint_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Compute cosine similarity between two fingerprints"""
    norm1 = np.linalg.norm(fp1)
    norm2 = np.linalg.norm(fp2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(fp1, fp2) / (norm1 * norm2))


def compare_llm_fingerprints(author1: str, author2: str) -> Dict[str, Any]:
    """Compare two agents' fingerprints with component breakdown"""
    fp1 = load_llm_fingerprint(author1)
    fp2 = load_llm_fingerprint(author2)

    if not fp1:
        fp1_vec = compute_llm_fingerprint(author1)
        save_llm_fingerprint(author1, fp1_vec)
    else:
        fp1_vec = fp1[0]

    if not fp2:
        fp2_vec = compute_llm_fingerprint(author2)
        save_llm_fingerprint(author2, fp2_vec)
    else:
        fp2_vec = fp2[0]

    overall = fingerprint_similarity(fp1_vec, fp2_vec)

    def component_sim(start, length):
        c1 = fp1_vec[start:start+length]
        c2 = fp2_vec[start:start+length]
        return fingerprint_similarity(c1, c2)

    components = {
        'template': component_sim(0, DIM_TEMPLATE),
        'semantic': component_sim(DIM_TEMPLATE, DIM_SEMANTIC),
        'interaction': component_sim(DIM_TEMPLATE + DIM_SEMANTIC, DIM_INTERACTION),
        'originality': component_sim(DIM_TEMPLATE + DIM_SEMANTIC + DIM_INTERACTION, DIM_ORIGINALITY),
    }

    return {
        'author1': author1,
        'author2': author2,
        'overall_similarity': overall,
        'components': components,
    }


def compute_all_llm_fingerprints(min_activity: int = 5) -> Dict[str, Any]:
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
            fp = compute_llm_fingerprint(author)
            save_llm_fingerprint(author, fp)
            computed += 1
        except Exception:
            errors += 1

    return {
        'total_authors': len(authors),
        'computed': computed,
        'errors': errors,
        'completed_at': datetime.now().isoformat()
    }


def find_similar_llm_agents(author: str, top_k: int = 10) -> List[Tuple[str, float, Dict[str, float]]]:
    """Find agents most similar by fingerprint"""
    target_fp = load_llm_fingerprint(author)
    if not target_fp:
        target_fp_vec = compute_llm_fingerprint(author)
        save_llm_fingerprint(author, target_fp_vec)
    else:
        target_fp_vec = target_fp[0]

    ensure_llm_fingerprint_tables()
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT author_name, full_fingerprint FROM llm_fingerprints
        WHERE author_name != ?
    """, (author,))

    results = []
    for row in cursor.fetchall():
        other_author = row[0]
        other_fp = pickle.loads(row[1])

        sim = fingerprint_similarity(target_fp_vec, other_fp)

        if sim > 0.5:
            comparison = compare_llm_fingerprints(author, other_author)
            results.append((other_author, sim, comparison['components']))
        else:
            results.append((other_author, sim, {}))

    conn.close()

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python llm_fingerprints.py [compute-all|compare AUTHOR1 AUTHOR2|similar AUTHOR|clusters]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "compute-all":
        print("Computing all fingerprints...")
        result = compute_all_llm_fingerprints()
        print(f"  Computed: {result['computed']}")
        print(f"  Errors: {result['errors']}")

    elif command == "compare" and len(sys.argv) >= 4:
        author1 = sys.argv[2]
        author2 = sys.argv[3]
        result = compare_llm_fingerprints(author1, author2)
        print(f"Fingerprint Comparison: {author1} vs {author2}")
        print(f"  Overall similarity: {result['overall_similarity']:.3f}")
        print("  Components:")
        for k, v in result['components'].items():
            print(f"    {k}: {v:.3f}")

    elif command == "similar" and len(sys.argv) >= 3:
        author = sys.argv[2]
        print(f"Agents similar to {author}:")
        for other, sim, components in find_similar_llm_agents(author):
            print(f"  {other}: {sim:.3f}")

    elif command == "clusters":
        print("Clustering agents by template patterns...")
        detector = PromptTemplateDetector()
        clusters = detector.cluster_by_template()
        print(f"Found {len(clusters)} clusters:")
        for cluster in clusters[:10]:
            print(f"  {cluster.cluster_id}: {len(cluster.members)} members, avg_sim={cluster.avg_similarity:.3f}")

    else:
        print(f"Unknown command: {command}")
