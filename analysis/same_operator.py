#!/usr/bin/env python3
"""
Same-Operator Detection for Moltbook Agent Analysis
Detect multiple agents run by the same person/organization

Grounded signals only:
- Template similarity: Same system prompt patterns (observable)
- Activation patterns: Never active together → same person (observable)
- Topic alignment: Similar content via embeddings (grounded)

Removed handwavy speculation:
- Infrastructure correlation (timing patterns don't reliably indicate shared API)
- Error mode similarity (gap correlation is speculation)
"""

import sqlite3
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass
import os

try:
    from sklearn.cluster import AgglomerativeClustering
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

DB_PATH = Path(os.getenv("MOLTMIRROR_DB_PATH", "analysis.db"))


@dataclass
class OperatorSimilarity:
    """Similarity metrics between two agents suggesting same operator"""
    author1: str
    author2: str
    overall_score: float
    template_similarity: float
    activation_overlap: float
    topic_alignment: float
    never_concurrent: bool
    evidence: Dict[str, Any]


@dataclass
class OperatorCluster:
    """Cluster of agents likely run by the same operator"""
    cluster_id: str
    members: List[str]
    confidence: float
    evidence_summary: Dict[str, Any]
    detected_at: str


@dataclass
class CoordinationEvidence:
    """Evidence of coordinated account creation or activity"""
    authors: List[str]
    creation_window_hours: float
    naming_pattern_match: bool
    initial_topics_similarity: float
    activity_correlation: float


def get_db():
    """Get database connection"""
    return sqlite3.connect(DB_PATH)


def ensure_same_operator_tables():
    """Ensure same-operator detection tables exist"""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS same_operator_candidates (
            author1 TEXT,
            author2 TEXT,
            overall_score REAL,
            template_similarity REAL,
            activation_overlap REAL,
            topic_alignment REAL,
            evidence TEXT,
            detected_at TEXT,
            PRIMARY KEY (author1, author2)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS operator_clusters (
            cluster_id TEXT PRIMARY KEY,
            members TEXT,
            evidence_summary TEXT,
            confidence REAL,
            detected_at TEXT
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_same_op_score
        ON same_operator_candidates(overall_score)
    """)

    conn.commit()
    conn.close()


def get_author_activity_timeline(author: str, days: int = 30) -> Dict[str, int]:
    """Get hourly activity counts for an author"""
    conn = get_db()
    cursor = conn.cursor()

    cutoff = (datetime.now() - timedelta(days=days)).isoformat()

    cursor.execute("""
        SELECT created_at FROM (
            SELECT created_at FROM posts WHERE author_name = ? AND created_at > ?
            UNION ALL
            SELECT created_at FROM comments WHERE author_name = ? AND created_at > ?
        )
    """, (author, cutoff, author, cutoff))

    hourly = defaultdict(int)
    for row in cursor.fetchall():
        if row[0]:
            try:
                dt = datetime.fromisoformat(row[0].replace('Z', '+00:00'))
                hour_key = dt.strftime('%Y-%m-%d %H')
                hourly[hour_key] += 1
            except (ValueError, AttributeError):
                continue

    conn.close()
    return dict(hourly)


def compute_activation_overlap(author1: str, author2: str, days: int = 30) -> Tuple[float, bool]:
    """
    Compute activation overlap between two agents.
    Returns (overlap_score, never_concurrent).

    - Low overlap with never_concurrent=True suggests same operator
    - High overlap suggests different operators
    """
    timeline1 = get_author_activity_timeline(author1, days)
    timeline2 = get_author_activity_timeline(author2, days)

    if not timeline1 or not timeline2:
        return 0.0, False

    all_hours = set(timeline1.keys()) | set(timeline2.keys())

    concurrent_hours = 0
    active1_only = 0
    active2_only = 0

    for hour in all_hours:
        a1 = timeline1.get(hour, 0) > 0
        a2 = timeline2.get(hour, 0) > 0

        if a1 and a2:
            concurrent_hours += 1
        elif a1:
            active1_only += 1
        else:
            active2_only += 1

    total_active_hours = active1_only + active2_only + concurrent_hours

    if total_active_hours == 0:
        return 0.0, True

    overlap = concurrent_hours / total_active_hours
    never_concurrent = concurrent_hours == 0

    return overlap, never_concurrent


def compute_topic_alignment(author1: str, author2: str) -> float:
    """
    Compute topic alignment based on semantic similarity of content.
    Uses embedding centroids - grounded in observable content.
    """
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT e.embedding
        FROM embeddings e
        JOIN posts p ON e.content_id = p.id AND e.content_type = 'post'
        WHERE p.author_name = ?
    """, (author1,))

    embeddings1 = [np.frombuffer(row[0], dtype=np.float32)
                   for row in cursor.fetchall() if row[0]]

    cursor.execute("""
        SELECT e.embedding
        FROM embeddings e
        JOIN posts p ON e.content_id = p.id AND e.content_type = 'post'
        WHERE p.author_name = ?
    """, (author2,))

    embeddings2 = [np.frombuffer(row[0], dtype=np.float32)
                   for row in cursor.fetchall() if row[0]]

    conn.close()

    if not embeddings1 or not embeddings2:
        return 0.0

    centroid1 = np.mean(embeddings1, axis=0)
    centroid2 = np.mean(embeddings2, axis=0)

    similarity = np.dot(centroid1, centroid2) / (
        np.linalg.norm(centroid1) * np.linalg.norm(centroid2) + 1e-8
    )

    return float(similarity)


def compute_template_similarity(author1: str, author2: str) -> float:
    """
    Compute template/prompt similarity using LLM fingerprints.
    Uses the simplified fingerprint (template component).
    """
    try:
        from .llm_fingerprints import load_llm_fingerprint, compute_llm_fingerprint, fingerprint_similarity, DIM_TEMPLATE
    except ImportError:
        try:
            from analysis.llm_fingerprints import load_llm_fingerprint, compute_llm_fingerprint, fingerprint_similarity, DIM_TEMPLATE
        except ImportError:
            return 0.0

    fp1 = load_llm_fingerprint(author1)
    fp2 = load_llm_fingerprint(author2)

    if not fp1:
        fp1_vec = compute_llm_fingerprint(author1)
    else:
        fp1_vec = fp1[0]

    if not fp2:
        fp2_vec = compute_llm_fingerprint(author2)
    else:
        fp2_vec = fp2[0]

    # Focus on template component (first DIM_TEMPLATE dims)
    template1 = fp1_vec[:DIM_TEMPLATE]
    template2 = fp2_vec[:DIM_TEMPLATE]

    return fingerprint_similarity(template1, template2)


class SameOperatorDetector:
    """Main class for detecting same-operator accounts"""

    def __init__(self):
        ensure_same_operator_tables()

    def compute_operator_similarity(self, author1: str, author2: str) -> OperatorSimilarity:
        """Compute similarity metrics suggesting same operator"""

        # Template similarity from LLM fingerprints
        template_sim = compute_template_similarity(author1, author2)

        # Activation overlap (low overlap = suspicious)
        overlap, never_concurrent = compute_activation_overlap(author1, author2)

        # Topic alignment via embeddings
        topic_align = compute_topic_alignment(author1, author2)

        # Compute overall score using only grounded metrics
        # Never concurrent is highly suspicious
        overlap_factor = 1.0 - overlap if never_concurrent else 0.5 - abs(overlap - 0.5)

        overall = (
            0.40 * template_sim +
            0.35 * overlap_factor +
            0.25 * topic_align
        )

        # Boost if never concurrent with high template similarity
        if never_concurrent and template_sim > 0.7:
            overall = min(1.0, overall * 1.3)

        evidence = {
            'template_sim': round(template_sim, 3),
            'overlap': round(overlap, 3),
            'never_concurrent': never_concurrent,
            'topic_align': round(topic_align, 3),
        }

        return OperatorSimilarity(
            author1=author1,
            author2=author2,
            overall_score=round(overall, 3),
            template_similarity=round(template_sim, 3),
            activation_overlap=round(overlap, 3),
            topic_alignment=round(topic_align, 3),
            never_concurrent=never_concurrent,
            evidence=evidence
        )

    def detect_all_candidates(self, min_activity: int = 5,
                               min_score: float = 0.6) -> List[OperatorSimilarity]:
        """Detect all same-operator candidate pairs"""
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
            ORDER BY activity DESC
            LIMIT 500
        """, (min_activity,))

        authors = [row[0] for row in cursor.fetchall()]
        conn.close()

        candidates = []
        now = datetime.now().isoformat()

        for i, author1 in enumerate(authors):
            for author2 in authors[i+1:]:
                try:
                    similarity = self.compute_operator_similarity(author1, author2)

                    if similarity.overall_score >= min_score:
                        candidates.append(similarity)
                        self._save_candidate(similarity, now)
                except Exception:
                    continue

        candidates.sort(key=lambda x: x.overall_score, reverse=True)
        return candidates

    def _save_candidate(self, similarity: OperatorSimilarity, timestamp: str):
        """Save a candidate pair to database"""
        conn = get_db()
        cursor = conn.cursor()

        a1, a2 = sorted([similarity.author1, similarity.author2])

        cursor.execute("""
            INSERT OR REPLACE INTO same_operator_candidates
            (author1, author2, overall_score, template_similarity,
             activation_overlap, topic_alignment, evidence, detected_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            a1, a2,
            similarity.overall_score,
            similarity.template_similarity,
            similarity.activation_overlap,
            similarity.topic_alignment,
            json.dumps(similarity.evidence),
            timestamp
        ))

        conn.commit()
        conn.close()

    def cluster_by_operator(self, min_score: float = 0.6) -> List[OperatorCluster]:
        """Cluster agents by likely operator"""
        if not SKLEARN_AVAILABLE:
            return []

        ensure_same_operator_tables()
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT author1, author2, overall_score, evidence
            FROM same_operator_candidates
            WHERE overall_score >= ?
        """, (min_score,))

        pairs = cursor.fetchall()
        conn.close()

        if not pairs:
            return []

        authors = set()
        edges = {}
        for a1, a2, score, evidence in pairs:
            authors.add(a1)
            authors.add(a2)
            edges[(a1, a2)] = (score, json.loads(evidence) if evidence else {})

        authors = list(authors)
        n = len(authors)

        if n < 2:
            return []

        distance_matrix = np.ones((n, n))
        for i, a1 in enumerate(authors):
            for j, a2 in enumerate(authors):
                if i != j:
                    key = tuple(sorted([a1, a2]))
                    if key in edges:
                        distance_matrix[i, j] = 1.0 - edges[key][0]

        np.fill_diagonal(distance_matrix, 0)

        try:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.4,
                metric='precomputed',
                linkage='average'
            )
            labels = clustering.fit_predict(distance_matrix)
        except Exception:
            return []

        clusters = defaultdict(list)
        for author, label in zip(authors, labels):
            clusters[label].append(author)

        now = datetime.now().isoformat()
        result = []

        for cluster_id, members in clusters.items():
            if len(members) < 2:
                continue

            internal_scores = []
            for i, m1 in enumerate(members):
                for m2 in members[i+1:]:
                    key = tuple(sorted([m1, m2]))
                    if key in edges:
                        internal_scores.append(edges[key][0])

            if not internal_scores:
                continue

            confidence = np.mean(internal_scores)

            evidence_summary = {
                'avg_score': round(confidence, 3),
                'min_score': round(min(internal_scores), 3),
                'max_score': round(max(internal_scores), 3),
                'pair_count': len(internal_scores)
            }

            never_concurrent_count = sum(
                1 for i, m1 in enumerate(members)
                for m2 in members[i+1:]
                if tuple(sorted([m1, m2])) in edges and edges[tuple(sorted([m1, m2]))][1].get('never_concurrent', False)
            )
            evidence_summary['never_concurrent_pairs'] = never_concurrent_count

            result.append(OperatorCluster(
                cluster_id=f"operator_{cluster_id}",
                members=members,
                confidence=confidence,
                evidence_summary=evidence_summary,
                detected_at=now
            ))

        conn = get_db()
        cursor = conn.cursor()

        for cluster in result:
            cursor.execute("""
                INSERT OR REPLACE INTO operator_clusters
                (cluster_id, members, evidence_summary, confidence, detected_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                cluster.cluster_id,
                json.dumps(cluster.members),
                json.dumps(cluster.evidence_summary),
                cluster.confidence,
                cluster.detected_at
            ))

        conn.commit()
        conn.close()

        return sorted(result, key=lambda x: len(x.members), reverse=True)

    def detect_coordinated_creation(self, time_window_hours: float = 24) -> List[CoordinationEvidence]:
        """Detect agents that were created in coordinated fashion"""
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT author_name, MIN(created_at) as first_seen
            FROM (
                SELECT author_name, created_at FROM posts WHERE author_name IS NOT NULL
                UNION ALL
                SELECT author_name, created_at FROM comments WHERE author_name IS NOT NULL
            )
            GROUP BY author_name
            ORDER BY first_seen
        """)

        authors_by_time = []
        for name, first_seen in cursor.fetchall():
            if first_seen:
                try:
                    dt = datetime.fromisoformat(first_seen.replace('Z', '+00:00'))
                    authors_by_time.append((name, dt))
                except (ValueError, AttributeError):
                    continue

        conn.close()

        if len(authors_by_time) < 2:
            return []

        results = []
        window = timedelta(hours=time_window_hours)

        i = 0
        while i < len(authors_by_time):
            cluster = [authors_by_time[i]]
            start_time = authors_by_time[i][1]

            j = i + 1
            while j < len(authors_by_time):
                if authors_by_time[j][1] - start_time <= window:
                    cluster.append(authors_by_time[j])
                    j += 1
                else:
                    break

            if len(cluster) >= 3:
                authors = [a[0] for a in cluster]
                naming_match = self._check_naming_pattern(authors)
                topic_sim = self._compute_initial_topic_similarity(authors)
                activity_corr = self._compute_activity_correlation(authors)

                creation_hours = (cluster[-1][1] - cluster[0][1]).total_seconds() / 3600

                if naming_match or topic_sim > 0.7 or activity_corr > 0.6:
                    results.append(CoordinationEvidence(
                        authors=authors,
                        creation_window_hours=round(creation_hours, 2),
                        naming_pattern_match=naming_match,
                        initial_topics_similarity=round(topic_sim, 3),
                        activity_correlation=round(activity_corr, 3)
                    ))

            i = j if j > i + 1 else i + 1

        return results

    def _check_naming_pattern(self, authors: List[str]) -> bool:
        """Check if authors have similar naming patterns"""
        import re

        patterns = []
        for name in authors:
            base = re.sub(r'\d+', '', name.lower())
            patterns.append(base)

        unique = set(patterns)
        if len(unique) < len(patterns):
            return True

        if len(authors) >= 2:
            prefix = os.path.commonprefix(authors)
            if len(prefix) >= 3:
                return True

        return False

    def _compute_initial_topic_similarity(self, authors: List[str]) -> float:
        """Compute topic similarity of authors' first few posts"""
        conn = get_db()
        cursor = conn.cursor()

        centroids = []

        for author in authors:
            cursor.execute("""
                SELECT e.embedding
                FROM embeddings e
                JOIN posts p ON e.content_id = p.id AND e.content_type = 'post'
                WHERE p.author_name = ?
                ORDER BY p.created_at
                LIMIT 5
            """, (author,))

            embeddings = [np.frombuffer(row[0], dtype=np.float32)
                         for row in cursor.fetchall() if row[0]]

            if embeddings:
                centroids.append(np.mean(embeddings, axis=0))

        conn.close()

        if len(centroids) < 2:
            return 0.0

        sims = []
        for i, c1 in enumerate(centroids):
            for c2 in centroids[i+1:]:
                sim = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-8)
                sims.append(sim)

        return float(np.mean(sims))

    def _compute_activity_correlation(self, authors: List[str]) -> float:
        """Compute correlation of activity patterns among authors"""
        timelines = [get_author_activity_timeline(a, days=30) for a in authors]

        if not all(timelines):
            return 0.0

        all_hours = set()
        for t in timelines:
            all_hours.update(t.keys())

        all_hours = sorted(all_hours)

        if len(all_hours) < 10:
            return 0.0

        vectors = []
        for t in timelines:
            vec = np.array([t.get(h, 0) for h in all_hours])
            vectors.append(vec)

        correlations = []
        for i, v1 in enumerate(vectors):
            for v2 in vectors[i+1:]:
                if np.std(v1) > 0 and np.std(v2) > 0:
                    corr = np.corrcoef(v1, v2)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)

        if not correlations:
            return 0.0

        return float(np.mean(correlations))

    def analyze_activation_overlap(self, cluster_members: List[str]) -> Dict[str, Any]:
        """Analyze activation overlap for a cluster of agents"""
        if len(cluster_members) < 2:
            return {'error': 'insufficient_members'}

        n = len(cluster_members)
        overlap_matrix = np.zeros((n, n))
        never_concurrent_matrix = np.zeros((n, n), dtype=bool)

        for i, a1 in enumerate(cluster_members):
            for j, a2 in enumerate(cluster_members):
                if i < j:
                    overlap, never = compute_activation_overlap(a1, a2)
                    overlap_matrix[i, j] = overlap
                    overlap_matrix[j, i] = overlap
                    never_concurrent_matrix[i, j] = never
                    never_concurrent_matrix[j, i] = never

        upper_triangle = overlap_matrix[np.triu_indices(n, k=1)]

        return {
            'members': cluster_members,
            'avg_overlap': round(float(np.mean(upper_triangle)), 3),
            'min_overlap': round(float(np.min(upper_triangle)), 3),
            'max_overlap': round(float(np.max(upper_triangle)), 3),
            'never_concurrent_pairs': int(never_concurrent_matrix.sum() // 2),
            'total_pairs': n * (n - 1) // 2,
            'single_operator_score': round(1.0 - float(np.mean(upper_triangle)), 3)
        }


def get_same_operator_candidates(limit: int = 50, min_score: float = 0.6) -> List[Dict[str, Any]]:
    """Get same-operator candidate pairs from database"""
    ensure_same_operator_tables()
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT author1, author2, overall_score, template_similarity,
               activation_overlap, topic_alignment, evidence, detected_at
        FROM same_operator_candidates
        WHERE overall_score >= ?
        ORDER BY overall_score DESC
        LIMIT ?
    """, (min_score, limit))

    results = []
    for row in cursor.fetchall():
        results.append({
            'author1': row[0],
            'author2': row[1],
            'overall_score': row[2],
            'template_similarity': row[3],
            'activation_overlap': row[4],
            'topic_alignment': row[5],
            'evidence': json.loads(row[6]) if row[6] else {},
            'detected_at': row[7]
        })

    conn.close()
    return results


def get_operator_clusters(limit: int = 20) -> List[Dict[str, Any]]:
    """Get operator clusters from database"""
    ensure_same_operator_tables()
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT cluster_id, members, evidence_summary, confidence, detected_at
        FROM operator_clusters
        ORDER BY confidence DESC
        LIMIT ?
    """, (limit,))

    results = []
    for row in cursor.fetchall():
        results.append({
            'cluster_id': row[0],
            'members': json.loads(row[1]) if row[1] else [],
            'evidence_summary': json.loads(row[2]) if row[2] else {},
            'confidence': row[3],
            'detected_at': row[4]
        })

    conn.close()
    return results


def run_same_operator_detection(min_activity: int = 5,
                                  min_score: float = 0.6) -> Dict[str, Any]:
    """Run complete same-operator detection pipeline"""
    detector = SameOperatorDetector()

    results = {}

    try:
        candidates = detector.detect_all_candidates(min_activity, min_score)
        results['candidates'] = {
            'status': 'success',
            'count': len(candidates),
            'top_score': candidates[0].overall_score if candidates else 0
        }
    except Exception as e:
        results['candidates'] = {'status': 'error', 'error': str(e)}

    try:
        clusters = detector.cluster_by_operator(min_score)
        results['clusters'] = {
            'status': 'success',
            'count': len(clusters),
            'largest': len(clusters[0].members) if clusters else 0
        }
    except Exception as e:
        results['clusters'] = {'status': 'error', 'error': str(e)}

    try:
        coordination = detector.detect_coordinated_creation()
        results['coordinated_creation'] = {
            'status': 'success',
            'count': len(coordination)
        }
    except Exception as e:
        results['coordinated_creation'] = {'status': 'error', 'error': str(e)}

    results['completed_at'] = datetime.now().isoformat()

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python same_operator.py [detect|clusters|compare AUTHOR1 AUTHOR2|coordination]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "detect":
        print("Detecting same-operator candidates...")
        results = run_same_operator_detection()
        for k, v in results.items():
            print(f"  {k}: {v}")

    elif command == "clusters":
        print("Getting operator clusters...")
        clusters = get_operator_clusters()
        for c in clusters:
            print(f"  {c['cluster_id']}: {len(c['members'])} members, confidence={c['confidence']:.3f}")
            print(f"    Members: {', '.join(c['members'][:5])}")

    elif command == "compare" and len(sys.argv) >= 4:
        author1 = sys.argv[2]
        author2 = sys.argv[3]
        detector = SameOperatorDetector()
        result = detector.compute_operator_similarity(author1, author2)
        print(f"Same-operator analysis: {author1} vs {author2}")
        print(f"  Overall score: {result.overall_score}")
        print(f"  Template similarity: {result.template_similarity}")
        print(f"  Activation overlap: {result.activation_overlap}")
        print(f"  Topic alignment: {result.topic_alignment}")
        print(f"  Never concurrent: {result.never_concurrent}")

    elif command == "coordination":
        print("Detecting coordinated account creation...")
        detector = SameOperatorDetector()
        evidence = detector.detect_coordinated_creation()
        for e in evidence[:10]:
            print(f"  {len(e.authors)} accounts in {e.creation_window_hours:.1f}h window")
            print(f"    Naming pattern: {e.naming_pattern_match}, Topic sim: {e.initial_topics_similarity}")
            print(f"    Authors: {', '.join(e.authors[:5])}")

    else:
        print(f"Unknown command: {command}")
