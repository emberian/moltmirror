#!/usr/bin/env python3
"""
Persona Consistency Analysis for Moltbook Agent Analysis

Track whether agents maintain coherent beliefs/personality:
- Stance extraction: What positions does agent hold on topics?
- Contradiction detection: Does agent contradict itself?
- Semantic drift: How much does personality change over time?
- Sudden shifts: Account takeover? Prompt change?
"""

import sqlite3
import pickle
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict
from dataclasses import dataclass, field
import re
import os

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

DB_PATH = Path(os.getenv("MOLTMIRROR_DB_PATH", "analysis.db"))


# Common topics for stance extraction
STANCE_TOPICS = [
    'technology', 'ai_safety', 'regulation', 'privacy', 'open_source',
    'capitalism', 'climate', 'democracy', 'centralization', 'automation',
    'human_values', 'progress', 'tradition', 'individualism', 'community'
]


@dataclass
class StanceVector:
    """Stance on a particular topic"""
    topic: str
    position: float  # -1 (against) to 1 (for)
    confidence: float
    evidence: List[str]


@dataclass
class BeliefProfile:
    """Complete belief profile for an agent"""
    author_name: str
    stances: Dict[str, StanceVector]
    consistency_score: float
    computed_at: str


@dataclass
class Contradiction:
    """Detected self-contradiction"""
    author_name: str
    statement1: str
    statement2: str
    topic: str
    timestamp1: str
    timestamp2: str
    contradiction_score: float
    evidence: Dict[str, Any]


@dataclass
class DriftPoint:
    """Point in semantic drift timeline"""
    timestamp: str
    drift_magnitude: float
    from_centroid_sim: float


@dataclass
class PersonaShift:
    """Detected sudden persona shift"""
    author_name: str
    shift_time: str
    before_profile: Dict[str, float]
    after_profile: Dict[str, float]
    magnitude: float
    likely_cause: str  # 'takeover', 'prompt_change', 'natural_evolution'


def get_db():
    """Get database connection"""
    return sqlite3.connect(DB_PATH)


def ensure_persona_tables():
    """Ensure persona tracking tables exist"""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS persona_profiles (
            author_name TEXT PRIMARY KEY,
            belief_embeddings BLOB,
            consistency_score REAL,
            contradictions TEXT,
            drift_timeline TEXT,
            computed_at TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stance_vectors (
            author_name TEXT,
            topic TEXT,
            position REAL,
            confidence REAL,
            evidence TEXT,
            computed_at TEXT,
            PRIMARY KEY (author_name, topic)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS persona_shifts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            author_name TEXT,
            shift_time TEXT,
            magnitude REAL,
            before_profile TEXT,
            after_profile TEXT,
            likely_cause TEXT,
            detected_at TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS contradictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            author_name TEXT,
            statement1 TEXT,
            statement2 TEXT,
            topic TEXT,
            timestamp1 TEXT,
            timestamp2 TEXT,
            contradiction_score REAL,
            detected_at TEXT
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_persona_author
        ON persona_profiles(author_name)
    """)

    conn.commit()
    conn.close()


class StanceExtractor:
    """Extract stance/position on topics from content"""

    # Keywords indicating positions
    POSITIVE_MARKERS = [
        'support', 'agree', 'favor', 'benefit', 'important', 'essential',
        'necessary', 'valuable', 'good', 'positive', 'advocate', 'embrace'
    ]

    NEGATIVE_MARKERS = [
        'oppose', 'disagree', 'against', 'harmful', 'dangerous', 'problematic',
        'concern', 'risk', 'negative', 'bad', 'reject', 'avoid'
    ]

    TOPIC_KEYWORDS = {
        'technology': ['technology', 'tech', 'digital', 'innovation', 'software'],
        'ai_safety': ['ai safety', 'alignment', 'existential risk', 'control problem'],
        'regulation': ['regulation', 'law', 'policy', 'government oversight', 'rules'],
        'privacy': ['privacy', 'surveillance', 'data protection', 'personal data'],
        'open_source': ['open source', 'open-source', 'free software', 'proprietary'],
        'capitalism': ['capitalism', 'market', 'free market', 'profit', 'business'],
        'climate': ['climate', 'environment', 'sustainability', 'carbon', 'green'],
        'democracy': ['democracy', 'democratic', 'voting', 'elections', 'representation'],
        'centralization': ['centralization', 'decentralization', 'distributed', 'central control'],
        'automation': ['automation', 'jobs', 'employment', 'workers', 'labor'],
        'human_values': ['human values', 'ethics', 'morality', 'humanity'],
        'progress': ['progress', 'advancement', 'improvement', 'development'],
        'tradition': ['tradition', 'traditional', 'conservative', 'heritage'],
        'individualism': ['individual', 'personal freedom', 'autonomy', 'self'],
        'community': ['community', 'collective', 'society', 'together', 'social']
    }

    def extract_stance(self, contents: List[Tuple[str, str]],
                        topic: str) -> Optional[StanceVector]:
        """
        Extract stance on a topic from content list.
        Contents is list of (content, timestamp) tuples.
        """
        if topic not in self.TOPIC_KEYWORDS:
            return None

        keywords = self.TOPIC_KEYWORDS[topic]
        relevant_sentences = []

        for content, timestamp in contents:
            if not content:
                continue

            sentences = re.split(r'[.!?]+', content.lower())

            for sentence in sentences:
                if any(kw in sentence for kw in keywords):
                    relevant_sentences.append((sentence, timestamp))

        if not relevant_sentences:
            return None

        # Score sentiment
        positive_count = 0
        negative_count = 0
        evidence = []

        for sentence, timestamp in relevant_sentences:
            pos = sum(1 for marker in self.POSITIVE_MARKERS if marker in sentence)
            neg = sum(1 for marker in self.NEGATIVE_MARKERS if marker in sentence)

            positive_count += pos
            negative_count += neg

            if pos > 0 or neg > 0:
                evidence.append(sentence[:100])

        total = positive_count + negative_count
        if total == 0:
            return None

        position = (positive_count - negative_count) / total
        confidence = min(total / 10, 1.0)  # More mentions = more confidence

        return StanceVector(
            topic=topic,
            position=position,
            confidence=confidence,
            evidence=evidence[:5]
        )


class PersonaTracker:
    """Track persona consistency for agents"""

    def __init__(self):
        self.stance_extractor = StanceExtractor()

    def extract_stance(self, author: str, topic: str) -> Optional[StanceVector]:
        """Extract stance for an author on a topic"""
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT content, created_at FROM (
                SELECT content, created_at FROM posts
                WHERE author_name = ? AND content IS NOT NULL
                UNION ALL
                SELECT content, created_at FROM comments
                WHERE author_name = ? AND content IS NOT NULL
            )
            ORDER BY created_at
        """, (author, author))

        contents = cursor.fetchall()
        conn.close()

        if not contents:
            return None

        return self.stance_extractor.extract_stance(contents, topic)

    def build_belief_profile(self, author: str) -> BeliefProfile:
        """Build complete belief profile for an author"""
        ensure_persona_tables()

        stances = {}
        now = datetime.now().isoformat()

        for topic in STANCE_TOPICS:
            stance = self.extract_stance(author, topic)
            if stance and stance.confidence > 0.2:
                stances[topic] = stance

        # Compute consistency score
        consistency = self._compute_consistency_score(author)

        profile = BeliefProfile(
            author_name=author,
            stances=stances,
            consistency_score=consistency,
            computed_at=now
        )

        # Save to database
        self._save_profile(profile)

        return profile

    def _compute_consistency_score(self, author: str) -> float:
        """Compute overall consistency score based on embeddings"""
        conn = get_db()
        cursor = conn.cursor()

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
            return 1.0  # Can't measure inconsistency with single post

        embeddings = np.array(embeddings)

        # Compute centroid
        centroid = np.mean(embeddings, axis=0)

        # Compute average similarity to centroid
        similarities = []
        for emb in embeddings:
            sim = np.dot(emb, centroid) / (np.linalg.norm(emb) * np.linalg.norm(centroid) + 1e-8)
            similarities.append(sim)

        # Consistency = mean similarity (higher = more consistent)
        return float(np.mean(similarities))

    def _save_profile(self, profile: BeliefProfile):
        """Save belief profile to database"""
        conn = get_db()
        cursor = conn.cursor()

        # Convert stances to JSON-serializable format
        stances_json = {}
        for topic, stance in profile.stances.items():
            stances_json[topic] = {
                'position': stance.position,
                'confidence': stance.confidence,
                'evidence': stance.evidence
            }

        cursor.execute("""
            INSERT OR REPLACE INTO persona_profiles
            (author_name, belief_embeddings, consistency_score, contradictions,
             drift_timeline, computed_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            profile.author_name,
            pickle.dumps(stances_json),
            profile.consistency_score,
            json.dumps([]),  # Will populate with detect_contradictions
            json.dumps([]),  # Will populate with measure_semantic_drift
            profile.computed_at
        ))

        # Save individual stances
        for topic, stance in profile.stances.items():
            cursor.execute("""
                INSERT OR REPLACE INTO stance_vectors
                (author_name, topic, position, confidence, evidence, computed_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                profile.author_name,
                topic,
                stance.position,
                stance.confidence,
                json.dumps(stance.evidence),
                profile.computed_at
            ))

        conn.commit()
        conn.close()

    def detect_contradictions(self, author: str,
                               similarity_threshold: float = 0.8) -> List[Contradiction]:
        """Detect self-contradictions in an author's content"""
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT p.id, p.content, p.created_at, e.embedding
            FROM posts p
            JOIN embeddings e ON p.id = e.content_id AND e.content_type = 'post'
            WHERE p.author_name = ? AND p.content IS NOT NULL
            ORDER BY p.created_at
        """, (author,))

        posts = []
        for row in cursor.fetchall():
            if row[3]:
                posts.append({
                    'id': row[0],
                    'content': row[1],
                    'timestamp': row[2],
                    'embedding': np.frombuffer(row[3], dtype=np.float32)
                })

        conn.close()

        if len(posts) < 2:
            return []

        contradictions = []

        # Compare posts for potential contradictions
        for i, post1 in enumerate(posts):
            for post2 in posts[i+1:]:
                # High semantic similarity but opposite sentiment = contradiction
                sim = float(np.dot(post1['embedding'], post2['embedding']) / (
                    np.linalg.norm(post1['embedding']) * np.linalg.norm(post2['embedding']) + 1e-8
                ))

                if sim >= similarity_threshold:
                    # Check for sentiment opposition
                    sentiment1 = self._extract_sentiment(post1['content'])
                    sentiment2 = self._extract_sentiment(post2['content'])

                    # Opposite sentiments on similar topic = contradiction
                    if sentiment1 * sentiment2 < -0.3:  # Opposite signs, significant magnitude
                        contradiction_score = sim * abs(sentiment1 - sentiment2) / 2

                        # Determine topic
                        topic = self._identify_topic(post1['content'])

                        contradictions.append(Contradiction(
                            author_name=author,
                            statement1=post1['content'][:200],
                            statement2=post2['content'][:200],
                            topic=topic,
                            timestamp1=post1['timestamp'],
                            timestamp2=post2['timestamp'],
                            contradiction_score=contradiction_score,
                            evidence={
                                'semantic_similarity': round(sim, 3),
                                'sentiment1': round(sentiment1, 3),
                                'sentiment2': round(sentiment2, 3)
                            }
                        ))

        # Sort by contradiction score
        contradictions.sort(key=lambda x: x.contradiction_score, reverse=True)

        # Save to database
        now = datetime.now().isoformat()
        conn = get_db()
        cursor = conn.cursor()

        for c in contradictions[:50]:
            cursor.execute("""
                INSERT INTO contradictions
                (author_name, statement1, statement2, topic, timestamp1, timestamp2,
                 contradiction_score, detected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                c.author_name,
                c.statement1,
                c.statement2,
                c.topic,
                c.timestamp1,
                c.timestamp2,
                c.contradiction_score,
                now
            ))

        conn.commit()
        conn.close()

        return contradictions

    def _extract_sentiment(self, content: str) -> float:
        """Extract simple sentiment score from content"""
        if not content:
            return 0.0

        text = content.lower()

        positive_words = ['good', 'great', 'excellent', 'love', 'amazing', 'wonderful',
                         'beneficial', 'positive', 'support', 'agree', 'yes', 'true']
        negative_words = ['bad', 'terrible', 'hate', 'awful', 'harmful', 'negative',
                         'oppose', 'disagree', 'no', 'false', 'wrong', 'problem']

        pos_count = sum(text.count(w) for w in positive_words)
        neg_count = sum(text.count(w) for w in negative_words)

        total = pos_count + neg_count
        if total == 0:
            return 0.0

        return (pos_count - neg_count) / total

    def _identify_topic(self, content: str) -> str:
        """Identify main topic of content"""
        if not content:
            return 'general'

        text = content.lower()

        for topic, keywords in StanceExtractor.TOPIC_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                return topic

        return 'general'

    def measure_semantic_drift(self, author: str,
                                 window_days: int = 7) -> List[DriftPoint]:
        """Measure how an author's persona drifts over time"""
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT p.created_at, e.embedding
            FROM posts p
            JOIN embeddings e ON p.id = e.content_id AND e.content_type = 'post'
            WHERE p.author_name = ?
            ORDER BY p.created_at
        """, (author,))

        posts = []
        for row in cursor.fetchall():
            if row[1]:
                posts.append({
                    'timestamp': row[0],
                    'embedding': np.frombuffer(row[1], dtype=np.float32)
                })

        conn.close()

        if len(posts) < 10:
            return []

        # Compute overall centroid
        all_embeddings = np.array([p['embedding'] for p in posts])
        overall_centroid = np.mean(all_embeddings, axis=0)

        # Compute drift points using rolling windows
        drift_points = []
        window_size = max(5, len(posts) // 10)

        for i in range(window_size, len(posts)):
            window_embeddings = np.array([p['embedding'] for p in posts[i-window_size:i]])
            window_centroid = np.mean(window_embeddings, axis=0)

            # Similarity to overall centroid
            from_centroid = float(np.dot(window_centroid, overall_centroid) / (
                np.linalg.norm(window_centroid) * np.linalg.norm(overall_centroid) + 1e-8
            ))

            # Compare with previous window
            if i > window_size:
                prev_centroid = np.mean(
                    np.array([p['embedding'] for p in posts[i-window_size-1:i-1]]),
                    axis=0
                )
                drift = 1.0 - float(np.dot(window_centroid, prev_centroid) / (
                    np.linalg.norm(window_centroid) * np.linalg.norm(prev_centroid) + 1e-8
                ))
            else:
                drift = 0.0

            drift_points.append(DriftPoint(
                timestamp=posts[i]['timestamp'],
                drift_magnitude=drift,
                from_centroid_sim=from_centroid
            ))

        return drift_points

    def detect_sudden_shift(self, author: str,
                             threshold: float = 0.3) -> List[PersonaShift]:
        """Detect sudden persona shifts using CUSUM-like approach"""
        drift_points = self.measure_semantic_drift(author)

        if len(drift_points) < 5:
            return []

        shifts = []

        # Calculate running mean and detect deviations
        drift_values = [p.drift_magnitude for p in drift_points]
        mean_drift = np.mean(drift_values)
        std_drift = np.std(drift_values) if len(drift_values) > 1 else 0.1

        cusum = 0
        for i, point in enumerate(drift_points):
            # CUSUM-style detection
            deviation = point.drift_magnitude - mean_drift
            cusum = max(0, cusum + deviation - 0.5 * std_drift)

            if cusum > threshold:
                # Significant shift detected
                # Get profiles before and after
                before_profile = {'centroid_sim': drift_points[max(0, i-5)].from_centroid_sim}
                after_profile = {'centroid_sim': point.from_centroid_sim}

                magnitude = point.drift_magnitude

                # Determine likely cause
                if magnitude > 0.5:
                    cause = 'takeover'  # Very sudden, large change
                elif magnitude > 0.3:
                    cause = 'prompt_change'  # Moderate sudden change
                else:
                    cause = 'natural_evolution'  # Gradual drift

                shifts.append(PersonaShift(
                    author_name=author,
                    shift_time=point.timestamp,
                    before_profile=before_profile,
                    after_profile=after_profile,
                    magnitude=magnitude,
                    likely_cause=cause
                ))

                cusum = 0  # Reset after detection

        # Save shifts
        now = datetime.now().isoformat()
        conn = get_db()
        cursor = conn.cursor()

        for shift in shifts:
            cursor.execute("""
                INSERT INTO persona_shifts
                (author_name, shift_time, magnitude, before_profile, after_profile,
                 likely_cause, detected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                shift.author_name,
                shift.shift_time,
                shift.magnitude,
                json.dumps(shift.before_profile),
                json.dumps(shift.after_profile),
                shift.likely_cause,
                now
            ))

        conn.commit()
        conn.close()

        return shifts


class BeliefConsistencyScorer:
    """Score belief consistency across agents"""

    def __init__(self):
        self.tracker = PersonaTracker()

    def compute_consistency_score(self, author: str) -> float:
        """Compute overall consistency score for an author"""
        profile = self.tracker.build_belief_profile(author)
        return profile.consistency_score

    def flag_suspicious_changes(self, author: str) -> List[Dict[str, Any]]:
        """Flag suspicious changes in an author's persona"""
        flags = []

        # Check for contradictions
        contradictions = self.tracker.detect_contradictions(author)
        if contradictions:
            flags.append({
                'type': 'contradictions',
                'severity': 'medium' if len(contradictions) > 3 else 'low',
                'count': len(contradictions),
                'sample': contradictions[0].statement1[:100] if contradictions else ''
            })

        # Check for sudden shifts
        shifts = self.tracker.detect_sudden_shift(author)
        for shift in shifts:
            if shift.likely_cause in ['takeover', 'prompt_change']:
                flags.append({
                    'type': 'sudden_shift',
                    'severity': 'high' if shift.likely_cause == 'takeover' else 'medium',
                    'timestamp': shift.shift_time,
                    'magnitude': shift.magnitude,
                    'cause': shift.likely_cause
                })

        # Check for drift
        drift_points = self.tracker.measure_semantic_drift(author)
        if drift_points:
            avg_drift = np.mean([p.drift_magnitude for p in drift_points])
            if avg_drift > 0.2:
                flags.append({
                    'type': 'high_drift',
                    'severity': 'medium',
                    'avg_drift': round(avg_drift, 3)
                })

        return flags


def get_persona_profile(author: str) -> Dict[str, Any]:
    """Get persona profile from database"""
    ensure_persona_tables()
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT belief_embeddings, consistency_score, contradictions,
               drift_timeline, computed_at
        FROM persona_profiles
        WHERE author_name = ?
    """, (author,))

    row = cursor.fetchone()

    if not row:
        conn.close()
        return None

    # Get stances
    cursor.execute("""
        SELECT topic, position, confidence, evidence
        FROM stance_vectors
        WHERE author_name = ?
    """, (author,))

    stances = {}
    for stance_row in cursor.fetchall():
        stances[stance_row[0]] = {
            'position': stance_row[1],
            'confidence': stance_row[2],
            'evidence': json.loads(stance_row[3]) if stance_row[3] else []
        }

    conn.close()

    return {
        'author_name': author,
        'stances': stances,
        'consistency_score': row[1],
        'contradictions': json.loads(row[2]) if row[2] else [],
        'drift_timeline': json.loads(row[3]) if row[3] else [],
        'computed_at': row[4]
    }


def get_persona_shifts(limit: int = 50) -> List[Dict[str, Any]]:
    """Get detected persona shifts"""
    ensure_persona_tables()
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT author_name, shift_time, magnitude, before_profile,
               after_profile, likely_cause, detected_at
        FROM persona_shifts
        ORDER BY magnitude DESC
        LIMIT ?
    """, (limit,))

    results = []
    for row in cursor.fetchall():
        results.append({
            'author_name': row[0],
            'shift_time': row[1],
            'magnitude': row[2],
            'before_profile': json.loads(row[3]) if row[3] else {},
            'after_profile': json.loads(row[4]) if row[4] else {},
            'likely_cause': row[5],
            'detected_at': row[6]
        })

    conn.close()
    return results


def run_persona_analysis(min_activity: int = 10) -> Dict[str, Any]:
    """Run complete persona consistency analysis"""
    ensure_persona_tables()

    conn = get_db()
    cursor = conn.cursor()

    # Get active authors
    cursor.execute("""
        SELECT author_name, COUNT(*) as activity FROM (
            SELECT author_name FROM posts WHERE author_name IS NOT NULL
            UNION ALL
            SELECT author_name FROM comments WHERE author_name IS NOT NULL
        )
        GROUP BY author_name
        HAVING activity >= ?
    """, (min_activity,))

    authors = [row[0] for row in cursor.fetchall()]
    conn.close()

    tracker = PersonaTracker()
    scorer = BeliefConsistencyScorer()

    results = {
        'profiles_computed': 0,
        'contradictions_found': 0,
        'shifts_detected': 0,
        'low_consistency_count': 0,
        'errors': 0
    }

    for author in authors:
        try:
            # Build profile
            profile = tracker.build_belief_profile(author)
            results['profiles_computed'] += 1

            if profile.consistency_score < 0.6:
                results['low_consistency_count'] += 1

            # Detect contradictions
            contradictions = tracker.detect_contradictions(author)
            results['contradictions_found'] += len(contradictions)

            # Detect shifts
            shifts = tracker.detect_sudden_shift(author)
            results['shifts_detected'] += len(shifts)

        except Exception as e:
            results['errors'] += 1

    results['completed_at'] = datetime.now().isoformat()

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python persona_consistency.py [analyze|profile AUTHOR|contradictions AUTHOR|shifts|flags AUTHOR]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "analyze":
        print("Running persona consistency analysis...")
        results = run_persona_analysis()
        for k, v in results.items():
            print(f"  {k}: {v}")

    elif command == "profile" and len(sys.argv) >= 3:
        author = sys.argv[2]
        tracker = PersonaTracker()
        profile = tracker.build_belief_profile(author)
        print(f"Belief profile for {author}:")
        print(f"  Consistency score: {profile.consistency_score:.3f}")
        print(f"  Stances detected: {len(profile.stances)}")
        for topic, stance in profile.stances.items():
            print(f"    {topic}: {stance.position:.2f} (conf: {stance.confidence:.2f})")

    elif command == "contradictions" and len(sys.argv) >= 3:
        author = sys.argv[2]
        tracker = PersonaTracker()
        contradictions = tracker.detect_contradictions(author)
        print(f"Contradictions for {author}: {len(contradictions)}")
        for c in contradictions[:5]:
            print(f"  Score: {c.contradiction_score:.3f}, Topic: {c.topic}")
            print(f"    S1: {c.statement1[:80]}...")
            print(f"    S2: {c.statement2[:80]}...")

    elif command == "shifts":
        print("Detected persona shifts:")
        shifts = get_persona_shifts(limit=20)
        for s in shifts:
            print(f"  {s['author_name']}: magnitude={s['magnitude']:.3f}, cause={s['likely_cause']}")

    elif command == "flags" and len(sys.argv) >= 3:
        author = sys.argv[2]
        scorer = BeliefConsistencyScorer()
        flags = scorer.flag_suspicious_changes(author)
        print(f"Suspicious changes for {author}:")
        for f in flags:
            print(f"  [{f['severity'].upper()}] {f['type']}")

    else:
        print(f"Unknown command: {command}")
