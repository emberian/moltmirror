#!/usr/bin/env python3
"""
Authorship Classification for Moltbook Agent Analysis

Use embeddings for author identification:
- AuthorshipClassifier: Train classifier on (author, content) pairs
- AuthorCentroidAnalyzer: Cluster authors by their embedding centroids
- CopyPasteChainDetector: Detect copy-paste chains using embeddings
"""

import sqlite3
import pickle
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict
from dataclasses import dataclass
import os

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

DB_PATH = Path(os.getenv("MOLTMIRROR_DB_PATH", "analysis.db"))


@dataclass
class AuthorPrediction:
    """Prediction of likely author for content"""
    content: str
    predicted_author: str
    confidence: float
    top_candidates: List[Tuple[str, float]]


@dataclass
class CopyChain:
    """Detected copy-paste chain"""
    chain_id: str
    original_content_id: str
    original_author: str
    copies: List[Dict[str, Any]]
    chain_length: int
    similarity_scores: List[float]


@dataclass
class AuthorCluster:
    """Cluster of semantically similar authors"""
    cluster_id: str
    members: List[str]
    centroid: np.ndarray
    cohesion: float
    sample_content: List[str]


def get_db():
    """Get database connection"""
    return sqlite3.connect(DB_PATH)


def ensure_authorship_tables():
    """Ensure authorship analysis tables exist"""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS author_centroids (
            author_name TEXT PRIMARY KEY,
            centroid BLOB,
            post_count INTEGER,
            computed_at TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS author_clusters (
            cluster_id TEXT PRIMARY KEY,
            members TEXT,
            centroid BLOB,
            cohesion REAL,
            created_at TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS copy_chains (
            chain_id TEXT PRIMARY KEY,
            original_content_id TEXT,
            original_author TEXT,
            copies TEXT,
            chain_length INTEGER,
            similarity_scores TEXT,
            detected_at TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS authorship_model (
            model_id TEXT PRIMARY KEY,
            model_type TEXT,
            model_data BLOB,
            label_encoder BLOB,
            accuracy REAL,
            trained_at TEXT
        )
    """)

    conn.commit()
    conn.close()


class AuthorshipClassifier:
    """Train classifier on (author, content) pairs for author prediction"""

    def __init__(self, model_type: str = 'logistic'):
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for AuthorshipClassifier")

        self.model_type = model_type
        self.model = None
        self.label_encoder = LabelEncoder()
        self.trained = False

    def build_training_set(self, min_posts: int = 10,
                            max_authors: int = 100) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Build training set from database.
        Returns (X, y, author_names)
        """
        conn = get_db()
        cursor = conn.cursor()

        # Get authors with sufficient posts
        cursor.execute("""
            SELECT author_name, COUNT(*) as post_count
            FROM posts
            WHERE author_name IS NOT NULL
            GROUP BY author_name
            HAVING post_count >= ?
            ORDER BY post_count DESC
            LIMIT ?
        """, (min_posts, max_authors))

        authors = [row[0] for row in cursor.fetchall()]

        if len(authors) < 3:
            conn.close()
            return None, None, []

        # Get embeddings for each author
        X = []
        y = []
        author_names = []

        for author in authors:
            cursor.execute("""
                SELECT e.embedding
                FROM embeddings e
                JOIN posts p ON e.content_id = p.id AND e.content_type = 'post'
                WHERE p.author_name = ?
            """, (author,))

            embeddings = []
            for row in cursor.fetchall():
                if row[0]:
                    embeddings.append(np.frombuffer(row[0], dtype=np.float32))

            if embeddings:
                X.extend(embeddings)
                y.extend([author] * len(embeddings))
                author_names.append(author)

        conn.close()

        if not X:
            return None, None, []

        return np.array(X), np.array(y), list(set(author_names))

    def train_classifier(self, X: np.ndarray, y: np.ndarray,
                          test_size: float = 0.2) -> Dict[str, Any]:
        """Train the authorship classifier"""
        if X is None or len(X) == 0:
            return {'error': 'no training data'}

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )

        # Train model
        if self.model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000,
                multi_class='multinomial',
                random_state=42
            )
        else:
            self.model = MLPClassifier(
                hidden_layer_sizes=(256, 128),
                max_iter=500,
                random_state=42
            )

        self.model.fit(X_train, y_train)
        self.trained = True

        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Save model
        self._save_model(accuracy)

        return {
            'accuracy': accuracy,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'num_authors': len(self.label_encoder.classes_)
        }

    def _save_model(self, accuracy: float):
        """Save trained model to database"""
        ensure_authorship_tables()
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO authorship_model
            (model_id, model_type, model_data, label_encoder, accuracy, trained_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            'primary',
            self.model_type,
            pickle.dumps(self.model),
            pickle.dumps(self.label_encoder),
            accuracy,
            datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()

    def load_model(self) -> bool:
        """Load trained model from database"""
        ensure_authorship_tables()
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT model_data, label_encoder, accuracy
            FROM authorship_model
            WHERE model_id = 'primary'
        """)

        row = cursor.fetchone()
        conn.close()

        if not row:
            return False

        self.model = pickle.loads(row[0])
        self.label_encoder = pickle.loads(row[1])
        self.trained = True

        return True

    def predict_author(self, content: str) -> AuthorPrediction:
        """Predict author for given content"""
        if not self.trained:
            if not self.load_model():
                return AuthorPrediction(
                    content=content[:100],
                    predicted_author='unknown',
                    confidence=0.0,
                    top_candidates=[]
                )

        # Get embedding for content
        try:
            from .embeddings import get_model
            model = get_model()
            embedding = model.encode([content])[0]
        except ImportError:
            return AuthorPrediction(
                content=content[:100],
                predicted_author='unknown',
                confidence=0.0,
                top_candidates=[]
            )

        # Predict
        probs = self.model.predict_proba([embedding])[0]
        top_indices = np.argsort(probs)[::-1][:5]

        top_candidates = [
            (self.label_encoder.inverse_transform([i])[0], float(probs[i]))
            for i in top_indices
        ]

        predicted = top_candidates[0][0]
        confidence = top_candidates[0][1]

        return AuthorPrediction(
            content=content[:100],
            predicted_author=predicted,
            confidence=confidence,
            top_candidates=top_candidates
        )

    def predict_author_from_embedding(self, embedding: np.ndarray) -> Dict[str, float]:
        """Predict author probabilities from embedding"""
        if not self.trained:
            if not self.load_model():
                return {}

        probs = self.model.predict_proba([embedding])[0]

        return {
            self.label_encoder.inverse_transform([i])[0]: float(probs[i])
            for i in range(len(probs))
        }


class AuthorCentroidAnalyzer:
    """Cluster authors by their embedding centroids"""

    def compute_author_centroid(self, author: str) -> Optional[np.ndarray]:
        """Compute centroid of author's content embeddings"""
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT e.embedding
            FROM embeddings e
            JOIN posts p ON e.content_id = p.id AND e.content_type = 'post'
            WHERE p.author_name = ?
        """, (author,))

        embeddings = []
        for row in cursor.fetchall():
            if row[0]:
                embeddings.append(np.frombuffer(row[0], dtype=np.float32))

        conn.close()

        if not embeddings:
            return None

        centroid = np.mean(embeddings, axis=0)

        # Save centroid
        ensure_authorship_tables()
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO author_centroids
            (author_name, centroid, post_count, computed_at)
            VALUES (?, ?, ?, ?)
        """, (
            author,
            pickle.dumps(centroid),
            len(embeddings),
            datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()

        return centroid

    def load_author_centroid(self, author: str) -> Optional[np.ndarray]:
        """Load cached author centroid"""
        ensure_authorship_tables()
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT centroid FROM author_centroids
            WHERE author_name = ?
        """, (author,))

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return pickle.loads(row[0])

    def cluster_authors_by_centroid(self, min_posts: int = 5,
                                      n_clusters: int = 10) -> List[AuthorCluster]:
        """Cluster authors by their content centroids"""
        if not SKLEARN_AVAILABLE:
            return []

        conn = get_db()
        cursor = conn.cursor()

        # Get authors with sufficient content
        cursor.execute("""
            SELECT author_name, COUNT(*) as post_count
            FROM posts
            WHERE author_name IS NOT NULL
            GROUP BY author_name
            HAVING post_count >= ?
        """, (min_posts,))

        authors = [row[0] for row in cursor.fetchall()]
        conn.close()

        if len(authors) < n_clusters:
            return []

        # Compute/load centroids
        centroids = []
        valid_authors = []

        for author in authors:
            centroid = self.load_author_centroid(author)
            if centroid is None:
                centroid = self.compute_author_centroid(author)

            if centroid is not None:
                centroids.append(centroid)
                valid_authors.append(author)

        if len(centroids) < n_clusters:
            return []

        X = np.array(centroids)

        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # Group by cluster
        clusters = defaultdict(list)
        for author, label, centroid in zip(valid_authors, labels, centroids):
            clusters[label].append((author, centroid))

        # Create cluster objects
        now = datetime.now().isoformat()
        results = []

        conn = get_db()
        cursor = conn.cursor()

        for cluster_id, members_data in clusters.items():
            if len(members_data) < 2:
                continue

            members = [m[0] for m in members_data]
            member_centroids = np.array([m[1] for m in members_data])
            cluster_centroid = np.mean(member_centroids, axis=0)

            # Compute cohesion
            sims = []
            for c in member_centroids:
                sim = np.dot(c, cluster_centroid) / (
                    np.linalg.norm(c) * np.linalg.norm(cluster_centroid) + 1e-8
                )
                sims.append(sim)
            cohesion = float(np.mean(sims))

            cluster = AuthorCluster(
                cluster_id=f"author_cluster_{cluster_id}",
                members=members,
                centroid=cluster_centroid,
                cohesion=cohesion,
                sample_content=[]  # Could populate with sample posts
            )
            results.append(cluster)

            # Save to database
            cursor.execute("""
                INSERT OR REPLACE INTO author_clusters
                (cluster_id, members, centroid, cohesion, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                cluster.cluster_id,
                json.dumps(members),
                pickle.dumps(cluster_centroid),
                cohesion,
                now
            ))

        conn.commit()
        conn.close()

        return sorted(results, key=lambda x: len(x.members), reverse=True)

    def find_semantic_near_duplicates(self, threshold: float = 0.95) -> List[Tuple[str, str, float]]:
        """Find authors with nearly identical content centroids"""
        ensure_authorship_tables()
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT author_name, centroid FROM author_centroids
        """)

        authors = []
        centroids = []

        for row in cursor.fetchall():
            if row[1]:
                authors.append(row[0])
                centroids.append(pickle.loads(row[1]))

        conn.close()

        if len(centroids) < 2:
            return []

        # Find pairs above threshold
        near_duplicates = []

        for i in range(len(centroids)):
            for j in range(i+1, len(centroids)):
                sim = float(np.dot(centroids[i], centroids[j]) / (
                    np.linalg.norm(centroids[i]) * np.linalg.norm(centroids[j]) + 1e-8
                ))

                if sim >= threshold:
                    near_duplicates.append((authors[i], authors[j], sim))

        return sorted(near_duplicates, key=lambda x: x[2], reverse=True)


class CopyPasteChainDetector:
    """Detect copy-paste chains using embeddings"""

    def detect_copy_chains(self, min_similarity: float = 0.98,
                            min_chain_length: int = 2) -> List[CopyChain]:
        """Detect copy-paste chains across posts"""
        conn = get_db()
        cursor = conn.cursor()

        # Get all posts with embeddings
        cursor.execute("""
            SELECT p.id, p.author_name, p.created_at, p.content, e.embedding
            FROM posts p
            JOIN embeddings e ON p.id = e.content_id AND e.content_type = 'post'
            WHERE p.content IS NOT NULL
            ORDER BY p.created_at
        """)

        posts = []
        for row in cursor.fetchall():
            if row[4]:
                posts.append({
                    'id': row[0],
                    'author': row[1],
                    'timestamp': row[2],
                    'content': row[3][:200],
                    'embedding': np.frombuffer(row[4], dtype=np.float32)
                })

        conn.close()

        if len(posts) < 2:
            return []

        chains = []
        used_posts = set()
        chain_count = 0

        # Find chains starting from each post
        for i, origin in enumerate(posts):
            if origin['id'] in used_posts:
                continue

            chain = [origin]
            used_posts.add(origin['id'])

            # Find similar later posts
            for j in range(i + 1, len(posts)):
                candidate = posts[j]
                if candidate['id'] in used_posts:
                    continue

                # Must be different author
                if candidate['author'] == origin['author']:
                    continue

                sim = float(np.dot(origin['embedding'], candidate['embedding']) / (
                    np.linalg.norm(origin['embedding']) * np.linalg.norm(candidate['embedding']) + 1e-8
                ))

                if sim >= min_similarity:
                    chain.append(candidate)
                    used_posts.add(candidate['id'])

            if len(chain) >= min_chain_length:
                chain_id = f"chain_{chain_count}"
                chain_count += 1

                similarity_scores = []
                for k in range(1, len(chain)):
                    sim = float(np.dot(chain[0]['embedding'], chain[k]['embedding']) / (
                        np.linalg.norm(chain[0]['embedding']) * np.linalg.norm(chain[k]['embedding']) + 1e-8
                    ))
                    similarity_scores.append(sim)

                chains.append(CopyChain(
                    chain_id=chain_id,
                    original_content_id=origin['id'],
                    original_author=origin['author'],
                    copies=[{
                        'content_id': c['id'],
                        'author': c['author'],
                        'timestamp': c['timestamp'],
                        'content_preview': c['content']
                    } for c in chain[1:]],
                    chain_length=len(chain),
                    similarity_scores=similarity_scores
                ))

        # Save chains
        now = datetime.now().isoformat()
        ensure_authorship_tables()
        conn = get_db()
        cursor = conn.cursor()

        for chain in chains:
            cursor.execute("""
                INSERT OR REPLACE INTO copy_chains
                (chain_id, original_content_id, original_author, copies,
                 chain_length, similarity_scores, detected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                chain.chain_id,
                chain.original_content_id,
                chain.original_author,
                json.dumps(chain.copies),
                chain.chain_length,
                json.dumps(chain.similarity_scores),
                now
            ))

        conn.commit()
        conn.close()

        return sorted(chains, key=lambda x: x.chain_length, reverse=True)

    def attribute_original_source(self, chain: CopyChain) -> str:
        """Attribute the original source of a copy chain"""
        # The original is the first post in the chain (by timestamp)
        return chain.original_author


def get_author_clusters(limit: int = 20) -> List[Dict[str, Any]]:
    """Get author clusters from database"""
    ensure_authorship_tables()
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT cluster_id, members, cohesion, created_at
        FROM author_clusters
        ORDER BY cohesion DESC
        LIMIT ?
    """, (limit,))

    results = []
    for row in cursor.fetchall():
        results.append({
            'cluster_id': row[0],
            'members': json.loads(row[1]) if row[1] else [],
            'cohesion': row[2],
            'created_at': row[3]
        })

    conn.close()
    return results


def get_copy_chains(limit: int = 50) -> List[Dict[str, Any]]:
    """Get copy chains from database"""
    ensure_authorship_tables()
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT chain_id, original_content_id, original_author, copies,
               chain_length, similarity_scores, detected_at
        FROM copy_chains
        ORDER BY chain_length DESC
        LIMIT ?
    """, (limit,))

    results = []
    for row in cursor.fetchall():
        results.append({
            'chain_id': row[0],
            'original_content_id': row[1],
            'original_author': row[2],
            'copies': json.loads(row[3]) if row[3] else [],
            'chain_length': row[4],
            'similarity_scores': json.loads(row[5]) if row[5] else [],
            'detected_at': row[6]
        })

    conn.close()
    return results


def run_authorship_analysis(min_posts: int = 10) -> Dict[str, Any]:
    """Run complete authorship analysis pipeline"""
    results = {}

    # Train classifier
    try:
        classifier = AuthorshipClassifier()
        X, y, authors = classifier.build_training_set(min_posts=min_posts)

        if X is not None and len(X) > 0:
            training_result = classifier.train_classifier(X, y)
            results['classifier'] = {
                'status': 'success',
                'accuracy': training_result.get('accuracy', 0),
                'num_authors': training_result.get('num_authors', 0)
            }
        else:
            results['classifier'] = {'status': 'error', 'error': 'insufficient data'}
    except Exception as e:
        results['classifier'] = {'status': 'error', 'error': str(e)}

    # Cluster authors
    try:
        analyzer = AuthorCentroidAnalyzer()
        clusters = analyzer.cluster_authors_by_centroid(min_posts=min_posts)
        results['clustering'] = {
            'status': 'success',
            'cluster_count': len(clusters),
            'largest_cluster': len(clusters[0].members) if clusters else 0
        }
    except Exception as e:
        results['clustering'] = {'status': 'error', 'error': str(e)}

    # Find near-duplicates
    try:
        analyzer = AuthorCentroidAnalyzer()
        duplicates = analyzer.find_semantic_near_duplicates()
        results['near_duplicates'] = {
            'status': 'success',
            'count': len(duplicates)
        }
    except Exception as e:
        results['near_duplicates'] = {'status': 'error', 'error': str(e)}

    # Detect copy chains
    try:
        detector = CopyPasteChainDetector()
        chains = detector.detect_copy_chains()
        results['copy_chains'] = {
            'status': 'success',
            'count': len(chains),
            'longest_chain': chains[0].chain_length if chains else 0
        }
    except Exception as e:
        results['copy_chains'] = {'status': 'error', 'error': str(e)}

    results['completed_at'] = datetime.now().isoformat()

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python authorship.py [analyze|train|predict CONTENT|clusters|duplicates|chains]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "analyze":
        print("Running authorship analysis...")
        results = run_authorship_analysis()
        for k, v in results.items():
            print(f"  {k}: {v}")

    elif command == "train":
        print("Training authorship classifier...")
        classifier = AuthorshipClassifier()
        X, y, authors = classifier.build_training_set()
        if X is not None:
            result = classifier.train_classifier(X, y)
            print(f"  Accuracy: {result.get('accuracy', 0):.3f}")
            print(f"  Authors: {result.get('num_authors', 0)}")
        else:
            print("  Insufficient data for training")

    elif command == "predict" and len(sys.argv) >= 3:
        content = ' '.join(sys.argv[2:])
        classifier = AuthorshipClassifier()
        prediction = classifier.predict_author(content)
        print(f"Predicted author: {prediction.predicted_author}")
        print(f"Confidence: {prediction.confidence:.3f}")
        print("Top candidates:")
        for author, prob in prediction.top_candidates:
            print(f"  {author}: {prob:.3f}")

    elif command == "clusters":
        print("Author clusters:")
        clusters = get_author_clusters()
        for c in clusters:
            print(f"  {c['cluster_id']}: {len(c['members'])} members, cohesion={c['cohesion']:.3f}")
            print(f"    Sample: {', '.join(c['members'][:5])}")

    elif command == "duplicates":
        print("Finding semantic near-duplicates...")
        analyzer = AuthorCentroidAnalyzer()
        dups = analyzer.find_semantic_near_duplicates()
        print(f"Found {len(dups)} near-duplicate pairs:")
        for a1, a2, sim in dups[:20]:
            print(f"  {a1} <-> {a2}: {sim:.4f}")

    elif command == "chains":
        print("Copy-paste chains:")
        chains = get_copy_chains()
        for c in chains[:10]:
            print(f"  {c['chain_id']}: {c['original_author']} -> {c['chain_length']} copies")
            for copy in c['copies'][:3]:
                print(f"    -> {copy['author']}")

    else:
        print(f"Unknown command: {command}")
