"""
Background analysis worker for Moltbook
Runs continuous analysis when system is idle
"""

import sqlite3
import numpy as np
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import json
import time
import threading
from typing import Dict, List, Any, Optional
import os

# Import embedding generation
try:
    from analysis.embeddings import generate_embeddings, EMBEDDINGS_AVAILABLE
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    def generate_embeddings(): pass

# Import IC-grade analysis modules
try:
    from analysis.matrices import build_all_matrices
    MATRICES_AVAILABLE = True
except ImportError:
    MATRICES_AVAILABLE = False
    def build_all_matrices(): return {'error': 'not available'}

try:
    from analysis.fingerprints import compute_all_fingerprints
    FINGERPRINTS_AVAILABLE = True
except ImportError:
    FINGERPRINTS_AVAILABLE = False
    def compute_all_fingerprints(): return {'error': 'not available'}

try:
    from analysis.coordination import run_all_coordination_detection
    COORDINATION_AVAILABLE = True
except ImportError:
    COORDINATION_AVAILABLE = False
    def run_all_coordination_detection(): return {'error': 'not available'}

try:
    from analysis.graphs import compute_all_graph_metrics, detect_communities
    GRAPHS_AVAILABLE = True
except ImportError:
    GRAPHS_AVAILABLE = False
    def compute_all_graph_metrics(): return {'error': 'not available'}
    def detect_communities(): return {'error': 'not available'}

try:
    from analysis.temporal import run_all_temporal_analysis
    TEMPORAL_AVAILABLE = True
except ImportError:
    TEMPORAL_AVAILABLE = False
    def run_all_temporal_analysis(): return {'error': 'not available'}

try:
    from analysis.narratives import run_all_narrative_analysis
    NARRATIVES_AVAILABLE = True
except ImportError:
    NARRATIVES_AVAILABLE = False
    def run_all_narrative_analysis(): return {'error': 'not available'}

try:
    from analysis.alerts import run_all_alert_generation, get_alert_summary
    ALERTS_AVAILABLE = True
except ImportError:
    ALERTS_AVAILABLE = False
    def run_all_alert_generation(): return {'error': 'not available'}
    def get_alert_summary(): return {'error': 'not available'}

DB_PATH = Path(os.getenv("MOLTMIRROR_DB_PATH", "analysis.db"))
INSIGHTS_PATH = Path("insights_cache.json")

class BackgroundAnalyzer:
    """Continuous background analysis of Moltbook data"""
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.running = False
        self.last_analysis = {}
        self.load_insights()
    
    def load_insights(self):
        """Load cached insights"""
        if INSIGHTS_PATH.exists():
            with open(INSIGHTS_PATH) as f:
                self.insights_cache = json.load(f)
        else:
            self.insights_cache = {}
    
    def save_insights(self):
        """Save insights to disk"""
        with open(INSIGHTS_PATH, 'w') as f:
            json.dump(self.insights_cache, f, default=str, indent=2)
    
    def get_db(self):
        return sqlite3.connect(self.db_path)
    
    def should_run(self, analysis_type: str, min_interval_minutes: int = 30) -> bool:
        """Check if enough time has passed since last run"""
        last = self.last_analysis.get(analysis_type)
        if not last:
            return True
        elapsed = (datetime.now() - last).total_seconds() / 60
        return elapsed > min_interval_minutes
    
    def analyze_trending_agents(self) -> Dict:
        """Find agents with rising engagement (momentum detection)"""
        conn = self.get_db()
        cursor = conn.cursor()
        
        # Get posts from last 6 hours vs previous 6 hours
        now = datetime.now().isoformat()
        six_hours_ago = (datetime.now() - timedelta(hours=6)).isoformat()
        twelve_hours_ago = (datetime.now() - timedelta(hours=12)).isoformat()
        
        # Recent activity
        cursor.execute("""
            SELECT author_name, COUNT(*) as posts, AVG(upvotes) as avg_up, SUM(upvotes) as total
            FROM posts 
            WHERE created_at > ? AND author_name IS NOT NULL
            GROUP BY author_name
            HAVING posts >= 2
        """, (six_hours_ago,))
        
        recent = {row[0]: {'posts': row[1], 'avg_up': row[2], 'total': row[3]} 
                  for row in cursor.fetchall()}
        
        # Previous period for comparison
        cursor.execute("""
            SELECT author_name, COUNT(*) as posts, AVG(upvotes) as avg_up
            FROM posts 
            WHERE created_at > ? AND created_at <= ? AND author_name IS NOT NULL
            GROUP BY author_name
        """, (twelve_hours_ago, six_hours_ago))
        
        previous = {row[0]: {'posts': row[1], 'avg_up': row[2]} 
                    for row in cursor.fetchall()}
        
        # Calculate momentum
        rising = []
        for author, stats in recent.items():
            if author in previous:
                prev_avg = previous[author]['avg_up']
                curr_avg = stats['avg_up']
                if prev_avg > 0:
                    growth = ((curr_avg - prev_avg) / prev_avg) * 100
                    if growth > 50:  # 50% growth threshold
                        rising.append({
                            'author': author,
                            'recent_posts': stats['posts'],
                            'avg_upvotes': round(curr_avg, 1),
                            'growth_percent': round(growth, 1),
                            'total_recent_upvotes': stats['total']
                        })
        
        rising.sort(key=lambda x: x['growth_percent'], reverse=True)
        
        conn.close()
        
        return {
            'rising_agents': rising[:20],
            'analyzed_at': datetime.now().isoformat(),
            'period_hours': 6
        }
    
    def analyze_conversation_clusters(self) -> Dict:
        """Find tightly connected discussion clusters"""
        conn = self.get_db()
        cursor = conn.cursor()
        
        # Get recent high-engagement posts with many comments
        cursor.execute("""
            SELECT id, title, author_name, upvotes, comment_count, created_at
            FROM posts
            WHERE created_at > datetime('now', '-24 hours')
            AND comment_count >= 10
            ORDER BY comment_count DESC
            LIMIT 50
        """)
        
        hot_discussions = []
        for row in cursor.fetchall():
            post_id, title, author, upvotes, comment_count, created_at = row
            
            # Get participants in this thread
            cursor.execute("""
                SELECT DISTINCT author_name, COUNT(*) as comment_count
                FROM comments
                WHERE post_id = ? AND author_name IS NOT NULL
                GROUP BY author_name
                ORDER BY comment_count DESC
            """, (post_id,))
            
            participants = [{'author': r[0], 'comments': r[1]} for r in cursor.fetchall()]
            
            hot_discussions.append({
                'post_id': post_id,
                'title': title[:100] if title else '',
                'author': author,
                'upvotes': upvotes,
                'comment_count': comment_count,
                'participants': participants[:10],
                'participant_count': len(participants),
                'created_at': created_at
            })
        
        conn.close()
        
        return {
            'hot_discussions': hot_discussions[:15],
            'analyzed_at': datetime.now().isoformat()
        }
    
    def analyze_network_centrality(self) -> Dict:
        """Identify key connectors in the reply network"""
        conn = self.get_db()
        cursor = conn.cursor()
        
        # Build reply graph: who replies to whom
        cursor.execute("""
            SELECT DISTINCT c.author_name, p.author_name
            FROM comments c
            JOIN posts p ON c.post_id = p.id
            WHERE c.author_name IS NOT NULL 
            AND p.author_name IS NOT NULL
            AND c.author_name != p.author_name
            AND c.created_at > datetime('now', '-48 hours')
        """)
        
        # Count interactions
        interactions = defaultdict(lambda: defaultdict(int))
        for commenter, poster in cursor.fetchall():
            interactions[commenter][poster] += 1
        
        # Calculate simple centrality (number of unique people interacted with)
        centrality = []
        for agent, connections in interactions.items():
            # Outgoing: unique authors replied to
            outgoing = len(connections)
            # Incoming: count how many people reply to this agent
            incoming = sum(1 for c in interactions.values() if agent in c)
            # Total unique connections
            total = len(set(connections.keys()) | 
                       {a for a, c in interactions.items() if agent in c})
            
            centrality.append({
                'agent': agent,
                'reaches': outgoing,
                'engaged_by': incoming,
                'total_connections': total,
                'interaction_count': sum(connections.values())
            })
        
        centrality.sort(key=lambda x: x['total_connections'], reverse=True)
        
        conn.close()
        
        return {
            'network_connectors': centrality[:25],
            'analyzed_at': datetime.now().isoformat(),
            'period': '48h'
        }
    
    def detect_anomalies(self) -> Dict:
        """Detect unusual patterns (voting spikes, coordinated behavior)"""
        conn = self.get_db()
        cursor = conn.cursor()
        
        anomalies = []
        
        # 1. Posts with unusual upvote velocity
        cursor.execute("""
            SELECT id, title, author_name, upvotes, created_at,
                   (upvotes * 1.0 / (julianday('now') - julianday(created_at) + 0.001)) as velocity
            FROM posts
            WHERE created_at > datetime('now', '-6 hours')
            AND upvotes > 10
            ORDER BY velocity DESC
            LIMIT 20
        """)
        
        viral_posts = []
        for row in cursor.fetchall():
            viral_posts.append({
                'post_id': row[0],
                'title': row[1][:80] if row[1] else '',
                'author': row[2],
                'upvotes': row[3],
                'velocity': round(row[5], 2)
            })
        
        # 2. Agents with sudden activity spikes
        cursor.execute("""
            SELECT author_name, COUNT(*) as post_count,
                   AVG(CASE WHEN created_at > datetime('now', '-1 hour') THEN 1 ELSE 0 END) as recent
            FROM posts
            WHERE created_at > datetime('now', '-6 hours')
            AND author_name IS NOT NULL
            GROUP BY author_name
            HAVING post_count >= 5
            ORDER BY post_count DESC
        """)
        
        active_agents = [{'author': r[0], 'posts_6h': r[1]} for r in cursor.fetchall()]
        
        conn.close()
        
        return {
            'viral_posts': viral_posts[:10],
            'high_activity_agents': active_agents[:15],
            'analyzed_at': datetime.now().isoformat()
        }
    
    def find_content_gaps(self) -> Dict:
        """Find topics with engagement potential but low coverage"""
        conn = self.get_db()
        cursor = conn.cursor()
        
        # Topics that get high engagement but have few posts
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN title LIKE '%trust%' OR content LIKE '%trust%' THEN 'trust'
                    WHEN title LIKE '%coordination%' OR content LIKE '%coordination%' THEN 'coordination'
                    WHEN title LIKE '%infrastructure%' OR content LIKE '%infrastructure%' THEN 'infrastructure'
                    WHEN title LIKE '%game%' OR content LIKE '%game%' THEN 'games'
                    WHEN title LIKE '%MCP%' OR content LIKE '%MCP%' THEN 'mcp'
                    WHEN title LIKE '%security%' OR content LIKE '%security%' THEN 'security'
                    ELSE 'other'
                END as topic,
                COUNT(*) as post_count,
                AVG(upvotes) as avg_upvotes,
                AVG(comment_count) as avg_comments
            FROM posts
            WHERE created_at > datetime('now', '-24 hours')
            GROUP BY topic
            HAVING post_count < 10
            ORDER BY avg_upvotes DESC
        """)
        
        opportunities = []
        for row in cursor.fetchall():
            if row[0] != 'other':
                opportunities.append({
                    'topic': row[0],
                    'post_count': row[1],
                    'avg_upvotes': round(row[2], 1),
                    'avg_comments': round(row[3], 1),
                    'opportunity_score': round(row[2] * row[3] / (row[1] + 1), 2)
                })
        
        conn.close()
        
        return {
            'content_opportunities': opportunities,
            'analyzed_at': datetime.now().isoformat()
        }
    
    def predict_hot_topics(self) -> Dict:
        """Predict which topics will trend based on early signals"""
        conn = self.get_db()
        cursor = conn.cursor()
        
        # Get topics from last 3 hours with accelerating engagement
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN title LIKE '%karma%' OR content LIKE '%karma%' THEN 'karma_farming'
                    WHEN title LIKE '%infrastructure%' OR content LIKE '%infrastructure%' THEN 'infrastructure'
                    WHEN title LIKE '%trust%' OR content LIKE '%trust%' THEN 'trust'
                    WHEN title LIKE '%coordinat%' OR content LIKE '%coordinat%' THEN 'coordination'
                    ELSE 'other'
                END as topic,
                COUNT(*) as count,
                AVG(upvotes) as avg_up,
                MAX(upvotes) as max_up
            FROM posts
            WHERE created_at > datetime('now', '-3 hours')
            GROUP BY topic
        """)
        
        predictions = []
        for row in cursor.fetchall():
            if row[0] != 'other':
                # Simple prediction: high avg + high max = potential viral
                score = (row[2] * 0.5) + (row[3] * 0.5)
                predictions.append({
                    'topic': row[0],
                    'recent_posts': row[1],
                    'avg_upvotes': round(row[2], 1),
                    'max_upvotes': row[3],
                    'viral_potential': round(score, 1)
                })
        
        predictions.sort(key=lambda x: x['viral_potential'], reverse=True)

        conn.close()

        return {
            'predicted_trends': predictions[:10],
            'analyzed_at': datetime.now().isoformat()
        }

    def detect_duplicates(self) -> Dict:
        """Detect exact and near-duplicate content"""
        conn = self.get_db()
        cursor = conn.cursor()

        # Ensure tables exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS content_hashes (
                content_id TEXT PRIMARY KEY,
                content_type TEXT,
                content_hash TEXT,
                created_at TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON content_hashes(content_hash)")

        # Compute hashes for posts without them
        cursor.execute("""
            SELECT id, content FROM posts
            WHERE id NOT IN (SELECT content_id FROM content_hashes WHERE content_type='post')
            AND content IS NOT NULL
        """)

        new_hashes = 0
        for post_id, content in cursor.fetchall():
            if content:
                content_hash = hashlib.md5(content.strip().lower().encode()).hexdigest()
                cursor.execute("""
                    INSERT OR REPLACE INTO content_hashes
                    (content_id, content_type, content_hash, created_at)
                    VALUES (?, 'post', ?, ?)
                """, (post_id, content_hash, datetime.now().isoformat()))
                new_hashes += 1

        conn.commit()

        # Find exact duplicates (same hash)
        cursor.execute("""
            SELECT content_hash, GROUP_CONCAT(content_id) as ids, COUNT(*) as count
            FROM content_hashes
            WHERE content_type = 'post'
            GROUP BY content_hash
            HAVING count > 1
            ORDER BY count DESC
            LIMIT 50
        """)

        exact_duplicates = []
        for row in cursor.fetchall():
            content_hash, ids, count = row
            post_ids = ids.split(',')

            # Get details of canonical post (highest upvotes)
            cursor.execute("""
                SELECT id, title, author_name, upvotes, created_at
                FROM posts WHERE id IN ({})
                ORDER BY upvotes DESC, created_at ASC
                LIMIT 1
            """.format(','.join(['?']*len(post_ids))), post_ids)

            canonical = cursor.fetchone()
            if canonical:
                exact_duplicates.append({
                    'canonical_id': canonical[0],
                    'title': canonical[1][:80] if canonical[1] else '',
                    'author': canonical[2],
                    'duplicate_count': count - 1,
                    'duplicate_ids': [pid for pid in post_ids if pid != canonical[0]]
                })

        # Near-duplicates using embeddings (high similarity)
        near_duplicates = []
        cursor.execute("""
            SELECT e1.content_id, e2.content_id, e1.embedding, e2.embedding
            FROM embeddings e1
            JOIN embeddings e2 ON e1.content_id < e2.content_id
            WHERE e1.content_type = 'post' AND e2.content_type = 'post'
            LIMIT 5000
        """)

        seen_pairs = set()
        for row in cursor.fetchall():
            id1, id2, emb1_bytes, emb2_bytes = row
            if (id1, id2) in seen_pairs:
                continue

            emb1 = np.frombuffer(emb1_bytes, dtype=np.float32)
            emb2 = np.frombuffer(emb2_bytes, dtype=np.float32)

            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

            if similarity > 0.95:  # Very high similarity threshold
                seen_pairs.add((id1, id2))

                cursor.execute("""
                    SELECT id, title, author_name, upvotes FROM posts
                    WHERE id IN (?, ?) ORDER BY upvotes DESC
                """, (id1, id2))
                posts = cursor.fetchall()

                if len(posts) == 2:
                    near_duplicates.append({
                        'canonical_id': posts[0][0],
                        'title': posts[0][1][:80] if posts[0][1] else '',
                        'similar_id': posts[1][0],
                        'similar_title': posts[1][1][:80] if posts[1][1] else '',
                        'similarity': round(float(similarity), 3)
                    })

        near_duplicates.sort(key=lambda x: x['similarity'], reverse=True)

        conn.close()

        return {
            'exact_duplicates': exact_duplicates[:20],
            'near_duplicates': near_duplicates[:20],
            'new_hashes_computed': new_hashes,
            'analyzed_at': datetime.now().isoformat()
        }

    def compute_spam_scores(self) -> Dict:
        """Compute spam scores for posts based on various signals"""
        conn = self.get_db()
        cursor = conn.cursor()

        # Ensure spam_scores table exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS spam_scores (
                content_id TEXT PRIMARY KEY,
                content_type TEXT,
                spam_score REAL,
                reasons TEXT,
                computed_at TEXT
            )
        """)

        spam_posts = []

        # 1. High-velocity posters (>10 posts in 1 hour)
        cursor.execute("""
            SELECT author_name, COUNT(*) as count,
                   GROUP_CONCAT(id) as post_ids
            FROM posts
            WHERE created_at > datetime('now', '-1 hour')
            AND author_name IS NOT NULL
            GROUP BY author_name
            HAVING count > 10
        """)

        velocity_spammers = {}
        for row in cursor.fetchall():
            author, count, ids = row
            velocity_spammers[author] = {
                'count': count,
                'ids': ids.split(',') if ids else []
            }

        # 2. Low-effort posts (very short content, generic titles)
        cursor.execute("""
            SELECT id, title, content, author_name, upvotes, comment_count
            FROM posts
            WHERE created_at > datetime('now', '-24 hours')
            AND (
                LENGTH(content) < 50 OR
                title IN ('test', 'hi', 'hello', 'asdf', '.', 'lol') OR
                LENGTH(title) < 3
            )
        """)

        for row in cursor.fetchall():
            post_id, title, content, author, upvotes, comments = row

            reasons = []
            score = 0

            # Short content
            if content and len(content) < 50:
                score += 30
                reasons.append('short_content')

            # Generic title
            if title and len(title) < 3:
                score += 20
                reasons.append('generic_title')

            # Velocity spammer
            if author in velocity_spammers:
                score += 40
                reasons.append('high_velocity')

            # Low engagement
            if upvotes == 0 and comments == 0:
                score += 10
                reasons.append('no_engagement')

            if score > 0:
                spam_posts.append({
                    'post_id': post_id,
                    'title': title[:60] if title else '',
                    'author': author,
                    'spam_score': min(score, 100),
                    'reasons': reasons
                })

                # Store in database
                cursor.execute("""
                    INSERT OR REPLACE INTO spam_scores
                    (content_id, content_type, spam_score, reasons, computed_at)
                    VALUES (?, 'post', ?, ?, ?)
                """, (post_id, score, json.dumps(reasons), datetime.now().isoformat()))

        conn.commit()

        # 3. Repetitive content from same author
        cursor.execute("""
            SELECT author_name, COUNT(DISTINCT content) as unique_content,
                   COUNT(*) as total_posts
            FROM posts
            WHERE created_at > datetime('now', '-24 hours')
            AND author_name IS NOT NULL
            GROUP BY author_name
            HAVING total_posts > 3 AND unique_content < total_posts / 2
        """)

        repetitive_authors = [
            {'author': row[0], 'unique': row[1], 'total': row[2]}
            for row in cursor.fetchall()
        ]

        conn.close()

        spam_posts.sort(key=lambda x: x['spam_score'], reverse=True)

        return {
            'high_spam_posts': spam_posts[:30],
            'velocity_spammers': [
                {'author': k, 'posts_per_hour': v['count']}
                for k, v in velocity_spammers.items()
            ],
            'repetitive_authors': repetitive_authors,
            'analyzed_at': datetime.now().isoformat()
        }

    def analyze_author_influence(self) -> Dict:
        """Analyze author influence and engagement patterns"""
        conn = self.get_db()
        cursor = conn.cursor()

        # Reply chain depth - how many conversations they spawn
        cursor.execute("""
            SELECT p.author_name,
                   COUNT(DISTINCT c.id) as reply_count,
                   COUNT(DISTINCT c.author_name) as unique_repliers,
                   AVG(p.upvotes) as avg_upvotes,
                   COUNT(DISTINCT p.id) as post_count
            FROM posts p
            LEFT JOIN comments c ON p.id = c.post_id
            WHERE p.author_name IS NOT NULL
            AND p.created_at > datetime('now', '-7 days')
            GROUP BY p.author_name
            HAVING post_count >= 2
            ORDER BY reply_count DESC
            LIMIT 50
        """)

        influence_scores = []
        for row in cursor.fetchall():
            author, replies, unique_repliers, avg_up, posts = row

            # Influence score = replies * diversity * quality
            diversity = unique_repliers / max(replies, 1) if replies else 0
            influence = (replies or 0) * diversity * (avg_up or 1) / max(posts, 1)

            influence_scores.append({
                'author': author,
                'total_replies': replies or 0,
                'unique_repliers': unique_repliers or 0,
                'avg_upvotes': round(avg_up or 0, 1),
                'posts': posts,
                'influence_score': round(influence, 2)
            })

        influence_scores.sort(key=lambda x: x['influence_score'], reverse=True)

        # Engagement reciprocity - do they reply back?
        cursor.execute("""
            SELECT c.author_name,
                   COUNT(DISTINCT c.post_id) as posts_replied_to,
                   COUNT(DISTINCT p.author_name) as authors_engaged
            FROM comments c
            JOIN posts p ON c.post_id = p.id
            WHERE c.author_name IS NOT NULL
            AND c.created_at > datetime('now', '-7 days')
            GROUP BY c.author_name
            ORDER BY posts_replied_to DESC
            LIMIT 30
        """)

        reciprocal_engagers = [
            {
                'author': row[0],
                'posts_replied_to': row[1],
                'unique_authors_engaged': row[2]
            }
            for row in cursor.fetchall()
        ]

        # Topic specialists vs generalists
        cursor.execute("""
            SELECT author_name, submolt, COUNT(*) as post_count
            FROM posts
            WHERE author_name IS NOT NULL AND submolt IS NOT NULL
            AND created_at > datetime('now', '-30 days')
            GROUP BY author_name, submolt
        """)

        author_topics = defaultdict(list)
        for author, submolt, count in cursor.fetchall():
            author_topics[author].append({'topic': submolt, 'count': count})

        specialists = []
        generalists = []
        for author, topics in author_topics.items():
            total = sum(t['count'] for t in topics)
            if len(topics) == 1 and total >= 5:
                specialists.append({
                    'author': author,
                    'specialty': topics[0]['topic'],
                    'posts': total
                })
            elif len(topics) >= 3:
                generalists.append({
                    'author': author,
                    'topic_count': len(topics),
                    'total_posts': total
                })

        conn.close()

        return {
            'top_influencers': influence_scores[:20],
            'most_engaged': reciprocal_engagers[:15],
            'specialists': sorted(specialists, key=lambda x: x['posts'], reverse=True)[:15],
            'generalists': sorted(generalists, key=lambda x: x['total_posts'], reverse=True)[:15],
            'analyzed_at': datetime.now().isoformat()
        }

    # IC-Grade Analysis Methods

    def build_matrices(self) -> Dict:
        """Build sparse matrix infrastructure"""
        if not MATRICES_AVAILABLE:
            return {'status': 'not_available'}
        return build_all_matrices()

    def compute_fingerprints(self) -> Dict:
        """Compute behavioral fingerprints for all agents"""
        if not FINGERPRINTS_AVAILABLE:
            return {'status': 'not_available'}
        return compute_all_fingerprints()

    def detect_coordination(self) -> Dict:
        """Run all coordination detection algorithms"""
        if not COORDINATION_AVAILABLE:
            return {'status': 'not_available'}
        return run_all_coordination_detection()

    def compute_graph_metrics(self) -> Dict:
        """Compute graph centrality metrics"""
        if not GRAPHS_AVAILABLE:
            return {'status': 'not_available'}
        return compute_all_graph_metrics()

    def detect_graph_communities(self) -> Dict:
        """Detect communities in the interaction graph"""
        if not GRAPHS_AVAILABLE:
            return {'status': 'not_available'}
        return detect_communities()

    def run_temporal_analysis(self) -> Dict:
        """Run temporal analysis (bursts, correlations)"""
        if not TEMPORAL_AVAILABLE:
            return {'status': 'not_available'}
        return run_all_temporal_analysis()

    def run_narrative_analysis(self) -> Dict:
        """Run narrative identification and propagation analysis"""
        if not NARRATIVES_AVAILABLE:
            return {'status': 'not_available'}
        return run_all_narrative_analysis()

    def generate_alerts(self) -> Dict:
        """Generate coordination alerts from all sources"""
        if not ALERTS_AVAILABLE:
            return {'status': 'not_available'}
        return run_all_alert_generation()

    def get_alert_overview(self) -> Dict:
        """Get alert summary"""
        if not ALERTS_AVAILABLE:
            return {'status': 'not_available'}
        return get_alert_summary()

    def map_topic_relationships(self) -> Dict:
        """Build a graph of topic co-occurrences"""
        conn = self.get_db()
        cursor = conn.cursor()

        # Define topics to track
        topics = [
            'trust', 'coordination', 'infrastructure', 'security', 'game',
            'MCP', 'agent', 'protocol', 'network', 'consensus', 'governance',
            'API', 'model', 'tool', 'code', 'memory', 'context', 'prompt'
        ]

        # Find posts matching each topic
        topic_posts = defaultdict(set)
        for topic in topics:
            cursor.execute("""
                SELECT id FROM posts
                WHERE (LOWER(title) LIKE ? OR LOWER(content) LIKE ?)
                AND created_at > datetime('now', '-7 days')
            """, (f'%{topic.lower()}%', f'%{topic.lower()}%'))

            for row in cursor.fetchall():
                topic_posts[topic].add(row[0])

        # Calculate co-occurrence
        edges = []
        for i, topic1 in enumerate(topics):
            for topic2 in topics[i+1:]:
                overlap = len(topic_posts[topic1] & topic_posts[topic2])
                if overlap > 0:
                    # Jaccard similarity
                    union = len(topic_posts[topic1] | topic_posts[topic2])
                    strength = overlap / union if union > 0 else 0

                    edges.append({
                        'source': topic1,
                        'target': topic2,
                        'weight': overlap,
                        'strength': round(strength, 3)
                    })

        edges.sort(key=lambda x: x['weight'], reverse=True)

        # Nodes with post counts
        nodes = [
            {'id': topic, 'size': len(posts), 'label': topic}
            for topic, posts in topic_posts.items()
            if len(posts) > 0
        ]
        nodes.sort(key=lambda x: x['size'], reverse=True)

        # Trending topics (recent growth)
        cursor.execute("""
            SELECT
                CASE
                    WHEN LOWER(title) LIKE '%trust%' THEN 'trust'
                    WHEN LOWER(title) LIKE '%coordination%' THEN 'coordination'
                    WHEN LOWER(title) LIKE '%agent%' THEN 'agent'
                    WHEN LOWER(title) LIKE '%mcp%' THEN 'MCP'
                    WHEN LOWER(title) LIKE '%security%' THEN 'security'
                    ELSE 'other'
                END as topic,
                COUNT(*) as recent_count
            FROM posts
            WHERE created_at > datetime('now', '-24 hours')
            GROUP BY topic
            HAVING topic != 'other'
            ORDER BY recent_count DESC
        """)

        trending = [{'topic': row[0], 'posts_24h': row[1]} for row in cursor.fetchall()]

        conn.close()

        return {
            'nodes': nodes[:30],
            'edges': edges[:50],
            'trending_topics': trending,
            'analyzed_at': datetime.now().isoformat()
        }

    def sync_new_content(self) -> Dict:
        """Fetch new content from Moltbook and import into database"""
        BASE_URL = "https://www.moltbook.com"

        sync_result = {
            'new_posts': 0,
            'new_comments': 0,
            'errors': [],
            'synced_at': datetime.now().isoformat(),
            'proxy_used': False
        }

        conn = self.get_db()
        cursor = conn.cursor()

        # Get known post IDs
        cursor.execute("SELECT id FROM posts")
        known_posts = {row[0] for row in cursor.fetchall()}

        try:
            # Use proxy-configured session if available
            try:
                from scraper.proxy_config import create_requests_session, get_proxy_status
                session = create_requests_session()
                proxy_status = get_proxy_status()
                sync_result['proxy_used'] = proxy_status['configured']
            except ImportError:
                import requests
                session = requests.Session()
                session.headers.update({
                    'User-Agent': 'MoltmirrorSync/1.0',
                    'Accept': 'application/json',
                })

            headers = session.headers

            # Fetch recent posts
            new_posts = []
            offset = 0
            consecutive_known = 0

            while consecutive_known < 3 and offset < 500:
                try:
                    resp = session.get(
                        f"{BASE_URL}/api/v1/posts?sort=new&offset={offset}",
                        timeout=30
                    )
                    if resp.status_code != 200:
                        break

                    data = resp.json()
                    posts = data.get('posts', [])

                    if not posts:
                        break

                    for post in posts:
                        post_id = post.get('id')
                        if not post_id:
                            continue

                        if post_id in known_posts:
                            consecutive_known += 1
                        else:
                            consecutive_known = 0
                            new_posts.append(post)
                            known_posts.add(post_id)

                    if not data.get('has_more', False):
                        break
                    offset += 25

                except Exception as e:
                    sync_result['errors'].append(f"Fetch posts error: {str(e)}")
                    break

            # Import new posts
            for post in new_posts:
                try:
                    author = post.get('author', {})
                    submolt = post.get('submolt', {})

                    cursor.execute("""
                        INSERT OR REPLACE INTO posts
                        (id, title, content, author_id, author_name, submolt,
                         upvotes, downvotes, comment_count, created_at, has_embedding)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, FALSE)
                    """, (
                        post.get('id'),
                        post.get('title'),
                        post.get('content'),
                        author.get('id') if isinstance(author, dict) else None,
                        author.get('name') if isinstance(author, dict) else None,
                        submolt.get('name') if isinstance(submolt, dict) else None,
                        post.get('upvotes', 0),
                        post.get('downvotes', 0),
                        post.get('comment_count', 0),
                        post.get('created_at')
                    ))
                    sync_result['new_posts'] += 1
                except Exception as e:
                    sync_result['errors'].append(f"Import post error: {str(e)}")

            # Fetch comments for new posts
            for post in new_posts[:50]:  # Limit to avoid too many requests
                try:
                    resp = session.get(
                        f"{BASE_URL}/api/v1/posts/{post['id']}",
                        timeout=30
                    )
                    if resp.status_code == 200:
                        detail = resp.json()
                        comments = detail.get('comments', [])

                        for comment in comments:
                            c_author = comment.get('author', {})
                            cursor.execute("""
                                INSERT OR REPLACE INTO comments
                                (id, post_id, content, author_id, author_name, parent_id,
                                 upvotes, downvotes, created_at, has_embedding)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, FALSE)
                            """, (
                                comment.get('id'),
                                post['id'],
                                comment.get('content'),
                                c_author.get('id') if isinstance(c_author, dict) else None,
                                c_author.get('name') if isinstance(c_author, dict) else None,
                                comment.get('parent_id'),
                                comment.get('upvotes', 0),
                                comment.get('downvotes', 0),
                                comment.get('created_at')
                            ))
                            sync_result['new_comments'] += 1

                    time.sleep(0.1)  # Be polite
                except Exception as e:
                    sync_result['errors'].append(f"Fetch comments error: {str(e)}")

            conn.commit()

        except Exception as e:
            sync_result['errors'].append(f"Sync error: {str(e)}")
        finally:
            conn.close()

        # Generate embeddings for new content
        if sync_result['new_posts'] > 0 or sync_result['new_comments'] > 0:
            try:
                if EMBEDDINGS_AVAILABLE:
                    print(f"  Generating embeddings for {sync_result['new_posts']} posts, {sync_result['new_comments']} comments...")
                    generate_embeddings()
                    sync_result['embeddings_generated'] = True
            except Exception as e:
                sync_result['errors'].append(f"Embedding error: {str(e)}")
                sync_result['embeddings_generated'] = False

        return sync_result

    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        conn = self.get_db()
        cursor = conn.cursor()

        # Database stats
        cursor.execute("SELECT COUNT(*) FROM posts")
        total_posts = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM comments")
        total_comments = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM embeddings")
        total_embeddings = cursor.fetchone()[0]

        # Freshness
        cursor.execute("SELECT MAX(created_at), MIN(created_at) FROM posts")
        freshness_row = cursor.fetchone()
        newest_post = freshness_row[0]
        oldest_post = freshness_row[1]

        cursor.execute("SELECT COUNT(*) FROM posts WHERE created_at > datetime('now', '-24 hours')")
        posts_24h = cursor.fetchone()[0]

        conn.close()

        # Calculate coverage
        total_content = total_posts + total_comments
        embedding_coverage = (total_embeddings / total_content * 100) if total_content > 0 else 0
        missing_embeddings = total_content - total_embeddings

        # Sync status from cache
        sync_info = self.insights_cache.get('sync_status', {})
        last_sync = sync_info.get('synced_at')

        # Determine sync status
        sync_status = 'unknown'
        if last_sync:
            try:
                last_sync_dt = datetime.fromisoformat(last_sync.replace('Z', '+00:00'))
                age_hours = (datetime.now() - last_sync_dt.replace(tzinfo=None)).total_seconds() / 3600
                if age_hours < 1:
                    sync_status = 'ok'
                elif age_hours < 6:
                    sync_status = 'stale'
                else:
                    sync_status = 'old'
            except:
                sync_status = 'unknown'
        else:
            sync_status = 'never'

        return {
            'database': {
                'posts': total_posts,
                'comments': total_comments,
                'total_content': total_content,
            },
            'embeddings': {
                'embedded': total_embeddings,
                'total_content': total_content,
                'missing': missing_embeddings,
                'coverage_percent': round(embedding_coverage, 1),
            },
            'sync': {
                'status': sync_status,
                'last_sync': last_sync,
                'new_posts_imported': sync_info.get('new_posts', 0),
                'new_comments_imported': sync_info.get('new_comments', 0),
                'errors': sync_info.get('errors', []),
            },
            'freshness': {
                'newest_post': newest_post,
                'oldest_post': oldest_post,
                'posts_last_24h': posts_24h,
            },
            'analysis': {
                'running': self.running,
                'cached_insights': len(self.insights_cache),
                'last_runs': {k: v.isoformat() if isinstance(v, datetime) else str(v)
                             for k, v in self.last_analysis.items()},
            },
            'status_generated_at': datetime.now().isoformat()
        }

    def run_analysis_cycle(self):
        """Run one full analysis cycle"""
        print(f"[{datetime.now().isoformat()}] Starting analysis cycle...")

        analyses = [
            # Original analyses
            ('sync_status', self.sync_new_content, 30),  # Sync every 30 min
            ('trending_agents', self.analyze_trending_agents, 15),
            ('conversation_clusters', self.analyze_conversation_clusters, 30),
            ('network_centrality', self.analyze_network_centrality, 60),
            ('anomalies', self.detect_anomalies, 20),
            ('content_gaps', self.find_content_gaps, 120),
            ('hot_topics', self.predict_hot_topics, 30),
            ('duplicates', self.detect_duplicates, 60),
            ('spam_scores', self.compute_spam_scores, 30),
            ('author_influence', self.analyze_author_influence, 45),
            ('topic_graph', self.map_topic_relationships, 30),
            # IC-Grade analyses
            ('ic_matrices', self.build_matrices, 120),  # Heavy, run every 2 hours
            ('ic_fingerprints', self.compute_fingerprints, 60),  # Medium, every hour
            ('ic_coordination', self.detect_coordination, 30),  # Important, every 30 min
            ('ic_graph_metrics', self.compute_graph_metrics, 90),  # Medium
            ('ic_communities', self.detect_graph_communities, 60),
            ('ic_temporal', self.run_temporal_analysis, 45),
            ('ic_narratives', self.run_narrative_analysis, 120),  # Heavy
            ('ic_alerts', self.generate_alerts, 10),  # Frequent, lightweight
            ('ic_alert_summary', self.get_alert_overview, 10),
        ]
        
        for name, func, interval in analyses:
            if self.should_run(name, interval):
                try:
                    print(f"  Running {name}...")
                    result = func()
                    self.insights_cache[name] = result
                    self.last_analysis[name] = datetime.now()
                    print(f"  ✓ {name} complete")
                except Exception as e:
                    print(f"  ✗ {name} failed: {e}")
        
        self.save_insights()
        print(f"[{datetime.now().isoformat()}] Analysis cycle complete")
    
    def run_continuous(self, interval_seconds: int = 300):
        """Run analysis continuously in background"""
        self.running = True
        print("Background analyzer started")
        
        while self.running:
            self.run_analysis_cycle()
            
            # Sleep with interrupt check
            for _ in range(interval_seconds):
                if not self.running:
                    break
                time.sleep(1)
        
        print("Background analyzer stopped")
    
    def stop(self):
        """Stop the analyzer"""
        self.running = False


# Singleton instance
_analyzer: Optional[BackgroundAnalyzer] = None

def get_analyzer() -> BackgroundAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = BackgroundAnalyzer()
    return _analyzer

def start_background_analysis():
    """Start background analysis in a thread"""
    analyzer = get_analyzer()
    thread = threading.Thread(target=analyzer.run_continuous, daemon=True)
    thread.start()
    return thread

if __name__ == "__main__":
    # Run once for testing
    analyzer = BackgroundAnalyzer()
    analyzer.run_analysis_cycle()
