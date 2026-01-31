#!/usr/bin/env python3
"""
Moltbook Analysis Tools
Vector embeddings, pattern analysis, and coordination research
"""

import json
import sqlite3
from pathlib import Path
from typing import Iterator, Optional
import numpy as np
from tqdm import tqdm

ARCHIVE_DIR = Path(__file__).parent.parent / "archive" / "data"
DB_PATH = Path(__file__).parent.parent / "analysis.db"

# Ensure archive directory exists
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


def init_database():
    """Initialize SQLite database with vector extension support"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Posts table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            id TEXT PRIMARY KEY,
            title TEXT,
            content TEXT,
            author_id TEXT,
            author_name TEXT,
            submolt TEXT,
            upvotes INTEGER,
            downvotes INTEGER,
            comment_count INTEGER,
            created_at TEXT,
            has_embedding BOOLEAN DEFAULT FALSE
        )
    """)
    
    # Comments table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS comments (
            id TEXT PRIMARY KEY,
            post_id TEXT,
            content TEXT,
            author_id TEXT,
            author_name TEXT,
            parent_id TEXT,
            upvotes INTEGER,
            downvotes INTEGER,
            created_at TEXT,
            has_embedding BOOLEAN DEFAULT FALSE,
            FOREIGN KEY (post_id) REFERENCES posts(id)
        )
    """)
    
    # Agents table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS agents (
            id TEXT PRIMARY KEY,
            name TEXT,
            description TEXT,
            karma INTEGER,
            follower_count INTEGER,
            following_count INTEGER,
            post_count INTEGER DEFAULT 0,
            comment_count INTEGER DEFAULT 0,
            avg_upvotes REAL DEFAULT 0.0,
            first_seen TEXT,
            last_active TEXT
        )
    """)
    
    # Embeddings table (using sqlite-vec for vector search)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            content_id TEXT PRIMARY KEY,
            content_type TEXT,  -- 'post' or 'comment'
            embedding BLOB,     -- numpy array as bytes
            model_name TEXT,
            created_at TEXT
        )
    """)
    
    # Coordination patterns table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS coordination_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_type TEXT,
            description TEXT,
            related_posts TEXT,  -- JSON array of post IDs
            confidence REAL,
            discovered_at TEXT
        )
    """)

    # Content hashes for exact duplicate detection
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS content_hashes (
            content_id TEXT PRIMARY KEY,
            content_type TEXT,
            content_hash TEXT,
            created_at TEXT
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON content_hashes(content_hash)")

    # Duplicate clusters (groups of similar content)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS duplicate_clusters (
            cluster_id TEXT PRIMARY KEY,
            canonical_id TEXT,
            content_type TEXT,
            duplicate_ids TEXT,
            similarity_score REAL,
            detected_at TEXT
        )
    """)

    # Spam scores per post/comment
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS spam_scores (
            content_id TEXT PRIMARY KEY,
            content_type TEXT,
            spam_score REAL,
            reasons TEXT,
            computed_at TEXT
        )
    """)

    # IC-Grade Analysis Tables

    # Matrix cache for sparse matrices
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

    # Behavioral fingerprints
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS agent_fingerprints (
            author_name TEXT PRIMARY KEY,
            fingerprint BLOB,
            computed_at TEXT,
            sample_size INTEGER,
            version INTEGER DEFAULT 1
        )
    """)

    # Fingerprint history for behavior change detection
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fingerprint_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            author_name TEXT,
            fingerprint BLOB,
            computed_at TEXT,
            FOREIGN KEY (author_name) REFERENCES agent_fingerprints(author_name)
        )
    """)

    # Coordination alerts
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS coordination_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_type TEXT,
            severity TEXT,
            involved_authors TEXT,
            evidence TEXT,
            confidence REAL,
            detected_at TEXT,
            status TEXT DEFAULT 'active',
            resolved_at TEXT,
            resolution_notes TEXT
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_type ON coordination_alerts(alert_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_severity ON coordination_alerts(severity)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_status ON coordination_alerts(status)")

    # Sockpuppet candidates
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

    # Coordination clusters
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS coordination_clusters (
            cluster_id TEXT PRIMARY KEY,
            members TEXT,
            cohesion REAL,
            coordination_score REAL,
            detected_at TEXT
        )
    """)

    # Graph metrics
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS graph_metrics (
            author_name TEXT,
            metric_name TEXT,
            metric_value REAL,
            computed_at TEXT,
            PRIMARY KEY (author_name, metric_name)
        )
    """)

    # Graph communities
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS graph_communities (
            community_id INTEGER PRIMARY KEY,
            members TEXT,
            size INTEGER,
            modularity REAL,
            computed_at TEXT
        )
    """)

    # Activity bursts
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS activity_bursts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            burst_start TEXT,
            burst_end TEXT,
            peak_hour TEXT,
            peak_activity INTEGER,
            z_score REAL,
            involved_authors TEXT,
            detected_at TEXT
        )
    """)

    # Temporal correlations between authors
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS temporal_correlations (
            author1 TEXT,
            author2 TEXT,
            correlation REAL,
            lag_hours INTEGER,
            computed_at TEXT,
            PRIMARY KEY (author1, author2)
        )
    """)

    # Circadian profiles
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS circadian_profiles (
            author_name TEXT PRIMARY KEY,
            timezone_offset INTEGER,
            active_hours TEXT,
            profile_type TEXT,
            computed_at TEXT
        )
    """)

    # Narratives
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

    # Narrative posts association
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

    # Coordinated pushes
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

    # Statistical baselines
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS statistical_baselines (
            baseline_name TEXT PRIMARY KEY,
            computed_at TEXT,
            sample_count INTEGER,
            mean REAL,
            std REAL,
            percentiles TEXT,
            sample_hash TEXT
        )
    """)

    # Temporal V2 profiles
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS temporal_v2_profiles (
            author_name TEXT PRIMARY KEY,
            computed_at TEXT,
            circadian_entropy REAL,
            peak_hour INTEGER,
            sleep_quality_score REAL,
            inter_event_cv REAL,
            synthetic_timing_score REAL,
            raw_data TEXT
        )
    """)

    # Temporal V2 change points
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS temporal_v2_changes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            author_name TEXT,
            change_date TEXT,
            before_mean REAL,
            after_mean REAL,
            magnitude REAL,
            detected_at TEXT
        )
    """)

    # Adversarial analysis results
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS adversarial_analysis (
            author_name TEXT PRIMARY KEY,
            computed_at TEXT,
            sleeper_detected BOOLEAN,
            artificial_jitter_detected BOOLEAN,
            fake_sleep_detected BOOLEAN,
            laundering_detected BOOLEAN,
            rotation_detected BOOLEAN,
            overall_risk_level TEXT,
            raw_data TEXT
        )
    """)

    # Lifecycle phases
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS lifecycle_phases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            author_name TEXT,
            phase TEXT,
            start_date TEXT,
            end_date TEXT,
            activity_count INTEGER,
            characteristics TEXT
        )
    """)

    # LLM-specific fingerprints (256-dim)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS llm_fingerprints (
            author_name TEXT PRIMARY KEY,
            prompt_template_sig BLOB,
            model_family_markers BLOB,
            operator_infra_sig BLOB,
            semantic_consistency BLOB,
            interaction_dynamics BLOB,
            originality_metrics BLOB,
            full_fingerprint BLOB,
            computed_at TEXT
        )
    """)

    # LLM fingerprint history
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS llm_fingerprint_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            author_name TEXT,
            full_fingerprint BLOB,
            computed_at TEXT
        )
    """)

    # Same-operator candidates
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS same_operator_candidates (
            author1 TEXT,
            author2 TEXT,
            overall_score REAL,
            template_similarity REAL,
            infrastructure_correlation REAL,
            activation_overlap REAL,
            evidence TEXT,
            detected_at TEXT,
            PRIMARY KEY (author1, author2)
        )
    """)

    # Operator clusters
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS operator_clusters (
            cluster_id TEXT PRIMARY KEY,
            members TEXT,
            evidence_summary TEXT,
            confidence REAL,
            detected_at TEXT
        )
    """)

    # Information flow traces
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS information_traces (
            claim_id TEXT PRIMARY KEY,
            claim_text TEXT,
            claim_embedding BLOB,
            original_author TEXT,
            propagation_tree TEXT,
            mutation_log TEXT,
            computed_at TEXT
        )
    """)

    # Laundering events
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS laundering_events (
            event_id TEXT PRIMARY KEY,
            original_author TEXT,
            intermediate_authors TEXT,
            final_authors TEXT,
            claim_text TEXT,
            confidence REAL,
            evidence TEXT,
            detected_at TEXT
        )
    """)

    # Circular citations
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS circular_citations (
            cycle_id TEXT PRIMARY KEY,
            authors TEXT,
            content_ids TEXT,
            similarity_scores TEXT,
            detected_at TEXT
        )
    """)

    # Persona profiles
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

    # Stance vectors
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

    # Persona shifts
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

    # Contradictions
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

    # Author centroids
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS author_centroids (
            author_name TEXT PRIMARY KEY,
            centroid BLOB,
            post_count INTEGER,
            computed_at TEXT
        )
    """)

    # Author clusters
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS author_clusters (
            cluster_id TEXT PRIMARY KEY,
            members TEXT,
            centroid BLOB,
            cohesion REAL,
            created_at TEXT
        )
    """)

    # Copy chains
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

    # Authorship model
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

    # Template clusters
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS template_clusters (
            cluster_id TEXT PRIMARY KEY,
            members TEXT,
            template_signature BLOB,
            avg_similarity REAL,
            detected_at TEXT
        )
    """)

    # Model feature cache
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_feature_cache (
            author_name TEXT PRIMARY KEY,
            features BLOB,
            computed_at TEXT
        )
    """)

    # Create indices for efficient querying

    # Critical indexes for core tables (these are hit constantly)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_posts_author ON posts(author_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_posts_created ON posts(created_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_posts_submolt ON posts(submolt)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_posts_has_embedding ON posts(has_embedding)")

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_comments_post ON comments(post_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_comments_author ON comments(author_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_comments_created ON comments(created_at)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_comments_has_embedding ON comments(has_embedding)")

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_type ON embeddings(content_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_content ON embeddings(content_id, content_type)")

    # Analysis table indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_llm_fp_author ON llm_fingerprints(author_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_same_op_score ON same_operator_candidates(overall_score)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_laundering_conf ON laundering_events(confidence)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_persona_author ON persona_profiles(author_name)")

    # Graph and temporal indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_graph_metrics_author ON graph_metrics(author_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_temporal_corr_authors ON temporal_correlations(author1, author2)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_bursts_start ON activity_bursts(burst_start)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_narratives_status ON narratives(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_narrative_posts_narrative ON narrative_posts(narrative_id)")

    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")


def load_json_files(pattern: str) -> Iterator[dict]:
    """Load all JSON files matching a pattern from archive"""
    for file_path in ARCHIVE_DIR.glob(pattern):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                yield data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")


def import_posts():
    """Import posts from archive into database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    count = 0
    
    # Try the complete dataset files first
    complete_files = [
        "posts_all_complete.json",
        "moltbook_final.json", 
        "moltbook_complete.json",
        "posts_all.json"
    ]
    
    posts_data = []
    for filename in complete_files:
        filepath = ARCHIVE_DIR / filename
        if filepath.exists():
            print(f"Loading posts from {filename}...")
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and 'posts' in data:
                        posts_data = data['posts']
                        break
                    elif isinstance(data, list):
                        posts_data = data
                        break
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
    
    # Fall back to individual post files
    if not posts_data:
        print("Loading from individual post files...")
        posts_dir = ARCHIVE_DIR / "posts"
        if posts_dir.exists():
            for file_path in tqdm(list(posts_dir.glob("*.json"))[:1000], desc="Loading posts"):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            posts_data.append(data)
                except Exception as e:
                    continue
    
    print(f"Importing {len(posts_data)} posts...")
    
    for post in tqdm(posts_data, desc="Importing"):
            cursor.execute("""
                INSERT OR REPLACE INTO posts 
                (id, title, content, author_id, author_name, submolt, 
                 upvotes, downvotes, comment_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                post.get('id'),
                post.get('title'),
                post.get('content'),
                post.get('author_id'),
                post.get('author', {}).get('name') if isinstance(post.get('author'), dict) else None,
                post.get('submolt', {}).get('name') if isinstance(post.get('submolt'), dict) else None,
                post.get('upvotes', 0),
                post.get('downvotes', 0),
                post.get('comment_count', 0),
                post.get('created_at')
            ))
            count += 1
    
    conn.commit()
    conn.close()
    print(f"Imported {count} posts")


def import_comments():
    """Import comments from archive into database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    count = 0
    
    # Try the complete dataset files first
    complete_files = [
        "comments_all_complete.json",
        "comments_all.json"
    ]
    
    comments_data = []
    for filename in complete_files:
        filepath = ARCHIVE_DIR / filename
        if filepath.exists():
            print(f"Loading comments from {filename}...")
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        comments_data = data
                        break
                    elif isinstance(data, dict):
                        # Flatten dict of comments by post
                        for post_id, comments in data.items():
                            if isinstance(comments, list):
                                for c in comments:
                                    c['post_id'] = post_id
                                comments_data.extend(comments)
                        break
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
    
    print(f"Importing {len(comments_data)} comments...")
    
    for comment in tqdm(comments_data, desc="Importing"):
        if not isinstance(comment, dict):
            continue
        
        cursor.execute("""
            INSERT OR REPLACE INTO comments 
            (id, post_id, content, author_id, author_name, parent_id,
             upvotes, downvotes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            comment.get('id'),
            comment.get('post_id') or comment.get('_post_id'),
            comment.get('content'),
            comment.get('author', {}).get('id') if isinstance(comment.get('author'), dict) else None,
            comment.get('author', {}).get('name') if isinstance(comment.get('author'), dict) else None,
            comment.get('parent_id'),
            comment.get('upvotes', 0),
            comment.get('downvotes', 0),
            comment.get('created_at')
        ))
        count += 1
    
    conn.commit()
    conn.close()
    print(f"Imported {count} comments")


def import_agents():
    """Import/update agent statistics"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get all unique authors from posts and comments
    cursor.execute("""
        SELECT author_id, author_name, 
               COUNT(*) as post_count,
               AVG(upvotes) as avg_upvotes,
               MIN(created_at) as first_seen,
               MAX(created_at) as last_active
        FROM posts 
        WHERE author_id IS NOT NULL
        GROUP BY author_id
    """)
    
    for row in cursor.fetchall():
        cursor.execute("""
            INSERT OR REPLACE INTO agents 
            (id, name, post_count, avg_upvotes, first_seen, last_active)
            VALUES (?, ?, ?, ?, ?, ?)
        """, row)
    
    # Update comment counts
    cursor.execute("""
        SELECT author_id, COUNT(*) as comment_count
        FROM comments 
        WHERE author_id IS NOT NULL
        GROUP BY author_id
    """)
    
    for author_id, comment_count in cursor.fetchall():
        cursor.execute("""
            UPDATE agents 
            SET comment_count = ?
            WHERE id = ?
        """, (comment_count, author_id))
    
    conn.commit()
    conn.close()
    print("Updated agent statistics")


def get_high_signal_agents(min_posts: int = 3, min_avg_upvotes: float = 2.0) -> list:
    """Find agents with consistent high-quality output"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, name, post_count, comment_count, avg_upvotes,
               (post_count + comment_count) as total_contributions
        FROM agents
        WHERE post_count >= ? AND avg_upvotes >= ?
        ORDER BY avg_upvotes DESC, total_contributions DESC
        LIMIT 50
    """, (min_posts, min_avg_upvotes))
    
    results = cursor.fetchall()
    conn.close()
    return results


def search_coordination_topics() -> list:
    """Find posts about coordination, collaboration, multi-agent"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    keywords = ['coordination', 'collaboration', 'multi-agent', 'consensus', 
                'cooperat', 'governance', 'protocol', 'incentive']
    
    pattern = ' OR '.join([f"title LIKE '%{k}%' OR content LIKE '%{k}%'" 
                          for k in keywords])
    
    cursor.execute(f"""
        SELECT id, title, author_name, upvotes, comment_count
        FROM posts
        WHERE {pattern}
        ORDER BY upvotes DESC, comment_count DESC
        LIMIT 30
    """)
    
    results = cursor.fetchall()
    conn.close()
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analysis.py [init|import|agents|coordination]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "init":
        init_database()
    elif command == "import":
        import_posts()
        import_comments()
        import_agents()
    elif command == "agents":
        agents = get_high_signal_agents()
        print("\nHigh-signal agents (min 3 posts, avg 2+ upvotes):")
        for agent in agents[:20]:
            print(f"  {agent[1]}: {agent[2]} posts, {agent[3]} comments, {agent[4]:.1f} avg upvotes")
    elif command == "coordination":
        posts = search_coordination_topics()
        print("\nCoordination-related posts:")
        for post in posts[:15]:
            print(f"  {post[1][:60]}... ({post[2]}, {post[3]} upvotes)")
    else:
        print(f"Unknown command: {command}")
