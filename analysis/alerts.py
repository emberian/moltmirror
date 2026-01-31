#!/usr/bin/env python3
"""
Alert System for IC-Grade Analysis
Aggregates alerts from all detection modules with severity levels
"""

import sqlite3
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from enum import Enum
import os

DB_PATH = Path(os.getenv("MOLTMIRROR_DB_PATH", "analysis.db"))


class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    SOCKPUPPET = "sockpuppet"
    SYNCHRONIZED_POSTING = "synchronized_posting"
    COORDINATION_CLUSTER = "coordination_cluster"
    ACTIVITY_BURST = "activity_burst"
    NARRATIVE_PUSH = "narrative_push"
    BEHAVIOR_CHANGE = "behavior_change"
    VOTE_MANIPULATION = "vote_manipulation"
    BOT_FARM = "bot_farm"


def get_db():
    """Get database connection"""
    return sqlite3.connect(DB_PATH)


def ensure_alert_tables():
    """Ensure alert tables exist"""
    conn = get_db()
    cursor = conn.cursor()

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

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_alerts_type ON coordination_alerts(alert_type)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_alerts_severity ON coordination_alerts(severity)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_alerts_status ON coordination_alerts(status)
    """)

    conn.commit()
    conn.close()


def create_alert(alert_type: AlertType,
                 severity: AlertSeverity,
                 involved_authors: List[str],
                 evidence: Dict[str, Any],
                 confidence: float) -> int:
    """Create a new alert"""
    ensure_alert_tables()
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO coordination_alerts
        (alert_type, severity, involved_authors, evidence, confidence, detected_at, status)
        VALUES (?, ?, ?, ?, ?, ?, 'active')
    """, (
        alert_type.value,
        severity.value,
        json.dumps(involved_authors),
        json.dumps(evidence),
        confidence,
        datetime.now().isoformat()
    ))

    alert_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return alert_id


def check_duplicate_alert(alert_type: str,
                           authors: List[str],
                           hours: int = 24) -> bool:
    """Check if a similar alert already exists"""
    conn = get_db()
    cursor = conn.cursor()

    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()

    cursor.execute("""
        SELECT involved_authors FROM coordination_alerts
        WHERE alert_type = ?
        AND detected_at > ?
        AND status = 'active'
    """, (alert_type, cutoff))

    for row in cursor.fetchall():
        existing_authors = set(json.loads(row[0]))
        new_authors = set(authors)

        # Check overlap
        overlap = len(existing_authors & new_authors) / max(len(existing_authors | new_authors), 1)
        if overlap > 0.7:
            conn.close()
            return True

    conn.close()
    return False


def generate_sockpuppet_alerts() -> List[Dict[str, Any]]:
    """Generate alerts from sockpuppet detection"""
    try:
        from .coordination import detect_sockpuppets
    except ImportError:
        from analysis.coordination import detect_sockpuppets

    candidates = detect_sockpuppets()
    alerts = []

    for candidate in candidates:
        # Determine severity based on confidence
        confidence = candidate['confidence']

        if confidence >= 0.9:
            severity = AlertSeverity.HIGH
        elif confidence >= 0.8:
            severity = AlertSeverity.MEDIUM
        else:
            severity = AlertSeverity.LOW

        # Skip duplicates
        authors = [candidate['author1'], candidate['author2']]
        if check_duplicate_alert(AlertType.SOCKPUPPET.value, authors):
            continue

        # Create alert
        alert_id = create_alert(
            AlertType.SOCKPUPPET,
            severity,
            authors,
            candidate['evidence'],
            confidence
        )

        alerts.append({
            'id': alert_id,
            'type': AlertType.SOCKPUPPET.value,
            'severity': severity.value,
            'authors': authors,
            'confidence': confidence
        })

    return alerts


def generate_synchronized_alerts() -> List[Dict[str, Any]]:
    """Generate alerts from synchronized posting detection"""
    try:
        from .coordination import detect_synchronized_posting
    except ImportError:
        from analysis.coordination import detect_synchronized_posting

    groups = detect_synchronized_posting()
    alerts = []

    for group in groups:
        score = group['coordination_score']

        # Score > 10: high, > 5: medium, else low
        if score > 10:
            severity = AlertSeverity.HIGH
        elif score > 5:
            severity = AlertSeverity.MEDIUM
        else:
            continue  # Skip low-score groups

        authors = group['authors']
        if check_duplicate_alert(AlertType.SYNCHRONIZED_POSTING.value, authors):
            continue

        confidence = min(score / 15, 1.0)

        alert_id = create_alert(
            AlertType.SYNCHRONIZED_POSTING,
            severity,
            authors,
            {
                'group_size': group['group_size'],
                'avg_similarity': group['avg_similarity'],
                'time_spread_minutes': group['time_spread_minutes'],
                'coordination_score': score
            },
            confidence
        )

        alerts.append({
            'id': alert_id,
            'type': AlertType.SYNCHRONIZED_POSTING.value,
            'severity': severity.value,
            'authors': authors,
            'confidence': confidence
        })

    return alerts


def generate_cluster_alerts() -> List[Dict[str, Any]]:
    """Generate alerts from coordination cluster detection"""
    try:
        from .coordination import detect_coordination_clusters
    except ImportError:
        from analysis.coordination import detect_coordination_clusters

    clusters = detect_coordination_clusters()
    alerts = []

    for cluster in clusters:
        if 'error' in cluster:
            continue

        score = cluster['coordination_score']

        # Score > 8: medium, > 12: high
        if score > 12:
            severity = AlertSeverity.HIGH
        elif score > 8:
            severity = AlertSeverity.MEDIUM
        else:
            continue

        authors = cluster['members']
        if check_duplicate_alert(AlertType.COORDINATION_CLUSTER.value, authors[:5]):
            continue

        confidence = min(score / 15, 1.0)

        alert_id = create_alert(
            AlertType.COORDINATION_CLUSTER,
            severity,
            authors[:20],  # Limit stored authors
            {
                'cluster_id': cluster['cluster_id'],
                'size': cluster['size'],
                'cohesion': cluster['cohesion'],
                'coordination_score': score
            },
            confidence
        )

        alerts.append({
            'id': alert_id,
            'type': AlertType.COORDINATION_CLUSTER.value,
            'severity': severity.value,
            'author_count': len(authors),
            'confidence': confidence
        })

    return alerts


def generate_burst_alerts() -> List[Dict[str, Any]]:
    """Generate alerts from activity burst detection"""
    try:
        from .temporal import detect_activity_bursts
    except ImportError:
        from analysis.temporal import detect_activity_bursts

    bursts = detect_activity_bursts()
    alerts = []

    for burst in bursts:
        z_score = burst['z_score']

        # z > 4: medium, z > 5: high
        if z_score > 5:
            severity = AlertSeverity.HIGH
        elif z_score > 4:
            severity = AlertSeverity.MEDIUM
        else:
            continue

        authors = [a['author'] for a in burst['top_authors']]
        if check_duplicate_alert(AlertType.ACTIVITY_BURST.value, authors[:3]):
            continue

        confidence = min((z_score - 3) / 3, 1.0)

        alert_id = create_alert(
            AlertType.ACTIVITY_BURST,
            severity,
            authors[:10],
            {
                'peak_hour': burst['peak_hour'],
                'peak_activity': burst['peak_activity'],
                'z_score': z_score,
                'duration_hours': burst['duration_hours']
            },
            confidence
        )

        alerts.append({
            'id': alert_id,
            'type': AlertType.ACTIVITY_BURST.value,
            'severity': severity.value,
            'z_score': z_score,
            'confidence': confidence
        })

    return alerts


def generate_narrative_push_alerts() -> List[Dict[str, Any]]:
    """Generate alerts from narrative push detection"""
    try:
        from .narratives import detect_coordinated_pushes
    except ImportError:
        from analysis.narratives import detect_coordinated_pushes

    pushes = detect_coordinated_pushes()
    alerts = []

    for push in pushes:
        score = push['coordination_score']

        # Critical if score > 10 and fingerprint similarity high
        fp_sim = push.get('fingerprint_similarity') or 0

        if score > 10 and fp_sim > 0.8:
            severity = AlertSeverity.CRITICAL
        elif score > 8:
            severity = AlertSeverity.HIGH
        elif score > 5:
            severity = AlertSeverity.MEDIUM
        else:
            continue

        authors = push['authors']
        if check_duplicate_alert(AlertType.NARRATIVE_PUSH.value, authors[:3]):
            continue

        confidence = min(score / 12, 1.0)

        alert_id = create_alert(
            AlertType.NARRATIVE_PUSH,
            severity,
            authors,
            {
                'narrative_id': push['narrative_id'],
                'coordination_score': score,
                'fingerprint_similarity': fp_sim,
                'time_spread_hours': push['time_spread_hours']
            },
            confidence
        )

        alerts.append({
            'id': alert_id,
            'type': AlertType.NARRATIVE_PUSH.value,
            'severity': severity.value,
            'authors': authors,
            'confidence': confidence
        })

    return alerts


def generate_behavior_change_alerts(min_delta: float = 0.5) -> List[Dict[str, Any]]:
    """Generate alerts for significant behavior changes"""
    try:
        from .fingerprints import detect_behavior_change
    except ImportError:
        from analysis.fingerprints import detect_behavior_change

    conn = get_db()
    cursor = conn.cursor()

    # Get authors with fingerprint history
    cursor.execute("""
        SELECT DISTINCT author_name FROM fingerprint_history
    """)

    authors = [row[0] for row in cursor.fetchall()]
    conn.close()

    alerts = []

    for author in authors:
        result = detect_behavior_change(author, threshold=min_delta)

        if not result.get('changes_detected'):
            continue

        changes = result.get('change_events', [])
        if not changes:
            continue

        # Get max delta
        max_delta = max(c['delta'] for c in changes)

        if max_delta >= 0.7:
            severity = AlertSeverity.MEDIUM
        else:
            severity = AlertSeverity.LOW

        if check_duplicate_alert(AlertType.BEHAVIOR_CHANGE.value, [author]):
            continue

        alert_id = create_alert(
            AlertType.BEHAVIOR_CHANGE,
            severity,
            [author],
            {
                'max_delta': max_delta,
                'change_count': len(changes),
                'changes': changes[-3:]  # Last 3 changes
            },
            max_delta
        )

        alerts.append({
            'id': alert_id,
            'type': AlertType.BEHAVIOR_CHANGE.value,
            'severity': severity.value,
            'author': author,
            'max_delta': max_delta
        })

    return alerts


def generate_bot_farm_alerts() -> List[Dict[str, Any]]:
    """Generate alerts for potential bot farms"""
    try:
        from .temporal import detect_bot_farm_patterns
    except ImportError:
        from analysis.temporal import detect_bot_farm_patterns

    farms = detect_bot_farm_patterns()
    alerts = []

    for farm in farms:
        score = farm.get('suspicion_score', 0)

        if score >= 2:
            severity = AlertSeverity.HIGH
        elif score >= 1:
            severity = AlertSeverity.MEDIUM
        else:
            continue

        authors = farm['members']
        if check_duplicate_alert(AlertType.BOT_FARM.value, authors[:3]):
            continue

        confidence = min(score / 3, 1.0)

        alert_id = create_alert(
            AlertType.BOT_FARM,
            severity,
            authors[:20],
            {
                'group_size': farm['size'],
                'suspicious_flags': farm['suspicious_flags'],
                'suspicion_score': score
            },
            confidence
        )

        alerts.append({
            'id': alert_id,
            'type': AlertType.BOT_FARM.value,
            'severity': severity.value,
            'author_count': len(authors),
            'confidence': confidence
        })

    return alerts


def run_all_alert_generation() -> Dict[str, Any]:
    """Run all alert generators and return summary"""
    results = {}

    generators = [
        ('sockpuppet', generate_sockpuppet_alerts),
        ('synchronized', generate_synchronized_alerts),
        ('cluster', generate_cluster_alerts),
        ('burst', generate_burst_alerts),
        ('narrative_push', generate_narrative_push_alerts),
        ('behavior_change', generate_behavior_change_alerts),
        ('bot_farm', generate_bot_farm_alerts)
    ]

    for name, generator in generators:
        try:
            alerts = generator()
            results[name] = {
                'status': 'success',
                'alerts_created': len(alerts),
                'high_severity': sum(1 for a in alerts if a.get('severity') in ['high', 'critical'])
            }
        except Exception as e:
            results[name] = {'status': 'error', 'error': str(e)}

    results['completed_at'] = datetime.now().isoformat()

    return results


def get_active_alerts(severity: Optional[str] = None,
                       alert_type: Optional[str] = None,
                       limit: int = 50) -> List[Dict[str, Any]]:
    """Get active alerts with optional filtering"""
    ensure_alert_tables()
    conn = get_db()
    cursor = conn.cursor()

    query = """
        SELECT id, alert_type, severity, involved_authors, evidence, confidence, detected_at
        FROM coordination_alerts
        WHERE status = 'active'
    """
    params = []

    if severity:
        query += " AND severity = ?"
        params.append(severity)

    if alert_type:
        query += " AND alert_type = ?"
        params.append(alert_type)

    query += " ORDER BY detected_at DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)

    alerts = []
    for row in cursor.fetchall():
        alerts.append({
            'id': row[0],
            'type': row[1],
            'severity': row[2],
            'authors': json.loads(row[3]),
            'evidence': json.loads(row[4]),
            'confidence': row[5],
            'detected_at': row[6]
        })

    conn.close()

    return alerts


def get_alert_summary() -> Dict[str, Any]:
    """Get summary statistics of active alerts"""
    ensure_alert_tables()
    conn = get_db()
    cursor = conn.cursor()

    # Count by severity
    cursor.execute("""
        SELECT severity, COUNT(*) FROM coordination_alerts
        WHERE status = 'active'
        GROUP BY severity
    """)
    by_severity = {row[0]: row[1] for row in cursor.fetchall()}

    # Count by type
    cursor.execute("""
        SELECT alert_type, COUNT(*) FROM coordination_alerts
        WHERE status = 'active'
        GROUP BY alert_type
    """)
    by_type = {row[0]: row[1] for row in cursor.fetchall()}

    # Recent alerts (24h)
    cursor.execute("""
        SELECT COUNT(*) FROM coordination_alerts
        WHERE status = 'active'
        AND detected_at > datetime('now', '-24 hours')
    """)
    recent_24h = cursor.fetchone()[0]

    # Total active
    cursor.execute("""
        SELECT COUNT(*) FROM coordination_alerts
        WHERE status = 'active'
    """)
    total_active = cursor.fetchone()[0]

    conn.close()

    return {
        'total_active': total_active,
        'by_severity': by_severity,
        'by_type': by_type,
        'new_24h': recent_24h,
        'critical_count': by_severity.get('critical', 0),
        'high_count': by_severity.get('high', 0)
    }


def resolve_alert(alert_id: int, notes: str = "") -> bool:
    """Mark an alert as resolved"""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE coordination_alerts
        SET status = 'resolved',
            resolved_at = ?,
            resolution_notes = ?
        WHERE id = ?
    """, (datetime.now().isoformat(), notes, alert_id))

    success = cursor.rowcount > 0
    conn.commit()
    conn.close()

    return success


def get_author_alert_history(author: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Get alerts involving a specific author"""
    ensure_alert_tables()
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, alert_type, severity, evidence, confidence, detected_at, status
        FROM coordination_alerts
        WHERE involved_authors LIKE ?
        ORDER BY detected_at DESC
        LIMIT ?
    """, (f'%"{author}"%', limit))

    alerts = []
    for row in cursor.fetchall():
        alerts.append({
            'id': row[0],
            'type': row[1],
            'severity': row[2],
            'evidence': json.loads(row[3]),
            'confidence': row[4],
            'detected_at': row[5],
            'status': row[6]
        })

    conn.close()

    return alerts


def compute_author_risk_score(author: str) -> Dict[str, Any]:
    """
    Compute overall risk score for an author based on:
    - Alert history
    - Sockpuppet connections
    - Coordination cluster membership
    - Behavior changes
    """
    alerts = get_author_alert_history(author)

    if not alerts:
        return {
            'author': author,
            'risk_score': 0,
            'risk_level': 'low',
            'alert_count': 0
        }

    # Weight by severity and recency
    severity_weights = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
    type_weights = {
        'sockpuppet': 3,
        'synchronized_posting': 2.5,
        'narrative_push': 3,
        'coordination_cluster': 2,
        'bot_farm': 3,
        'behavior_change': 1.5,
        'activity_burst': 1
    }

    now = datetime.now()
    score = 0

    for alert in alerts:
        if alert['status'] == 'resolved':
            continue

        base = severity_weights.get(alert['severity'], 1)
        type_mult = type_weights.get(alert['type'], 1)

        # Decay by age
        try:
            alert_time = datetime.fromisoformat(alert['detected_at'])
            age_days = (now - alert_time).days
            decay = 1 / (1 + age_days * 0.1)
        except (ValueError, AttributeError):
            decay = 0.5

        score += base * type_mult * decay * alert['confidence']

    # Normalize to 0-100
    score = min(score * 10, 100)

    if score >= 70:
        level = 'critical'
    elif score >= 50:
        level = 'high'
    elif score >= 25:
        level = 'medium'
    else:
        level = 'low'

    return {
        'author': author,
        'risk_score': round(score, 1),
        'risk_level': level,
        'alert_count': len([a for a in alerts if a['status'] == 'active']),
        'alert_types': list(set(a['type'] for a in alerts if a['status'] == 'active'))
    }


def get_high_risk_authors(limit: int = 30) -> List[Dict[str, Any]]:
    """Get authors with highest risk scores"""
    ensure_alert_tables()
    conn = get_db()
    cursor = conn.cursor()

    # Get unique authors from active alerts
    cursor.execute("""
        SELECT DISTINCT involved_authors FROM coordination_alerts
        WHERE status = 'active'
    """)

    all_authors = set()
    for row in cursor.fetchall():
        authors = json.loads(row[0])
        all_authors.update(authors)

    conn.close()

    # Compute risk scores
    risk_scores = []
    for author in all_authors:
        score = compute_author_risk_score(author)
        if score['risk_score'] > 0:
            risk_scores.append(score)

    # Sort by risk score
    risk_scores.sort(key=lambda x: x['risk_score'], reverse=True)

    return risk_scores[:limit]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python alerts.py [generate|list|summary|risk|author AUTHOR_NAME]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "generate":
        print("Generating alerts from all sources...")
        results = run_all_alert_generation()
        for k, v in results.items():
            print(f"  {k}: {v}")

    elif command == "list":
        print("Active alerts:")
        alerts = get_active_alerts()
        for a in alerts[:20]:
            print(f"  [{a['severity'].upper()}] {a['type']}: {len(a['authors'])} authors (conf={a['confidence']:.2f})")

    elif command == "summary":
        summary = get_alert_summary()
        print("Alert Summary:")
        print(f"  Total active: {summary['total_active']}")
        print(f"  New (24h): {summary['new_24h']}")
        print(f"  By severity: {summary['by_severity']}")
        print(f"  By type: {summary['by_type']}")

    elif command == "risk":
        print("High-risk authors:")
        authors = get_high_risk_authors()
        for a in authors[:15]:
            print(f"  {a['author']}: {a['risk_score']:.1f} ({a['risk_level']}) - {a['alert_count']} alerts")

    elif command == "author" and len(sys.argv) >= 3:
        author = sys.argv[2]
        print(f"Risk profile for {author}:")
        score = compute_author_risk_score(author)
        print(f"  Score: {score['risk_score']:.1f}")
        print(f"  Level: {score['risk_level']}")
        print(f"  Active alerts: {score['alert_count']}")

    else:
        print(f"Unknown command: {command}")
