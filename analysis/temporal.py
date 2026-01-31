#!/usr/bin/env python3
"""
Temporal Analysis for IC-Grade Analysis
Burst detection, cross-correlation, and circadian profiling
"""

import sqlite3
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import os

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

DB_PATH = Path(os.getenv("MOLTMIRROR_DB_PATH", "analysis.db"))


def get_db():
    """Get database connection"""
    return sqlite3.connect(DB_PATH)


def ensure_temporal_tables():
    """Ensure temporal analysis tables exist"""
    conn = get_db()
    cursor = conn.cursor()

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

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS circadian_profiles (
            author_name TEXT PRIMARY KEY,
            timezone_offset INTEGER,
            active_hours TEXT,
            profile_type TEXT,
            computed_at TEXT
        )
    """)

    conn.commit()
    conn.close()


def build_hourly_activity(days: int = 7) -> Dict[str, np.ndarray]:
    """
    Build hourly activity time series for each author
    Returns dict of author -> 24*days array
    """
    conn = get_db()
    cursor = conn.cursor()

    cutoff = (datetime.now() - timedelta(days=days)).isoformat()

    cursor.execute("""
        SELECT author_name, created_at FROM (
            SELECT author_name, created_at FROM posts
            WHERE created_at > ? AND author_name IS NOT NULL
            UNION ALL
            SELECT author_name, created_at FROM comments
            WHERE created_at > ? AND author_name IS NOT NULL
        )
    """, (cutoff, cutoff))

    # Parse timestamps
    author_activity = defaultdict(lambda: defaultdict(int))
    start_time = datetime.now() - timedelta(days=days)

    for author, created_at in cursor.fetchall():
        try:
            dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            # Calculate hour offset from start
            offset_hours = int((dt.replace(tzinfo=None) - start_time).total_seconds() / 3600)
            if 0 <= offset_hours < days * 24:
                author_activity[author][offset_hours] += 1
        except (ValueError, AttributeError):
            continue

    conn.close()

    # Convert to numpy arrays
    total_hours = days * 24
    result = {}
    for author, hourly in author_activity.items():
        arr = np.zeros(total_hours, dtype=np.float32)
        for hour, count in hourly.items():
            arr[hour] = count
        result[author] = arr

    return result


def build_global_hourly_activity(days: int = 7) -> np.ndarray:
    """Build total hourly activity across all authors"""
    conn = get_db()
    cursor = conn.cursor()

    cutoff = (datetime.now() - timedelta(days=days)).isoformat()

    cursor.execute("""
        SELECT created_at FROM (
            SELECT created_at FROM posts WHERE created_at > ?
            UNION ALL
            SELECT created_at FROM comments WHERE created_at > ?
        )
    """, (cutoff, cutoff))

    total_hours = days * 24
    hourly = np.zeros(total_hours, dtype=np.float32)
    start_time = datetime.now() - timedelta(days=days)

    for (created_at,) in cursor.fetchall():
        try:
            dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            offset_hours = int((dt.replace(tzinfo=None) - start_time).total_seconds() / 3600)
            if 0 <= offset_hours < total_hours:
                hourly[offset_hours] += 1
        except (ValueError, AttributeError):
            continue

    conn.close()

    return hourly


def detect_activity_bursts(z_threshold: float = 3.0, days: int = 7) -> List[Dict[str, Any]]:
    """
    Detect activity bursts where activity > mean + z_threshold * std

    Returns list of burst events with:
    - Start/end time
    - Peak activity
    - Z-score
    - Involved authors
    """
    ensure_temporal_tables()

    global_activity = build_global_hourly_activity(days)

    if len(global_activity) == 0 or global_activity.sum() == 0:
        return []

    mean = np.mean(global_activity)
    std = np.std(global_activity)

    if std == 0:
        return []

    z_scores = (global_activity - mean) / std

    # Find burst hours
    burst_hours = np.where(z_scores > z_threshold)[0]

    if len(burst_hours) == 0:
        return []

    # Group consecutive hours into burst events
    bursts = []
    start_time = datetime.now() - timedelta(days=days)

    current_burst = [burst_hours[0]]
    for i in range(1, len(burst_hours)):
        if burst_hours[i] == burst_hours[i-1] + 1:
            current_burst.append(burst_hours[i])
        else:
            bursts.append(current_burst)
            current_burst = [burst_hours[i]]
    bursts.append(current_burst)

    # Get authors active during each burst
    author_activity = build_hourly_activity(days)

    result = []
    for burst_hours_list in bursts:
        burst_start = start_time + timedelta(hours=int(burst_hours_list[0]))
        burst_end = start_time + timedelta(hours=int(burst_hours_list[-1]) + 1)

        peak_hour_offset = max(burst_hours_list, key=lambda h: global_activity[h])
        peak_activity = int(global_activity[peak_hour_offset])
        peak_z = float(z_scores[peak_hour_offset])

        # Find active authors during burst
        active_authors = []
        for author, activity in author_activity.items():
            burst_activity = sum(activity[h] for h in burst_hours_list if h < len(activity))
            if burst_activity > 0:
                active_authors.append({
                    'author': author,
                    'activity': int(burst_activity)
                })

        active_authors.sort(key=lambda x: x['activity'], reverse=True)

        result.append({
            'burst_start': burst_start.isoformat(),
            'burst_end': burst_end.isoformat(),
            'duration_hours': len(burst_hours_list),
            'peak_hour': (start_time + timedelta(hours=int(peak_hour_offset))).isoformat(),
            'peak_activity': peak_activity,
            'z_score': round(peak_z, 2),
            'total_activity': int(sum(global_activity[h] for h in burst_hours_list)),
            'top_authors': active_authors[:10],
            'author_count': len(active_authors)
        })

    # Sort by z-score
    result.sort(key=lambda x: x['z_score'], reverse=True)

    # Save to database
    conn = get_db()
    cursor = conn.cursor()
    now = datetime.now().isoformat()

    for burst in result[:20]:
        cursor.execute("""
            INSERT INTO activity_bursts
            (burst_start, burst_end, peak_hour, peak_activity, z_score, involved_authors, detected_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            burst['burst_start'],
            burst['burst_end'],
            burst['peak_hour'],
            burst['peak_activity'],
            burst['z_score'],
            json.dumps([a['author'] for a in burst['top_authors']]),
            now
        ))

    conn.commit()
    conn.close()

    return result[:20]


def compute_temporal_correlation(author1: str, author2: str, max_lag: int = 12) -> Dict[str, Any]:
    """
    Compute cross-correlation between two authors' activity patterns

    High correlation + small lag = potential coordination
    """
    author_activity = build_hourly_activity()

    if author1 not in author_activity or author2 not in author_activity:
        return {'error': 'author not found'}

    a1 = author_activity[author1]
    a2 = author_activity[author2]

    # Normalize
    a1_norm = a1 - np.mean(a1)
    a2_norm = a2 - np.mean(a2)

    std1 = np.std(a1)
    std2 = np.std(a2)

    if std1 == 0 or std2 == 0:
        return {
            'author1': author1,
            'author2': author2,
            'correlation': 0,
            'best_lag': 0,
            'reason': 'insufficient_variance'
        }

    # Compute cross-correlation at different lags
    correlations = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            c = np.corrcoef(a1_norm[-lag:], a2_norm[:lag])[0, 1]
        elif lag > 0:
            c = np.corrcoef(a1_norm[:-lag], a2_norm[lag:])[0, 1]
        else:
            c = np.corrcoef(a1_norm, a2_norm)[0, 1]

        if not np.isnan(c):
            correlations.append((lag, c))

    if not correlations:
        return {
            'author1': author1,
            'author2': author2,
            'correlation': 0,
            'best_lag': 0
        }

    # Find best correlation
    best_lag, best_corr = max(correlations, key=lambda x: abs(x[1]))

    return {
        'author1': author1,
        'author2': author2,
        'correlation': round(best_corr, 4),
        'best_lag_hours': best_lag,
        'all_correlations': {str(lag): round(corr, 4) for lag, corr in correlations}
    }


def find_correlated_pairs(min_correlation: float = 0.7,
                           min_activity: int = 10) -> List[Dict[str, Any]]:
    """
    Find pairs of authors with highly correlated activity patterns
    """
    ensure_temporal_tables()

    author_activity = build_hourly_activity()

    # Filter by minimum activity
    active_authors = [
        author for author, activity in author_activity.items()
        if activity.sum() >= min_activity
    ]

    if len(active_authors) < 2:
        return []

    # Compute pairwise correlations
    correlated_pairs = []

    for i, author1 in enumerate(active_authors):
        a1 = author_activity[author1]
        a1_norm = a1 - np.mean(a1)
        std1 = np.std(a1)

        if std1 == 0:
            continue

        for author2 in active_authors[i+1:]:
            a2 = author_activity[author2]
            a2_norm = a2 - np.mean(a2)
            std2 = np.std(a2)

            if std2 == 0:
                continue

            # Zero-lag correlation
            corr = np.corrcoef(a1_norm, a2_norm)[0, 1]

            if not np.isnan(corr) and abs(corr) >= min_correlation:
                correlated_pairs.append({
                    'author1': author1,
                    'author2': author2,
                    'correlation': round(corr, 4),
                    'activity1': int(a1.sum()),
                    'activity2': int(a2.sum())
                })

    # Sort by correlation
    correlated_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)

    # Save to database
    conn = get_db()
    cursor = conn.cursor()
    now = datetime.now().isoformat()

    for pair in correlated_pairs[:50]:
        cursor.execute("""
            INSERT OR REPLACE INTO temporal_correlations
            (author1, author2, correlation, lag_hours, computed_at)
            VALUES (?, ?, ?, 0, ?)
        """, (pair['author1'], pair['author2'], pair['correlation'], now))

    conn.commit()
    conn.close()

    return correlated_pairs[:30]


def compute_circadian_profile(author: str) -> Dict[str, Any]:
    """
    Compute circadian profile for an author:
    - Likely timezone (based on activity pattern)
    - Active hours
    - Profile type (daytime, nighttime, irregular)
    """
    conn = get_db()
    cursor = conn.cursor()

    # Get all timestamps for author
    cursor.execute("""
        SELECT created_at FROM (
            SELECT created_at FROM posts WHERE author_name = ?
            UNION ALL
            SELECT created_at FROM comments WHERE author_name = ?
        )
    """, (author, author))

    timestamps = cursor.fetchall()
    conn.close()

    if len(timestamps) < 5:
        return {
            'author': author,
            'error': 'insufficient_data',
            'sample_size': len(timestamps)
        }

    # Count activity by hour of day
    hourly = np.zeros(24, dtype=np.float32)

    for (created_at,) in timestamps:
        try:
            dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            hourly[dt.hour] += 1
        except (ValueError, AttributeError):
            continue

    if hourly.sum() == 0:
        return {'author': author, 'error': 'no_valid_timestamps'}

    # Normalize
    hourly = hourly / hourly.sum()

    # Find peak activity hours
    peak_hour = int(np.argmax(hourly))

    # Find active hours (>5% of activity)
    active_hours = [h for h in range(24) if hourly[h] > 0.05]

    # Estimate timezone offset (assuming peak should be ~14:00 local time)
    # This is a rough heuristic
    expected_peak = 14
    timezone_offset = (peak_hour - expected_peak) % 24
    if timezone_offset > 12:
        timezone_offset -= 24

    # Determine profile type
    daytime_activity = sum(hourly[8:20])
    nighttime_activity = sum(hourly[0:8]) + sum(hourly[20:24])

    if daytime_activity > 0.7:
        profile_type = 'daytime'
    elif nighttime_activity > 0.7:
        profile_type = 'nighttime'
    elif len(active_hours) <= 8:
        profile_type = 'focused'
    else:
        profile_type = 'irregular'

    result = {
        'author': author,
        'sample_size': len(timestamps),
        'peak_hour_utc': peak_hour,
        'estimated_timezone_offset': timezone_offset,
        'active_hours_utc': active_hours,
        'profile_type': profile_type,
        'daytime_ratio': round(daytime_activity, 3),
        'hourly_distribution': hourly.tolist()
    }

    # Save to database
    ensure_temporal_tables()
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO circadian_profiles
        (author_name, timezone_offset, active_hours, profile_type, computed_at)
        VALUES (?, ?, ?, ?, ?)
    """, (
        author,
        timezone_offset,
        json.dumps(active_hours),
        profile_type,
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()

    return result


def detect_bot_farm_patterns() -> List[Dict[str, Any]]:
    """
    Detect potential bot farm patterns:
    - Groups of authors with identical circadian profiles
    - Unusual activity patterns (24/7, very regular intervals)
    """
    conn = get_db()
    cursor = conn.cursor()

    # Get all authors with sufficient activity
    cursor.execute("""
        SELECT author_name, COUNT(*) as activity FROM (
            SELECT author_name FROM posts WHERE author_name IS NOT NULL
            UNION ALL
            SELECT author_name FROM comments WHERE author_name IS NOT NULL
        )
        GROUP BY author_name
        HAVING activity >= 10
    """)

    active_authors = [row[0] for row in cursor.fetchall()]
    conn.close()

    if len(active_authors) < 3:
        return []

    # Compute profiles
    profiles = {}
    for author in active_authors:
        profile = compute_circadian_profile(author)
        if 'hourly_distribution' in profile:
            profiles[author] = np.array(profile['hourly_distribution'])

    if len(profiles) < 3:
        return []

    # Find similar profiles
    similar_groups = []
    processed = set()

    for author1, p1 in profiles.items():
        if author1 in processed:
            continue

        group = [author1]
        processed.add(author1)

        for author2, p2 in profiles.items():
            if author2 in processed:
                continue

            # Compute cosine similarity
            sim = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2) + 1e-8)

            if sim > 0.95:  # Very similar profiles
                group.append(author2)
                processed.add(author2)

        if len(group) >= 3:
            # Check for suspicious patterns
            hourly = profiles[author1]
            is_24_7 = hourly.min() > 0.02  # Active in all hours
            is_regular = np.std(hourly) < 0.03  # Very even distribution

            similar_groups.append({
                'members': group,
                'size': len(group),
                'avg_profile': hourly.tolist(),
                'suspicious_flags': {
                    'active_24_7': is_24_7,
                    'very_regular': is_regular
                },
                'suspicion_score': sum([is_24_7, is_regular, len(group) > 5])
            })

    # Sort by suspicion
    similar_groups.sort(key=lambda x: x['suspicion_score'], reverse=True)

    return similar_groups[:10]


def get_activity_heatmap(author: str) -> Dict[str, Any]:
    """
    Get activity heatmap data for visualization:
    - Day of week × Hour of day matrix
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

    # Build 7×24 matrix (day × hour)
    heatmap = np.zeros((7, 24), dtype=np.int32)

    for (created_at,) in timestamps:
        try:
            dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            heatmap[dt.weekday(), dt.hour] += 1
        except (ValueError, AttributeError):
            continue

    return {
        'author': author,
        'heatmap': heatmap.tolist(),
        'total_activity': int(heatmap.sum()),
        'peak_day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][int(np.argmax(heatmap.sum(axis=1)))],
        'peak_hour': int(np.argmax(heatmap.sum(axis=0)))
    }


def run_all_temporal_analysis() -> Dict[str, Any]:
    """Run all temporal analyses and return results"""
    results = {}

    try:
        bursts = detect_activity_bursts()
        results['activity_bursts'] = {
            'status': 'success',
            'count': len(bursts),
            'max_z_score': bursts[0]['z_score'] if bursts else None
        }
    except Exception as e:
        results['activity_bursts'] = {'status': 'error', 'error': str(e)}

    try:
        pairs = find_correlated_pairs()
        results['correlated_pairs'] = {
            'status': 'success',
            'count': len(pairs),
            'max_correlation': pairs[0]['correlation'] if pairs else None
        }
    except Exception as e:
        results['correlated_pairs'] = {'status': 'error', 'error': str(e)}

    try:
        bot_farms = detect_bot_farm_patterns()
        results['bot_farm_patterns'] = {
            'status': 'success',
            'groups_found': len(bot_farms)
        }
    except Exception as e:
        results['bot_farm_patterns'] = {'status': 'error', 'error': str(e)}

    results['completed_at'] = datetime.now().isoformat()

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python temporal.py [bursts|correlations|profile AUTHOR|bot-farms|all]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "bursts":
        print("Detecting activity bursts...")
        bursts = detect_activity_bursts()
        print(f"Found {len(bursts)} bursts:")
        for b in bursts[:5]:
            print(f"  {b['peak_hour']}: z={b['z_score']:.2f}, activity={b['peak_activity']}")
            print(f"    Top authors: {', '.join(a['author'] for a in b['top_authors'][:3])}")

    elif command == "correlations":
        print("Finding correlated pairs...")
        pairs = find_correlated_pairs()
        print(f"Found {len(pairs)} correlated pairs:")
        for p in pairs[:10]:
            print(f"  {p['author1']} <-> {p['author2']}: r={p['correlation']:.3f}")

    elif command == "profile" and len(sys.argv) >= 3:
        author = sys.argv[2]
        print(f"Computing circadian profile for {author}...")
        profile = compute_circadian_profile(author)
        print(f"  Peak hour UTC: {profile.get('peak_hour_utc')}")
        print(f"  Timezone offset: {profile.get('estimated_timezone_offset')}")
        print(f"  Profile type: {profile.get('profile_type')}")

    elif command == "bot-farms":
        print("Detecting bot farm patterns...")
        farms = detect_bot_farm_patterns()
        print(f"Found {len(farms)} suspicious groups:")
        for f in farms[:5]:
            print(f"  Group of {f['size']}: {', '.join(f['members'][:5])}")
            print(f"    Flags: {f['suspicious_flags']}")

    elif command == "all":
        print("Running all temporal analysis...")
        results = run_all_temporal_analysis()
        for k, v in results.items():
            print(f"  {k}: {v}")

    else:
        print(f"Unknown command: {command}")
