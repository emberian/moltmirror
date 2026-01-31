#!/usr/bin/env python3
"""
Graph Analytics for IC-Grade Analysis
Network analysis using networkx for influence, communities, and bridges
"""

import sqlite3
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict
import os

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("networkx not installed. Run: pip install networkx")

try:
    from community import community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False

DB_PATH = Path(os.getenv("MOLTMIRROR_DB_PATH", "analysis.db"))


def get_db():
    """Get database connection"""
    return sqlite3.connect(DB_PATH)


def ensure_graph_tables():
    """Ensure graph metrics tables exist"""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS graph_metrics (
            author_name TEXT,
            metric_name TEXT,
            metric_value REAL,
            computed_at TEXT,
            PRIMARY KEY (author_name, metric_name)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS graph_communities (
            community_id INTEGER,
            members TEXT,
            size INTEGER,
            modularity REAL,
            computed_at TEXT,
            PRIMARY KEY (community_id)
        )
    """)

    conn.commit()
    conn.close()


def build_interaction_graph(days: int = 30) -> nx.DiGraph:
    """
    Build directed graph of author interactions from comments on posts
    Edge weight = number of interactions
    """
    if not NETWORKX_AVAILABLE:
        raise ImportError("networkx not available")

    conn = get_db()
    cursor = conn.cursor()

    # Get all interactions: commenter -> post_author
    cursor.execute("""
        SELECT c.author_name as commenter, p.author_name as poster, COUNT(*) as weight
        FROM comments c
        JOIN posts p ON c.post_id = p.id
        WHERE c.author_name IS NOT NULL
        AND p.author_name IS NOT NULL
        AND c.author_name != p.author_name
        AND c.created_at > datetime('now', ? || ' days')
        GROUP BY c.author_name, p.author_name
    """, (f'-{days}',))

    interactions = cursor.fetchall()
    conn.close()

    G = nx.DiGraph()

    for commenter, poster, weight in interactions:
        G.add_edge(commenter, poster, weight=weight)

    return G


def build_undirected_graph(days: int = 30) -> nx.Graph:
    """Build undirected version for community detection"""
    DG = build_interaction_graph(days)
    return DG.to_undirected()


def compute_pagerank(days: int = 30, damping: float = 0.85) -> Dict[str, float]:
    """
    Compute PageRank centrality for all authors
    Higher score = more influential (receives more interactions from influential people)
    """
    if not NETWORKX_AVAILABLE:
        return {}

    G = build_interaction_graph(days)

    if len(G.nodes()) == 0:
        return {}

    # PageRank
    pagerank = nx.pagerank(G, alpha=damping, weight='weight')

    return pagerank


def compute_betweenness_centrality(days: int = 30) -> Dict[str, float]:
    """
    Compute betweenness centrality for all authors
    Higher score = more important as a bridge between communities
    """
    if not NETWORKX_AVAILABLE:
        return {}

    G = build_undirected_graph(days)

    if len(G.nodes()) == 0:
        return {}

    # Sample for large graphs
    if len(G.nodes()) > 500:
        betweenness = nx.betweenness_centrality(G, k=min(100, len(G.nodes())), weight='weight')
    else:
        betweenness = nx.betweenness_centrality(G, weight='weight')

    return betweenness


def compute_eigenvector_centrality(days: int = 30) -> Dict[str, float]:
    """
    Compute eigenvector centrality
    Higher score = connected to other well-connected nodes
    """
    if not NETWORKX_AVAILABLE:
        return {}

    G = build_undirected_graph(days)

    if len(G.nodes()) == 0:
        return {}

    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=500, weight='weight')
    except nx.PowerIterationFailedConvergence:
        # Fallback to degree centrality
        eigenvector = nx.degree_centrality(G)

    return eigenvector


def detect_communities() -> Dict[str, Any]:
    """
    Detect communities using Louvain algorithm
    Returns communities with modularity scores
    """
    if not NETWORKX_AVAILABLE:
        return {'error': 'networkx not available'}

    G = build_undirected_graph()

    if len(G.nodes()) < 3:
        return {'error': 'insufficient nodes', 'count': len(G.nodes())}

    # Remove isolated nodes
    G.remove_nodes_from(list(nx.isolates(G)))

    if len(G.nodes()) < 3:
        return {'error': 'insufficient connected nodes'}

    if LOUVAIN_AVAILABLE:
        partition = community_louvain.best_partition(G)
        modularity = community_louvain.modularity(partition, G)
    else:
        # Fallback to connected components
        partition = {}
        for i, component in enumerate(nx.connected_components(G)):
            for node in component:
                partition[node] = i
        modularity = None

    # Group by community
    communities = defaultdict(list)
    for node, comm_id in partition.items():
        communities[comm_id].append(node)

    # Sort by size
    community_list = []
    for comm_id, members in communities.items():
        community_list.append({
            'community_id': comm_id,
            'members': sorted(members),
            'size': len(members)
        })

    community_list.sort(key=lambda x: x['size'], reverse=True)

    # Save to database
    ensure_graph_tables()
    conn = get_db()
    cursor = conn.cursor()

    now = datetime.now().isoformat()
    for comm in community_list:
        cursor.execute("""
            INSERT OR REPLACE INTO graph_communities
            (community_id, members, size, modularity, computed_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            comm['community_id'],
            json.dumps(comm['members']),
            comm['size'],
            modularity,
            now
        ))

    conn.commit()
    conn.close()

    return {
        'total_communities': len(community_list),
        'modularity': modularity,
        'communities': community_list[:20],  # Top 20
        'largest_community_size': community_list[0]['size'] if community_list else 0
    }


def find_bridges() -> List[Dict[str, Any]]:
    """
    Find bridge agents who connect different communities
    Based on betweenness centrality + community membership
    """
    if not NETWORKX_AVAILABLE:
        return []

    G = build_undirected_graph()

    if len(G.nodes()) < 3:
        return []

    # Get betweenness
    betweenness = compute_betweenness_centrality()

    # Get community membership
    if LOUVAIN_AVAILABLE:
        partition = community_louvain.best_partition(G)
    else:
        partition = {node: 0 for node in G.nodes()}

    bridges = []
    for node, bc in betweenness.items():
        if bc < 0.01:  # Skip low betweenness
            continue

        # Count neighbors in different communities
        node_comm = partition.get(node, -1)
        neighbors = list(G.neighbors(node))

        cross_community = sum(1 for n in neighbors if partition.get(n, -1) != node_comm)
        same_community = len(neighbors) - cross_community

        if cross_community > 0:
            bridge_score = bc * (cross_community / len(neighbors))
            bridges.append({
                'author': node,
                'betweenness': round(bc, 4),
                'bridge_score': round(bridge_score, 4),
                'total_connections': len(neighbors),
                'cross_community_connections': cross_community,
                'same_community_connections': same_community,
                'community_id': node_comm
            })

    bridges.sort(key=lambda x: x['bridge_score'], reverse=True)

    return bridges[:30]


def find_cliques(min_size: int = 3) -> List[Dict[str, Any]]:
    """
    Find cliques (fully connected subgraphs) using Bron-Kerbosch algorithm
    These represent tight coordination groups
    """
    if not NETWORKX_AVAILABLE:
        return []

    G = build_undirected_graph()

    if len(G.nodes()) < min_size:
        return []

    # Find all maximal cliques
    cliques = list(nx.find_cliques(G))

    # Filter by size
    cliques = [c for c in cliques if len(c) >= min_size]

    # Sort by size
    cliques.sort(key=len, reverse=True)

    result = []
    for i, clique in enumerate(cliques[:30]):  # Top 30
        # Calculate total edge weight in clique
        total_weight = 0
        for j, a in enumerate(clique):
            for b in clique[j+1:]:
                if G.has_edge(a, b):
                    total_weight += G[a][b].get('weight', 1)

        result.append({
            'clique_id': i,
            'members': sorted(clique),
            'size': len(clique),
            'total_edge_weight': total_weight,
            'avg_edge_weight': round(total_weight / max(len(clique) * (len(clique) - 1) / 2, 1), 2)
        })

    return result


def find_gatekeepers() -> List[Dict[str, Any]]:
    """
    Find gatekeepers - nodes that control information flow
    High betweenness + low degree = gatekeeper
    """
    if not NETWORKX_AVAILABLE:
        return []

    G = build_undirected_graph()

    if len(G.nodes()) < 3:
        return []

    betweenness = compute_betweenness_centrality()
    degrees = dict(G.degree())

    gatekeepers = []
    for node in G.nodes():
        bc = betweenness.get(node, 0)
        degree = degrees.get(node, 0)

        if bc > 0.01 and degree > 0:
            # Gatekeeper score: high betweenness relative to degree
            gk_score = bc / np.log(degree + 1)

            gatekeepers.append({
                'author': node,
                'gatekeeper_score': round(gk_score, 4),
                'betweenness': round(bc, 4),
                'degree': degree
            })

    gatekeepers.sort(key=lambda x: x['gatekeeper_score'], reverse=True)

    return gatekeepers[:30]


def get_author_network_position(author: str) -> Dict[str, Any]:
    """Get comprehensive network position for a specific author"""
    if not NETWORKX_AVAILABLE:
        return {'error': 'networkx not available'}

    G = build_undirected_graph()
    DG = build_interaction_graph()

    if author not in G.nodes():
        return {'error': 'author not in network'}

    # Basic metrics
    degree = G.degree(author)
    in_degree = DG.in_degree(author) if author in DG.nodes() else 0
    out_degree = DG.out_degree(author) if author in DG.nodes() else 0

    # Centrality metrics
    pagerank = compute_pagerank()
    betweenness = compute_betweenness_centrality()
    eigenvector = compute_eigenvector_centrality()

    # Community
    if LOUVAIN_AVAILABLE:
        partition = community_louvain.best_partition(G)
        community_id = partition.get(author, -1)
        community_size = sum(1 for v in partition.values() if v == community_id)
    else:
        community_id = -1
        community_size = 0

    # Neighbors
    neighbors = list(G.neighbors(author))
    neighbor_communities = defaultdict(int)
    for n in neighbors:
        neighbor_communities[partition.get(n, -1) if LOUVAIN_AVAILABLE else -1] += 1

    return {
        'author': author,
        'degree': degree,
        'in_degree': in_degree,
        'out_degree': out_degree,
        'pagerank': round(pagerank.get(author, 0), 6),
        'betweenness': round(betweenness.get(author, 0), 6),
        'eigenvector': round(eigenvector.get(author, 0), 6),
        'community_id': community_id,
        'community_size': community_size,
        'neighbor_count': len(neighbors),
        'top_neighbors': neighbors[:10],
        'neighbor_communities': dict(neighbor_communities)
    }


def compute_all_graph_metrics() -> Dict[str, Any]:
    """Compute and store all graph metrics"""
    ensure_graph_tables()

    results = {}

    try:
        pagerank = compute_pagerank()
        results['pagerank'] = {'status': 'success', 'nodes': len(pagerank)}

        # Save top results
        conn = get_db()
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        for author, score in pagerank.items():
            cursor.execute("""
                INSERT OR REPLACE INTO graph_metrics
                (author_name, metric_name, metric_value, computed_at)
                VALUES (?, 'pagerank', ?, ?)
            """, (author, score, now))

        conn.commit()
        conn.close()

    except Exception as e:
        results['pagerank'] = {'status': 'error', 'error': str(e)}

    try:
        betweenness = compute_betweenness_centrality()
        results['betweenness'] = {'status': 'success', 'nodes': len(betweenness)}

        conn = get_db()
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        for author, score in betweenness.items():
            cursor.execute("""
                INSERT OR REPLACE INTO graph_metrics
                (author_name, metric_name, metric_value, computed_at)
                VALUES (?, 'betweenness', ?, ?)
            """, (author, score, now))

        conn.commit()
        conn.close()

    except Exception as e:
        results['betweenness'] = {'status': 'error', 'error': str(e)}

    try:
        eigenvector = compute_eigenvector_centrality()
        results['eigenvector'] = {'status': 'success', 'nodes': len(eigenvector)}

        conn = get_db()
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        for author, score in eigenvector.items():
            cursor.execute("""
                INSERT OR REPLACE INTO graph_metrics
                (author_name, metric_name, metric_value, computed_at)
                VALUES (?, 'eigenvector', ?, ?)
            """, (author, score, now))

        conn.commit()
        conn.close()

    except Exception as e:
        results['eigenvector'] = {'status': 'error', 'error': str(e)}

    try:
        communities = detect_communities()
        results['communities'] = {
            'status': 'success',
            'count': communities.get('total_communities', 0)
        }
    except Exception as e:
        results['communities'] = {'status': 'error', 'error': str(e)}

    results['completed_at'] = datetime.now().isoformat()

    return results


def get_top_influencers(metric: str = 'pagerank', limit: int = 20) -> List[Dict[str, Any]]:
    """Get top influencers by a specific metric"""
    ensure_graph_tables()
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT author_name, metric_value FROM graph_metrics
        WHERE metric_name = ?
        ORDER BY metric_value DESC
        LIMIT ?
    """, (metric, limit))

    results = [
        {'author': row[0], metric: round(row[1], 6)}
        for row in cursor.fetchall()
    ]

    conn.close()

    return results


def get_graph_summary() -> Dict[str, Any]:
    """Get summary statistics about the graph"""
    if not NETWORKX_AVAILABLE:
        return {'error': 'networkx not available'}

    G = build_undirected_graph()
    DG = build_interaction_graph()

    if len(G.nodes()) == 0:
        return {'error': 'empty graph'}

    # Basic stats
    stats = {
        'nodes': len(G.nodes()),
        'edges': len(G.edges()),
        'directed_edges': len(DG.edges()),
        'density': round(nx.density(G), 6),
        'average_degree': round(sum(dict(G.degree()).values()) / len(G.nodes()), 2)
    }

    # Connected components
    components = list(nx.connected_components(G))
    stats['connected_components'] = len(components)
    stats['largest_component_size'] = len(max(components, key=len)) if components else 0

    # Clustering coefficient
    try:
        stats['avg_clustering'] = round(nx.average_clustering(G), 4)
    except Exception:
        stats['avg_clustering'] = None

    return stats


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python graphs.py [metrics|communities|bridges|cliques|summary|influencers]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "metrics":
        print("Computing all graph metrics...")
        results = compute_all_graph_metrics()
        for k, v in results.items():
            print(f"  {k}: {v}")

    elif command == "communities":
        print("Detecting communities...")
        result = detect_communities()
        print(f"Found {result.get('total_communities', 0)} communities")
        if 'communities' in result:
            for c in result['communities'][:5]:
                print(f"  Community {c['community_id']}: {c['size']} members")
                print(f"    Sample: {', '.join(c['members'][:5])}")

    elif command == "bridges":
        print("Finding bridge agents...")
        bridges = find_bridges()
        for b in bridges[:10]:
            print(f"  {b['author']}: score={b['bridge_score']:.4f}, cross={b['cross_community_connections']}")

    elif command == "cliques":
        print("Finding cliques...")
        cliques = find_cliques()
        for c in cliques[:10]:
            print(f"  Clique {c['clique_id']}: {c['size']} members - {', '.join(c['members'][:5])}")

    elif command == "summary":
        print("Graph summary:")
        stats = get_graph_summary()
        for k, v in stats.items():
            print(f"  {k}: {v}")

    elif command == "influencers":
        print("Top influencers by PageRank:")
        influencers = get_top_influencers()
        for i in influencers[:15]:
            print(f"  {i['author']}: {i['pagerank']:.6f}")

    else:
        print(f"Unknown command: {command}")
