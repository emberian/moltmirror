#!/usr/bin/env python3
"""
Moltbook Analysis CLI
Entry point for all analysis tools
"""

import click
from pathlib import Path
import sys

# Add analysis module to path
sys.path.insert(0, str(Path(__file__).parent))

from analysis import (
    init_database, import_posts, import_comments, import_agents,
    get_high_signal_agents, search_coordination_topics
)


@click.group()
def cli():
    """Moltbook Analysis Tools"""
    pass


@cli.command()
def init():
    """Initialize the analysis database"""
    init_database()


@cli.command()
def import_data():
    """Import all data from archive"""
    import_posts()
    import_comments()
    import_agents()


@cli.command()
@click.option('--min-posts', default=3, help='Minimum posts required')
@click.option('--min-upvotes', default=2.0, help='Minimum average upvotes')
def agents(min_posts, min_upvotes):
    """Find high-signal agents"""
    results = get_high_signal_agents(min_posts, min_upvotes)
    click.echo(f"\nHigh-signal agents (≥{min_posts} posts, ≥{min_upvotes} avg upvotes):")
    click.echo("-" * 70)
    for agent in results[:30]:
        click.echo(f"{agent[1]:20} | {agent[2]} posts | {agent[3]} comments | {agent[4]:.1f} avg ⬆")


@cli.command()
def coordination():
    """Find coordination-related posts"""
    posts = search_coordination_topics()
    click.echo(f"\nCoordination-related posts ({len(posts)} found):")
    click.echo("-" * 70)
    for post in posts[:20]:
        title = post[1][:50] + "..." if len(post[1]) > 50 else post[1]
        click.echo(f"{title:55} | {post[2]:15} | {post[3]}⬆ | {post[4]}💬")


@cli.command()
def embeddings():
    """Generate embeddings for all content"""
    try:
        from analysis.embeddings import generate_embeddings
        generate_embeddings()
    except ImportError as e:
        click.echo(f"Error: {e}")
        click.echo("Install with: uv pip install sentence-transformers")


@cli.command()
@click.argument('query')
@click.option('--top-k', default=10, help='Number of results')
def search(query, top_k):
    """Semantic search across all content"""
    try:
        from analysis.embeddings import semantic_search
        results = semantic_search(query, top_k)
        click.echo(f"\nResults for: {query}")
        click.echo("-" * 70)
        for sim, content_type, content_id, text in results:
            prefix = "📄" if content_type == "post" else "💬"
            click.echo(f"{prefix} [{sim:.3f}] {text[:70]}...")
    except ImportError:
        click.echo("sentence-transformers not available")


@cli.command()
@click.argument('post_id')
def similar(post_id):
    """Find posts similar to a given post"""
    try:
        from analysis.embeddings import find_similar_posts
        results = find_similar_posts(post_id)
        click.echo(f"\nPosts similar to {post_id}:")
        click.echo("-" * 70)
        for sim, content_id, title, author in results:
            click.echo(f"[{sim:.3f}] {title[:60]}... ({author})")
    except ImportError:
        click.echo("sentence-transformers not available")


@cli.command()
def clusters():
    """Find content clusters using embeddings"""
    try:
        from analysis.embeddings import find_content_clusters
        clusters = find_content_clusters(n_clusters=10)
        click.echo("\nContent Clusters:")
        click.echo("-" * 70)
        for i, cluster in enumerate(clusters):
            if cluster:
                click.echo(f"\nCluster {i+1} ({len(cluster)} posts):")
                for post in cluster[:5]:
                    click.echo(f"  - {post['title'][:60]}... ({post['author']})")
    except ImportError as e:
        click.echo(f"Error: {e}")


@cli.command()
@click.argument('query')
def trends(query):
    """Track topic trends over time"""
    try:
        from analysis.embeddings import topic_trends
        trends = topic_trends(query)
        click.echo(f"\nTopic trends for: {query}")
        click.echo(f"Found {len(trends)} related posts")
        click.echo("-" * 70)
        for t in trends[-10:]:
            click.echo(f"[{t['created_at'][:16]}] {t['title'][:50]}... (sim: {t['similarity']:.2f})")
    except ImportError:
        click.echo("sentence-transformers not available")


@cli.command()
@click.argument('author1')
@click.argument('author2')
def compare(author1, author2):
    """Compare semantic similarity between two authors"""
    try:
        from analysis.embeddings import find_author_similarity
        result = find_author_similarity(author1, author2)
        if result:
            click.echo(f"\nAuthor similarity: {result['author1']} vs {result['author2']}")
            click.echo("-" * 70)
            click.echo(f"Similarity: {result['similarity']:.3f}")
            click.echo(f"{result['author1']} posts: {result['author1_posts']}")
            click.echo(f"{result['author2']} posts: {result['author2_posts']}")
        else:
            click.echo("Could not calculate similarity (insufficient data)")
    except ImportError:
        click.echo("sentence-transformers not available")


@cli.command()
def stats():
    """Show database statistics"""
    import sqlite3
    from analysis import DB_PATH
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM posts")
    post_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM comments")
    comment_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM agents")
    agent_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM embeddings")
    embedding_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT SUM(upvotes) FROM posts")
    total_upvotes = cursor.fetchone()[0] or 0
    
    conn.close()
    
    click.echo("\n📊 Moltbook Archive Statistics")
    click.echo("-" * 40)
    click.echo(f"Posts:       {post_count:,}")
    click.echo(f"Comments:    {comment_count:,}")
    click.echo(f"Agents:      {agent_count:,}")
    click.echo(f"Embeddings:  {embedding_count:,}")
    click.echo(f"Total ⬆:     {total_upvotes:,}")


if __name__ == '__main__':
    cli()
