# Moltbook Analysis Tools

Vector embeddings, pattern analysis, and coordination research for Moltbook data.

## Setup

```bash
cd ~/dev/moltmirror

# Install dependencies
uv pip install -e ".[dev]"

# Or install specific packages
uv pip install sentence-transformers numpy tqdm click
```

## Quick Start

```bash
# Initialize database
python moltanalyze.py init

# Import data from archive
python moltanalyze.py import-data

# Show statistics
python moltanalyze.py stats

# Find high-signal agents
python moltanalyze.py agents

# Find coordination-related posts
python moltanalyze.py coordination
```

## Embeddings & Semantic Search

```bash
# Generate embeddings for all posts/comments (requires sentence-transformers)
python moltanalyze.py embeddings

# Search semantically
python moltanalyze.py search "agent coordination patterns"

# Find similar posts
python analysis/embeddings.py similar POST_ID
```

## Analysis Modules

### `analysis/__init__.py`
- Database schema and initialization
- Data import from archive
- High-signal agent detection
- Coordination topic search

### `analysis/embeddings.py`
- Vector embedding generation using local models
- Semantic search across all content
- Similarity matching for posts

## Database Schema

### Tables
- **posts** — All posts with metadata
- **comments** — All comments with parent relationships
- **agents** — Agent statistics and reputation metrics
- **embeddings** — Vector embeddings for semantic search
- **coordination_patterns** — Discovered coordination patterns

## Features

### High-Signal Agent Detection
Finds agents with:
- Minimum post threshold (default: 3)
- Minimum average upvotes (default: 2.0)
- Sorted by engagement quality

### Coordination Topic Search
Finds posts about:
- Coordination, collaboration
- Multi-agent systems
- Consensus and governance
- Incentives and protocols

### Semantic Search
Uses sentence-transformers (e5-base-v2) to:
- Generate embeddings locally (no API calls)
- Search by meaning, not just keywords
- Find similar posts

## Research Questions These Tools Can Answer

1. **Who are the most influential agents?**
   ```bash
   python moltanalyze.py agents --min-posts 5 --min-upvotes 5.0
   ```

2. **What coordination mechanisms are being discussed?**
   ```bash
   python moltanalyze.py coordination
   ```

3. **Find posts similar to a given post:**
   ```bash
   python analysis/embeddings.py similar POST_ID
   ```

4. **Search for specific concepts:**
   ```bash
   python moltanalyze.py search "trust formation incomplete information"
   ```

## Future Enhancements

- [ ] Network graph of agent interactions
- [ ] Topic modeling (LDA/NMF)
- [ ] Sentiment analysis over time
- [ ] Coordination pattern detection
- [ ] Agent reputation dynamics
- [ ] Cross-reference with tulip data

## Data Sources

- Archive: `~/dev/moltmirror/archive/data/`
- Database: `~/dev/moltmirror/analysis.db`
- Models: `~/.cache/huggingface/hub/`
