# Moltbook Analysis Platform - Complete System

## 🎯 What Was Built

### 1. **Data Pipeline**
- Imports 13,307 posts + 96,616 comments from Moltbook archive
- Generates 768-dimensional vector embeddings using E5 model
- SQLite database with full-text + semantic indexing

### 2. **API Server** (`api.py`)
**Core Endpoints:**
- `POST /api/search` — Semantic search
- `GET /api/similar/{post_id}` — Find similar content
- `POST /api/compare` — Author similarity
- `POST /api/trends` — Topic tracking
- `GET /api/agents` — Leaderboard

**Background Insights Endpoints:**
- `GET /api/insights/trending` — Rising agents with momentum
- `GET /api/insights/discussions` — Hot conversation clusters
- `GET /api/insights/network` — Network centrality analysis
- `GET /api/insights/anomalies` — Viral content detection
- `GET /api/insights/opportunities` — Underserved topics
- `GET /api/insights/predictions` — Trend forecasting
- `GET /api/insights/all` — All insights in one call

### 3. **Background Analyzer** (`background_analysis.py`)
Continuously runs when CPU is idle:
- **Trending Agents** (15 min): Detects 50%+ engagement growth
- **Discussion Clusters** (30 min): Maps conversation networks
- **Network Centrality** (60 min): Identifies key connectors
- **Anomaly Detection** (20 min): Spots viral posts early
- **Content Opportunities** (2 hrs): High-value topic gaps
- **Trend Prediction** (30 min): Forecasts hot topics

### 4. **Web Dashboard** (`static/index.html`)
- 🔍 **Search**: Semantic queries with similarity scores
- 💡 **Insights**: 6 AI-powered analysis cards
- 👥 **Compare**: Author similarity visualization
- 📈 **Trends**: Topic evolution over time
- 🌟 **Agents**: High-signal agent leaderboard

### 5. **Deployment** 
- **Dockerfile**: Python 3.11 + ML stack
- **docker-compose.yml**: Full stack with resource limits
- **nginx.conf**: SSL termination
- **cloud-init.sh**: AWS auto-configuration

## 💰 $60/month Server Spec

**Recommended: c6i.xlarge**
- 4 vCPUs @ 3.5GHz (Intel Ice Lake)
- 8 GB RAM
- 12.5 Gbps network
- ~$55/month + storage/transfer = ~$64/month

**Resource Allocation:**
- API server: 0.5 CPU (handles requests)
- Background analysis: up to 3.5 CPU (when idle)
- Memory: 7GB usable

## 🚀 Deployment Commands

```bash
# 1. Launch EC2 with user-data
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type c6i.xlarge \
  --key-name your-key \
  --security-groups moltmirror-sg \
  --user-data file://cloud-init.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=moltmirror}]'

# 2. Wait for instance, then copy data
scp analysis.db ubuntu@<IP>:/opt/moltmirror/
scp -r api.py background_analysis.py static/ docker-compose.yml nginx.conf ubuntu@<IP>:/opt/moltmirror/

# 3. SSH and start
ssh ubuntu@<IP>
cd /opt/moltmirror
docker-compose up -d

# 4. Point DNS
# moltmirror.fg-goose.online A-record → <EC2_IP>

# 5. Get SSL with Let's Encrypt (on server)
docker run -it --rm \
  -v /opt/moltmirror/ssl:/etc/letsencrypt \
  -v /opt/moltmirror/nginx.conf:/etc/nginx/nginx.conf \
  certbot/certbot certonly --standalone -d moltmirror.fg-goose.online
```

## 📊 Background Analysis Output

The system continuously generates:

```json
{
  "trending_agents": {
    "rising_agents": [
      {"author": "NewVoice", "growth_percent": 340, "recent_posts": 5}
    ]
  },
  "conversation_clusters": {
    "hot_discussions": [
      {"title": "Infrastructure debate", "participants": 23, "comment_count": 156}
    ]
  },
  "network_centrality": {
    "network_connectors": [
      {"agent": "eudaemon_0", "total_connections": 147, "reaches": 89}
    ]
  },
  "anomalies": {
    "viral_posts": [
      {"title": "Agentic Karma", "velocity": 3547, "upvotes": 21284}
    ]
  },
  "content_gaps": {
    "content_opportunities": [
      {"topic": "formal_verification", "avg_upvotes": 45.2, "post_count": 3}
    ]
  },
  "hot_topics": {
    "predicted_trends": [
      {"topic": "infrastructure", "viral_potential": 892.5}
    ]
  }
}
```

## 🎮 Usage Examples

**Find coordination-related posts:**
```bash
curl -X POST https://moltmirror.fg-goose.online/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "coordination games multi-agent", "top_k": 5}'
```

**Compare two authors:**
```bash
curl -X POST https://moltmirror.fg-goose.online/api/compare \
  -H "Content-Type: application/json" \
  -d '{"author1": "eudaemon_0", "author2": "Senator_Tommy"}'
```

**Get latest insights:**
```bash
curl https://moltmirror.fg-goose.online/api/insights/all
```

## 🔧 Maintenance

```bash
# View logs
ssh ubuntu@<IP> "cd /opt/moltmirror && docker-compose logs -f"

# Update data
scp new_analysis.db ubuntu@<IP>:/opt/moltmirror/analysis.db
ssh ubuntu@<IP> "cd /opt/moltmirror && docker-compose restart"

# Force analysis cycle
curl -X POST https://moltmirror.fg-goose.online/api/admin/trigger-analysis

# Check health
curl https://moltmirror.fg-goose.online/api/health
```

## 📝 Files Created

```
moltmirror/
├── api.py                    # FastAPI server (300+ lines)
├── background_analysis.py    # Continuous analyzer (400+ lines)
├── moltanalyze.py            # CLI tool
├── analysis/
│   ├── __init__.py          # Database & import pipeline
│   └── embeddings.py        # Vector operations
├── static/
│   └── index.html           # Web dashboard (600+ lines)
├── Dockerfile               # Container definition
├── docker-compose.yml       # Orchestration
├── nginx.conf               # SSL proxy config
├── cloud-init.sh            # AWS setup script
├── DEPLOYMENT.md            # This guide
└── analysis.db              # SQLite database (portable)
```

## 🎯 Key Features

1. **Semantic Search**: Find posts by meaning, not just keywords
2. **Background Intelligence**: Continuous analysis when idle
3. **Predictive Insights**: Forecast trends before they peak
4. **Network Analysis**: Map conversation clusters & connectors
5. **Content Opportunities**: Identify underserved high-value topics
6. **Real-time Dashboard**: Live insights via web UI
7. **REST API**: Full programmatic access
8. **Resource Efficient**: Uses idle CPU for analysis

Total: ~2,500 lines of Python + JavaScript, fully containerized, production-ready.
