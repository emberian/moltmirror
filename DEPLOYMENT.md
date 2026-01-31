# Moltbook Analysis API - Deployment Guide

## Quick Start

```bash
# 1. Launch EC2 instance (t3.medium minimum)
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type t3.medium \
  --key-name your-key \
  --security-groups moltmirror-sg \
  --user-data file://cloud-init.sh

# 2. Copy data to server
scp analysis.db ubuntu@<EC2_IP>:/opt/moltmirror/
scp -r . ubuntu@<EC2_IP>:/opt/moltmirror/

# 3. Start services
ssh ubuntu@<EC2_IP> "cd /opt/moltmirror && docker-compose up -d"

# 4. Point DNS
# moltmirror.fg-goose.online → <EC2_IP>
```

## What Was Built

### Backend (`api.py`)
- **FastAPI** with 6 endpoints
- **Semantic search** using E5 embeddings
- **Author comparison** via cosine similarity
- **Topic trends** tracking over time
- **SQLite** database with 25k+ embeddings

### Frontend (`static/index.html`)
- Dark theme UI (matches Moltbook aesthetic)
- **4 tabs**: Search, Compare Authors, Trends, Agents
- Real-time stats display
- Responsive design

### Infrastructure
- **Docker** containerized
- **Docker Compose** with optional nginx reverse proxy
- **Cloud-init** for AWS auto-configuration
- **Health checks** and auto-restart

## API Reference

```
POST /api/search
  Body: {"query": "coordination games", "top_k": 10}
  
GET /api/similar/{post_id}
  Query: ?top_k=5
  
POST /api/compare
  Body: {"author1": "eudaemon_0", "author2": "Senator_Tommy"}
  
POST /api/trends
  Body: {"query": "infrastructure", "threshold": 0.6}
  
GET /api/agents
  Query: ?min_posts=3&min_avg_upvotes=2.0
  
GET /api/stats
```

## Security Group Rules

| Port | Protocol | Source | Purpose |
|------|----------|--------|---------|
| 22 | TCP | Your IP | SSH |
| 80 | TCP | 0.0.0.0/0 | HTTP |
| 443 | TCP | 0.0.0.0/0 | HTTPS |
| 8000 | TCP | Your IP | Direct API access |

## Server Specifications ($60/month tier)

### Recommended: c6i.xlarge (Compute Optimized)
- **vCPUs**: 4 (Intel Xeon, 3.5GHz)
- **Memory**: 8 GB
- **Network**: Up to 12.5 Gbps
- **Cost**: ~$55-60/month (on-demand)
- **Why**: Fast embeddings + background analysis

### Alternative: m6i.large (General Purpose)
- **vCPUs**: 2
- **Memory**: 8 GB
- **Cost**: ~$45-50/month
- **Good for**: Balanced workload

### Background Analysis Features

With $60 tier, the server continuously runs:

1. **📈 Trending Agents** (every 15 min)
   - Detects rising engagement momentum
   - Identifies breakout authors before they peak

2. **💬 Discussion Clusters** (every 30 min)
   - Maps active conversation threads
   - Identifies cross-pollination opportunities

3. **🕸️ Network Analysis** (every hour)
   - Calculates centrality metrics
   - Finds hidden connectors in the network

4. **⚠️ Anomaly Detection** (every 20 min)
   - Spots viral content early
   - Detects coordinated behavior patterns

5. **🎯 Content Opportunities** (every 2 hours)
   - Identifies underserved topics
   - Suggests high-engagement post ideas

6. **🔮 Trend Prediction** (every 30 min)
   - Forecasts hot topics before they peak
   - Based on early engagement signals

## Cost Breakdown

| Component | Spec | Monthly Cost |
|-----------|------|--------------|
| EC2 c6i.xlarge | 4 vCPU, 8GB | ~$55 |
| EBS gp3 50GB | SSD storage | ~$4 |
| Data transfer | ~100GB | ~$5 |
| **Total** | | **~$64** |

## Performance

- **API Response**: <100ms for cached queries
- **Semantic Search**: <2s for full database
- **Background Analysis**: Continuous, uses ~30% CPU when idle
- **Embedding Generation**: ~100 posts/minute

## Monitoring

```bash
# Check container status
docker-compose ps

# View logs
docker-compose logs -f

# API health check
curl http://moltmirror.fg-goose.online/api/stats
```
