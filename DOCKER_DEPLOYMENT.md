# Docker Deployment Guide

Complete guide for deploying Bahar using Docker and Docker Compose.

## 📋 Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- At least 8GB RAM available
- 20GB free disk space (for models and caches)

## 🚀 Quick Start

### Using Helper Scripts (Recommended)

```bash
# Development mode
./docker-start.sh

# Production mode with Nginx
./docker-start.sh --prod

# Stop services
./docker-stop.sh

# View help
./docker-start.sh --help
```

### Using Makefile (Alternative)

```bash
# Start development
make start

# Start production
make start-prod

# View logs
make logs

# Stop services
make stop

# See all commands
make help
```

### Using Docker Compose Directly

**Development Mode:**
```bash
# Build and start the application
docker-compose up -d

# View logs
docker-compose logs -f bahar

# Access the application
open http://localhost:8501
```

**Production Mode (with Nginx):**
```bash
# Build and start with Nginx reverse proxy
docker-compose -f docker-compose.prod.yml up -d

# View logs
docker-compose -f docker-compose.prod.yml logs -f

# Access the application
open http://your-server-ip
```

## 📦 What's Included

### Docker Files

1. **Dockerfile** - Main application image
   - Python 3.12 slim base
   - UV package manager for fast installs
   - Pre-downloaded spaCy models (en_core_web_lg, nl_core_news_lg)
   - Pre-downloaded NLTK data
   - Optimized layer caching

2. **docker-compose.yml** - Development configuration
   - Single service (Bahar app)
   - Port 8501 exposed directly
   - Volume mounts for persistence

3. **docker-compose.prod.yml** - Production configuration
   - Bahar app + Nginx reverse proxy
   - SSL/TLS support ready
   - Rate limiting
   - Better logging

4. **.dockerignore** - Excludes unnecessary files from image

5. **nginx/nginx.conf** - Nginx configuration
   - WebSocket support for Streamlit
   - Rate limiting
   - SSL/TLS configuration (commented)

## 💾 Volume Management

### Local Directory Volumes (Persistent Data)

All model caches and data are stored in local directories under `volumes/`:

| Directory | Purpose | Size (approx) |
|-----------|---------|---------------|
| `volumes/transformers_cache/` | HuggingFace transformers models | 2-5 GB |
| `volumes/huggingface_cache/` | HuggingFace hub cache | 1-2 GB |
| `volumes/torch_cache/` | PyTorch models | 500 MB |
| `volumes/nltk_data/` | NLTK datasets | 200 MB |
| `volumes/spacy_data/` | spaCy models | 1 GB |
| `volumes/app_data/` | Application data | 100 MB |
| `volumes/nginx_logs/` | Nginx logs (production) | 50-100 MB |

### View Volumes

```bash
# Check total size
du -sh volumes/

# Check individual directories
du -sh volumes/*/

# List contents
ls -lah volumes/transformers_cache/
```

### Backup Volumes

```bash
# Backup all volumes (recommended)
tar czf bahar-volumes-backup-$(date +%Y%m%d).tar.gz volumes/

# Backup specific directory
tar czf transformers-backup.tar.gz volumes/transformers_cache/

# Restore from backup
tar xzf bahar-volumes-backup-YYYYMMDD.tar.gz
```

### Clean Up Volumes

```bash
# Remove all cached data (WARNING: will be re-downloaded on next start!)
rm -rf volumes/*/

# Remove specific cache
rm -rf volumes/transformers_cache/

# Clean up old logs
rm -rf volumes/nginx_logs/*
```

## 🔧 Configuration

### Environment Variables

Edit `docker-compose.yml` to customize:

```yaml
environment:
  # Cache directories
  - TRANSFORMERS_CACHE=/app/cache/transformers
  - HF_HOME=/app/cache/huggingface

  # Streamlit settings
  - STREAMLIT_SERVER_PORT=8501
  - STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

### Resource Limits

Adjust CPU and memory limits in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '4.0'      # Maximum CPUs
      memory: 8G       # Maximum RAM
    reservations:
      cpus: '2.0'      # Minimum CPUs
      memory: 4G       # Minimum RAM
```

### Port Configuration

Change the exposed port:

```yaml
ports:
  - "8080:8501"  # Access on port 8080 instead of 8501
```

## 🛠️ Common Commands

### Build and Start

```bash
# Build image
docker-compose build

# Start services
docker-compose up -d

# Build and start (fresh build)
docker-compose up -d --build

# Start with specific compose file
docker-compose -f docker-compose.prod.yml up -d
```

### Manage Services

```bash
# Stop services
docker-compose stop

# Start services
docker-compose start

# Restart services
docker-compose restart

# Stop and remove containers
docker-compose down

# Stop and remove containers + volumes
docker-compose down -v
```

### Logs and Monitoring

```bash
# View logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f bahar

# View last 100 lines
docker-compose logs --tail=100 bahar

# Check service status
docker-compose ps

# Check resource usage
docker stats bahar-app
```

### Execute Commands in Container

```bash
# Open shell in container
docker-compose exec bahar bash

# Run Python command
docker-compose exec bahar python -c "import bahar; print(bahar.__version__)"

# Check installed packages
docker-compose exec bahar pip list

# Download additional spaCy model
docker-compose exec bahar python -m spacy download de_core_news_lg
```

## 🔒 Production Deployment

### 1. Configure SSL/TLS

Create SSL certificates (using Let's Encrypt):

```bash
# Create nginx/ssl directory
mkdir -p nginx/ssl

# Generate self-signed certificate (for testing)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/privkey.pem \
  -out nginx/ssl/fullchain.pem

# For production, use Let's Encrypt:
# certbot certonly --standalone -d your-domain.com
# cp /etc/letsencrypt/live/your-domain.com/*.pem nginx/ssl/
```

### 2. Update Nginx Configuration

Edit `nginx/nginx.conf`:

1. Uncomment the HTTPS server block
2. Update `server_name` with your domain
3. Uncomment the HTTP → HTTPS redirect

### 3. Deploy

```bash
# Start with production configuration
docker-compose -f docker-compose.prod.yml up -d

# Check logs
docker-compose -f docker-compose.prod.yml logs -f

# Access via HTTPS
open https://your-domain.com
```

### 4. Enable Auto-Restart

Services are configured with `restart: always` in production mode, so they'll automatically restart on:
- Container crash
- Docker daemon restart
- Server reboot

## 📊 Monitoring

### Health Checks

The application includes built-in health checks:

```bash
# Check health status
docker inspect --format='{{.State.Health.Status}}' bahar-app

# View health check logs
docker inspect --format='{{range .State.Health.Log}}{{.Output}}{{end}}' bahar-app
```

### Resource Monitoring

```bash
# Real-time stats
docker stats bahar-app

# Export stats to file
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" > stats.txt
```

## 🐛 Troubleshooting

### Container Won't Start

```bash
# Check logs
docker-compose logs bahar

# Check if port is already in use
lsof -i :8501

# Remove and rebuild
docker-compose down
docker-compose up -d --build
```

### Out of Memory

```bash
# Increase memory limit in docker-compose.yml
# Or free up system memory
docker system prune -a
```

### Models Not Loading

```bash
# Check volume mounts
docker volume inspect bahar_transformers_cache

# Re-download models
docker-compose exec bahar python -m spacy download en_core_web_lg
```

### Permission Issues

```bash
# Fix volume permissions
docker-compose exec bahar chown -R root:root /app/cache
```

## 🔄 Updates and Maintenance

### Update Application

```bash
# Pull latest code
git pull

# Rebuild and restart
docker-compose down
docker-compose up -d --build
```

### Update Dependencies

```bash
# Edit pyproject.toml
# Then rebuild
docker-compose build --no-cache
docker-compose up -d
```

### Clean Up

```bash
# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune

# Full cleanup (careful!)
docker system prune -a --volumes
```

## 📈 Performance Optimization

### 1. Pre-download Models

To speed up first startup, pre-download models during build:

```dockerfile
# Add to Dockerfile
RUN python -c "from transformers import AutoModel, AutoTokenizer; \
    AutoModel.from_pretrained('bhadresh-savani/distilbert-base-uncased-emotion'); \
    AutoTokenizer.from_pretrained('bhadresh-savani/distilbert-base-uncased-emotion')"
```

### 2. Use BuildKit

Enable Docker BuildKit for faster builds:

```bash
DOCKER_BUILDKIT=1 docker-compose build
```

### 3. Multi-stage Builds

For smaller images, consider multi-stage builds (already optimized in Dockerfile).

### 4. Resource Allocation

Adjust based on your server:

- **Small server (4GB RAM)**: Set limits to 2GB
- **Medium server (8GB RAM)**: Set limits to 4-6GB
- **Large server (16GB+ RAM)**: Set limits to 8GB+

## 🔐 Security Best Practices

1. **Don't expose port 8501 directly in production** - Use Nginx
2. **Enable SSL/TLS** - Use Let's Encrypt certificates
3. **Set up rate limiting** - Already configured in nginx.conf
4. **Keep images updated** - Regularly rebuild with latest base images
5. **Use secrets for sensitive data** - Don't hardcode in docker-compose.yml
6. **Enable firewall** - Only allow ports 80/443

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Streamlit Deployment Guide](https://docs.streamlit.io/knowledge-base/tutorials/deploy)
- [Nginx Documentation](https://nginx.org/en/docs/)

## 💡 Tips

- **First startup is slow** - Models are being downloaded and cached
- **Subsequent startups are fast** - Models are cached in volumes
- **Use production mode for public deployment** - Better security and performance
- **Monitor disk space** - Model caches can grow large
- **Backup volumes regularly** - Especially before major updates

## 🆘 Support

If you encounter issues:

1. Check logs: `docker-compose logs -f bahar`
2. Check health: `docker inspect bahar-app`
3. Check resources: `docker stats bahar-app`
4. Check volumes: `docker volume ls`
5. Check network: `docker network inspect bahar_bahar-network`

For more help, see the main README.md or open an issue.

