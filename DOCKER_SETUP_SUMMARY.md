# Docker Setup Summary

## ✅ What Was Created

### Core Docker Files
1. **Dockerfile** - Multi-stage optimized image with Python 3.12
   - Pre-installs all dependencies (transformers, torch, spaCy, NLTK)
   - Downloads spaCy models (en_core_web_lg, nl_core_news_lg)
   - Downloads NLTK data
   - Health checks configured
   - Optimized layer caching

2. **docker-compose.yml** - Development configuration
   - Direct port mapping (8501)
   - Local volume mounts in `volumes/` directory
   - Resource limits (4 CPU, 8GB RAM)
   - Auto-restart policy

3. **docker-compose.prod.yml** - Production configuration
   - Nginx reverse proxy
   - SSL/TLS ready
   - Rate limiting
   - Better logging
   - Same volume structure

4. **.dockerignore** - Optimizes build context
   - Excludes cache, venv, tests, etc.
   - Reduces image size

### Nginx Configuration
5. **nginx/nginx.conf** - Reverse proxy setup
   - WebSocket support for Streamlit
   - Rate limiting (10 req/s)
   - SSL/TLS configuration (commented, ready to enable)
   - Security headers
   - HTTP → HTTPS redirect (ready)

### Helper Scripts
6. **docker-start.sh** - Quick start script
   - Auto-creates volume directories
   - Supports --prod and --build flags
   - Shows helpful information after start
   - Checks Docker installation

7. **docker-stop.sh** - Stop script
   - Supports --dev, --prod flags
   - Optional volume removal (--remove-volumes)
   - Safe with confirmation prompts

8. **Makefile** - Convenient commands
   - `make start` - Start development
   - `make start-prod` - Start production
   - `make logs` - View logs
   - `make backup` - Backup volumes
   - `make status` - Check services
   - `make help` - Show all commands

### Documentation
9. **DOCKER_DEPLOYMENT.md** - Complete guide (400+ lines)
   - Installation instructions
   - Volume management
   - Configuration options
   - Troubleshooting
   - Security best practices
   - Performance optimization

10. **env.example** - Environment template
    - All configurable variables
    - Well-commented

### Volume Management
11. **volumes/.gitignore** - Excludes cached data from git
12. **volumes/README.md** - Volume documentation

## 📁 Volume Structure

All data stored in local `volumes/` directory:

```
volumes/
├── transformers_cache/    # HuggingFace transformers (2-5 GB)
├── huggingface_cache/     # HuggingFace hub (1-2 GB)
├── torch_cache/           # PyTorch models (500 MB)
├── nltk_data/             # NLTK datasets (200 MB)
├── spacy_data/            # spaCy models (1 GB)
├── app_data/              # Application data (100 MB)
└── nginx_logs/            # Nginx logs (production only)
```

**Benefits:**
- ✅ Easy backup: `tar czf backup.tar.gz volumes/`
- ✅ Easy migration: Copy `volumes/` to new server
- ✅ Easy inspection: Browse files directly
- ✅ Easy cleanup: `rm -rf volumes/transformers_cache/`
- ✅ No Docker volume commands needed

## 🚀 Usage

### Quick Start
```bash
# Development (simplest)
./docker-start.sh

# Production with Nginx
./docker-start.sh --prod

# Using Make
make start
make start-prod
```

### Common Commands
```bash
# View logs
make logs
docker-compose logs -f

# Stop services
make stop
./docker-stop.sh

# Restart
make restart

# Check status
make status
docker-compose ps

# Backup data
make backup
tar czf backup.tar.gz volumes/

# Clean up
make clean          # Remove containers only
make clean-all      # Remove containers + volumes
```

## 🔧 Configuration

### Resource Limits
Edit `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      cpus: '4.0'      # Adjust based on server
      memory: 8G       # Adjust based on server
```

### Port Configuration
```yaml
ports:
  - "8080:8501"  # Change 8080 to your preferred port
```

### Environment Variables
Copy `env.example` to `.env` and customize.

## 🔒 Production Deployment

### 1. SSL/TLS Setup
```bash
# Create SSL directory
mkdir -p nginx/ssl

# Add your certificates
cp /path/to/fullchain.pem nginx/ssl/
cp /path/to/privkey.pem nginx/ssl/

# Or use Let's Encrypt
certbot certonly --standalone -d your-domain.com
cp /etc/letsencrypt/live/your-domain.com/*.pem nginx/ssl/
```

### 2. Update Nginx Config
Edit `nginx/nginx.conf`:
- Uncomment HTTPS server block
- Update `server_name` with your domain
- Uncomment HTTP → HTTPS redirect

### 3. Deploy
```bash
./docker-start.sh --prod
# or
make start-prod
```

## 💾 Backup & Restore

### Backup
```bash
# Full backup
tar czf bahar-backup-$(date +%Y%m%d).tar.gz volumes/

# Specific cache
tar czf transformers-backup.tar.gz volumes/transformers_cache/

# Using Make
make backup
```

### Restore
```bash
# Extract backup
tar xzf bahar-backup-YYYYMMDD.tar.gz

# Restart services
./docker-start.sh
```

## 📊 Monitoring

### Resource Usage
```bash
# Real-time stats
make stats
docker stats bahar-app

# Disk usage
make disk
du -sh volumes/
```

### Health Check
```bash
# Check health
make health
docker inspect --format='{{.State.Health.Status}}' bahar-app

# View logs
make logs
```

## 🐛 Troubleshooting

### Container Won't Start
```bash
# Check logs
make logs
docker-compose logs bahar

# Rebuild
./docker-start.sh --build
```

### Port Already in Use
```bash
# Find process using port 8501
lsof -i :8501

# Or change port in docker-compose.yml
```

### Out of Disk Space
```bash
# Check usage
make disk

# Clean old images
docker image prune -a

# Remove unused volumes
rm -rf volumes/transformers_cache/
```

### Models Not Loading
```bash
# Check volumes
ls -lah volumes/transformers_cache/

# Re-download
docker-compose exec bahar python -m spacy download en_core_web_lg
```

## 🎯 Key Features

✅ **Easy Deployment**
- One command start: `./docker-start.sh`
- All dependencies included
- No manual setup required

✅ **Production Ready**
- Nginx reverse proxy
- SSL/TLS support
- Rate limiting
- Auto-restart
- Health checks

✅ **Easy Maintenance**
- Local volumes (easy backup)
- Helper scripts
- Makefile commands
- Comprehensive docs

✅ **Optimized**
- Layer caching
- Resource limits
- Logging configured
- Security headers

## 📚 Documentation

- **DOCKER_DEPLOYMENT.md** - Complete deployment guide
- **volumes/README.md** - Volume management
- **README.md** - Updated with Docker instructions
- **This file** - Quick reference

## 🔗 Quick Links

- Main README: [README.md](README.md)
- Full Docker Guide: [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md)
- Volume Info: [volumes/README.md](volumes/README.md)

## ✨ Next Steps

1. **Test locally:**
   ```bash
   ./docker-start.sh
   open http://localhost:8501
   ```

2. **Deploy to server:**
   ```bash
   # Copy project to server
   scp -r bahar/ user@server:/path/

   # SSH to server
   ssh user@server

   # Start production
   cd /path/bahar
   ./docker-start.sh --prod
   ```

3. **Set up SSL:**
   - Add certificates to `nginx/ssl/`
   - Update `nginx/nginx.conf`
   - Restart: `make restart`

4. **Monitor:**
   ```bash
   make status
   make logs
   make disk
   ```

---

**Created:** $(date +%Y-%m-%d)
**Version:** 1.0
**Status:** ✅ Ready for production
