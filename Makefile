.PHONY: help start stop restart logs build clean backup status

# Default target
help:
	@echo "🌸 Bahar - Docker Management Commands"
	@echo "======================================"
	@echo ""
	@echo "Development:"
	@echo "  make start          - Start development services"
	@echo "  make stop           - Stop all services"
	@echo "  make restart        - Restart services"
	@echo "  make logs           - View logs (follow mode)"
	@echo "  make build          - Rebuild Docker images"
	@echo ""
	@echo "Production:"
	@echo "  make start-prod     - Start production services (with Nginx)"
	@echo "  make stop-prod      - Stop production services"
	@echo "  make logs-prod      - View production logs"
	@echo ""
	@echo "Maintenance:"
	@echo "  make status         - Show service status"
	@echo "  make backup         - Backup volumes to tar.gz"
	@echo "  make clean          - Stop and remove containers"
	@echo "  make clean-all      - Stop, remove containers and volumes"
	@echo "  make shell          - Open shell in container"
	@echo ""
	@echo "Monitoring:"
	@echo "  make stats          - Show resource usage"
	@echo "  make disk           - Show disk usage of volumes"
	@echo "  make health         - Check health status"

# Development commands
start:
	@echo "🚀 Starting development services..."
	@./docker-start.sh

stop:
	@echo "🛑 Stopping services..."
	@./docker-stop.sh

restart:
	@echo "🔄 Restarting services..."
	@docker-compose restart

logs:
	@docker-compose logs -f

build:
	@echo "🔨 Building Docker images..."
	@docker-compose build

# Production commands
start-prod:
	@echo "🚀 Starting production services..."
	@./docker-start.sh --prod

stop-prod:
	@echo "🛑 Stopping production services..."
	@./docker-stop.sh --prod

logs-prod:
	@docker-compose -f docker-compose.prod.yml logs -f

# Maintenance commands
status:
	@echo "📊 Service Status:"
	@echo ""
	@docker-compose ps
	@echo ""
	@docker-compose -f docker-compose.prod.yml ps 2>/dev/null || true

backup:
	@echo "💾 Creating backup..."
	@tar czf bahar-backup-$$(date +%Y%m%d-%H%M%S).tar.gz volumes/
	@echo "✓ Backup created: bahar-backup-$$(date +%Y%m%d-%H%M%S).tar.gz"

clean:
	@echo "🧹 Cleaning up containers..."
	@docker-compose down
	@docker-compose -f docker-compose.prod.yml down 2>/dev/null || true
	@echo "✓ Containers removed (volumes preserved)"

clean-all:
	@echo "⚠️  WARNING: This will remove all cached data!"
	@echo "Press Ctrl+C to cancel, or wait 5 seconds..."
	@sleep 5
	@docker-compose down -v
	@docker-compose -f docker-compose.prod.yml down -v 2>/dev/null || true
	@rm -rf volumes/*
	@echo "✓ Containers and volumes removed"

shell:
	@docker-compose exec bahar bash

# Monitoring commands
stats:
	@docker stats bahar-app --no-stream

disk:
	@echo "💿 Volume Disk Usage:"
	@du -sh volumes/ 2>/dev/null || echo "No volumes directory"
	@echo ""
	@du -sh volumes/*/ 2>/dev/null || true

health:
	@echo "🏥 Health Status:"
	@docker inspect --format='{{.State.Health.Status}}' bahar-app 2>/dev/null || echo "Container not running"

