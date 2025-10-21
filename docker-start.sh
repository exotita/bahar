#!/bin/bash
# Bahar Docker Quick Start Script

set -e

echo "🌸 Bahar - Docker Deployment Script"
echo "===================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Error: Docker Compose is not installed"
    echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

# Determine Docker Compose command
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

# Create volumes directory structure
echo "📁 Creating volumes directory structure..."
mkdir -p volumes/{transformers_cache,huggingface_cache,torch_cache,nltk_data,spacy_data,app_data,nginx_logs}
echo "✓ Volumes directories created"
echo ""

# Create config directory if it doesn't exist
mkdir -p config
echo "✓ Config directory ready"
echo ""

# Create nginx directory for production
mkdir -p nginx/ssl
echo "✓ Nginx directories ready"
echo ""

# Parse arguments
MODE="dev"
BUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --prod|--production)
            MODE="prod"
            shift
            ;;
        --build)
            BUILD=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --prod, --production    Use production configuration (with Nginx)"
            echo "  --build                 Force rebuild of Docker images"
            echo "  --help, -h              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                      # Start in development mode"
            echo "  $0 --prod               # Start in production mode"
            echo "  $0 --build              # Rebuild and start"
            echo "  $0 --prod --build       # Rebuild and start in production"
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Select compose file
if [ "$MODE" = "prod" ]; then
    COMPOSE_FILE="docker-compose.prod.yml"
    echo "🚀 Starting in PRODUCTION mode with Nginx reverse proxy"
else
    COMPOSE_FILE="docker-compose.yml"
    echo "🔧 Starting in DEVELOPMENT mode"
fi
echo ""

# Build if requested
if [ "$BUILD" = true ]; then
    echo "🔨 Building Docker images..."
    $DOCKER_COMPOSE -f $COMPOSE_FILE build
    echo "✓ Build complete"
    echo ""
fi

# Start services
echo "🚀 Starting services..."
$DOCKER_COMPOSE -f $COMPOSE_FILE up -d

# Wait for services to be healthy
echo ""
echo "⏳ Waiting for services to be ready..."
sleep 5

# Check service status
if $DOCKER_COMPOSE -f $COMPOSE_FILE ps | grep -q "Up"; then
    echo "✓ Services are running"
    echo ""

    # Display access information
    echo "=========================================="
    echo "✅ Bahar is now running!"
    echo "=========================================="
    echo ""

    if [ "$MODE" = "prod" ]; then
        echo "🌐 Access the application:"
        echo "   HTTP:  http://localhost"
        echo "   HTTPS: https://localhost (if SSL configured)"
    else
        echo "🌐 Access the application:"
        echo "   http://localhost:8501"
    fi

    echo ""
    echo "📊 Useful commands:"
    echo "   View logs:     $DOCKER_COMPOSE -f $COMPOSE_FILE logs -f"
    echo "   Stop services: $DOCKER_COMPOSE -f $COMPOSE_FILE stop"
    echo "   Restart:       $DOCKER_COMPOSE -f $COMPOSE_FILE restart"
    echo "   Remove:        $DOCKER_COMPOSE -f $COMPOSE_FILE down"
    echo ""
    echo "📁 Data is stored in: ./volumes/"
    echo "💾 Backup command:    tar czf backup.tar.gz volumes/"
    echo ""

    # Show disk usage
    if command -v du &> /dev/null; then
        VOLUME_SIZE=$(du -sh volumes/ 2>/dev/null | cut -f1 || echo "0")
        echo "💿 Current volume size: $VOLUME_SIZE"
        echo ""
    fi

else
    echo "❌ Error: Services failed to start"
    echo ""
    echo "Check logs with:"
    echo "  $DOCKER_COMPOSE -f $COMPOSE_FILE logs"
    exit 1
fi

