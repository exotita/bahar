#!/bin/bash
# Bahar Docker Stop Script

set -e

echo "🌸 Bahar - Docker Stop Script"
echo "=============================="
echo ""

# Determine Docker Compose command
if docker compose version &> /dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

# Parse arguments
REMOVE_VOLUMES=false
MODE="both"

while [[ $# -gt 0 ]]; do
    case $1 in
        --remove-volumes|-v)
            REMOVE_VOLUMES=true
            shift
            ;;
        --dev)
            MODE="dev"
            shift
            ;;
        --prod)
            MODE="prod"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dev                   Stop only development services"
            echo "  --prod                  Stop only production services"
            echo "  --remove-volumes, -v    Remove volumes (WARNING: deletes cached data)"
            echo "  --help, -h              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                      # Stop all services"
            echo "  $0 --dev                # Stop development services only"
            echo "  $0 --prod               # Stop production services only"
            echo "  $0 --remove-volumes     # Stop and remove all data"
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Function to stop services
stop_services() {
    local compose_file=$1
    local mode_name=$2

    if [ -f "$compose_file" ]; then
        echo "🛑 Stopping $mode_name services..."

        if [ "$REMOVE_VOLUMES" = true ]; then
            echo "⚠️  WARNING: This will remove all cached data!"
            echo "   Press Ctrl+C to cancel, or wait 5 seconds to continue..."
            sleep 5
            $DOCKER_COMPOSE -f $compose_file down -v
            echo "✓ $mode_name services stopped and volumes removed"
        else
            $DOCKER_COMPOSE -f $compose_file down
            echo "✓ $mode_name services stopped"
        fi
    fi
}

# Stop services based on mode
if [ "$MODE" = "both" ] || [ "$MODE" = "dev" ]; then
    stop_services "docker-compose.yml" "development"
    echo ""
fi

if [ "$MODE" = "both" ] || [ "$MODE" = "prod" ]; then
    stop_services "docker-compose.prod.yml" "production"
    echo ""
fi

echo "=========================================="
echo "✅ Services stopped successfully"
echo "=========================================="
echo ""

if [ "$REMOVE_VOLUMES" = false ]; then
    echo "💾 Data preserved in ./volumes/"
    echo "   To remove cached data, run: rm -rf volumes/*"
else
    echo "🗑️  Cached data removed"
    echo "   Data will be re-downloaded on next start"
fi
echo ""

