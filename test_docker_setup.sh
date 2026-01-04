#!/bin/bash
echo "Testing Docker setup for NYC Taxi MLOps project..."
echo ""

# Check if Docker is installed
if command -v docker &> /dev/null; then
    echo "✓ Docker is installed"
else
    echo "✗ Docker is not installed"
    echo "  Install Docker from: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
    echo "✓ Docker Compose is installed"
else
    echo "✗ Docker Compose is not installed"
    exit 1
fi

# Check Dockerfile exists
if [ -f "Dockerfile" ]; then
    echo "✓ Dockerfile exists"
else
    echo "✗ Dockerfile not found"
    exit 1
fi

# Check docker-compose.yml exists
if [ -f "docker-compose.yml" ]; then
    echo "✓ docker-compose.yml exists"
else
    echo "✗ docker-compose.yml not found"
    exit 1
fi

echo ""
echo "✅ Docker setup looks good!"
echo ""
echo "To build and run:"
echo "  docker-compose build"
echo "  docker-compose up"
echo ""
echo "To run training in Docker:"
echo "  docker-compose run app train"
