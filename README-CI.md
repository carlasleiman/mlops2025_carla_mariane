# CI/CD Setup Guide

## Local Development

\`\`\`bash
# Install dependencies
make install

# Run tests
make test

# Run linting
make lint

# Build Docker
make docker-build
\`\`\`

## GitHub Actions Pipeline

The CI/CD pipeline runs automatically on:
- Push to main, feat/*, ops/*, or ci/* branches
- Pull requests to main

### Pipeline Jobs:
1. **lint-test**: Runs flake8 linting and pytest tests
2. **build**: Builds Docker image if tests pass

## Pre-commit Hooks

Install and use pre-commit:
\`\`\`bash
pip install pre-commit
pre-commit install
\`\`\`

Hooks include:
- Trailing whitespace removal
- End-of-file fixer
- Black code formatting

## Testing

Run tests locally:
\`\`\`bash
make test
\`\`\`

Test structure:
- tests/test_example.py: Example tests
- tests/conftest.py: Test fixtures
- tests/__init__.py: Package initialization

## Makefile Commands

| Command | Description |
|---------|-------------|
| make install | Install dependencies |
| make test | Run tests |
| make lint | Run linting |
| make format | Format code with Black |
| make docker-build | Build Docker image |
| make docker-run | Run Docker container |
| make ci-local | Run local CI (lint + test + build) |
