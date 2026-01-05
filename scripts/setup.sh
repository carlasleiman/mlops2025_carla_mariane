#!/bin/bash

echo "Setting up CI/CD Pipeline..."

# Install pre-commit
pip install pre-commit
pre-commit install

# Run tests
pytest tests/

echo "Setup complete!"
