# Wave Theory Chatbot - Deployment Guide

This guide covers how to deploy the Wave Theory Chatbot using Docker and other deployment methods.

## Quick Start

### Using Docker Compose (Recommended)

1. **Clone and navigate to the project:**
   ```bash
   cd wavetheory
   ```

2. **Set up environment variables:**
   ```bash
   cp env.example .env
   # Edit .env with your actual API keys
   ```

3. **Deploy with Docker Compose:**
   ```bash
   make docker-run
   # or
   docker-compose up -d
   ```

4. **Access the application:**
   - Main app: http://localhost:8501
   - Jupyter (dev): http://localhost:8888

### Using Make Commands

```bash
# Install dependencies
make install

# Run in development mode
make dev

# Build and deploy with Docker
make deploy

# Stop containers
make docker-stop

# View logs
make docker-logs
```

## Manual Docker Deployment

### Build the Image

```bash
docker build -t wave-theory-chatbot .
```

### Run the Container

```bash
docker run -d \
  --name wave-theory-chatbot \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/logs:/app/logs \
  wave-theory-chatbot
```

## Environment Configuration

### Required Environment Variables

Create a `.env` file based on `env.example`:

```bash
# API Keys
HUGGINGFACE_TOKEN=your_actual_token
OPENAI_API_KEY=your_actual_key

# Paths
MODEL_PATH=/app/models
CHECKPOINT_PATH=/app/checkpoints
DATA_PATH=/app/data

# JAX Configuration
JAX_PLATFORM_NAME=cpu
JAX_ENABLE_X64=true

# Streamlit
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

## Production Deployment

### Using Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml wave-theory
```

### Using Kubernetes

1. **Create namespace:**
   ```bash
   kubectl create namespace wave-theory
   ```

2. **Deploy application:**
   ```bash
   kubectl apply -f k8s/ -n wave-theory
   ```

### Using Cloud Platforms

#### AWS ECS

1. Build and push to ECR
2. Create ECS task definition
3. Deploy service with load balancer

#### Google Cloud Run

```bash
# Build and deploy
gcloud run deploy wave-theory-chatbot \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### Azure Container Instances

```bash
# Build and push to ACR
az acr build --registry myregistry --image wave-theory-chatbot .

# Deploy container instance
az container create \
  --resource-group myResourceGroup \
  --name wave-theory-chatbot \
  --image myregistry.azurecr.io/wave-theory-chatbot:latest \
  --ports 8501
```

## Monitoring and Logging

### Health Checks

The application includes health checks:
- Docker: `HEALTHCHECK` in Dockerfile
- Streamlit: `/_stcore/health` endpoint

### Logging

Logs are written to:
- Container: `/app/logs/wave_theory.log`
- Host: `./logs/wave_theory.log`

### Monitoring Commands

```bash
# View container logs
docker logs wave-theory-chatbot

# View real-time logs
docker logs -f wave-theory-chatbot

# Check container status
docker ps

# View resource usage
docker stats wave-theory-chatbot
```

## Troubleshooting

### Common Issues

1. **Port already in use:**
   ```bash
   # Change port in docker-compose.yml
   ports:
     - "8502:8501"  # Use different host port
   ```

2. **Permission issues:**
   ```bash
   # Fix permissions
   sudo chown -R $USER:$USER data models checkpoints logs
   ```

3. **Memory issues:**
   ```bash
   # Increase memory limits in docker-compose.yml
   deploy:
     resources:
       limits:
         memory: 16G
   ```

4. **JAX platform issues:**
   ```bash
   # Set JAX platform
   export JAX_PLATFORM_NAME=cpu
   ```

### Debug Mode

Run in debug mode:

```bash
# Development
STREAMLIT_SERVER_HEADLESS=false make dev

# Docker
docker run -it --rm \
  -e DEBUG=true \
  -e STREAMLIT_SERVER_HEADLESS=false \
  -p 8501:8501 \
  wave-theory-chatbot
```

## Scaling

### Horizontal Scaling

Use a load balancer (nginx, traefik) with multiple container instances:

```yaml
# docker-compose.yml
services:
  wave-theory-app:
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
```

### Vertical Scaling

Adjust resource limits in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '8'
      memory: 16G
    reservations:
      cpus: '4'
      memory: 8G
```

## Security Considerations

1. **API Keys:** Store in environment variables, not in code
2. **Network:** Use internal networks for container communication
3. **Volumes:** Mount only necessary directories
4. **Updates:** Regularly update base images and dependencies
5. **Secrets:** Use Docker secrets or external secret management

## Backup and Recovery

### Data Backup

```bash
# Backup data directory
tar -czf wave-theory-backup-$(date +%Y%m%d).tar.gz data/ models/ checkpoints/

# Restore from backup
tar -xzf wave-theory-backup-20231201.tar.gz
```

### Model Checkpoints

Models and checkpoints are automatically saved to mounted volumes. Ensure regular backups of these directories.

## Support

For deployment issues:
1. Check logs: `make docker-logs`
2. Verify environment variables
3. Check resource usage: `docker stats`
4. Review configuration files

For more help, see the main README.md or create an issue in the repository.
