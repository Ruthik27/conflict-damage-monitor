# Deploy

## Local API dev
uvicorn src.api.main:app --reload --port 8000

## Docker
docker build -t conflict-damage-monitor .
docker run -p 8000:8000 conflict-damage-monitor

## Frontend dev
cd frontend && npm install && npm start
