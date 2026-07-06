# Gen-AI Reading App

A sophisticated multi-service application for semantic document search and AI-powered analysis. The architecture uses a Node.js gateway to route requests between a React frontend and a Python FastAPI backend, all secured with Supabase authentication.

## Architecture

```
┌─────────────┐
│   Browser   │
│   (React)   │
└──────┬──────┘
       │ HTTP
       ▼
┌─────────────────────────────┐
│ Gateway (Node.js/Express)   │
│ - JWT Authentication        │
│ - Request Routing           │
│ - Error Handling            │
└──────┬──────────────────────┘
       │ HTTP
       ▼
┌──────────────────────────────┐
│ Backend (Python/FastAPI)     │
│ - Semantic Search            │
│ - Multi-Agent Reasoning      │
│ - Document Processing        │
└──────┬──────────────────────┘
       │ API
       ▼
┌──────────────────────────────┐
│ Supabase                     │
│ - PostgreSQL Database        │
│ - Authentication             │
│ - Vector Storage             │
└──────────────────────────────┘
```

## Project Structure

```
gen-ai-reading-app/
├── frontend/                 # React + Vite application
│   ├── src/
│   ├── package.json
│   ├── Dockerfile
│   └── vite.config.js
├── gateway/                  # Node.js Express API Gateway
│   ├── src/
│   │   ├── index.js        # Main server
│   │   ├── middleware/     # Auth, logging, errors
│   │   ├── routes/         # API endpoints
│   │   └── services/       # Backend client
│   ├── package.json
│   └── Dockerfile
├── backend/                  # Python FastAPI backend
│   ├── app/
│   │   ├── main.py         # FastAPI app
│   │   ├── agents.py       # Multi-agent system
│   │   ├── embeddings.py   # Vector search
│   │   └── models.py       # Pydantic models
│   ├── requirements.txt
│   └── Dockerfile
├── database/
│   └── migrations/         # SQL schema files
├── docker-compose.yml      # Multi-container orchestration
└── .env.example           # Environment variables template
```

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Supabase account with a project (free tier works)
- OpenRouter API key (for LLM access)

### 1. Clone and Setup

```bash
git clone <repository>
cd gen-ai-reading-app

# Copy environment template
cp .env.example .env

# Copy individual service templates
cp backend/.env.example backend/.env
cp gateway/.env.example gateway/.env
cp frontend/.env.example frontend/.env
```

### 2. Configure Environment Variables

Edit `.env` with your credentials:

```bash
# Supabase (get from https://supabase.com)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SECRET_KEY=your-secret-key
VITE_SUPABASE_ANON_KEY=your-publishable-key

# OpenRouter (get from https://openrouter.ai)
OPENROUTER_API_KEY=your-api-key

# Frontend
VITE_GATEWAY_URL=http://localhost:3001

# Gateway
NODE_ENV=development
BACKEND_URL=http://backend:8000
```

### 3. Setup Supabase Database

Run the SQL schema in your Supabase SQL editor:

```bash
cat database/migrations/001_initial.sql
```

This creates:
- `documents` table - stores uploaded PDFs
- `embeddings` table - stores vector embeddings for semantic search
- `security_logs` table - tracks security events

### 4. Run with Docker Compose

```bash
# Build and start all services
docker-compose up --build

# Access the application
# Frontend:  http://localhost:3000
# Gateway:   http://localhost:3001
# Backend:   http://localhost:8000 (API docs at /docs)
```

### 5. Use the Application

1. Open http://localhost:3000 in your browser
2. Click "Upload PDF" to add a document
3. Select a document from the history list
4. Ask questions about the document
5. The multi-agent system will:
   - Verify the query is safe
   - Perform semantic search to find relevant content
   - Reason about the context using an LLM
   - Validate and format the response

## Development

Running services individually (useful during development):

### Backend
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

### Gateway
```bash
cd gateway
npm install
npm run dev
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## API Endpoints

### Gateway (Public API)

- `GET /health` - Health check
- `POST /documents/upload` - Upload a PDF
- `GET /documents` - List all documents
- `POST /query/process` - Query a document

### Backend (Internal)

- `GET /health` - Health check
- `POST /upload` - Direct file upload (called by gateway)
- `POST /process` - Process query (called by gateway)
- `GET /documents` - List documents (called by gateway)

## Features

### Multi-Agent System

1. **SecurityAgent** - Validates input for injection attacks
2. **LibrarianAgent** - Semantic search using embeddings (cosine similarity)
3. **AnalystAgent** - Reasoning with context using LLM
4. **EditorAgent** - Formats response and prevents hallucinations

### Semantic Search

- PDFs are chunked into ~1000 character segments
- Each chunk is embedded using OpenAI's text-embedding-3-small
- User queries are embedded and compared using cosine similarity
- Top 5 most relevant chunks are passed to the analyst

### Security

- Supabase JWT authentication (optional auth in development)
- Input validation for jailbreak attempts
- Security event logging

## Troubleshooting

### Backend can't connect to Supabase

```bash
# Check your environment variables
docker-compose exec backend env | grep SUPABASE
```

### Gateway can't reach backend

```bash
# Ensure backend is healthy
docker-compose exec gateway curl http://backend:8000/health
```

### Frontend can't reach gateway

```bash
# Check gateway health
curl http://localhost:3001/health

# Check console in browser for CORS errors
```

### PDF upload fails

- Ensure file is a valid PDF
- Check file size (backend timeout after 5 minutes)
- Review backend logs: `docker-compose logs backend`

## Deployment to Railway

### Prerequisites

- Railway account
- GitHub repository with this code

### Deploy Steps

1. Connect your GitHub repo to Railway
2. Create a new Railway project
3. Add services:
   - Frontend (Node.js)
   - Gateway (Node.js)
   - Backend (Python)
4. Set environment variables in each service
5. Add plugins: PostgreSQL (optional, use Supabase instead)
6. Configure domains for frontend and gateway

### Configuration for Railway

Each service's `Dockerfile` is already optimized for Railway:
- Multi-stage builds for smaller images
- Health checks included
- Environment variable support
- Port configuration via ENV or code

## Contributing

1. Create a feature branch
2. Make changes
3. Test with `docker-compose up`
4. Submit a pull request

## License

MIT

## Support

For issues and questions:
1. Check the README files in each service folder
2. Review Docker logs: `docker-compose logs -f [service-name]`
3. Check Supabase dashboard for database issues
