# AI Data Analyst

Production-style full-stack web application for uploading any tabular dataset (CSV/Excel), profiling it, cleaning it, running EDA, training ML models, chatting with an LLM using summarized context, and viewing an auto-generated dashboard.

## Features

- **Dataset upload** — CSV/Excel with preview, type detection, and summary stats
- **Automatic profiling** — column types, missing values, duplicates, ML task hints
- **Cleaning pipeline** — imputation, deduplication, datetime parsing, categorical encoding
- **EDA** — Plotly histograms, correlation heatmap, boxplots, category bars, time series
- **AutoML lite** — linear/logistic regression, random forest, XGBoost with metrics
- **Explainability** — feature importance + optional SHAP summaries
- **LLM chat** — natural language Q&A using structured summaries (no raw rows sent)
- **Dashboard** — KPIs, charts, report sections, downloadable JSON report

## Tech Stack

| Layer | Stack |
|-------|-------|
| Backend | Python, FastAPI, Pandas, NumPy, Scikit-learn, XGBoost, Plotly, SHAP, Google Gemini API |
| Frontend | Next.js (App Router), TypeScript, Tailwind CSS, shadcn-style UI, Plotly.js |
| DevOps | Docker, docker-compose, `.env` configuration |

## Project Structure

```
ai-data-analyst/
├── backend/
│   ├── app/
│   │   ├── api/routes/       # FastAPI endpoints
│   │   ├── services/         # Business logic
│   │   ├── ml/               # Explainability
│   │   ├── llm/              # LLM client + context builder
│   │   └── utils/            # Loaders, type detection, session store
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/app/              # Pages: upload, overview, eda, ml, chat, dashboard
│   ├── src/components/
│   └── Dockerfile
├── sample-data/
├── docker-compose.yml
├── .env.example
└── README.md
```

## Quick Start (Docker)

1. **Clone / enter the project**

   ```bash
   cd ai-data-analyst
   ```

2. **Configure environment**

   ```bash
   cp .env.example .env
   ```

   Set `GEMINI_API_KEYS` (comma-separated) for LLM chat (optional — app works without it using fallback summaries).

3. **Run**

   ```bash
   docker-compose up --build
   ```

4. **Open the app**

   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API docs: http://localhost:8000/docs

5. **Try sample data**

   Upload `sample-data/sales_sample.csv` and walk through Upload → Overview → EDA → ML → Chat → Dashboard.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/upload` | Upload CSV/Excel, profile dataset |
| GET | `/profile?session_id=` | Get/refine profile (optional `target_column`) |
| POST | `/clean` | Run cleaning pipeline |
| GET | `/eda?session_id=` | Generate EDA charts |
| POST | `/train` | Train ML models on target column |
| POST | `/ask` | Ask LLM question (summarized context only) |
| GET | `/dashboard?session_id=` | Auto-generated dashboard |
| GET | `/health` | Health check |

## Local Development (without Docker)

### Backend

Requires **Python 3.11 or 3.12** (scikit-learn/XGBoost do not yet support 3.14).

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
npm run dev
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEYS` | Comma-separated Google AI API keys | empty |
| `GEMINI_BASE_URL` | Gemini API base URL | `https://generativelanguage.googleapis.com/v1beta` |
| `GEMINI_MODELS` | Comma-separated models, priority order | see `.env.example` |
| `GEMINI_MAX_RETRIES` | Retries per model/key before fallback | `2` |
| `CORS_ORIGINS` | Allowed frontend origins | `http://localhost:3000` |
| `NEXT_PUBLIC_API_URL` | Frontend → backend URL | `http://localhost:8000` |
| `MAX_UPLOAD_SIZE_MB` | Upload limit | `50` |

## Architecture Notes

- **Dataset-agnostic** — no fixed schema; types inferred per column
- **Layered backend** — routes → services → ml/llm/utils
- **Session-based** — in-memory session store (swap for Redis/DB in production)
- **LLM safety** — context builder sends aggregates/stats only, never raw rows
- **Modular services** — profiling, cleaning, EDA, ML, dashboard run independently

## Suggested Workflow

1. Upload dataset on **Upload** page
2. Review types and run cleaning on **Overview**
3. Explore charts on **EDA**
4. Select target column and train on **ML**
5. Ask questions on **Chat**
6. Review KPIs and download report on **Dashboard**

## License

MIT — portfolio / educational use.
