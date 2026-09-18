# GridWise Hackathon - Energy Optimizer

This is the LLM-assisted energy scheduling and optimization service for the GridWise challenge.

## Architecture

1. **API Layer**: Built with FastAPI. Receives `POST /optimize-energy` requests.
2. **LLM Interpreter**: Uses Groq-hosted LLMs (default `openai/gpt-oss-120b`, fallback `qwen/qwen3.8-27b`) to translate natural-language operator notes into structured directive rules. Handles multi-model fallback and rate-limit backoffs.
3. **Guardrails**: Deterministic validation blocks impossible inputs, enforces numeric rules, and maps `no_op` irrelevant notes.
4. **Mathematical Optimizer**: Uses `scipy.optimize.linprog` (HiGHS solver) to build a robust two-phase Linear Programming model. The first phase minimizes cost; the second phase minimizes battery cycling while holding optimal cost.
5. **Replay Validation**: An independent simulation runs over the final schedule to guarantee the plan absolutely conforms to energy balances, limits, and LLM constraints before the HTTP response is served. 

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd BUP

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## Environment Variables

Copy `.env.example` to `.env` and fill in your keys:

- `GROQ_API_KEY`: Your Groq API token (required).
- `GROQ_MODEL`: Primary model (default: `openai/gpt-oss-120b`).
- `GROQ_FALLBACK_MODEL`: Comma-separated list of fallback models.
- `GROQ_REASONING_EFFORT`: Reasoning effort (`low`, `medium`, `high`).
- `LLM_TIMEOUT_SECONDS`: Per-call timeout budget.
- `LLM_BUDGET_SECONDS`: Total LLM optimization deadline.
- `PORT`: Port to bind the server (default: `8000`).

## Running Locally

Start the application locally using Uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Endpoints

- **Health Test**: `GET /health`
  ```bash
  curl http://localhost:8000/health
  ```
  Returns: `{"status": "ok"}`

- **Optimization Test**: `POST /optimize-energy`
  ```bash
  curl -X POST "http://localhost:8000/optimize-energy" \
       -H "Content-Type: application/json" \
       -d @BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json
  ```
  Note: Extract a single scenario from the JSON file to test the endpoint.

## Docker Fallback

To build and run using Docker:

```bash
# Build the image
docker build -t gridwise-optimizer .

# Run the container
docker run -p 8000:8000 --env-file .env gridwise-optimizer
```

## Known Limitations

- Sub-minute precision is not supported; all directives round to the inclusive-start hour block.
- Fallback models might have varying capabilities in parsing JSON exactly as requested, though the system uses resilient JSON extraction heuristics.
