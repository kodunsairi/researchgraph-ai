# ResearchGraph AI

GraphRAG pipeline for tabular-data papers applying SHAP/LIME to Random Forest, XGBoost, and Logistic Regression.

## Quick Start

```powershell
# 1. Create and activate venv
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt
pip install -e .

# 3. Configure environment
Copy-Item .env.example .env
# Edit .env — set NEO4J_PASSWORD (8+ chars) and OLLAMA_MODEL

# 4. Start infrastructure
docker compose up -d

# 5. Smoke test
python scripts/check_services.py

# 6. Run tests
pytest -q
```

## Project Layout

```
researchgraph-ai/
├── data/
│   ├── raw/          # PDFs (git-ignored; re-download from corpus list)
│   ├── processed/    # Extracted text chunks (git-ignored)
│   └── metadata/     # Paper metadata JSON
├── src/researchgraph/
│   ├── config.py         # Typed settings via pydantic-settings
│   ├── logging_setup.py  # Shared logging configuration
│   ├── ingestion/        # PDF download & text extraction (Milestone 1.1)
│   ├── extraction/       # LLM-based entity/relation extraction (Milestone 2.x)
│   ├── graph/            # Neo4j schema & write helpers (Milestone 3.x)
│   ├── vectorstore/      # Qdrant ingestion & search (Milestone 4.x)
│   ├── retrieval/        # GraphRAG hybrid retrieval (Milestone 5.x)
│   ├── generation/       # Answer generation + citations (Milestone 6.x)
│   ├── evaluation/       # Retrieval & faithfulness evals (Milestone 7.x)
│   ├── api/              # FastAPI endpoints (Milestone 8.x)
│   └── llm/              # LLM provider abstraction (Milestone 2.4)
├── scripts/
│   └── check_services.py # Smoke test: Neo4j, Qdrant, Ollama
├── tests/
│   └── test_config.py
├── frontend/             # (Phase 8) React/Svelte UI
├── docker-compose.yml    # Neo4j + Qdrant
├── pyproject.toml
└── requirements.txt
```

## Schema (v2)

```
(Paper)-[:RUNS]->(Experiment)
(Experiment)-[:USES_ALGORITHM]->(Algorithm)
(Experiment)-[:ON_DATASET]->(Dataset)
(Experiment)-[:EXPLAINED_WITH]->(ExplainabilityMethod)   # optional
(Experiment)-[:PRODUCES]->(Result)-[:MEASURED_BY]->(Metric)
```

Pydantic models formalized in Milestone 2.1.

## Infrastructure

| Service | URL | Purpose |
|---------|-----|---------|
| Neo4j | bolt://localhost:7687, http://localhost:7474 | Knowledge graph |
| Qdrant | http://localhost:6333 | Vector store |
| Ollama | http://localhost:11434 | Local LLM (host, GPU) |
