"""Smoke test: are Neo4j, Qdrant and the LLM backend reachable?

Run:  python scripts/check_services.py
"""
from __future__ import annotations

import sys

import httpx
from neo4j import GraphDatabase
from qdrant_client import QdrantClient

from researchgraph.config import Settings, get_settings
from researchgraph.logging_setup import get_logger, setup_logging

logger = get_logger("check_services")


def check_neo4j(s: Settings) -> bool:
    auth = (s.neo4j_user, s.neo4j_password.get_secret_value())
    try:
        with GraphDatabase.driver(s.neo4j_uri, auth=auth) as driver:
            driver.verify_connectivity()
            with driver.session(database=s.neo4j_database) as session:
                session.run("RETURN 1").consume()
                try:
                    version = session.run("RETURN apoc.version() AS v").single()["v"]
                    logger.info("Neo4j OK (APOC %s)", version)
                except Exception:
                    logger.warning("Neo4j OK, but APOC is not available yet")
        return True
    except Exception as exc:
        logger.error("Neo4j FAILED: %s", exc)
        return False


def check_qdrant(s: Settings) -> bool:
    try:
        client = QdrantClient(url=s.qdrant_url, timeout=5)
        n = len(client.get_collections().collections)
        logger.info("Qdrant OK (%d collections)", n)
        return True
    except Exception as exc:
        logger.error("Qdrant FAILED: %s", exc)
        return False


def check_llm(s: Settings) -> bool:
    if s.llm_provider != "ollama":
        ok = s.openai_api_key is not None
        (logger.info if ok else logger.error)("OpenAI key %s", "present" if ok else "MISSING")
        return ok
    try:
        resp = httpx.get(f"{s.ollama_base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        installed = {m["name"] for m in resp.json().get("models", [])}
        if s.ollama_model not in installed:
            logger.error(
                "Ollama reachable, but model '%s' is not pulled. Installed: %s",
                s.ollama_model, sorted(installed) or "none",
            )
            return False
        logger.info("Ollama OK (model '%s' available)", s.ollama_model)
        return True
    except Exception as exc:
        logger.error("Ollama FAILED: %s", exc)
        return False


def main() -> int:
    s = get_settings()
    setup_logging(s.log_level)
    results = {
        "neo4j": check_neo4j(s),
        "qdrant": check_qdrant(s),
        "llm": check_llm(s),
    }
    failed = [name for name, ok in results.items() if not ok]
    if failed:
        logger.error("Failed checks: %s", ", ".join(failed))
        return 1
    logger.info("All services healthy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
