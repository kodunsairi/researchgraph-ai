"""Neo4j constraints and indices initialization."""
from __future__ import annotations

from typing import Any
from researchgraph.logging_setup import get_logger

log = get_logger(__name__)

CONSTRAINTS = [
    "CREATE CONSTRAINT paper_id_uniq IF NOT EXISTS FOR (p:Paper) REQUIRE p.arxiv_id IS UNIQUE",
    "CREATE CONSTRAINT exp_id_uniq IF NOT EXISTS FOR (e:Experiment) REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT algo_name_uniq IF NOT EXISTS FOR (a:Algorithm) REQUIRE a.name IS UNIQUE",
    "CREATE CONSTRAINT dataset_name_uniq IF NOT EXISTS FOR (d:Dataset) REQUIRE d.name IS UNIQUE",
    "CREATE CONSTRAINT xai_name_uniq IF NOT EXISTS FOR (x:ExplainabilityMethod) REQUIRE x.name IS UNIQUE",
]


def setup_constraints(driver: Any, database: str = "neo4j") -> None:
    """Create uniqueness constraints for schema v2 nodes."""
    log.info("Setting up Neo4j constraints...")
    with driver.session(database=database) as session:
        for cypher in CONSTRAINTS:
            try:
                session.run(cypher)
                log.debug("Executed: %s", cypher)
            except Exception as exc:
                log.warning("Constraint creation warning (%s): %s", cypher, exc)
    log.info("Neo4j constraints verified.")
