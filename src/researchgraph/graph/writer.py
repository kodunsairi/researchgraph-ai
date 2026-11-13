"""Neo4j graph writer for persisting extracted research entities."""
from __future__ import annotations

import hashlib
from typing import Any, Optional
from neo4j import GraphDatabase, Driver

from researchgraph.config import Settings, get_settings
from researchgraph.graph.constraints import setup_constraints
from researchgraph.graph.schema import Experiment, Paper
from researchgraph.logging_setup import get_logger

log = get_logger(__name__)


class GraphWriter:
    """Handles connecting to Neo4j and writing schema v2 nodes/relationships."""

    def __init__(self, settings: Optional[Settings] = None, driver: Optional[Driver] = None):
        self.settings = settings or get_settings()
        if driver is not None:
            self._driver = driver
            self._owns_driver = False
        else:
            self._driver = GraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_user, self.settings.neo4j_password.get_secret_value()),
            )
            self._owns_driver = True
        self.database = self.settings.neo4j_database

    @property
    def driver(self) -> Driver:
        return self._driver

    def setup_schema(self) -> None:
        """Apply uniqueness constraints."""
        setup_constraints(self._driver, self.database)

    def upsert_paper(self, paper: Paper) -> None:
        """Upsert a Paper node."""
        query = """
        MERGE (p:Paper {arxiv_id: $arxiv_id})
        ON CREATE SET
            p.title = $title,
            p.authors = $authors,
            p.abstract = $abstract,
            p.published = $published
        ON MATCH SET
            p.title = $title,
            p.authors = $authors,
            p.abstract = $abstract,
            p.published = $published
        """
        with self._driver.session(database=self.database) as session:
            session.run(
                query,
                arxiv_id=paper.arxiv_id,
                title=paper.title,
                authors=paper.authors,
                abstract=paper.abstract,
                published=paper.published,
            )
        log.debug("Upserted Paper: %s", paper.arxiv_id)

    def upsert_experiment(self, exp: Experiment) -> str:
        """Upsert an Experiment and all linked entities in one transaction."""
        exp_id = exp.get_experiment_id()

        def _tx_write(tx: Any) -> None:
            # 1. Ensure Paper, Algorithm, Dataset, and Experiment exist and are linked
            core_query = """
            MERGE (p:Paper {arxiv_id: $arxiv_id})
            MERGE (a:Algorithm {name: $algo_name})
            ON CREATE SET a.family = $algo_family
            MERGE (d:Dataset {name: $dataset_name})
            ON CREATE SET d.domain = $dataset_domain, d.is_tabular = $is_tabular
            MERGE (e:Experiment {id: $exp_id})
            ON CREATE SET e.paper_arxiv_id = $arxiv_id
            MERGE (p)-[:RUNS]->(e)
            MERGE (e)-[:USES_ALGORITHM]->(a)
            MERGE (e)-[:ON_DATASET]->(d)
            """
            tx.run(
                core_query,
                arxiv_id=exp.paper_arxiv_id,
                algo_name=exp.algorithm.name,
                algo_family=exp.algorithm.family,
                dataset_name=exp.dataset.name,
                dataset_domain=exp.dataset.domain,
                is_tabular=exp.dataset.is_tabular,
                exp_id=exp_id,
            )

            # 2. Link Explainability Method if provided
            if exp.explainability_method and exp.explainability_method.name:
                xai_query = """
                MATCH (e:Experiment {id: $exp_id})
                MERGE (x:ExplainabilityMethod {name: $xai_name})
                MERGE (e)-[:EXPLAINED_WITH]->(x)
                """
                tx.run(xai_query, exp_id=exp_id, xai_name=exp.explainability_method.name)

            # 3. Create Result and Metric nodes
            metric_query = """
            MATCH (e:Experiment {id: $exp_id})
            MERGE (r:Result {id: $result_id})
            ON CREATE SET r.value = $value, r.unit = $unit, r.metric_name = $metric_name
            MERGE (m:Metric {name: $metric_name})
            MERGE (e)-[:PRODUCES]->(r)
            MERGE (r)-[:MEASURED_BY]->(m)
            """
            for m in exp.metrics:
                res_key = f"{exp_id}::{m.name}"
                res_id = hashlib.sha256(res_key.encode("utf-8")).hexdigest()[:16]
                tx.run(
                    metric_query,
                    exp_id=exp_id,
                    result_id=res_id,
                    value=m.value,
                    unit=m.unit,
                    metric_name=m.name,
                )

        with self._driver.session(database=self.database) as session:
            session.execute_write(_tx_write)

        log.debug("Upserted Experiment %s for Paper %s", exp_id, exp.paper_arxiv_id)
        return exp_id

    def close(self) -> None:
        if self._owns_driver:
            self._driver.close()

    def __enter__(self) -> GraphWriter:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
