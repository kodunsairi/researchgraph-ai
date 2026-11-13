"""Graph-based retriever querying Neo4j knowledge graph."""
from __future__ import annotations

from typing import Any, List, Optional
from neo4j import Driver, GraphDatabase

from researchgraph.config import Settings, get_settings
from researchgraph.logging_setup import get_logger

log = get_logger(__name__)


class GraphRetriever:
    """Queries Neo4j for structured paper, experiment, and metric entities."""

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

    def query_experiments(
        self,
        algorithm: Optional[str] = None,
        dataset: Optional[str] = None,
        xai_method: Optional[str] = None,
        limit: int = 15,
    ) -> List[dict[str, Any]]:
        """Retrieve structured experiments matching optional filters."""
        query = """
        MATCH (p:Paper)-[:RUNS]->(e:Experiment)-[:USES_ALGORITHM]->(a:Algorithm)
        MATCH (e)-[:ON_DATASET]->(d:Dataset)
        OPTIONAL MATCH (e)-[:EXPLAINED_WITH]->(x:ExplainabilityMethod)
        OPTIONAL MATCH (e)-[:PRODUCES]->(r:Result)-[:MEASURED_BY]->(m:Metric)
        WHERE ($algo IS NULL OR toLower(a.name) CONTAINS toLower($algo))
          AND ($dataset IS NULL OR toLower(d.name) CONTAINS toLower($dataset))
          AND ($xai IS NULL OR toLower(x.name) CONTAINS toLower($xai))
        RETURN p.arxiv_id AS arxiv_id,
               p.title AS paper_title,
               a.name AS algorithm,
               d.name AS dataset,
               x.name AS xai_method,
               collect(DISTINCT {metric: m.name, value: r.value, unit: r.unit}) AS results
        LIMIT $limit
        """
        try:
            with self._driver.session(database=self.database) as session:
                result = session.run(
                    query,
                    algo=algorithm,
                    dataset=dataset,
                    xai=xai_method,
                    limit=limit,
                )
                return [record.data() for record in result]
        except Exception as exc:
            log.warning("Graph query failed: %s", exc)
            return []

    def close(self) -> None:
        if self._owns_driver:
            self._driver.close()
