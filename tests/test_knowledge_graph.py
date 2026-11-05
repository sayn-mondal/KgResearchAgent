import os

import pytest

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USERNAME", "test")
os.environ.setdefault("NEO4J_PASSWORD", "test")

from src.tools.knowledge_graph import KnowledgeGraph


@pytest.fixture
def knowledge_graph():
    return object.__new__(KnowledgeGraph)


def test_validate_cypher_ignores_blocked_words_in_values_and_comments(
    knowledge_graph
):
    query, parameters = knowledge_graph.validate_cypher(
        """
        MATCH (e:Entity)
        WHERE e.description = 'SET // CREATE'
        // DELETE is only a comment
        RETURN e.id AS id
        LIMIT 10
        """,
        parameters={},
        limit=10
    )

    assert "RETURN e.id AS id" in query
    assert parameters == {}


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (n) SET n.value = 1 RETURN n.value LIMIT 1",
        "MATCH (n) RETURN n LIMIT 1",
        "MATCH (n) RETURN * LIMIT 1",
        "MATCH (n) RETURN properties(n) AS data LIMIT 1",
        "MATCH (n)-[*]-(m) RETURN n.id AS id LIMIT 1",
        "MATCH (n) CALL db.labels() YIELD label RETURN label LIMIT 1",
    ]
)
def test_validate_cypher_rejects_unsafe_or_unbounded_output(
    knowledge_graph,
    query
):
    with pytest.raises(ValueError):
        knowledge_graph.validate_cypher(
            query,
            parameters={},
            limit=10
        )


def test_validate_cypher_enforces_result_limit(knowledge_graph):
    with pytest.raises(ValueError, match="LIMIT"):
        knowledge_graph.validate_cypher(
            "MATCH (e:Entity) RETURN e.id AS id LIMIT 11",
            parameters={},
            limit=10
        )


def test_schema_uses_metadata_without_returning_graph_examples():
    class FakeResult:
        def __init__(self, rows):
            self.rows = rows

        def data(self):
            return self.rows

    class FakeTransaction:
        def __init__(self):
            self.queries = []

        def run(self, query):
            self.queries.append(query)
            return FakeResult([])

    class FakeSession:
        def __init__(self):
            self.transaction = FakeTransaction()

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def execute_read(self, callback):
            return callback(self.transaction)

    class FakeDriver:
        def __init__(self):
            self.fake_session = FakeSession()

        def session(self, database):
            assert database == "neo4j"
            return self.fake_session

    graph = object.__new__(KnowledgeGraph)
    graph.read_driver = FakeDriver()
    graph.database = "neo4j"
    graph._schema_cache = None

    schema = graph.get_schema()

    assert set(schema) == {"nodes", "relationships"}
    assert len(graph.read_driver.fake_session.transaction.queries) == 2
    assert all(
        "db.schema." in query
        for query in graph.read_driver.fake_session.transaction.queries
    )
    assert all(
        "properties(n)" not in query
        and "properties(r)" not in query
        for query in graph.read_driver.fake_session.transaction.queries
    )


def test_failed_search_raises_instead_of_returning_empty_success(
    knowledge_graph
):
    from src.tools.knowledge_graph import CypherGeneration

    knowledge_graph.get_schema = lambda: {}
    knowledge_graph.generate_cypher = lambda **_: (
        CypherGeneration(
            cypher="MATCH (n) SET n.x = 1 RETURN n.x",
            explanation="invalid write query"
        ),
        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    )

    with pytest.raises(RuntimeError, match="two query attempts"):
        knowledge_graph.search_question("example")
