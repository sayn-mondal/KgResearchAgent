import os
import re
import json
import certifi

from typing import Any
from dotenv import load_dotenv
from neo4j import GraphDatabase, unit_of_work
from neo4j.exceptions import Neo4jError
from langchain.tools import tool
from pydantic import BaseModel, Field


load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()


# --------------------------------------------------
# STRUCTURED CYPHER OUTPUT
# --------------------------------------------------

class CypherGeneration(BaseModel):

    cypher: str = Field(
        description="A single read-only Neo4j Cypher query."
    )

    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters used by the Cypher query."
    )

    explanation: str = Field(
        description="Brief explanation of what the query retrieves."
    )


# --------------------------------------------------
# KNOWLEDGE GRAPH
# --------------------------------------------------

class KnowledgeGraph:

    MAX_RESULTS = 50
    QUERY_TIMEOUT_SECONDS = 15
    MAX_RESULT_BYTES = 256_000

    def __init__(self):

        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")

        if not all([uri, username, password]):
            raise ValueError(
                "Neo4j configuration is missing from .env"
            )

        readonly_username = os.getenv(
            "NEO4J_READONLY_USERNAME"
        )
        readonly_password = os.getenv(
            "NEO4J_READONLY_PASSWORD"
        )

        if bool(readonly_username) != bool(readonly_password):
            raise ValueError(
                "Set both NEO4J_READONLY_USERNAME and "
                "NEO4J_READONLY_PASSWORD, or neither."
            )

        self.driver = GraphDatabase.driver(
            uri,
            auth=(username, password)
        )

        self.read_driver = (
            GraphDatabase.driver(
                uri,
                auth=(readonly_username, readonly_password)
            )
            if readonly_username and readonly_password
            else self.driver
        )

        self.database = os.getenv(
            "NEO4J_DATABASE", "neo4j"
        )
        self._schema_cache = None

    # --------------------------------------------------
    # GET SHARED GEMINI MODEL
    # --------------------------------------------------

    @staticmethod
    def get_llm():

        # Lazy import avoids circular imports because
        # agent.py also imports search_knowledge_graph.
        from src.kg_research_agent.agent import get_llm

        return get_llm()

    # --------------------------------------------------
    # RETRIEVE CURRENT NEO4J SCHEMA
    # --------------------------------------------------

    def get_schema(self):

        if self._schema_cache is not None:
            return self._schema_cache

        @unit_of_work(timeout=self.QUERY_TIMEOUT_SECONDS)
        def read_schema(tx):

            node_schema = tx.run(
                """
                CALL db.schema.nodeTypeProperties()
                YIELD nodeLabels, propertyName, propertyTypes

                RETURN
                    nodeLabels AS labels,
                    propertyName AS property,
                    propertyTypes AS types

                ORDER BY labels, property
                """
            ).data()

            relationship_schema = tx.run(
                """
                CALL db.schema.relTypeProperties()
                YIELD relType, propertyName, propertyTypes

                RETURN
                    relType AS relationship_type,
                    propertyName AS property,
                    propertyTypes AS types

                ORDER BY relationship_type, property
                """
            ).data()

            return {
                "nodes": node_schema,
                "relationships": relationship_schema
            }

        with self.read_driver.session(
            database=self.database
        ) as session:

            self._schema_cache = session.execute_read(read_schema)
            return self._schema_cache

    # --------------------------------------------------
    # EXTRACT LLM TOKEN USAGE
    # --------------------------------------------------

    @staticmethod
    def extract_usage(raw_response, llm):

        usage = getattr(
            raw_response,
            "usage_metadata",
            None
        ) or {}

        if not usage:
            response_metadata = getattr(
                raw_response,
                "response_metadata",
                {}
            ) or {}

            usage = (
                response_metadata.get("usage_metadata")
                or response_metadata.get("token_usage")
                or {}
            )

        input_tokens = usage.get(
            "input_tokens",
            usage.get("prompt_token_count", 0)
        )

        output_tokens = usage.get(
            "output_tokens",
            usage.get("candidates_token_count", 0)
        )

        total_tokens = usage.get(
            "total_tokens",
            usage.get("total_token_count", 0)
        )

        return {
            "model": getattr(llm, "model", "Gemini"),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "calls": 1
        }

    @staticmethod
    def combine_usage(total, new_usage):

        for key in [
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "calls"
        ]:

            total[key] = (
                total.get(key, 0)
                + new_usage.get(key, 0)
            )

        total["model"] = new_usage.get(
            "model",
            total.get("model", "Gemini")
        )

        return total

    # --------------------------------------------------
    # GENERATE CYPHER USING GEMINI
    # --------------------------------------------------

    def generate_cypher(
        self,
        question: str,
        schema: dict,
        limit: int,
        previous_cypher: str = "",
        previous_error: str = ""
    ):

        llm = self.get_llm()

        structured_llm = llm.with_structured_output(
            CypherGeneration,
            include_raw=True
        )

        schema_text = json.dumps(
            schema,
            indent=2,
            ensure_ascii=False,
            default=str
        )

        prompt = f"""
You are an expert Neo4j Cypher query generator.

Your responsibility is to understand a user's natural-language
question and generate the correct Cypher query to retrieve
the required information from the existing Neo4j knowledge graph.

USER QUESTION:
{question}


CURRENT NEO4J SCHEMA:
{schema_text}


GRAPH QUERY REQUIREMENTS:

1. Understand the actual intent of the user's question.

2. Identify which entities, properties and relationships
   need to be retrieved to answer it.

3. Generate Cypher using ONLY the labels, relationship types
   and properties present in the supplied schema.

4. The graph contains concepts and technologies as entities.
   Their names, descriptions and aliases may help identify
   the correct entity.

5. Use case-insensitive matching when appropriate.

6. Select the appropriate relationship depth:
   - Use direct relationships when the user asks what is
     connected or related to an entity.
   - Use multiple relationship levels only when the question
     requires broader context.
   - Do not retrieve unnecessary relationships.

7. Return useful information such as entity names,
   descriptions, relationship descriptions and relevant
   connected entities.

8. Use parameterized Cypher wherever possible.
   Put parameter values inside the parameters field.

9. Return projected fields and collections rather than
   entire Neo4j nodes or paths whenever possible.

10. Return a maximum of {limit} records.

11. Include a numeric LIMIT clause in the query.

12. Generate exactly ONE read-only Cypher query.

SECURITY RESTRICTIONS:

- Never generate CREATE, MERGE, DELETE, SET, REMOVE or
  any other data-modification query.
- Never generate CALL or execute database procedures.
- Never generate LOAD CSV or administrative commands.
- Never attempt to modify the graph.
- Do not generate multiple statements.

IMPORTANT:

Do not simply search for the entire user question as an entity.

Understand what information is being requested and construct
a query that retrieves that information.

An empty result is acceptable when the knowledge graph
does not contain relevant information.


{f'''
PREVIOUS CYPHER QUERY:
{previous_cypher}

PREVIOUS ERROR:
{previous_error}

The previous query failed. Correct the query while following
all schema and security restrictions.
''' if previous_error else ''}


Return the result using the required structured output.
"""

        response = structured_llm.invoke(prompt)

        parsed = response.get("parsed")

        if parsed is None:

            raise ValueError(
                f"Gemini could not generate structured Cypher: "
                f"{response.get('parsing_error')}"
            )

        raw_response = response.get("raw")

        usage = self.extract_usage(
            raw_response,
            llm
        )

        return parsed, usage

    # --------------------------------------------------
    # CYPHER VALIDATION
    # --------------------------------------------------

    @staticmethod
    def remove_comments_and_strings(query: str):

        masked = list(query)
        index = 0

        def blank(start: int, end: int) -> None:
            for position in range(start, end):
                if masked[position] not in "\r\n":
                    masked[position] = " "

        while index < len(query):
            if query.startswith("//", index):
                end = query.find("\n", index)
                if end == -1:
                    end = len(query)
                blank(index, end)
                index = end
                continue

            if query.startswith("/*", index):
                end = query.find("*/", index + 2)
                end = len(query) if end == -1 else end + 2
                blank(index, end)
                index = end
                continue

            quote = query[index]
            if quote not in ("'", '"', "`"):
                index += 1
                continue

            start = index
            index += 1

            while index < len(query):
                if query[index] == "\\":
                    index = min(index + 2, len(query))
                    continue

                if query[index] == quote:
                    if index + 1 < len(query) and query[index + 1] == quote:
                        index += 2
                        continue
                    index += 1
                    break

                index += 1

            blank(start, index)

        return "".join(masked)

    def validate_cypher(
        self,
        query: str,
        parameters: dict,
        limit: int
    ):

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Generated Cypher query is empty.")

        if len(query) > 20_000:
            raise ValueError("Generated Cypher query is too large.")

        # Remove Markdown code fences if Gemini produces them.
        query = re.sub(
            r"^```(?:cypher)?\s*",
            "",
            query.strip(),
            flags=re.IGNORECASE
        )

        query = re.sub(
            r"\s*```$",
            "",
            query
        ).strip()

        # Remove one optional trailing semicolon.
        if query.endswith(";"):
            query = query[:-1].strip()

        safe_query = self.remove_comments_and_strings(query)

        # Reject multiple statements.
        if ";" in safe_query:
            raise ValueError(
                "Multiple Cypher statements are not allowed."
            )

        # Reject write operations and procedure execution.
        forbidden_keywords = (
            r"\b("
            r"CREATE|MERGE|DELETE|DETACH|SET|REMOVE|"
            r"DROP|ALTER|RENAME|GRANT|DENY|REVOKE|"
            r"TERMINATE|FOREACH|CALL|LOAD|USE|START|STOP|"
            r"UNION"
            r")\b"
        )

        if re.search(
            forbidden_keywords,
            safe_query,
            flags=re.IGNORECASE
        ):

            raise ValueError(
                "Generated query contains a prohibited "
                "Cypher operation."
            )

        # Reject external procedure namespaces.
        if re.search(
            r"\b(apoc|dbms|gds)\s*\.",
            safe_query,
            flags=re.IGNORECASE
        ):

            raise ValueError(
                "Database procedures are not allowed."
            )

        # Require a read query.
        if not re.match(
            r"^\s*(OPTIONAL\s+)?MATCH\b",
            safe_query,
            flags=re.IGNORECASE
        ):

            raise ValueError(
                "Generated query must start with MATCH "
                "or OPTIONAL MATCH."
            )

        if not re.search(
            r"\bRETURN\b",
            safe_query,
            flags=re.IGNORECASE
        ):

            raise ValueError(
                "Generated query must contain RETURN."
            )

        if re.search(
            r"\bRETURN\s+(?:DISTINCT\s+)?\*",
            safe_query,
            flags=re.IGNORECASE
        ):
            raise ValueError(
                "Returning all graph variables is not allowed."
            )

        if re.search(
            r"\bproperties\s*\(",
            safe_query,
            flags=re.IGNORECASE
        ):
            raise ValueError(
                "Returning complete property maps is not allowed."
            )

        if re.search(
            r"\[\s*(?:[A-Za-z_][A-Za-z0-9_]*\s*)?"
            r"(?::[A-Za-z_][A-Za-z0-9_]*\s*)?\*",
            safe_query
        ):
            raise ValueError(
                "Variable-length relationship patterns are not allowed."
            )

        return_clause = re.search(
            r"\bRETURN\b(.*?)(?:\bORDER\s+BY\b|\bSKIP\b|\bLIMIT\b|$)",
            safe_query,
            flags=re.IGNORECASE | re.DOTALL
        )

        if return_clause:
            projections = return_clause.group(1).strip()
            projections = re.sub(
                r"^DISTINCT\s+",
                "",
                projections,
                flags=re.IGNORECASE
            )

            if any(
                re.fullmatch(
                    r"[A-Za-z_][A-Za-z0-9_]*"
                    r"(?:\s+AS\s+[A-Za-z_][A-Za-z0-9_]*)?",
                    projection.strip(),
                    flags=re.IGNORECASE
                )
                for projection in projections.split(",")
            ):
                raise ValueError(
                    "Return explicit entity properties instead of "
                    "whole graph objects."
                )

        # Require an integer LIMIT.
        limit_match = re.search(
            r"\bLIMIT\s+(\d+)\b",
            safe_query,
            flags=re.IGNORECASE
        )

        if limit_match:

            generated_limit = int(
                limit_match.group(1)
            )

            if generated_limit < 1 or generated_limit > limit:

                raise ValueError(
                    f"Query LIMIT must be between 1 and {limit}."
                )

        elif re.search(
            r"\bLIMIT\b",
            safe_query,
            flags=re.IGNORECASE
        ):

            raise ValueError(
                "LIMIT must use a numeric integer."
            )

        else:

            query += f"\nLIMIT {limit}"

        # Validate parameter names and values.
        if not isinstance(parameters, dict):

            raise ValueError(
                "Cypher parameters must be a dictionary."
            )

        for key, value in parameters.items():

            if not isinstance(key, str) or not re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]*",
                key
            ):

                raise ValueError(
                    f"Invalid Cypher parameter name: {key}"
                )

            valid_scalar = (
                value is None
                or isinstance(value, (str, int, float, bool))
            )

            valid_list = (
                isinstance(value, list)
                and all(
                    item is None
                    or isinstance(item, (str, int, float, bool))
                    for item in value
                )
            )

            if not valid_scalar and not valid_list:

                raise ValueError(
                    f"Unsupported value for parameter: {key}"
                )

        return query, parameters

    # --------------------------------------------------
    # SERIALIZE NEO4J RESULTS
    # --------------------------------------------------

    @staticmethod
    def serialize_value(value):

        if value is None:
            return None

        if isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, dict):

            return {
                key: KnowledgeGraph.serialize_value(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple, set)):

            return [
                KnowledgeGraph.serialize_value(item)
                for item in value
            ]

        # Neo4j nodes.
        if hasattr(value, "labels") and hasattr(value, "element_id"):

            return {
                "labels": list(value.labels),
                "properties": dict(value)
            }

        # Neo4j paths.
        if hasattr(value, "nodes") and hasattr(value, "relationships"):

            return {
                "nodes": [
                    KnowledgeGraph.serialize_value(node)
                    for node in value.nodes
                ],
                "relationships": [
                    KnowledgeGraph.serialize_value(rel)
                    for rel in value.relationships
                ]
            }

        # Neo4j relationships.
        if hasattr(value, "start_node") and hasattr(value, "end_node"):

            return {
                "type": value.type,
                "properties": dict(value)
            }

        if hasattr(value, "iso_format"):

            return value.iso_format()

        if hasattr(value, "isoformat"):

            return value.isoformat()

        return str(value)

    # --------------------------------------------------
    # EXECUTE READ-ONLY CYPHER
    # --------------------------------------------------

    def execute_cypher(
        self,
        query: str,
        parameters: dict,
        limit: int = MAX_RESULTS
    ):

        @unit_of_work(timeout=self.QUERY_TIMEOUT_SECONDS)
        def read_query(tx):

            result = tx.run(
                query,
                parameters=parameters
            )

            records = []
            total_bytes = 0

            for record in result:
                serialized = self.serialize_value(record.data())
                record_bytes = len(
                    json.dumps(
                        serialized,
                        ensure_ascii=False,
                        default=str
                    ).encode("utf-8")
                )

                if len(records) >= limit:
                    raise ValueError(
                        f"Query returned more than {limit} records."
                    )

                total_bytes += record_bytes
                if total_bytes > self.MAX_RESULT_BYTES:
                    raise ValueError(
                        "Query result exceeds the maximum response size."
                    )

                records.append(serialized)

            return records

        with self.read_driver.session(
            database=self.database
        ) as session:

            return session.execute_read(read_query)

    # --------------------------------------------------
    # NATURAL LANGUAGE TO CYPHER SEARCH
    # --------------------------------------------------

    def search_question(
        self,
        question: str,
        limit: int = 50
    ):

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        limit = max(
            1,
            min(int(limit), self.MAX_RESULTS)
        )

        schema = self.get_schema()

        usage_total = {
            "model": "Gemini",
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "calls": 0
        }

        previous_cypher = ""
        previous_error = ""
        last_error = None

        # Initial generation + one correction attempt.
        for attempt in range(2):

            generation, usage = self.generate_cypher(
                question=question,
                schema=schema,
                limit=limit,
                previous_cypher=previous_cypher,
                previous_error=previous_error
            )

            self.combine_usage(
                usage_total,
                usage
            )

            previous_cypher = generation.cypher

            try:

                cypher, parameters = self.validate_cypher(
                    query=generation.cypher,
                    parameters=generation.parameters,
                    limit=limit
                )

                print("\n[KnowledgeGraph] Generated Cypher:")
                print(cypher)

                matches = self.execute_cypher(
                    query=cypher,
                    parameters=parameters,
                    limit=limit
                )

                print(
                    f"[KnowledgeGraph] Retrieved "
                    f"{len(matches)} records."
                )

                return {
                    "question": question,
                    "cypher": cypher,
                    "explanation": generation.explanation,
                    "matches": matches,
                    "match_count": len(matches),
                    "usage": usage_total,
                    "error": None
                }

            except ValueError as exc:
                last_error = exc
                previous_error = str(exc)

                print(
                    f"\n[KnowledgeGraph] Query attempt "
                    f"{attempt + 1} failed: {previous_error}"
                )

            except Neo4jError as exc:
                if not (exc.code or "").startswith(
                    "Neo.ClientError.Statement."
                ):
                    raise

                last_error = exc
                previous_error = str(exc)

                print(
                    f"\n[KnowledgeGraph] Query attempt "
                    f"{attempt + 1} failed: {previous_error}"
                )

        raise RuntimeError(
            "Unable to retrieve graph results after "
            f"two query attempts: {previous_error}"
        ) from last_error

    # --------------------------------------------------
    # BACKWARD-COMPATIBLE METHODS
    # --------------------------------------------------

    def search_many(
        self,
        search_terms: list[str],
        limit: int = 15
    ):

        question = " ".join(
            str(term).strip()
            for term in search_terms
            if term and str(term).strip()
        )

        result = self.search_question(
            question=question,
            limit=limit
        )

        result["search_terms"] = search_terms

        return result

    def search(self, query: str):

        result = self.search_question(query)

        return result["matches"]

    # --------------------------------------------------
    # INSERT OR UPDATE KNOWLEDGE
    # --------------------------------------------------

    def upsert(
        self,
        entities: list,
        relationships: list
    ):

        with self.driver.session(
            database=self.database
        ) as session:

            # Insert or update entities.
            for entity in entities:

                session.run(
                    """
                    MERGE (e:Entity {id: $name})

                    SET e.description =
                        CASE

                            WHEN $description IS NOT NULL
                            AND trim($description) <> ''

                            THEN $description

                            ELSE e.description

                        END

                    SET e.aliases =
                        CASE

                            WHEN size($aliases) > 0

                            THEN $aliases

                            ELSE e.aliases

                        END
                    """,

                    name=entity["name"],

                    description=entity.get(
                        "description", ""
                    ),

                    aliases=entity.get(
                        "aliases", []
                    )
                )

            # Insert or update relationships.
            for relationship in relationships:

                if (
                    relationship["source"]
                    == relationship["target"]
                ):
                    continue

                session.run(
                    """
                    MERGE (a:Entity {id: $source})
                    MERGE (b:Entity {id: $target})

                    MERGE (a)-[r:RELATED_TO]-(b)

                    SET r.description =
                        CASE

                            WHEN $description IS NOT NULL
                            AND trim($description) <> ''

                            THEN $description

                            ELSE r.description

                        END
                    """,

                    source=relationship["source"],
                    target=relationship["target"],

                    description=relationship.get(
                        "description", ""
                    )
                )

        self._schema_cache = None

        return {
            "entities_stored": len(entities),

            "relationships_stored": sum(
                1 for relationship in relationships
                if relationship["source"]
                != relationship["target"]
            )
        }

    # --------------------------------------------------
    # CLOSE CONNECTION
    # --------------------------------------------------

    def close(self):

        if self.read_driver is not self.driver:
            self.read_driver.close()

        self.driver.close()


# --------------------------------------------------
# GLOBAL KNOWLEDGE GRAPH INSTANCE
# --------------------------------------------------

graph = KnowledgeGraph()


# --------------------------------------------------
# LANGCHAIN SEARCH TOOL
# --------------------------------------------------

@tool
def search_knowledge_graph(query: str) -> str:

    """
    Understand a natural-language question, generate a
    read-only Cypher query using Gemini, and retrieve
    relevant information from the Neo4j knowledge graph.
    """

    result = graph.search_question(query)

    # Return graph context to the workflow.
    # The generated Cypher is printed in the terminal,
    # so it does not inflate the workflow's context length.

    if result["error"]:
        raise RuntimeError(result["error"])

    tool_result = {
        "match_count": result["match_count"],
        "matches": result["matches"],
        "cypher": result["cypher"],
        "explanation": result["explanation"],
        "usage": result["usage"],
        "error": None
    }

    return json.dumps(
        tool_result,
        indent=2,
        ensure_ascii=False,
        default=str
    )