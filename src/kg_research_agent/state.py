from typing import TypedDict


class ResearchState(TypedDict):
    question: str
    search_query: str
    graph_context: str
    graph_match_count: int
    graph_cypher: str
    graph_explanation: str
    sufficient: bool
    evaluation_reason: str

    research_results: str
    sources: list
    usage: dict

    answer: str