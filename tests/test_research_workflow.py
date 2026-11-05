import json
from types import SimpleNamespace

from src.workflows import research_workflow


def test_retrieve_graph_uses_full_question_and_tracks_query_usage(
    monkeypatch
):
    observed = {}

    class FakeTool:
        def invoke(self, payload):
            observed["query"] = payload["query"]
            return json.dumps({
                "match_count": 1,
                "matches": [{"entity": "Example", "description": "Fact"}],
                "cypher": "MATCH (e:Entity) RETURN e.id AS id LIMIT 1",
                "explanation": "Matched the requested entity.",
                "usage": {
                    "input_tokens": 3,
                    "output_tokens": 2,
                    "total_tokens": 5
                },
                "error": None
            })

    monkeypatch.setattr(
        research_workflow,
        "search_knowledge_graph",
        FakeTool()
    )

    question = "What does Example do?"
    result = research_workflow.retrieve_graph({
        "question": question,
        "usage": {}
    })

    assert observed["query"] == question
    assert json.loads(result["graph_context"]) == [
        {"entity": "Example", "description": "Fact"}
    ]
    assert result["graph_match_count"] == 1
    assert result["graph_cypher"].startswith("MATCH")
    assert result["usage"]["gemini"]["total_tokens"] == 5


def test_sufficiency_uses_graph_facts_not_context_length(monkeypatch):
    observed = {}

    class FakeEvaluator:
        def invoke(self, prompt):
            observed["prompt"] = prompt
            return {
                "parsing_error": None,
                "parsed": research_workflow.GraphSufficiency(
                    sufficient=True,
                    reason="Pricing models are described.",
                    missing_information=["specific monetary costs"]
                ),
                "raw": SimpleNamespace(
                    usage_metadata={
                        "input_tokens": 4,
                        "output_tokens": 2,
                        "total_tokens": 6
                    }
                )
            }

    class FakeLLM:
        def with_structured_output(self, schema, *, include_raw):
            observed["schema"] = schema
            observed["include_raw"] = include_raw
            return FakeEvaluator()

    monkeypatch.setattr(research_workflow, "llm", FakeLLM())

    result = research_workflow.evaluate_context({
        "question": "What are these pricing models' monetary costs?",
        "graph_context": json.dumps([{"fact": "x" * 500}]),
        "graph_match_count": 1,
        "usage": {}
    })

    assert result["sufficient"] is False
    assert "specific monetary costs" in result["evaluation_reason"]
    assert result["usage"]["gemini"]["total_tokens"] == 6
    assert observed["schema"] is research_workflow.GraphSufficiency
    assert observed["include_raw"] is True
    assert "monetary costs" in observed["prompt"]


def test_missing_costs_in_graph_answer_routes_to_web_research(monkeypatch):
    class FakeReviewer:
        def invoke(self, prompt):
            assert "specific monetary costs" in prompt
            return {
                "parsing_error": None,
                "parsed": research_workflow.GraphAnswerReview(
                    complete=True,
                    missing_information=["specific monetary costs"],
                    reason=(
                        "The draft explicitly says monetary costs "
                        "are unavailable."
                    )
                ),
                "raw": SimpleNamespace(
                    usage_metadata={
                        "input_tokens": 5,
                        "output_tokens": 2,
                        "total_tokens": 7
                    }
                )
            }

    class FakeLLM:
        def with_structured_output(self, schema, *, include_raw):
            assert schema is research_workflow.GraphAnswerReview
            assert include_raw is True
            return FakeReviewer()

    monkeypatch.setattr(research_workflow, "llm", FakeLLM())

    result = research_workflow.review_graph_answer({
        "question": "What are these pricing models' monetary costs?",
        "graph_context": '[{"name": "Model A"}]',
        "answer": (
            "The graph describes the pricing models but does not contain "
            "specific monetary costs."
        ),
        "usage": {}
    })

    assert result["sufficient"] is False
    assert "specific monetary costs" in result["evaluation_reason"]
    assert result["usage"]["gemini"]["total_tokens"] == 7
    assert research_workflow.route_after_graph_answer({
        "sufficient": result["sufficient"]
    }) == "research_needed"
