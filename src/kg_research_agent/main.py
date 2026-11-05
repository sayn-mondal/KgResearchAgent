from src.workflows.research_workflow import build_research_workflow


def main():

    workflow = build_research_workflow()

    print("KgResearchAgent started!")

    while True:

        question = input("\nYou: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if not question:
            continue

        try:

            result = workflow.invoke({
                "question": question,
                "search_query": "",
                "graph_context": "",
                "graph_match_count": 0,
                "graph_cypher": "",
                "graph_explanation": "",
                "sufficient": False,
                "evaluation_reason": "",
                "research_results": "",
                "sources": [],
                "usage": {},
                "answer": ""
            })

            print("\nAgent:", result["answer"])

            if result["sources"]:
                print("\nSources:")

                for source in result["sources"]:
                    print(f"- {source['title']}: {source['url']}")

            if result["usage"]:
                print("\nToken Usage:")
                print(result["usage"])

        except Exception as e:

            print(f"\nError: {e}")


if __name__ == "__main__":
    main()