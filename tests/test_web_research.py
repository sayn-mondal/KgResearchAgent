from src.tools.web_research import research_web


def main():

    query = "What is SAP S/4HANA and what are its main capabilities?"

    print("\nStarting web research...")

    result = research_web(query)

    print("\n========== RESEARCH RESULTS ==========")

    print(result["research"])

    print("\n========== TOKEN USAGE ==========")

    print(result["usage"])

    print("\n========== MODEL ==========")

    print(result["model"])


if __name__ == "__main__":
    main()