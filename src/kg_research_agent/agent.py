from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from src.tools.knowledge_graph import search_knowledge_graph

load_dotenv()
GEMINI_MODEL = "gemini-3.5-flash-lite"

def get_llm():

    return ChatGoogleGenerativeAI(
        model= GEMINI_MODEL,
        #temperature=0
    )


def create_research_agent():

    agent = create_agent(
        model=get_llm(),
        tools=[search_knowledge_graph],
        system_prompt=(
            "You are KgResearchAgent, an intelligent research assistant."
        )
    )

    return agent