from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from src.kg_research_agent.agent import GEMINI_MODEL


class Entity(BaseModel):
    name: str = Field(description="Name of the entity")
    description: str = Field(
        description="Short description of the entity"
    )


class Relationship(BaseModel):
    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    description: str = Field(
        description="How the two entities are related"
    )


class KnowledgeExtraction(BaseModel):
    entities: list[Entity]
    relationships: list[Relationship]


def extract_knowledge(research: str):

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0
    )

    structured_llm = llm.with_structured_output(
        KnowledgeExtraction,
        include_raw=True
    )

    prompt = f"""
    You are a knowledge graph extraction system.

    Extract entities and relationships from the research below.

    Research:
    {research}

    Rules:
    1. Extract only explicitly supported entities.
    2. Identify meaningful relationships between entities.
    3. Do not invent entities or relationships.
    4. Avoid duplicate entities.
    5. Use consistent entity names.
    6. Every relationship must reference extracted entities.
    7. Keep descriptions concise.
    """

    response = structured_llm.invoke(prompt)

    if response["parsing_error"]:
        raise response["parsing_error"]

    return {
        "knowledge": response["parsed"],
        "usage": response["raw"].usage_metadata
    }