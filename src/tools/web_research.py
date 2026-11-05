import os
import re

from urllib.parse import urlparse

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def to_dict(value):
    if value is None:
        return {}

    if isinstance(value, dict):
        return value

    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)

    return {}


class WebResearch:

    def __init__(self):

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing from .env"
            )

        self.client = Groq(api_key=api_key)

        self.model = "openai/gpt-oss-120b"


    def extract_sources(self, message):

        sources = []
        seen_urls = set()

        def add_source(title, url):

            if not url:
                return

            url = str(url).strip()

            url = url.rstrip(
                ".,;:!?)]}>'\"`"
            )

            parsed = urlparse(url)

            if (
                parsed.scheme not in ("http", "https")
                or not parsed.netloc
            ):
                return

            key = (
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                parsed.path.rstrip("/"),
                parsed.query
            )

            if key in seen_urls:
                return

            seen_urls.add(key)

            sources.append({
                "title": title or parsed.netloc,
                "url": url
            })


        message_data = to_dict(message)

        executed_tools = (
            message_data.get("executed_tools") or []
        )

        for tool in executed_tools:

            tool_data = to_dict(tool)

            # 1. URLs from opened browser pages.
            browser_results = (
                tool_data.get("browser_results") or []
            )

            for result in browser_results:

                result = to_dict(result)

                add_source(
                    result.get("title"),
                    result.get("url")
                )

            # 2. URLs from search results.
            search_results = to_dict(
                tool_data.get("search_results")
            )

            results = (
                search_results.get("results") or []
            )

            for result in results:

                result = to_dict(result)

                add_source(
                    result.get("title"),
                    result.get("url")
                )


        # 3. URLs from citation annotations, if available.
        annotations = message_data.get("annotations") or []

        for annotation in annotations:

            annotation = to_dict(annotation)

            citation = to_dict(
                annotation.get("url_citation")
            )

            if citation:

                add_source(
                    citation.get("title"),
                    citation.get("url")
                )


        # 4. Fallback: extract URLs from generated text.
        content = message_data.get("content") or ""

        url_pattern = r'https?://[^\s<>"\'\]\)]+'

        for url in re.findall(url_pattern, content):

            add_source(None, url)

        return sources


    def search(self, query: str) -> dict:

        print(f"\n[WebResearch] Searching for: {query}")

        response = self.client.chat.completions.create(

            model=self.model,

            messages=[
                {
                    "role": "system",
                    "content": """
                    You are a technical web research assistant.

                    Research the given topic using Browser Search.

                    Instructions:
                    - Use no more than 3 search results.
                    - Prefer official documentation and authoritative sources.
                    - Never use Wikipedia.
                    - Do not use news searches for general technical questions.
                    - Open at most 1 relevant webpage.
                    - Avoid unnecessary browsing.
                    - Extract only information relevant to the question.

                    Return:
                    - Key findings (maximum 8 bullet points).
                    - Relevant technologies.
                    - Sources.
                    """
                },
                {
                    "role": "user",
                    "content": f"Research this topic: {query}"
                }
            ],

            tools=[
                {"type": "browser_search"}
            ],

            tool_choice="required",
            temperature=0.2,
            max_completion_tokens=1500,
            reasoning_effort="low",
            stream=False
        )


        message = response.choices[0].message

        research_content = message.content or ""

        sources = self.extract_sources(message)


        usage = response.usage

        token_usage = {
            "prompt_tokens": (
                usage.prompt_tokens if usage else None
            ),

            "completion_tokens": (
                usage.completion_tokens if usage else None
            ),

            "total_tokens": (
                usage.total_tokens if usage else None
            )
        }


        print("[WebResearch] Research completed.")

        print(
            f"[WebResearch] Sources found: {len(sources)}"
        )

        print(
            f"[WebResearch] Tokens consumed: "
            f"{token_usage['total_tokens']}"
        )


        return {
            "query": query,
            "research": research_content,
            "sources": sources,
            "usage": token_usage,
            "model": self.model
        }


web_research = WebResearch()


def research_web(query: str) -> dict:

    return web_research.search(query)