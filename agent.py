from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from ddgs import DDGS


@tool("DuckDuckGo Search")
def duckduckgo_search(query: str) -> str:
    """Search DuckDuckGo for current information and return useful results."""

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
    except Exception as exc:
        return f"Search failed for query '{query}': {exc}"

    if not results:
        return f"No results found for '{query}'."

    formatted = []

    for i, result in enumerate(results, start=1):
        title = result.get("title", "No title")
        link = result.get("href", "")
        snippet = result.get("body", "")

        formatted.append(
            f"{i}. {title}\n"
            f"Link: {link}\n"
            f"{snippet}"
        )

    return "\n\n".join(formatted)


def build_crew(topic: str, groq_api_key: str) -> Crew:
    """Create the research agent, task, and crew."""

    if not topic or not topic.strip():
        raise ValueError("Research topic cannot be empty.")

    if not groq_api_key or not groq_api_key.strip():
        raise ValueError("Groq API key is missing.")

    llm = LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=groq_api_key.strip(),
        temperature=0.5,
    )

    researcher = Agent(
        role="Senior Research Analyst",
        goal=(
            f"Research '{topic}' thoroughly using web search and produce "
            "an accurate, well-organized, up-to-date report."
        ),
        backstory=(
            "You are a meticulous senior research analyst. "
            "You search the web, cross-check important claims, "
            "avoid inventing information, and cite the sources you use."
        ),
        tools=[duckduckgo_search],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    research_task = Task(
        description=(
            f"Research the following topic:\n\n"
            f"{topic}\n\n"
            "Instructions:\n"
            "1. Search DuckDuckGo multiple times using specific queries.\n"
            "2. Gather current and relevant information.\n"
            "3. Cross-check important claims when possible.\n"
            "4. Do not invent facts, statistics, URLs, or sources.\n"
            "5. Only cite URLs returned by the search tool.\n"
            "6. Write the final report in Markdown.\n\n"
            "Include:\n"
            "- Introduction\n"
            "- Key findings\n"
            "- Facts and statistics\n"
            "- Recent developments where relevant\n"
            "- Analysis\n"
            "- Conclusion\n"
            "- Sources"
        ),
        expected_output=(
            "A 500-800 word Markdown research report with clear headings, "
            "a conclusion, and a Sources section containing real URLs."
        ),
        agent=researcher,
    )

    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        process=Process.sequential,
        verbose=True,
    )

    return crew


def run_research(topic: str, groq_api_key: str) -> str:
    """Run the research crew and return its final report."""

    crew = build_crew(
        topic=topic,
        groq_api_key=groq_api_key,
    )

    result = crew.kickoff()

    return str(result)
