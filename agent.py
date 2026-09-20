"""
agent.py

This file defines:
1. A free DuckDuckGo search tool.
2. A single CrewAI Research Analyst agent.
3. A research task.
4. A run_research() function used by the Streamlit app.
"""

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from ddgs import DDGS


# ---------------------------------------------------------------------------
# 1. DUCKDUCKGO SEARCH TOOL
# ---------------------------------------------------------------------------

@tool("DuckDuckGo Search")
def duckduckgo_search(query: str) -> str:
    """
    Search the web using DuckDuckGo.

    Returns the top search results including title, URL, and snippet.
    Use this tool when current information, facts, statistics, sources,
    or additional context are required.
    """

    try:
        with DDGS() as ddgs:
            results = list(
                ddgs.text(
                    query,
                    max_results=5
                )
            )

    except Exception as exc:
        return f"Search failed for query '{query}': {exc}"

    if not results:
        return f"No results found for '{query}'."

    formatted_results = []

    for index, result in enumerate(results, start=1):
        title = result.get("title", "No title")
        link = result.get("href", "")
        snippet = result.get("body", "")

        formatted_results.append(
            f"{index}. {title}\n"
            f"   Link: {link}\n"
            f"   {snippet}"
        )

    return "\n\n".join(formatted_results)


# ---------------------------------------------------------------------------
# 2. BUILD THE CREW
# ---------------------------------------------------------------------------

def build_crew(topic: str, groq_api_key: str) -> Crew:
    """
    Create a CrewAI research crew for the supplied topic.
    """

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
            f"Research the topic '{topic}' thoroughly using web search "
            "and produce an accurate, well-organized, up-to-date report."
        ),

        backstory=(
            "You are a meticulous senior research analyst experienced in "
            "turning web research into clear and trustworthy reports. "
            "You verify important claims, compare multiple sources when "
            "possible, distinguish uncertain information from established "
            "facts, and provide the sources used in your research."
        ),

        tools=[duckduckgo_search],

        llm=llm,

        verbose=True,

        allow_delegation=False,
    )

    # -----------------------------------------------------------------------
    # RESEARCH TASK
    # -----------------------------------------------------------------------

    research_task = Task(
        description=(
            f"Research the following topic:\n\n"
            f"{topic}\n\n"

            "Research instructions:\n"
            "1. Use the DuckDuckGo Search tool several times using different "
            "and specific search queries.\n"
            "2. Gather recent and relevant information about the topic.\n"
            "3. Cross-check important claims against multiple sources whenever "
            "possible.\n"
            "4. Do not invent statistics, quotations, URLs, or sources.\n"
            "5. Only cite URLs that were actually returned by the search tool.\n"
            "6. Clearly distinguish uncertain claims from established facts.\n"
            "7. Produce the final report in Markdown.\n\n"

            "The report must contain:\n"
            "- Introduction\n"
            "- Key findings\n"
            "- Relevant facts and statistics\n"
            "- Recent developments where applicable\n"
            "- Analysis\n"
            "- Conclusion\n"
            "- Sources\n\n"

            "The Sources section must contain the real URLs used during "
            "the research."
        ),

        expected_output=(
            "A well-structured Markdown research report of approximately "
            "500-800 words containing clear headings, factual analysis, "
            "a conclusion, and a Sources section containing real URLs "
            "obtained through the search tool."
        ),

        agent=researcher,
    )

    # -----------------------------------------------------------------------
    # CREW
    # -----------------------------------------------------------------------

    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        process=Process.sequential,
        verbose=True,
    )

    return crew


# ---------------------------------------------------------------------------
# 3. PUBLIC FUNCTION CALLED BY STREAMLIT
# ---------------------------------------------------------------------------

def run_research(topic: str, groq_api_key: str) -> str:
    """
    Run the research agent and return the final Markdown report.
    """

    crew = build_crew(
        topic=topic,
        groq_api_key=groq_api_key,
    )

    result = crew.kickoff()

    return str(result)
