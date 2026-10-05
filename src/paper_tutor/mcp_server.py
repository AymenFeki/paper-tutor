"""MCP server exposing the paper database as a tool for Claude Desktop."""

import os

import httpx
from mcp.server import MCPServer

API_URL = os.getenv("PAPER_TUTOR_API", "http://127.0.0.1:8000")

mcp = MCPServer("paper-tutor")


def format_papers(papers):
    """Turn the API's paper list into numbered plain text for the model."""
    lines = []
    for i,paper in enumerate(papers, start=1):
        venue = paper.get("venue") or "unknown venue"
        authors = ", ".join(paper.get("authors") or []) or "unknown authors"
        url = f"https://openalex.org/{paper['id']}"
        lines.append(f"[{i}] {paper['title']} ({paper['year']}, {venue})\n{authors}\n{url}\n{paper['abstract']}")
    return "\n\n".join(lines)


@mcp.tool()
async def search_papers(question: str, k: int = 5) -> str:
    """Search a curated database of research papers in statistics, econometrics,
    machine learning, deep learning, economics, finance and cloud computing.

    Use this when the user asks about a statistical or ML method, concept or topic
    and wants answers grounded in real papers. Returns titles, years, venues,
    authors, OpenAlex links and abstracts. Cite papers by their number.

    Args:
        question: What to search for, in plain language.
        k: Number of papers to return (1-10).
    """
    k = max(1, min(k, 10))
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(f"{API_URL}/search", json={"question": question, "k": k})
            response.raise_for_status()
    except httpx.HTTPError as error:
        return f"The paper database could not be reached: {error}. Is the paper-tutor API running?"
    return format_papers(response.json()["papers"])


if __name__ == "__main__":
    mcp.run(transport="stdio")