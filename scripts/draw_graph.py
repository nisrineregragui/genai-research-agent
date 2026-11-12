"""Write the agent graph as a Mermaid diagram (for the README and the presentation).

Run: python -m scripts.draw_graph   ->   docs/agent_graph.mmd
"""

from __future__ import annotations

from app.agent import build_graph
from app.config import PROJECT_ROOT


def main() -> None:
    # Structure only: the nodes are never executed, so no component is needed.
    graph = build_graph(None, None, None, None, None, max_revisions=1)
    output = PROJECT_ROOT / "docs" / "agent_graph.mmd"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(graph.get_graph().draw_mermaid(), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
    