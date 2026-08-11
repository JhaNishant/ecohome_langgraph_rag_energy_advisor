"""LangGraph powered EcoHome Energy Advisor."""

from __future__ import annotations

import os
from typing import Any
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from models.preferences import load_user_preferences
from tools import TOOL_KIT


load_dotenv(Path(__file__).resolve().with_name(".env"))


class Agent:
    """A tool using advisor for practical home energy decisions."""

    def __init__(self, instructions: str, model: str | None = None):
        if not instructions or not instructions.strip():
            raise ValueError("instructions must be a nonempty system prompt")
        self.instructions = instructions.strip()
        self.model_name = model or os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
        self.api_key = os.getenv("VOCAREUM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=0,
            api_key=self.api_key or "missing_api_key",
            base_url=os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1"),
            reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT", "medium"),
            use_responses_api=True,
        )
        self.llm_with_tools = self.llm.bind_tools(TOOL_KIT)
        self.graph = self._build_graph()

    def _advisor_node(self, state: MessagesState) -> dict[str, Any]:
        return {"messages": [self.llm_with_tools.invoke(state["messages"])]}

    def _build_graph(self):
        workflow = StateGraph(MessagesState)
        workflow.add_node("advisor", self._advisor_node)
        workflow.add_node("tools", ToolNode(TOOL_KIT, handle_tool_errors=True))
        workflow.add_edge(START, "advisor")
        workflow.add_conditional_edges(
            "advisor",
            tools_condition,
            {"tools": "tools", "__end__": END},
        )
        workflow.add_edge("tools", "advisor")
        return workflow.compile()

    def invoke(self, question: str, context: str | None = None) -> dict[str, Any]:
        """Answer an energy question and return the complete LangGraph message trace."""
        if not question or not question.strip():
            raise ValueError("question must be nonempty")
        if not self.api_key:
            raise RuntimeError("Set VOCAREUM_API_KEY or OPENAI_API_KEY before invoking the advisor")
        preferences = load_user_preferences()
        profile_context = (
            f"Preferences: EV departure {preferences['ev']['departure_time']}; "
            f"comfort {preferences['comfort']['minimum_temperature_c']}–{preferences['comfort']['maximum_temperature_c']}°C; "
            f"battery reserve {preferences['battery']['reserve_percent']}%; "
            f"solar priority {preferences['priorities']['maximize_solar']}."
        )
        base_context = context.strip() if context else "Location: Berlin, Germany. Timezone: Europe/Berlin. Currency: EUR."
        context_text = f"{base_context}\n{profile_context}"
        system_prompt = f"{self.instructions}\n\nCurrent customer context:\n{context_text}"
        return self.graph.invoke(
            {"messages": [SystemMessage(content=system_prompt), HumanMessage(content=question.strip())]}
        )

    def get_agent_tools(self) -> list[str]:
        """Return the tools available to the advisor."""
        return [tool.name for tool in TOOL_KIT]
