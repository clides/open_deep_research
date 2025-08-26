"""Graph state definitions and data structures for the Codebase Consulting agent."""

import operator
from typing import Annotated, Optional

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


###################
# Structured Outputs
###################
class ExecuteTask(BaseModel):
    """Call this tool to execute a task on a specific topic."""
    task_description: str = Field(
        description="The task to execute. Should be a single task, and should be described in high detail (at least a paragraph).",
    )

class TaskComplete(BaseModel):
    """Call this tool to indicate that the task is complete."""

class Summary(BaseModel):
    """Analysis summary with key findings."""
    
    summary: str
    key_excerpts: str

class ClarifyWithUser(BaseModel):
    """Model for user clarification requests."""
    
    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the task scope",
    )
    verification: str = Field(
        description="Verify message that we will start the task after the user has provided the necessary information.",
    )

class TaskBrief(BaseModel):
    """Task description and brief for guiding the work."""
    
    task_brief: str = Field(
        description="A task description that will be used to guide the work.",
    )


###################
# State Definitions
###################

def override_reducer(current_value, new_value):
    """Reducer function that allows overriding values in state."""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)
    
class AgentInputState(MessagesState):
    """InputState is only 'messages'."""

class AgentState(MessagesState):
    """Main agent state containing messages and analysis data."""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    task_brief: Optional[str]
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    final_report: str

class SupervisorState(TypedDict):
    """State for the supervisor that manages tasks."""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    task_brief: str
    notes: Annotated[list[str], override_reducer] = []
    iterations: int = 0
    raw_notes: Annotated[list[str], override_reducer] = []

class AnalystState(TypedDict):
    """State for individual analysts conducting analysis."""
    
    analyst_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    task_description: str
    compressed_analysis: str
    raw_notes: Annotated[list[str], override_reducer] = []

class AnalystOutputState(BaseModel):
    """Output state from individual analysts."""
    
    compressed_analysis: str
    raw_notes: Annotated[list[str], override_reducer] = []
