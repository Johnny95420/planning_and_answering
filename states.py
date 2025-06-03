from dataclasses import dataclass, field
from typing import Annotated, Sequence, TypedDict
import operator
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from pydantic import BaseModel, Field
from typing_extensions import Annotated
from langchain_core.messages import AnyMessage


@dataclass
class InputState:
    topic: str
    history: str
    prev_summary_index: int
    iterations: int = field(default=0)
    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )


@dataclass
class State(InputState):
    is_last_step: IsLastStep = field(default=False)


class Step(BaseModel):
    goal: str = Field(description="The detailed goal of this step")
    content: str = Field(description="The content of this step")


class QuestionStateInput(TypedDict):
    question: str
    background_knowledge: str


class QuestionState(TypedDict):
    question: str
    steps: list[Step]
    background_knowledge: str
    completed_steps: Annotated[list, operator.add]
    can_answer: bool
    plan_solving_iterations: int
    report: str


class QustionOutputState(TypedDict):
    report: str
