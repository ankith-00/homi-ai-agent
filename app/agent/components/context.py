from dataclasses import dataclass, field

from pydantic_ai.messages import ModelMessage


@dataclass
class Session:
    user_id: str = ""
    profile: dict = field(default_factory=dict)
    history: list[ModelMessage] = field(default_factory=list)
    relevant_memories: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
