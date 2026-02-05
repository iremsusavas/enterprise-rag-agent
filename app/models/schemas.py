from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class UserContext(BaseModel):
    user_id: str
    role: str


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    citations: List[str]
    user_id: str
    steps: List[str] = []


class UserRequest(BaseModel):
    query: str
    user_id: str
    role: str


class AgentState(BaseModel):
    messages: List[str] = []
    context: List[str] = []
    steps: List[str] = []
