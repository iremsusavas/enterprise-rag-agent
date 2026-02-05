from __future__ import annotations

from app.models.schemas import UserContext


def get_current_user() -> UserContext:
    return UserContext(
        user_id="demo-user",
        role="admin",
    )
