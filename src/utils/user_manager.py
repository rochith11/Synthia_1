"""Simple user tracking — basic identification stored with datasets (no auth)."""

from typing import Optional


class UserManager:
    """Track the current user for audit and dataset ownership.

    V1 implementation: no authentication, just a configurable username.
    """

    def __init__(self, username: str = "default_user"):
        self.username = username

    def get_current_user(self) -> str:
        return self.username

    def set_user(self, username: str) -> None:
        self.username = username

    def get_user_info(self) -> dict:
        return {"username": self.username}
