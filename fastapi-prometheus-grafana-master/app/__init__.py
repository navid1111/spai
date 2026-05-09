"""FastAPI Retail Detection Application package."""

__all__ = ["app"]


def __getattr__(name: str):
	"""Lazily expose app to avoid import-time side effects during test collection."""
	if name == "app":
		from .main import app as fastapi_app

		return fastapi_app
	raise AttributeError(f"module 'app' has no attribute {name!r}")
