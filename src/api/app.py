"""WSGI/ASGI app export for deployment tools (gunicorn, etc.)"""
from api.main import app  # noqa: F401

__all__ = ["app"]
