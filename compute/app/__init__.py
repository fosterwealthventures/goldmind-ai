# compute/app/__init__.py
from .server import app

__all__ = ["app"]

def create_app():
    return app
