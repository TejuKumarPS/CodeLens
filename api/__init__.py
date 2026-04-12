"""
api — CodeLens Module 8
========================
FastAPI backend that exposes the full CodeLens retrieval pipeline
as HTTP endpoints consumed by the React frontend (M9).

Endpoints:
  POST /search          — main retrieval endpoint
  GET  /health          — liveness check
  GET  /index/stats     — collection metadata
  POST /evaluate        — run evaluation on test split (admin)

Public entry point:
  create_app()  — returns a configured FastAPI application instance
"""

from .app import create_app

__all__ = ["create_app"]