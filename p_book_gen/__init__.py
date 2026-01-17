# p_book_gen/__init__.py
"""
Package for the parallel book generator.

Exposes `root_agent` so ADK can discover it via `p_book_gen.agent.root_agent`.
"""

from .agent import root_agent  # noqa: F401

