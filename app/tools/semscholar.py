"""
Compatibility layer for Semantic Scholar module.

This module provides backward compatibility for code that imports
from app.tools.semscholar. It delegates to the new PaperPilot implementation.
"""

from app.tools.paperpilot import search_external
from app.state import Ref

__all__ = ["search_external", "Ref"]