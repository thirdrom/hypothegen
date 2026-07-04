"""
Инициализатор пакета tools.

Экспортирует внешние источники и утилиты, используемые другими узлами графа.
"""

from app.tools.paperpilot import search_external
from app.tools.semscholar import search_external as semscholar_search_external

__all__ = ["search_external", "semscholar_search_external"]
