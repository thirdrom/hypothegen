"""
Инициализатор пакета tools.

Экспортирует внешние источники и утилиты, используемые другими узлами графа.
"""

from app.tools.paperpilot import search_external

__all__ = ["search_external"]
