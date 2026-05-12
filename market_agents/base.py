from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

import requests


class MarketAgent(ABC):
    """Contrato base para agentes especializados."""

    def __init__(self, name: str, role: str, sources: list[dict[str, str]]) -> None:
        self.name = name
        self.role = role
        self.sources = sources

    @abstractmethod
    def fetch_data(self, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        """Obtiene información desde fuentes externas y/o simuladas."""

    @abstractmethod
    def analyze(self, data: dict[str, Any]) -> dict[str, Any]:
        """Procesa datos y genera métricas técnicas."""

    @abstractmethod
    def report(self, analysis: dict[str, Any]) -> dict[str, Any]:
        """Genera el reporte en formato estructurado."""

    def _check_sources(self) -> list[dict[str, str]]:
        checked_sources: list[dict[str, str]] = []
        for source in self.sources:
            url = source["url"]
            status = "available"
            try:
                requests.get(url, timeout=5)
            except requests.RequestException:
                status = "unreachable"

            checked_sources.append(
                {
                    "name": source["name"],
                    "url": url,
                    "type": source["type"],
                    "status": status,
                    "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
        return checked_sources
