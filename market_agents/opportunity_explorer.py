from __future__ import annotations

from typing import Any

import pandas as pd

from .base import MarketAgent


class OpportunityExplorer(MarketAgent):
    """Detecta oportunidades y nichos emergentes."""

    VALUATION_WEIGHT = 0.45
    MOMENTUM_WEIGHT = 0.45
    VOLATILITY_SCALE = 10
    VOLATILITY_PENALTY_WEIGHT = 0.10

    def __init__(self) -> None:
        super().__init__(
            name="Explorador de Oportunidades",
            role="Detección de activos y tendencias emergentes",
            sources=[
                {"name": "World Bank Data", "url": "https://data.worldbank.org/", "type": "database"},
                {"name": "IMF Data", "url": "https://www.imf.org/en/Data", "type": "report"},
                {"name": "Financial Times Markets", "url": "https://www.ft.com/markets", "type": "news"},
            ],
        )

    def fetch_data(self, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        source_checks = self._check_sources()
        opportunities = [
            {"asset": "Semiconductores Asia", "valuation_gap_pct": 14.0, "momentum_score": 78, "volatility": 0.26},
            {"asset": "Infraestructura Verde LATAM", "valuation_gap_pct": 19.5, "momentum_score": 69, "volatility": 0.21},
            {"asset": "Fintech Pagos SME", "valuation_gap_pct": 12.2, "momentum_score": 83, "volatility": 0.31},
            {"asset": "Bonos Corporativos IG Europa", "valuation_gap_pct": 8.4, "momentum_score": 62, "volatility": 0.14},
        ]
        return {"opportunities": opportunities, "sources": source_checks}

    def analyze(self, data: dict[str, Any]) -> dict[str, Any]:
        frame = pd.DataFrame(data["opportunities"])
        positive_component = (
            (frame["valuation_gap_pct"] * self.VALUATION_WEIGHT)
            + (frame["momentum_score"] * self.MOMENTUM_WEIGHT)
        )
        volatility_penalty = frame["volatility"] * self.VOLATILITY_SCALE * self.VOLATILITY_PENALTY_WEIGHT
        frame["composite_score"] = positive_component - volatility_penalty
        ranked = frame.sort_values(by="composite_score", ascending=False).reset_index(drop=True)
        short_term = ranked.head(2)["asset"].tolist()
        long_term = ranked.head(3)["asset"].tolist()

        return {
            "top_short_term": short_term,
            "top_long_term": long_term,
            "opportunities_table": ranked.round(3),
            "scenario_short_term": f"Priorizar asignación táctica en: {', '.join(short_term)}.",
            "scenario_long_term": f"Construir posición estructural en: {', '.join(long_term)}.",
            "sources": data["sources"],
        }

    def report(self, analysis: dict[str, Any]) -> dict[str, Any]:
        summary = (
            f"Oportunidades prioritarias en corto plazo: {', '.join(analysis['top_short_term'])}. "
            f"Consolidar visión de largo plazo en: {', '.join(analysis['top_long_term'])}."
        )
        deep_analysis = {
            "table": analysis["opportunities_table"].to_dict(orient="records"),
            "top_short_term": analysis["top_short_term"],
            "top_long_term": analysis["top_long_term"],
            "scenario_short_term": analysis["scenario_short_term"],
            "scenario_long_term": analysis["scenario_long_term"],
            "sources": analysis["sources"],
        }
        table_markdown = self._to_markdown(analysis["opportunities_table"])
        markdown = (
            "## 2) Explorador de Oportunidades\n\n"
            "### Resumen Ejecutivo\n"
            f"{summary}\n\n"
            "### Análisis Profundo\n"
            f"{table_markdown}\n\n"
            f"- Escenario corto plazo: {analysis['scenario_short_term']}\n"
            f"- Escenario largo plazo: {analysis['scenario_long_term']}\n\n"
            "### Fuentes Consultadas\n"
            + "\n".join([f"- {s['name']} ({s['type']}): {s['url']} [estado: {s['status']}]" for s in analysis["sources"]])
            + "\n"
        )
        return {"executive_summary": summary, "deep_analysis": deep_analysis, "markdown": markdown}

    @staticmethod
    def _to_markdown(frame: pd.DataFrame) -> str:
        headers = [str(col) for col in frame.columns]
        separator = ["---"] * len(headers)
        rows = frame.astype(str).values.tolist()
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(separator) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        return "\n".join(lines)
