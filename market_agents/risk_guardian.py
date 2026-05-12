from __future__ import annotations

from typing import Any

import pandas as pd

from .base import MarketAgent


class RiskGuardian(MarketAgent):
    """Clasifica riesgos por impacto y probabilidad."""

    def __init__(self) -> None:
        super().__init__(
            name="Guardia de Riesgos",
            role="Gestión de riesgos de inversión",
            sources=[
                {"name": "BIS Indicators", "url": "https://www.bis.org/statistics/index.htm", "type": "macro"},
                {"name": "ECB Supervision", "url": "https://www.bankingsupervision.europa.eu/home/html/index.en.html", "type": "regulatory"},
                {"name": "UN News Global", "url": "https://news.un.org/en/", "type": "geopolitics"},
            ],
        )

    def fetch_data(self, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        source_checks = self._check_sources()
        risks = [
            {"risk": "Volatilidad de tasas globales", "impact": 5, "probability": 4},
            {"risk": "Cambios regulatorios en fintech", "impact": 4, "probability": 3},
            {"risk": "Escalada geopolítica en rutas energéticas", "impact": 5, "probability": 3},
            {"risk": "Desaceleración de consumo global", "impact": 3, "probability": 4},
        ]
        return {"risks": risks, "sources": source_checks}

    def analyze(self, data: dict[str, Any]) -> dict[str, Any]:
        frame = pd.DataFrame(data["risks"])
        frame["risk_score"] = frame["impact"] * frame["probability"]
        frame["classification"] = frame["risk_score"].apply(
            lambda score: "alto" if score >= 16 else ("medio" if score >= 10 else "bajo")
        )
        risk_matrix = frame.sort_values(by=["risk_score", "impact"], ascending=False).reset_index(drop=True)

        return {
            "risk_table": risk_matrix,
            "critical_risks": risk_matrix[risk_matrix["classification"] == "alto"]["risk"].tolist(),
            "sources": data["sources"],
        }

    def report(self, analysis: dict[str, Any]) -> dict[str, Any]:
        summary = (
            "Riesgos críticos detectados: "
            + (", ".join(analysis["critical_risks"]) if analysis["critical_risks"] else "sin riesgos de clasificación alta.")
        )
        deep_analysis = {
            "table": analysis["risk_table"].to_dict(orient="records"),
            "critical_risks": analysis["critical_risks"],
            "sources": analysis["sources"],
        }
        markdown = (
            "## 3) Guardia de Riesgos\n\n"
            "### Resumen Ejecutivo\n"
            f"{summary}\n\n"
            "### Análisis Profundo\n"
            f"{analysis['risk_table'].to_markdown(index=False)}\n\n"
            "### Fuentes Consultadas\n"
            + "\n".join([f"- {s['name']} ({s['type']}): {s['url']} [estado: {s['status']}]" for s in analysis["sources"]])
            + "\n"
        )
        return {"executive_summary": summary, "deep_analysis": deep_analysis, "markdown": markdown}
