from __future__ import annotations

from typing import Any

from .base import MarketAgent


class StrategyArchitect(MarketAgent):
    """Integra hallazgos y propone estrategia sostenible."""

    RISK_PENALTY_PER_CRITICAL = 0.07
    MAX_RISK_PENALTY = 0.21
    TACTICAL_BUDGET_POSITIVE_SENTIMENT = 0.45
    TACTICAL_BUDGET_NON_POSITIVE_SENTIMENT = 0.35
    MIN_TACTICAL_BUDGET = 0.20

    def __init__(self) -> None:
        super().__init__(
            name="Arquitecto de Estrategia",
            role="Síntesis estratégica multiagente",
            sources=[
                {"name": "Sentiment Analyst Output", "url": "internal://sentiment", "type": "internal"},
                {"name": "Opportunity Explorer Output", "url": "internal://opportunities", "type": "internal"},
                {"name": "Risk Guardian Output", "url": "internal://risks", "type": "internal"},
            ],
        )

    def fetch_data(self, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"integrated_inputs": inputs or {}, "sources": self._check_sources()}

    def analyze(self, data: dict[str, Any]) -> dict[str, Any]:
        sentiment = data["integrated_inputs"]["sentiment"]
        opportunities = data["integrated_inputs"]["opportunities"]
        risks = data["integrated_inputs"]["risks"]

        risk_deep_analysis = risks.get("deep_analysis", {})
        critical_risks_list = risk_deep_analysis.get("critical_risks", [])
        if not isinstance(critical_risks_list, list):
            critical_risks_list = []

        risk_penalty = min(
            len(critical_risks_list) * self.RISK_PENALTY_PER_CRITICAL,
            self.MAX_RISK_PENALTY,
        )
        base_risk_budget = (
            self.TACTICAL_BUDGET_POSITIVE_SENTIMENT
            if sentiment["deep_analysis"]["metrics"]["average_sentiment"] > 0
            else self.TACTICAL_BUDGET_NON_POSITIVE_SENTIMENT
        )
        tactical_budget = round(max(base_risk_budget - risk_penalty, self.MIN_TACTICAL_BUDGET), 2)
        defensive_budget = round(1.0 - tactical_budget, 2)
        short_term_assets = opportunities["deep_analysis"]["top_short_term"]
        critical_risks = critical_risks_list
        critical_risk_text = ", ".join(critical_risks) if critical_risks else "sin riesgos críticos actuales"

        recommendations = [
            f"Asignar {int(tactical_budget * 100)}% a oportunidades de mayor score en corto plazo: "
            f"{', '.join(short_term_assets)}.",
            f"Reservar {int(defensive_budget * 100)}% en activos defensivos por riesgos críticos: "
            f"{critical_risk_text}.",
            "Rebalancear mensualmente y recalibrar la cartera ante eventos regulatorios y geopolíticos relevantes.",
        ]
        scenarios = {
            "short_term": "Ejecución táctica con cobertura parcial para controlar volatilidad de 30-90 días.",
            "long_term": "Escalamiento progresivo hacia tesis estructurales de crecimiento con disciplina de riesgo.",
        }
        return {
            "recommendations": recommendations,
            "scenarios": scenarios,
            "allocation": {"tactical_growth": tactical_budget, "defensive_core": defensive_budget},
            "sources": data["sources"],
        }

    def report(self, analysis: dict[str, Any]) -> dict[str, Any]:
        summary = (
            "Estrategia integrada: balance entre crecimiento táctico y núcleo defensivo, "
            "priorizando sostenibilidad de retornos."
        )
        deep_analysis = {
            "allocation": analysis["allocation"],
            "scenarios": analysis["scenarios"],
            "recommendations": analysis["recommendations"],
            "sources": analysis["sources"],
        }
        recommendations_md = "\n".join([f"- {r}" for r in analysis["recommendations"]])
        markdown = (
            "## 4) Arquitecto de Estrategia\n\n"
            "### Resumen Ejecutivo\n"
            f"{summary}\n\n"
            "### Análisis Profundo\n"
            f"- Asignación táctica (corto plazo): `{analysis['allocation']['tactical_growth']}`\n"
            f"- Asignación defensiva (visión largo plazo): `{analysis['allocation']['defensive_core']}`\n"
            f"- Escenario corto plazo: {analysis['scenarios']['short_term']}\n"
            f"- Escenario largo plazo: {analysis['scenarios']['long_term']}\n"
            f"- Recomendaciones:\n{recommendations_md}\n\n"
            "### Fuentes Consultadas\n"
            + "\n".join([f"- {s['name']} ({s['type']}): {s['url']} [estado: {s['status']}]" for s in analysis["sources"]])
            + "\n"
        )
        return {"executive_summary": summary, "deep_analysis": deep_analysis, "markdown": markdown}
