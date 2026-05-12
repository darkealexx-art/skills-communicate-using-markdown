from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .opportunity_explorer import OpportunityExplorer
from .risk_guardian import RiskGuardian
from .sentiment_analyst import SentimentAnalyst
from .strategy_architect import StrategyArchitect


class MarketAnalysisOrchestrator:
    """Coordina la ejecución de agentes especializados."""

    def __init__(self, output_dir: str = "outputs") -> None:
        self.output_dir = Path(output_dir)
        self.sentiment_agent = SentimentAnalyst(output_dir=output_dir)
        self.opportunity_agent = OpportunityExplorer()
        self.risk_agent = RiskGuardian()
        self.strategy_agent = StrategyArchitect()

    def run(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        sentiment_report = self.sentiment_agent.report(
            self.sentiment_agent.analyze(self.sentiment_agent.fetch_data())
        )
        opportunity_report = self.opportunity_agent.report(
            self.opportunity_agent.analyze(self.opportunity_agent.fetch_data())
        )
        risk_report = self.risk_agent.report(
            self.risk_agent.analyze(self.risk_agent.fetch_data())
        )

        strategy_inputs = {
            "sentiment": sentiment_report,
            "opportunities": opportunity_report,
            "risks": risk_report,
        }
        strategy_report = self.strategy_agent.report(
            self.strategy_agent.analyze(self.strategy_agent.fetch_data(strategy_inputs))
        )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        combined = {
            "generated_at_utc": timestamp,
            "agents": {
                "sentiment_analyst": sentiment_report,
                "opportunity_explorer": opportunity_report,
                "risk_guardian": risk_report,
                "strategy_architect": strategy_report,
            },
        }
        markdown_report = self._compose_markdown(combined)

        json_path = self.output_dir / f"market_report_{timestamp}.json"
        md_path = self.output_dir / f"market_report_{timestamp}.md"
        json_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(markdown_report, encoding="utf-8")

        return {
            "json_report_path": str(json_path),
            "markdown_report_path": str(md_path),
            "combined_report": combined,
            "markdown_content": markdown_report,
        }

    @staticmethod
    def _compose_markdown(report_data: dict[str, Any]) -> str:
        agents = report_data["agents"]
        return (
            "# Informe Integrado: Ecosistema de Analistas de Mercado\n\n"
            f"Generado (UTC): `{report_data['generated_at_utc']}`\n\n"
            + agents["sentiment_analyst"]["markdown"]
            + "\n"
            + agents["opportunity_explorer"]["markdown"]
            + "\n"
            + agents["risk_guardian"]["markdown"]
            + "\n"
            + agents["strategy_architect"]["markdown"]
            + "\n"
        )
