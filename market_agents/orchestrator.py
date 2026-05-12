from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

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
        run_suffix = uuid4().hex[:8]
        agent_reports = {
            "sentiment_analyst": self._write_agent_report(
                "sentiment_analyst",
                self.sentiment_agent.name,
                sentiment_report,
                timestamp,
                run_suffix,
            ),
            "opportunity_explorer": self._write_agent_report(
                "opportunity_explorer",
                self.opportunity_agent.name,
                opportunity_report,
                timestamp,
                run_suffix,
            ),
            "risk_guardian": self._write_agent_report(
                "risk_guardian",
                self.risk_agent.name,
                risk_report,
                timestamp,
                run_suffix,
            ),
            "strategy_architect": self._write_agent_report(
                "strategy_architect",
                self.strategy_agent.name,
                strategy_report,
                timestamp,
                run_suffix,
            ),
        }
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

        json_path = self.output_dir / f"market_report_{timestamp}_{run_suffix}.json"
        md_path = self.output_dir / f"market_report_{timestamp}_{run_suffix}.md"
        json_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(markdown_report, encoding="utf-8")

        return {
            "json_report_path": str(json_path),
            "markdown_report_path": str(md_path),
            "agent_reports": agent_reports,
            "combined_report": combined,
            "markdown_content": markdown_report,
        }

    def _write_agent_report(
        self,
        agent_key: str,
        agent_name: str,
        report: dict[str, Any],
        timestamp: str,
        run_suffix: str,
    ) -> dict[str, str]:
        detailed_markdown = f"# Informe Detallado: {agent_name}\n\n{report['markdown']}"
        json_path = self.output_dir / f"{agent_key}_report_{timestamp}_{run_suffix}.json"
        md_path = self.output_dir / f"{agent_key}_report_{timestamp}_{run_suffix}.md"
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(detailed_markdown, encoding="utf-8")
        return {"json_report_path": str(json_path), "markdown_report_path": str(md_path)}

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
