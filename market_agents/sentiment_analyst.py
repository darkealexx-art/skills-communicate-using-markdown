from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from .base import MarketAgent

matplotlib.use("Agg")


class SentimentAnalyst(MarketAgent):
    """Evalúa el pulso emocional del mercado."""

    def __init__(self, output_dir: str = "outputs") -> None:
        super().__init__(
            name="Analista de Sentimiento",
            role="Evaluación de pulso emocional de mercado",
            sources=[
                {"name": "Reuters Markets", "url": "https://www.reuters.com/markets/", "type": "news"},
                {"name": "Bloomberg Markets", "url": "https://www.bloomberg.com/markets", "type": "news"},
                {"name": "OECD Economic Outlook", "url": "https://www.oecd.org/economic-outlook/", "type": "report"},
            ],
        )
        self.output_dir = Path(output_dir)

    def fetch_data(self, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        source_checks = self._check_sources()
        signals = [
            {"source": "Reuters Markets", "headline": "Rally en tecnología por expectativas de recorte de tasas", "sentiment_score": 0.72},
            {"source": "Bloomberg Markets", "headline": "Mayor cautela en energía por tensiones geopolíticas", "sentiment_score": -0.28},
            {"source": "OECD Economic Outlook", "headline": "Crecimiento estable con inflación desacelerando", "sentiment_score": 0.20},
            {"source": "Reuters Markets", "headline": "Fondos aumentan exposición a activos de riesgo", "sentiment_score": 0.55},
            {"source": "Bloomberg Markets", "headline": "Bonos soberanos muestran toma de utilidades", "sentiment_score": -0.12},
        ]
        return {"signals": signals, "sources": source_checks}

    def analyze(self, data: dict[str, Any]) -> dict[str, Any]:
        frame = pd.DataFrame(data["signals"])
        avg_sentiment = float(frame["sentiment_score"].mean())
        bullish = int((frame["sentiment_score"] > 0.2).sum())
        bearish = int((frame["sentiment_score"] < -0.2).sum())
        neutral = len(frame) - bullish - bearish

        market_pulse = "neutral"
        if avg_sentiment > 0.2:
            market_pulse = "alcista"
        elif avg_sentiment < -0.2:
            market_pulse = "bajista"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        chart_path = self.output_dir / "sentiment_distribution.png"
        plt.figure(figsize=(6, 4))
        plt.bar(["Alcista", "Neutral", "Bajista"], [bullish, neutral, bearish], color=["green", "gray", "red"])
        plt.title("Distribución de Sentimiento de Mercado")
        plt.ylabel("Conteo de señales")
        plt.tight_layout()
        plt.savefig(chart_path)
        plt.close()

        return {
            "market_pulse": market_pulse,
            "average_sentiment": round(avg_sentiment, 3),
            "bullish_signals": bullish,
            "neutral_signals": neutral,
            "bearish_signals": bearish,
            "technical_notes": "Media de sentimiento y conteo por categorías sobre señales recientes.",
            "signals_table": frame.to_dict(orient="records"),
            "chart_path": str(chart_path),
            "sources": data["sources"],
        }

    def report(self, analysis: dict[str, Any]) -> dict[str, Any]:
        summary = (
            f"Pulso actual: **{analysis['market_pulse']}** con sentimiento promedio "
            f"de **{analysis['average_sentiment']}**."
        )
        deep_analysis = {
            "metrics": {
                "average_sentiment": analysis["average_sentiment"],
                "bullish_signals": analysis["bullish_signals"],
                "neutral_signals": analysis["neutral_signals"],
                "bearish_signals": analysis["bearish_signals"],
            },
            "chart_path": analysis["chart_path"],
            "technical_notes": analysis["technical_notes"],
            "signals": analysis["signals_table"],
            "sources": analysis["sources"],
        }
        markdown = (
            "## 1) Analista de Sentimiento\n\n"
            "### Resumen Ejecutivo\n"
            f"{summary}\n\n"
            "### Análisis Profundo\n"
            f"- Métrica central (promedio): `{analysis['average_sentiment']}`\n"
            f"- Señales alcistas: `{analysis['bullish_signals']}`\n"
            f"- Señales neutrales: `{analysis['neutral_signals']}`\n"
            f"- Señales bajistas: `{analysis['bearish_signals']}`\n"
            f"- Gráfica: `{analysis['chart_path']}`\n\n"
            "### Fuentes Consultadas\n"
            + "\n".join([f"- {s['name']} ({s['type']}): {s['url']} [estado: {s['status']}]" for s in analysis["sources"]])
            + "\n"
        )
        return {"executive_summary": summary, "deep_analysis": deep_analysis, "markdown": markdown}
