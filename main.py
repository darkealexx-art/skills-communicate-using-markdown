from __future__ import annotations

import json

from market_agents import MarketAnalysisOrchestrator


def run_markdown_example() -> None:
    orchestrator = MarketAnalysisOrchestrator(output_dir="outputs")
    result = orchestrator.run()
    print("Reporte Markdown generado en:", result["markdown_report_path"])
    print("Reporte JSON generado en:", result["json_report_path"])


def run_json_example() -> None:
    orchestrator = MarketAnalysisOrchestrator(output_dir="outputs")
    result = orchestrator.run()
    preview = {
        "generated_at_utc": result["combined_report"]["generated_at_utc"],
        "agent_keys": list(result["combined_report"]["agents"].keys()),
    }
    print("Vista resumida JSON:")
    print(json.dumps(preview, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # Ejemplo 1: ejecución completa y artefactos Markdown/JSON.
    run_markdown_example()

    # Ejemplo 2: consumo estructurado del resultado en JSON.
    run_json_example()