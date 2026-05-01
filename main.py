from __future__ import annotations

import csv
from pathlib import Path

from cross_analysis import generate_consensus_sequences
from data_engine import HISTORIAL_PROHIBIDO, load_and_preprocess
from models_engine import generate_all_models


def run_pipeline() -> Path:
    base_dir = Path(__file__).resolve().parent
    file_paths = [
        base_dir / "Melate (1).csv",
        base_dir / "Revancha.csv",
        base_dir / "Revanchita.csv",
    ]
    draws = load_and_preprocess(file_paths)
    model_results = generate_all_models(draws, HISTORIAL_PROHIBIDO)
    consensus_results = generate_consensus_sequences(
        model_results, HISTORIAL_PROHIBIDO
    )
    results = model_results + consensus_results

    output_path = base_dir / "resultados_melate_ai.csv"
    fieldnames = [
        "Modelo",
        "ID",
        "N1",
        "N2",
        "N3",
        "N4",
        "N5",
        "N6",
        "Suma",
        "Paridad",
        "Score",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row[key] for key in fieldnames})
    return output_path


if __name__ == "__main__":
    run_pipeline()
