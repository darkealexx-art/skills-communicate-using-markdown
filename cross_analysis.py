from __future__ import annotations

from typing import Iterable

import numpy as np

from data_engine import (
    HISTORIAL_PROHIBIDO,
    compute_biometrics,
    normalize_weights,
    passes_biometric_filters,
)

CONSENSUS_POOL_SIZE = 20
CONSENSUS_TARGET = 10
CONSENSUS_MAX_ATTEMPTS = 5000


def generate_consensus_sequences(
    model_results: Iterable[dict],
    forbidden: set[tuple[int, ...]] | None = None,
    seed: int = 123,
) -> list[dict]:
    if forbidden is None:
        forbidden = HISTORIAL_PROHIBIDO
    counts = np.zeros(56, dtype=float)
    for result in model_results:
        numbers = np.array([result[f"N{i}"] for i in range(1, 7)], dtype=int)
        counts[numbers - 1] += 1
    weights = normalize_weights(counts)
    if np.all(counts == 0):
        top_indices = np.arange(len(weights))
    else:
        top_indices = np.argsort(weights)[-CONSENSUS_POOL_SIZE:]
    choice_pool = top_indices + 1
    restricted_weights = normalize_weights(weights[top_indices])
    rng = np.random.default_rng(seed)

    sequences = []
    seen: set[tuple[int, ...]] = set()
    attempts = 0
    while len(sequences) < CONSENSUS_TARGET and attempts < CONSENSUS_MAX_ATTEMPTS:
        attempts += 1
        numbers = rng.choice(
            choice_pool, size=6, replace=False, p=restricted_weights
        )
        numbers.sort()
        combo = tuple(numbers.tolist())
        if combo in forbidden or combo in seen:
            continue
        if not passes_biometric_filters(numbers):
            continue
        score = float(weights[numbers - 1].sum())
        total_sum, parity, _, _ = compute_biometrics(numbers)
        sequences.append(
            {
                "Modelo": "Consenso",
                "ID": len(sequences) + 1,
                "N1": int(numbers[0]),
                "N2": int(numbers[1]),
                "N3": int(numbers[2]),
                "N4": int(numbers[3]),
                "N5": int(numbers[4]),
                "N6": int(numbers[5]),
                "Suma": total_sum,
                "Paridad": parity,
                "Score": score,
            }
        )
        seen.add(combo)
    return sequences
