from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
from sklearn.cluster import KMeans
import umap

from data_engine import (
    HISTORIAL_PROHIBIDO,
    compute_biometrics,
    normalize_weights,
    passes_biometric_filters,
)

NUMBERS = np.arange(1, 57)
TRANSFORMER_WINDOW_SIZE = 50
HOT_COLD_WINDOW_SIZE = 100
MONTECARLO_ITERATIONS = 1_000_000
MAX_ATTEMPTS_MULTIPLIER = 500
MIN_DEGREE_SUM = 1.0
FITNESS_PENALTY_VALID = 1.0
FITNESS_PENALTY_INVALID = 0.3


def generate_all_models(
    draws: np.ndarray,
    forbidden: set[tuple[int, ...]] | None = None,
    seed: int = 42,
) -> list[dict]:
    if forbidden is None:
        forbidden = HISTORIAL_PROHIBIDO
    rng = np.random.default_rng(seed)
    frequency = _frequency_vector(draws)
    one_hot = _one_hot_draws(draws)

    results: list[dict] = []
    results.extend(
        _build_model_results(
            "Transformer",
            transformer_model(draws, one_hot, rng, forbidden),
        )
    )
    results.extend(
        _build_model_results(
            "LSTM",
            lstm_model(one_hot, rng, forbidden),
        )
    )
    results.extend(
        _build_model_results(
            "Montecarlo",
            montecarlo_model(frequency, rng, forbidden),
        )
    )
    results.extend(
        _build_model_results(
            "Bayesiano",
            bayesian_model(frequency, rng, forbidden),
        )
    )
    results.extend(
        _build_model_results(
            "Entropia",
            shannon_entropy_model(frequency, rng, forbidden),
        )
    )
    results.extend(
        _build_model_results(
            "Grafos",
            graph_centrality_model(draws, rng, forbidden),
        )
    )
    results.extend(
        _build_model_results(
            "Genetico",
            genetic_algorithm_model(frequency, rng, forbidden),
        )
    )
    results.extend(
        _build_model_results(
            "Rezago",
            hot_cold_model(draws, frequency, rng, forbidden),
        )
    )
    results.extend(
        _build_model_results(
            "MeanReversion",
            mean_reversion_model(draws, frequency, rng, forbidden),
        )
    )
    results.extend(
        _build_model_results(
            "GNN",
            gnn_model(draws, rng, forbidden),
        )
    )
    results.extend(
        _build_model_results(
            "UMAP",
            umap_model(one_hot, frequency, rng, forbidden),
        )
    )
    return results


def transformer_model(
    draws: np.ndarray,
    one_hot: np.ndarray,
    rng: np.random.Generator,
    forbidden: set[tuple[int, ...]],
) -> list[tuple[np.ndarray, float]]:
    window = (
        one_hot[-TRANSFORMER_WINDOW_SIZE:]
        if len(one_hot) > TRANSFORMER_WINDOW_SIZE
        else one_hot
    )
    if len(window) == 0:
        weights = np.full(56, 1 / 56)
    else:
        torch.manual_seed(int(rng.integers(0, 1_000_000)))
        attn = torch.nn.MultiheadAttention(
            embed_dim=56, num_heads=4, batch_first=True
        )
        with torch.no_grad():
            tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
            output, _ = attn(tensor, tensor, tensor)
            scores = output[0, -1, :]
            weights = torch.softmax(scores, dim=0).cpu().numpy()
    return _generate_sequences_from_weights(weights, 10, rng, forbidden)


def lstm_model(
    one_hot: np.ndarray,
    rng: np.random.Generator,
    forbidden: set[tuple[int, ...]],
) -> list[tuple[np.ndarray, float]]:
    if len(one_hot) == 0:
        weights = np.full(56, 1 / 56)
    else:
        torch.manual_seed(int(rng.integers(0, 1_000_000)))
        lstm = torch.nn.LSTM(
            input_size=56, hidden_size=64, num_layers=1, batch_first=True
        )
        projection = torch.nn.Linear(64, 56)
        with torch.no_grad():
            tensor = torch.tensor(one_hot, dtype=torch.float32).unsqueeze(0)
            output, _ = lstm(tensor)
            scores = projection(output[0, -1, :])
            weights = torch.softmax(scores, dim=0).cpu().numpy()
    return _generate_sequences_from_weights(weights, 10, rng, forbidden)


def montecarlo_model(
    frequency: np.ndarray,
    rng: np.random.Generator,
    forbidden: set[tuple[int, ...]],
    iterations: int = MONTECARLO_ITERATIONS,
) -> list[tuple[np.ndarray, float]]:
    weights = normalize_weights(frequency)
    candidates: list[tuple[np.ndarray, float]] = []
    seen: set[tuple[int, ...]] = set()
    chunk = 10_000
    for start in range(0, iterations, chunk):
        size = min(chunk, iterations - start)
        random_matrix = rng.random((size, 56))
        idx = np.argsort(random_matrix, axis=1)[:, :6]
        numbers = np.sort(idx + 1, axis=1)
        scores = weights[idx].sum(axis=1)
        order = np.argsort(scores)[::-1]
        for row in order:
            combo = tuple(numbers[row].tolist())
            if combo in forbidden or combo in seen:
                continue
            if not passes_biometric_filters(numbers[row]):
                continue
            candidates.append((numbers[row], float(scores[row])))
            seen.add(combo)
            if len(candidates) >= 10:
                return candidates
    return candidates


def bayesian_model(
    frequency: np.ndarray,
    rng: np.random.Generator,
    forbidden: set[tuple[int, ...]],
) -> list[tuple[np.ndarray, float]]:
    alpha = frequency + 1.0
    weights = rng.dirichlet(alpha)
    return _generate_sequences_from_weights(weights, 10, rng, forbidden)


def shannon_entropy_model(
    frequency: np.ndarray,
    rng: np.random.Generator,
    forbidden: set[tuple[int, ...]],
) -> list[tuple[np.ndarray, float]]:
    weights = 1.0 / (frequency + 1.0)
    weights = normalize_weights(weights)
    return _generate_sequences_from_weights(weights, 10, rng, forbidden)


def graph_centrality_model(
    draws: np.ndarray,
    rng: np.random.Generator,
    forbidden: set[tuple[int, ...]],
) -> list[tuple[np.ndarray, float]]:
    adjacency = _cooccurrence_matrix(draws)
    if np.all(adjacency == 0):
        weights = np.full(56, 1 / 56)
    else:
        eigenvalues, eigenvectors = np.linalg.eig(adjacency)
        principal = eigenvectors[:, np.argmax(np.real(eigenvalues))]
        weights = np.abs(np.real(principal))
        weights = normalize_weights(weights)
    return _generate_sequences_from_weights(weights, 10, rng, forbidden)


def genetic_algorithm_model(
    frequency: np.ndarray,
    rng: np.random.Generator,
    forbidden: set[tuple[int, ...]],
) -> list[tuple[np.ndarray, float]]:
    weights = normalize_weights(frequency)
    return _genetic_algorithm(weights, rng, forbidden)


def hot_cold_model(
    draws: np.ndarray,
    frequency: np.ndarray,
    rng: np.random.Generator,
    forbidden: set[tuple[int, ...]],
    window: int = HOT_COLD_WINDOW_SIZE,
) -> list[tuple[np.ndarray, float]]:
    recent = draws[-window:] if len(draws) > window else draws
    recent_counts = _frequency_vector(recent)
    hot = normalize_weights(recent_counts)
    cold = normalize_weights(1.0 / (frequency + 1.0))
    weights = normalize_weights(0.6 * hot + 0.4 * cold)
    return _generate_sequences_from_weights(weights, 10, rng, forbidden)


def mean_reversion_model(
    draws: np.ndarray,
    frequency: np.ndarray,
    rng: np.random.Generator,
    forbidden: set[tuple[int, ...]],
) -> list[tuple[np.ndarray, float]]:
    expected = (len(draws) * 6) / 56 if len(draws) else 1.0
    deviation = expected - frequency
    weights = normalize_weights(np.clip(deviation, 0, None))
    return _generate_sequences_from_weights(weights, 10, rng, forbidden)


def gnn_model(
    draws: np.ndarray,
    rng: np.random.Generator,
    forbidden: set[tuple[int, ...]],
) -> list[tuple[np.ndarray, float]]:
    adjacency = _cooccurrence_matrix(draws)
    if np.all(adjacency == 0):
        weights = np.full(56, 1 / 56)
    else:
        torch.manual_seed(int(rng.integers(0, 1_000_000)))
        adj = torch.tensor(adjacency, dtype=torch.float32)
        adj = adj + torch.eye(56)
        degree_inv = torch.diag(
            1.0 / torch.clamp(adj.sum(dim=1), min=MIN_DEGREE_SUM)
        )
        norm_adj = degree_inv @ adj
        features = torch.eye(56)
        weight = torch.randn(56, 16)
        embeddings = torch.relu(norm_adj @ features @ weight)
        scores = torch.norm(embeddings, dim=1)
        weights = torch.softmax(scores, dim=0).cpu().numpy()
    return _generate_sequences_from_weights(weights, 10, rng, forbidden)


def umap_model(
    one_hot: np.ndarray,
    frequency: np.ndarray,
    rng: np.random.Generator,
    forbidden: set[tuple[int, ...]],
) -> list[tuple[np.ndarray, float]]:
    if len(one_hot) < 5:
        weights = normalize_weights(frequency)
        return _generate_sequences_from_weights(weights, 10, rng, forbidden)
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=int(rng.integers(0, 1_000_000)),
    )
    embedding = reducer.fit_transform(one_hot)
    n_clusters = min(10, len(one_hot))
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=10,
        random_state=int(rng.integers(0, 1_000_000)),
    )
    labels = kmeans.fit_predict(embedding)
    sequences: list[tuple[np.ndarray, float]] = []
    seen: set[tuple[int, ...]] = set()
    for cluster_id in range(n_clusters):
        cluster_rows = one_hot[labels == cluster_id]
        if len(cluster_rows) == 0:
            continue
        cluster_freq = cluster_rows.mean(axis=0)
        top_indices = np.argsort(cluster_freq)[-12:]
        weights = normalize_weights(cluster_freq[top_indices])
        for _ in range(10):
            numbers = rng.choice(top_indices + 1, size=6, replace=False, p=weights)
            numbers.sort()
            combo = tuple(numbers.tolist())
            if combo in forbidden or combo in seen:
                continue
            if not passes_biometric_filters(numbers):
                continue
            score = float(cluster_freq[numbers - 1].sum())
            sequences.append((numbers, score))
            seen.add(combo)
            if len(sequences) >= 10:
                return sequences
    if len(sequences) < 10:
        weights = normalize_weights(frequency)
        sequences.extend(
            _generate_sequences_from_weights(
                weights, 10 - len(sequences), rng, forbidden, seen
            )
        )
    return sequences


def _build_model_results(
    model_name: str, sequences: Iterable[tuple[np.ndarray, float]]
) -> list[dict]:
    results: list[dict] = []
    for idx, (numbers, score) in enumerate(sequences, start=1):
        total_sum, parity, _, _ = compute_biometrics(numbers)
        results.append(
            {
                "Modelo": model_name,
                "ID": idx,
                "N1": int(numbers[0]),
                "N2": int(numbers[1]),
                "N3": int(numbers[2]),
                "N4": int(numbers[3]),
                "N5": int(numbers[4]),
                "N6": int(numbers[5]),
                "Suma": total_sum,
                "Paridad": parity,
                "Score": float(score),
            }
        )
    return results


def _generate_sequences_from_weights(
    weights: np.ndarray,
    count: int,
    rng: np.random.Generator,
    forbidden: set[tuple[int, ...]],
    extra_seen: set[tuple[int, ...]] | None = None,
) -> list[tuple[np.ndarray, float]]:
    weights = normalize_weights(weights)
    sequences: list[tuple[np.ndarray, float]] = []
    seen: set[tuple[int, ...]] = set()
    if extra_seen:
        seen.update(extra_seen)
    attempts = 0
    while len(sequences) < count and attempts < count * MAX_ATTEMPTS_MULTIPLIER:
        attempts += 1
        numbers = rng.choice(NUMBERS, size=6, replace=False, p=weights)
        numbers.sort()
        combo = tuple(numbers.tolist())
        if combo in forbidden or combo in seen:
            continue
        if not passes_biometric_filters(numbers):
            continue
        score = float(weights[numbers - 1].sum())
        sequences.append((numbers, score))
        seen.add(combo)
    return sequences


def _frequency_vector(draws: np.ndarray) -> np.ndarray:
    if draws.size == 0:
        return np.ones(56)
    counts = np.bincount(draws.flatten(), minlength=57)[1:]
    return counts.astype(float)


def _one_hot_draws(draws: np.ndarray) -> np.ndarray:
    if draws.size == 0:
        return np.empty((0, 56), dtype=np.float32)
    one_hot = np.zeros((len(draws), 56), dtype=np.float32)
    rows = np.arange(len(draws))[:, None]
    one_hot[rows, draws - 1] = 1.0
    return one_hot


def _cooccurrence_matrix(draws: np.ndarray) -> np.ndarray:
    matrix = np.zeros((56, 56), dtype=float)
    for draw in draws:
        indices = draw - 1
        matrix[np.ix_(indices, indices)] += 1.0
    np.fill_diagonal(matrix, 0.0)
    return matrix


def _genetic_algorithm(
    weights: np.ndarray,
    rng: np.random.Generator,
    forbidden: set[tuple[int, ...]],
    population_size: int = 200,
    generations: int = 40,
    mutation_rate: float = 0.1,
) -> list[tuple[np.ndarray, float]]:
    population = _random_population(population_size, rng, weights)
    for _ in range(generations):
        fitness = _fitness(population, weights)
        elite_count = max(10, int(population_size * 0.3))
        elite_idx = np.argsort(fitness)[-elite_count:]
        elites = population[elite_idx]
        children = []
        while len(children) < population_size - elite_count:
            parents = rng.choice(elite_count, size=2, replace=False)
            child = _crossover(elites[parents[0]], elites[parents[1]], rng)
            child = _mutate(child, rng, mutation_rate, weights)
            children.append(child)
        population = np.vstack([elites, np.array(children)])
    fitness = _fitness(population, weights)
    sorted_idx = np.argsort(fitness)[::-1]
    sequences: list[tuple[np.ndarray, float]] = []
    seen: set[tuple[int, ...]] = set()
    for idx in sorted_idx:
        numbers = np.sort(population[idx])
        combo = tuple(numbers.tolist())
        if combo in forbidden or combo in seen:
            continue
        if not passes_biometric_filters(numbers):
            continue
        score = float(weights[numbers - 1].sum())
        sequences.append((numbers, score))
        seen.add(combo)
        if len(sequences) >= 10:
            break
    return sequences


def _random_population(
    population_size: int, rng: np.random.Generator, weights: np.ndarray
) -> np.ndarray:
    population = []
    for _ in range(population_size):
        numbers = rng.choice(NUMBERS, size=6, replace=False, p=weights)
        population.append(numbers)
    return np.array(population, dtype=int)


def _fitness(population: np.ndarray, weights: np.ndarray) -> np.ndarray:
    scores = weights[population - 1].sum(axis=1)
    penalties = np.array(
        [
            FITNESS_PENALTY_VALID
            if passes_biometric_filters(row)
            else FITNESS_PENALTY_INVALID
            for row in population
        ]
    )
    return scores * penalties


def _crossover(
    parent_a: np.ndarray, parent_b: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    pool = np.unique(np.concatenate([parent_a, parent_b]))
    if len(pool) >= 6:
        child = rng.choice(pool, size=6, replace=False)
    else:
        remaining = np.setdiff1d(NUMBERS, pool)
        fill = rng.choice(remaining, size=6 - len(pool), replace=False)
        child = np.concatenate([pool, fill])
    return child


def _mutate(
    child: np.ndarray,
    rng: np.random.Generator,
    mutation_rate: float,
    weights: np.ndarray,
) -> np.ndarray:
    child = child.copy()
    if rng.random() < mutation_rate:
        idx = int(rng.integers(0, 6))
        available = np.setdiff1d(NUMBERS, child)
        if len(available):
            child[idx] = rng.choice(available)
    return child
