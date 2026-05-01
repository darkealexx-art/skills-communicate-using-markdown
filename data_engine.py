from __future__ import annotations

import csv
import itertools
import re
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

MIN_NUMBER = 1
MAX_NUMBER = 56
ERA_START = date(2007, 12, 1)

HISTORIAL_PROHIBIDO: set[Tuple[int, ...]] = set()

_NUMBER_HEADER_RE = re.compile(
    r"(n|num|numero|number|bola)\s*([1-6])", re.IGNORECASE
)


def load_and_preprocess(file_paths: Sequence[str | Path]) -> np.ndarray:
    draws: list[Tuple[int, ...]] = []
    for path in file_paths:
        draws.extend(_load_draws_from_csv(Path(path)))
    if not draws:
        raise ValueError("No se encontraron sorteos válidos en los CSVs.")
    draws_array = np.array(draws, dtype=np.int16)
    global HISTORIAL_PROHIBIDO
    HISTORIAL_PROHIBIDO = {tuple(draw.tolist()) for draw in draws_array}
    return draws_array


def compute_biometrics(numbers: Sequence[int]) -> Tuple[int, str, int, float]:
    ordered = sorted(int(n) for n in numbers)
    total_sum = int(sum(ordered))
    even_count = sum(n % 2 == 0 for n in ordered)
    odd_count = len(ordered) - even_count
    parity = f"{even_count}:{odd_count}"
    max_consecutive = _max_consecutive_run(ordered)
    avg_gap = float(np.mean(np.diff(ordered))) if len(ordered) > 1 else 0.0
    return total_sum, parity, max_consecutive, avg_gap


def passes_biometric_filters(numbers: Sequence[int]) -> bool:
    total_sum, parity, max_consecutive, avg_gap = compute_biometrics(numbers)
    if not 120 <= total_sum <= 210:
        return False
    if parity not in {"3:3", "4:2", "2:4"}:
        return False
    if max_consecutive > 2:
        return False
    if avg_gap <= 5.4:
        return False
    return True


def _load_draws_from_csv(path: Path) -> list[Tuple[int, ...]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV no encontrado: {path}")
    draws: list[Tuple[int, ...]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        first_row = next(reader, None)
        if first_row is None:
            return draws
        if _row_looks_like_header(first_row):
            headers = [cell.strip() for cell in first_row]
            rows = reader
        else:
            headers = None
            rows = itertools.chain([first_row], reader)

        number_columns = _infer_number_columns(headers)
        date_column = _infer_date_column(headers)

        for row in rows:
            if not row:
                continue
            if date_column is not None and date_column < len(row):
                parsed_date = _parse_date(row[date_column])
                if parsed_date and parsed_date < ERA_START:
                    continue
            numbers = _extract_numbers(row, number_columns)
            if numbers is None:
                continue
            if len(numbers) != 6:
                continue
            if any(n < MIN_NUMBER or n > MAX_NUMBER for n in numbers):
                continue
            draws.append(tuple(sorted(numbers)))
    return draws


def _row_looks_like_header(row: Sequence[str]) -> bool:
    for cell in row:
        if cell is None:
            continue
        cell = cell.strip()
        if not cell:
            continue
        if re.search(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]", cell):
            return True
    return False


def _infer_number_columns(headers: Optional[Sequence[str]]) -> Optional[list[int]]:
    if not headers:
        return None
    positions: dict[int, int] = {}
    for idx, header in enumerate(headers):
        match = _NUMBER_HEADER_RE.search(header)
        if match:
            positions[int(match.group(2))] = idx
    if len(positions) == 6:
        return [positions[i] for i in range(1, 7)]
    return None


def _infer_date_column(headers: Optional[Sequence[str]]) -> Optional[int]:
    if not headers:
        return None
    for idx, header in enumerate(headers):
        if "fecha" in header.lower() or "date" in header.lower():
            return idx
    return None


def _extract_numbers(
    row: Sequence[str], number_columns: Optional[Sequence[int]]
) -> Optional[list[int]]:
    if number_columns:
        numbers: list[int] = []
        for idx in number_columns:
            if idx >= len(row):
                return None
            value = _parse_int(row[idx])
            if value is None:
                return None
            numbers.append(value)
        return numbers
    numbers = []
    for cell in row:
        value = _parse_int(cell)
        if value is not None and MIN_NUMBER <= value <= MAX_NUMBER:
            numbers.append(value)
    if len(numbers) != 6:
        return None
    return numbers


def _parse_int(value: str) -> Optional[int]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return int(float(value))
        except ValueError:
            return None


def _parse_date(value: str) -> Optional[date]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def _max_consecutive_run(numbers: Sequence[int]) -> int:
    if not numbers:
        return 0
    max_run = 1
    current_run = 1
    for prev, curr in zip(numbers, numbers[1:]):
        if curr == prev + 1:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    return max_run
