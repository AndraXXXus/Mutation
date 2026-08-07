import gc
import json
import itertools
from collections import Counter
from functools import lru_cache
import numpy as np
import spot


def extract_atomic_propositions(*formulas: str) -> tuple[str, ...]:
    aps = set()
    for f in formulas:
        try:
            parsed = spot.formula(f)
            for ap in spot.atomic_prop_collect(parsed):
                aps.add(ap.ap_name())
        except Exception:
            continue
    return tuple(sorted(list(aps)))


def generate_trace_stream(aps: tuple[str, ...], k: int):
    if not aps or k <= 0:
        yield ()
        return

    step_valuations = []
    for truth_values in itertools.product([True, False], repeat=len(aps)):
        literals = [ap if is_true else f"!{ap}" for ap, is_true in zip(aps, truth_values)]
        step_valuations.append(" & ".join(literals))

    for trace in itertools.product(step_valuations, repeat=k):
        yield trace


@lru_cache(maxsize=1024)
def get_satisfying_traces_cached(formula_str: str, aps_tuple: tuple[str, ...], k: int) -> set[tuple[str, ...]]:
    if (2 ** (len(aps_tuple) * k)) > 100_000:
        return set()

    try:
        aut = spot.translate(spot.from_ltlf(formula_str))
        dict_ptr = aut.get_dict()
    except Exception:
        return set()

    satisfying_traces = set()

    for i, trace in enumerate(generate_trace_stream(aps_tuple, k)):
        if trace:
            prefix = " ; ".join(f"({step}) & alive" for step in trace)
            word_str = f"{prefix} ; cycle{{!alive}}"
        else:
            word_str = "cycle{!alive}"

        word = spot.parse_word(word_str, dict_ptr)
        word_aut = word.as_automaton()

        if not spot.product(aut, word_aut).is_empty():
            satisfying_traces.add(trace)

        del word, word_aut
        if i % 5000 == 0 and i > 0:
            gc.collect()

    del aut, dict_ptr
    gc.collect()
    return satisfying_traces


def compute_pairwise_similarity_matrix(unique_formulas: list[str], k: int) -> list[list[float]]:
    n = len(unique_formulas)
    if n <= 1:
        return [[1.0]]

    aps_tuple = extract_atomic_propositions(*unique_formulas)
    matrix = [[1.0] * n for _ in range(n)]

    trace_sets = [get_satisfying_traces_cached(f, aps_tuple, k) for f in unique_formulas]

    for i in range(n):
        for j in range(i + 1, n):
            t1, t2 = trace_sets[i], trace_sets[j]
            union_len = len(t1.union(t2))
            sim = 1.0 if union_len == 0 else len(t1.intersection(t2)) / union_len
            matrix[i][j] = sim
            matrix[j][i] = sim

    return matrix


def compute_ambiguity_from_matrix(probabilities: list[float], similarity_matrix: list[list[float]]) -> float:
    p = np.array(probabilities, dtype=float)
    s = np.array(similarity_matrix, dtype=float)

    if len(p) <= 1 or np.isclose(np.sum(p**2), 1.0):
        return 0.0

    p_nz = p[p > 0]
    H = -np.sum(p_nz * np.log(p_nz))
    uncertainty_term = 1.0 - np.exp(-H)

    numerator = np.sum(np.outer(p, p) * (1.0 - s))
    denominator = 1.0 - np.sum(p**2)

    S_norm = 0.0 if denominator == 0 else numerator / denominator
    return float(uncertainty_term * S_norm)


def calculate_row_ambiguity(cleaned_output_data, k: int = 3) -> float:
    data = cleaned_output_data

    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return 0.0

    if not isinstance(data, list) or len(data) == 0:
        return 0.0

    responses = [item.get("Response") for item in data if isinstance(item, dict) and item.get("Response")]

    if not responses:
        return 0.0

    counts = Counter(responses)
    unique_responses = list(counts.keys())

    if len(unique_responses) <= 1:
        return 0.0

    total_responses = len(responses)
    probabilities = [count / total_responses for count in counts.values()]

    sim_matrix = compute_pairwise_similarity_matrix(unique_responses, k=k)
    return compute_ambiguity_from_matrix(probabilities, sim_matrix)



import os
import importlib
import ltl_worker
from joblib import Parallel, delayed
from tqdm import tqdm

# Force reload module in case you edited Cell 1
importlib.reload(ltl_worker)

horizon = 3
num_cores = max(1, os.cpu_count() - 1)
print(f"Parallelizing execution across {num_cores} CPU cores...")

cleaned_outputs = df["cleaned_result_output"].tolist()

results = Parallel(n_jobs=num_cores, batch_size=10)(
    delayed(ltl_worker.calculate_row_ambiguity)(output, k=horizon) 
    for output in tqdm(cleaned_outputs, desc="Parallel LTL Ambiguity")
)

df["ltl_ambiguity_score"] = results
