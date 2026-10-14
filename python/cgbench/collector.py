"""Correctness-gated Williams scheduling and completed wall timing samples."""

import math
import time


def williams_rows(names):
    """Balance positions and within-row predecessors over a complete cycle.

    Even candidate counts need k rows; odd counts need both orientations (2k).
    The construction follows the design used in solvcon.benchmark.collector.
    Reference: E. J. Williams (1949), https://doi.org/10.1071/CH9490149.
    """
    count = len(names)
    if count < 2:
        return (tuple(names),)
    pattern = [
        ((position + 1) // 2 if position % 2 else count - position // 2) % count
        for position in range(count)
    ]
    base = [tuple((index + offset) % count for index in pattern) for offset in range(count)]
    if count % 2:
        forward = (count + 1) // 2
        reverse = count - forward
        rows = [base[(offset * forward) % count] for offset in range(count)]
        rows += [
            tuple(reversed(base[(offset * reverse) % count])) for offset in range(1, count + 1)
        ]
    else:
        rows = base
    return tuple(tuple(names[index] for index in row) for row in rows)


def time_candidates(executors, sampling, clock=time.perf_counter_ns):
    """Exclude scheduling/warmup from blocks; include every solve's x0 reset."""
    names = tuple(executors)
    rows = williams_rows(names)
    if sampling["rounds"] % len(rows):
        raise ValueError(f"rounds must be a multiple of the complete Williams cycle ({len(rows)})")
    for index in range(-sampling["warmups"], 0):
        for name in rows[index % len(rows)]:
            executors[name]()
    elapsed = {name: [] for name in names}
    orders = []
    for index in range(sampling["rounds"]):
        row = rows[index % len(rows)]
        orders.append(list(row))
        for name in row:
            execute = executors[name]
            synchronize = getattr(execute, "synchronize", None)
            if synchronize is not None:
                synchronize()
            start = clock()
            for _ in range(sampling["repetitions"]):
                execute()
            if synchronize is not None:
                synchronize()
            elapsed[name].append(int(clock() - start))
    return orders, elapsed


def check_outcome(outcome, arrays, options):
    """Independently validate the true residual before admitting timing data."""
    import numpy as np

    from .numerics import stable_l2

    matrix, rhs, _initial = arrays
    x = np.asarray(outcome.x)
    if x.shape != rhs.shape or x.dtype != rhs.dtype:
        return {"eligible": False, "reason": "solution shape or dtype mismatch"}
    if not np.all(np.isfinite(x)):
        return {"eligible": False, "reason": "solution contains non-finite values"}
    true_norm = stable_l2(rhs - matrix @ x)
    threshold = max(options["atol"], options["rtol"] * stable_l2(rhs))
    # Permit only the rounding-scale difference between equivalent norm sums.
    rounding = 1.0 + 64.0 * np.finfo(rhs.dtype).eps
    eligible = outcome.converged and math.isfinite(true_norm) and true_norm <= threshold * rounding
    return {
        "eligible": bool(eligible),
        "reason": None if eligible else "solver did not converge to the requested true residual",
        "solver_status": outcome.status,
        "iterations": int(outcome.iterations),
        "matvecs": int(outcome.matvecs),
        "true_residual_norm": true_norm if math.isfinite(true_norm) else None,
        "reported_residual_norm": outcome.residual_norm
        if math.isfinite(outcome.residual_norm)
        else None,
        "residual_threshold": threshold,
    }


def _check_executor_outcome(execute, outcome, arrays, options):
    # Explicit GPU downloads occur only at untimed eligibility gates.
    if hasattr(execute, "host_outcome"):
        outcome = execute.host_outcome(outcome)
    return check_outcome(outcome, arrays, options)


def collect_case(case_spec, run_spec):
    """Use exactly the case builder and solver adapters exercised by pytest."""
    from .backends import make_executor
    from .cases import input_hash, make_case

    matrix, rhs, initial, known_solution = make_case(case_spec, run_spec["seed"])
    arrays = matrix, rhs, initial
    exact_hash = input_hash(case_spec, (matrix, rhs, initial, known_solution))
    expected_hash = run_spec.get("expected_hashes", {}).get(case_spec.name)
    if expected_hash is not None and exact_hash != expected_hash:
        raise ValueError("replay input hash mismatch: " + case_spec.name)
    candidates = {}
    results = []
    options = run_spec["solver"]
    for name in run_spec["implementations"]:
        try:
            execute = make_executor(
                name,
                arrays,
                options,
                run_spec["scope"],
                run_spec["threads"],
                device=run_spec.get("device", "cpu"),
                cuda_device=run_spec.get("cuda_device", 0),
            )
        except ImportError as exc:
            if run_spec.get("device", "cpu") == "cuda":
                raise RuntimeError(f"Requested CUDA candidate {name} is unavailable") from exc
            results.append(
                {
                    "name": name,
                    "status": "ineligible",
                    "reason": str(exc),
                    "round_elapsed_ns": [],
                }
            )
            continue
        outcome = execute()
        correctness = _check_executor_outcome(execute, outcome, arrays, options)
        eligible = correctness.pop("eligible")
        entry = {
            "name": name,
            "status": "measured" if eligible else "invalid",
            **correctness,
            "round_elapsed_ns": [],
            "setup": execute.benchmark_metadata() if hasattr(execute, "benchmark_metadata") else {},
        }
        results.append(entry)
        if eligible:
            candidates[name] = execute
    # Missing optional candidates may shorten the design. Round upward to a
    # full cycle, and record the resolved sampling rather than a partial design.
    sampling = dict(run_spec["sampling"])
    expected_candidates = run_spec.get("expected_candidates", {}).get(case_spec.name)
    if expected_candidates is not None and list(candidates) != expected_candidates:
        raise ValueError("replay eligible candidates changed: " + case_spec.name)
    saved_sampling = run_spec.get("resolved_sampling", {}).get(case_spec.name)
    if saved_sampling is not None:
        sampling = dict(saved_sampling)
    cycle = len(williams_rows(tuple(candidates)))
    sampling["rounds"] = ((sampling["rounds"] + cycle - 1) // cycle) * cycle
    expected_orders = run_spec.get("expected_round_orders", {}).get(case_spec.name)
    if expected_orders is not None:
        rows = williams_rows(tuple(candidates))
        generated = [list(rows[index % cycle]) for index in range(sampling["rounds"])]
        if generated != expected_orders:
            raise ValueError("replay timing schedule changed: " + case_spec.name)
    orders, elapsed = time_candidates(candidates, sampling)
    # Revalidate after measurement too, catching state-reset regressions.
    for entry in results:
        name = entry["name"]
        if name in candidates:
            after = _check_executor_outcome(candidates[name], candidates[name](), arrays, options)
            if not after.pop("eligible"):
                entry.update(after)
                entry["status"] = "invalid"
                entry["reason"] = "post-measurement check: " + entry["reason"]
            else:
                entry["round_elapsed_ns"] = elapsed[name]
    return {
        "case": case_spec.to_dict(),
        "input_sha256": exact_hash,
        "dtype": "float64",
        "layout": "CSR" if case_spec.storage == "csr" else "C",
        "nnz": int(matrix.nnz) if case_spec.storage == "csr" else None,
        "input_stored_bytes": int(
            rhs.nbytes
            + initial.nbytes
            + (
                matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
                if case_spec.storage == "csr"
                else matrix.nbytes
            )
        ),
        "input_storage_basis": "shared input NumPy/SciPy arrays, including original index dtype; excludes native owning copies and workspace",
        "sampling": sampling,
        "schedule": "williams",
        "cycle_rows": cycle,
        "round_orders": orders,
        "results": results,
    }
