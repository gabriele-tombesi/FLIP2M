# Copyright (c) 2011-2025 Columbia University, System Level Design Group
# SPDX-License-Identifier: Apache-2.0

from test_FLIP2M import *

# ----------------------------------------------------------
# Helper functions
# ----------------------------------------------------------
class ProgressCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self, bound_tol=0.01):
        super().__init__()
        self._start = time.time()
        self._best = None
        self._bound_tol = bound_tol
    def OnSolutionCallback(self):
        t = time.time() - self._start
        obj = self.ObjectiveValue()
        bound = self.BestObjectiveBound()
        self._best = obj

        print(f"[{t:.2f}s] New incumbent = {obj}, bound = {bound}")

        if bound != cp_model.INT_MAX and (obj - bound) / max(1, abs(bound)) < self._bound_tol:
            print(f"  Converged within {self._bound_tol*100:.1f}% of bound; stopping early.")
            self.StopSearch()


def compute_cp_metrics(all_chains, p_cost):
    energy = 0
    edp = 0
    off_chip_accesses=0
    for chain in all_chains:
        _, steps = chain
        for step in steps:
            multiplier = (step['resources'] * (p_cost["acc"]+p_cost["router"]) + step['mem_tiles'] * p_cost["mem"])
            energy += multiplier * step['latency']
            edp += multiplier * (step['latency'] ** 2)
            off_chip_accesses += step['off_chip_accesses']

    res_metrics = {"energy":energy, "edp":edp, "accesses": off_chip_accesses}
    return res_metrics


def dp_segment_cost(mode, p_cost, cost_objective):
    if cost_objective == "energy":
        return  int(( mode['resources'] * (p_cost["acc"]+p_cost["router"]) + mode['mem_tiles'] * p_cost["mem"]) * mode['latency'])
    elif cost_objective == "EDP":
        return int(( mode['resources'] * (p_cost["acc"]+p_cost["router"]) + mode['mem_tiles'] * p_cost["mem"]) * mode['latency'] * mode['latency'])
    else:
        return mode['latency']


def build_events_and_actions(all_chains):
    events = []
    for (tenant_id, chain) in all_chains:
        for seg in chain:
            if seg.get('mode_index') == 'dummy':
                continue
            st = seg['start_time']
            ft = seg['finish_time']
            seg_idx = seg['chain_idx']
            events.append( (st, 'start', tenant_id, seg_idx) )
            events.append( (ft, 'end', tenant_id, seg_idx) )
    def sort_key(ev):
        type_order = 0 if ev[1] == 'end' else 1
        return (ev[0], type_order)
    events.sort(key=sort_key)
    action_list_str = []
    action_list_num = []
    for (tval, ev_type, tenant, seg_idx) in events:
        if ev_type == 'start':
            action_list_str.append(f"start_tenant_{tenant}")
            action_list_num.append(5)
            action_list_num.append(tenant)
        else:
            action_list_str.append(f"wait_tenant_{tenant}")
            action_list_num.append(6)
            action_list_num.append(tenant)
    return events, action_list_str, action_list_num


def build_filtered_output(all_chains, sequences):
    filtered_output = {}
    for i, seq in enumerate(sequences):
        filtered_output[i] = {"length": seq["length"], "modes": {}}
    for (tenant, chain) in all_chains:
        for seg in chain:
            if seg.get('mode_index') == 'dummy':
                continue
            mode_info = {
                'latency': seg['latency'],
                'length': seg['end_index'] - seg['start_index'],
                'resources': seg['resources'],
                'mem_tiles': seg['mem_tiles'],
                'mapping': seg['mapping'],
                # 'off_chip_accesses': seg['off_chip_accesses']
            }
            if 'layers_in' in seg:
                mode_info['layers_in'] = seg['layers_in']
            filtered_output[tenant]["modes"][seg['start_index']] = [mode_info]
    return filtered_output


def collect_segmentation_statistics(solved_epoch):
  segs = np.zeros(3)
  for tenant, mapping in solved_epoch.items():
    for segment, segment_mapping in mapping['modes'].items():
      segs[segment_mapping[0]['length']-1]+=1

  return segs

# ----------------------------------------------------------
# DP Engine
# ----------------------------------------------------------
def compute_optimal_segmentation(L, segments_dict, p_cost, cost_objective):
    """
    Compute the optimal segmentation for a single network of L layers, under the chosen cost objective.
    :param L: Number of layers (indexed 0..L-1).
    :param segments_dict: dict where key = layer index, value = list of segments
                          (each with 'length', 'latency', 'resources', etc.).
    :param cost_objective: "latency" or "area-latency"
    :return: minimal total cost and the chosen segmentation path.
    """
    dp = [float('inf')] * (L + 1)
    dp_energy = [float('inf')] * (L + 1)
    dp_EDP = [float('inf')] * (L + 1)
    dp_acc = [float('inf')] * (L + 1)
    chosen_segment = [None] * (L + 1)
    dp[L] = 0
    dp_energy[L] = 0
    dp_EDP[L] = 0
    dp_acc[L] = 0
    for i in range(L - 1, -1, -1):
        possible_segments = segments_dict.get(i, [])
        for seg in possible_segments:
            length = seg['length']
            # compute cost depending on cost_objective
            cost_seg = dp_segment_cost(seg, p_cost, cost_objective)
            cost_seg_energy = dp_segment_cost(seg, p_cost, 'energy')
            cost_seg_EDP = dp_segment_cost(seg, p_cost, 'EDP')
            cost_seg_acc= (seg['off_chip_accesses'] * seg['batch'] *  seg['mem_tiles'])
            next_layer = i + length
            if next_layer <= L:
                cost = cost_seg + dp[next_layer]
                cost_energy = cost_seg_energy + dp_energy[next_layer]
                cost_EDP = cost_seg_EDP + dp_EDP[next_layer]
                cost_acc = cost_seg_acc + dp_acc[next_layer]
                if cost < dp[i]:
                    dp[i] = cost
                    dp_energy[i]=cost_energy
                    dp_EDP[i]=cost_EDP
                    dp_acc[i]=cost_acc
                    chosen_segment[i] = seg
    segmentation = []
    idx = 0
    while idx < L:
        seg = chosen_segment[idx]
        if seg is None:
            break
        segmentation.append((idx, seg))
        idx += seg['length']

    res_metrics={"energy":dp_energy[0], "edp":dp_EDP[0], "accesses": dp_acc[0]}
    return dp[0],res_metrics,segmentation

# ----------------------------------------------------------
# DP Solver
# ----------------------------------------------------------
def dp_solver(seq, tenant_id, p_cost, cost_objective):
    """
    Compute an optimal segmentation and execution chain for a single sequence via dynamic programming.

    Parameters:
        seq (dict): Sequence specification, with keys:
            - 'length': int, total number of layers
            - 'modes': dict mapping layer index to list of mode dicts
        tenant_id (int): Identifier for this sequence’s tenant/network.
        p_cost (dict): Per-unit costs (keys: 'acc', 'router', 'mem').
        cost_objective (str): Which metric to minimize ('latency', 'energy', 'EDP', etc.).

    Returns:
        best_cost (float): Objective value of the optimal segmentation.
        res_metrics (dict): Detailed metrics from `compute_optimal_segmentation`.
        optimal_segmentation (List[Tuple[int, dict]]):
            List of (start_layer, mode_info) tuples defining the segmentation.
        dp_chain (List[dict]): Execution chain with timing/resource details.
    """
    length = seq["length"]
    # Prune modes: keep only the best segment per supported depth at each layer
    segments_by_layer = {}
    for layer_idx in range(length):
        modes_at_layer = seq.get("modes", {}).get(layer_idx)
        if not modes_at_layer:
            continue

        best_modes = {}
        for mode in modes_at_layer:
            seg_len = mode["length"]
            # Only consider depths 1, 2, or 3
            if seg_len not in {1, 2, 3}:
                continue
            # Ensure segment fits
            if layer_idx + seg_len > length:
                continue

            existing = best_modes.get(seg_len)
            if existing:
                curr_cost = dp_segment_cost(mode, p_cost, cost_objective)
                exist_cost = dp_segment_cost(existing, p_cost, cost_objective)
                if curr_cost < exist_cost:
                    best_modes[seg_len] = mode
            else:
                best_modes[seg_len] = mode

        if best_modes:
            segments_by_layer[layer_idx] = list(best_modes.values())

    # Compute optimal segmentation via DP
    best_cost, res_metrics, optimal_segmentation = compute_optimal_segmentation(
        length,
        segments_by_layer,
        p_cost,
        cost_objective
    )

    # (Optional) Recompute individual segment costs if needed for metrics
    for start_layer, mode_info in optimal_segmentation:
        _ = dp_segment_cost(mode_info, p_cost, cost_objective)

    # Build the DP chain with back-to-back timing
    dp_chain = []
    current_time = 0
    for seg_idx, (start_layer, mode_info) in enumerate(optimal_segmentation):
        start_time = current_time
        finish_time = start_time + mode_info["latency"]
        current_time = finish_time

        dp_chain.append({
            "tenant": tenant_id,
            "chain_idx": seg_idx,
            "start_index": start_layer,
            "end_index": start_layer + mode_info["length"],
            "start_time": start_time,
            "finish_time": finish_time,
            "latency": mode_info["latency"],
            "resources": mode_info.get("resources", 0),
            "mem_tiles": mode_info.get("mem_tiles", 0),
            "mode_index": mode_info.get("mode_index"),
            "length": mode_info.get("length", 0),
            "mapping": mode_info.get("mapping", 0),
        })

    return best_cost, res_metrics, optimal_segmentation, dp_chain


# ----------------------------------------------------------
# CP Solver
# ----------------------------------------------------------
def cp_solver(sequences, total_resources, total_mem_tiles, p_cost, cost_objective, cost_lower_bound, horizon=1000):

    model = cp_model.CpModel()

    # Identify empty networks
    network_status = {}
    for i, seq in enumerate(sequences):
        network_status[i] = (seq["length"] == 0)

    candidates = {}
    candidate_list = []
    candidate_id = 0
    for i, seq in enumerate(sequences):
        if network_status[i]:
            candidates[i] = []
        else:
            L = seq["length"]
            candidates[i] = []
            for j in range(L):
                if j in seq["modes"]:
                    for m_idx, mode in enumerate(seq["modes"][j]):
                        seg_length = mode['length']
                        if j + seg_length <= L:
                            cdict = {
                                'seq': i,
                                'start_index': j,
                                'end_index': j + seg_length,
                                'resources': mode['resources'],
                                'latency': mode['latency'],
                                'mode_index': m_idx,
                                'candidate_id': candidate_id,
                                'mem_tiles': mode.get('mem_tiles', 0),
                                'length': mode.get('length', 0),
                                'mapping': mode.get('mapping', 0),
                                'off_chip_accesses': mode.get('off_chip_accesses', 0)
                            }
                            candidates[i].append(cdict)
                            candidate_list.append(cdict)
                            candidate_id += 1
    # Add dummy candidates
    for i, seq in enumerate(sequences):
        if not network_status[i]:
            L = seq["length"]
            starting_indices = {c['start_index'] for c in candidates[i]}
            reached_nodes = {c['end_index'] for c in candidates[i]}
            for j in reached_nodes:
                if j < L and j not in starting_indices:
                    cdict = {
                        'seq': i,
                        'start_index': j,
                        'end_index': j + 1,
                        'resources': 0,
                        'latency': 0,
                        'mode_index': 'dummy',
                        'candidate_id': candidate_id,
                        'mem_tiles': 0,
                        'length': 0,
                        'mapping': 0,
                        'off_chip_accesses': 0
                    }
                    candidates[i].append(cdict)
                    candidate_list.append(cdict)
                    candidate_id += 1

    for c in candidate_list:
        cid = c['candidate_id']
        c['selected'] = model.NewBoolVar(f"select_{cid}")
        c['start_time'] = model.NewIntVar(0, horizon, f"start_{cid}")
        c['interval'] = model.NewOptionalIntervalVar(
            c['start_time'],
            c['latency'],
            c['start_time'] + c['latency'],
            c['selected'],
            f"interval_{cid}"
        )

    # Flow constraints
    for i, seq in enumerate(sequences):
        if not network_status[i]:
            L = seq["length"]
            for j in range(L + 1):
                incoming = [cc['selected'] for cc in candidates[i] if cc['end_index'] == j]
                outgoing = [cc['selected'] for cc in candidates[i] if cc['start_index'] == j]
                if j == 0:
                    model.Add(sum(outgoing) == 1)
                elif j == L:
                    model.Add(sum(incoming) == 1)
                else:
                    model.Add(sum(incoming) == sum(outgoing))

    # Precedence constraints
    for i, seq in enumerate(sequences):
        if not network_status[i]:
            L = seq["length"]
            for j in range(1, L):
                incoming_candidates = [cc for cc in candidates[i] if cc['end_index'] == j]
                outgoing_candidates = [cc for cc in candidates[i] if cc['start_index'] == j]
                for c_in in incoming_candidates:
                    for c_out in outgoing_candidates:
                        model.Add(c_in['start_time'] + c_in['latency'] <= c_out['start_time']) \
                            .OnlyEnforceIf([c_in['selected'], c_out['selected']])

    # Cumulative resource constraints
    intervals = [cc['interval'] for cc in candidate_list]
    demands = [cc['resources'] for cc in candidate_list]
    model.AddCumulative(intervals, demands, total_resources)
    mem_demands = [cc['mem_tiles'] for cc in candidate_list]
    model.AddCumulative(intervals, mem_demands, total_mem_tiles)


    if cost_objective == "latency":
        # Minimize makespan
        makespan = model.NewIntVar(0, horizon, "makespan")
        M = horizon
        for i, seq in enumerate(sequences):
            if not network_status[i]:
                L = seq["length"]
                end_candidates = [c for c in candidates[i] if c['end_index'] == L]
                for c in end_candidates:
                    finish_time = model.NewIntVar(0, horizon, f"finish_{c['candidate_id']}")
                    model.Add(finish_time == c['start_time'] + c['latency']).OnlyEnforceIf(c['selected'])
                    model.Add(finish_time <= makespan + M * (1 - c['selected']))
        model.Minimize(makespan)
        model.Add(makespan >= cost_lower_bound)
    elif cost_objective in ("energy", "EDP"):
        is_edp = (cost_objective == "EDP")
        segment_costs = []
        BIG_M = 100_000_000
        for c in candidate_list:
            cost_c = int ( (c['resources'] * (p_cost["acc"] + p_cost["router"])
                            + c['mem_tiles'] * p_cost["mem"] ) *  (c['latency'] ** (2 if is_edp else 1)))
            cost_var = model.NewIntVar(0, BIG_M, f"cost_{c['candidate_id']}")
            model.Add(cost_var == cost_c).OnlyEnforceIf(c['selected'])
            model.Add(cost_var == 0).OnlyEnforceIf(c['selected'].Not())
            segment_costs.append(cost_var)
        total_cost = model.NewIntVar(0, BIG_M * len(candidate_list), "total_cost")
        model.Add(total_cost == sum(segment_costs))
        model.Minimize(total_cost)
        model.Add(total_cost >= cost_lower_bound)

    solver = cp_model.CpSolver()
    solver.parameters.random_seed = 42
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = 300
    callback = ProgressCallback(bound_tol=0.03)

    status = solver.Solve(model, callback) if any(not network_status[i] for i in range(len(sequences))) else None

    if status == cp_model.OPTIMAL:
      print("Proved optimal!")
    elif status == cp_model.FEASIBLE:
      print("Stopped early with feasible solution at gap <= 5%")

    if cost_objective == "latency":
      if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        total_cost_cp = solver.Value(makespan)
      else:
        total_cost_cp = None
    else:
      if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        total_cost_cp = solver.Value(total_cost)
      else:
        total_cost_cp = None

    all_chains = []
    if solver:
        for i, seq in enumerate(sequences):
            if network_status[i]:
                all_chains.append( (i, []) )
            else:
                chain = []
                current_index = 0
                seg_index = 0
                while current_index < seq["length"]:
                    selected_candidates = [cc for cc in candidates[i]
                                           if cc['start_index'] == current_index
                                           and solver.Value(cc['selected'])]
                    if not selected_candidates:
                        break
                    c = selected_candidates[0]
                    st = solver.Value(c['start_time'])
                    ft = st + c['latency']
                    if cost_objective == "area-latency":
                        cost_val = c['resources'] * c['latency']
                    chain_item = {
                        'tenant': i,
                        'chain_idx': seg_index,
                        'start_index': c['start_index'],
                        'end_index': c['end_index'],
                        'start_time': st,
                        'finish_time': ft,
                        'latency': c['latency'],
                        'resources': c['resources'],
                        'mem_tiles': c['mem_tiles'],
                        'mode_index': c['mode_index'],
                        'length': c['length'],
                        'mapping': c['mapping'],
                        'off_chip_accesses': c['off_chip_accesses']
                    }
                    chain.append(chain_item)
                    seg_index += 1
                    current_index = c['end_index']
                all_chains.append( (i, chain) )
    return all_chains, total_cost_cp


# ----------------------------------------------------------
# CP/DP Solver Wrapper
# ----------------------------------------------------------
def SolverEngine(sequences, total_resources, total_mem_tiles, cost_objective="latency"):
    """
    Solve mapping & scheduling, using DP for single-model
    sequences or CP for multi-model sequences.

    Parameters:
        sequences (List[Dict]): One dict per tenant/network, each with a 'length' key.
        total_resources (int): Total accelerators available.
        total_mem_tiles (int): Total memory tiles available.
        cost_objective (str): Which metric to optimize ('latency', 'energy', 'edp', etc.).

    Returns:
        dict: {
            'filtered_output': Dict[int, chain],
            'events': List[tuple],
            'action_list_str': List[str],
            'action_list_num': List[int],
            'total_cost': float,
            'total_rel_energy': float,
            'total_rel_edp': float,
            'total_rel_acc': float
        }
    """
    # Power unit costs for DP/CP solvers
    unit_costs = {"acc": 0.09, "router": 0.045, "mem": 1.065}

    # Find indices of non-empty sequences
    non_empty = [i for i, seq in enumerate(sequences) if seq.get("length", 0) > 0]

    # If nothing to schedule, return empty result
    if not non_empty:
        return {
            "filtered_output": {},
            "events": [],
            "action_list_str": [],
            "action_list_num": [],
            "total_cost": 0.0,
            "total_rel_energy": 0.0,
            "total_rel_edp": 0.0,
            "total_rel_acc": 0.0,
        }

    # Single-sequence: use DP solver
    if len(non_empty) == 1:
        idx = non_empty[0]
        print("Running DP solver on single sequence")
        best_cost, metrics, _, dp_chain = dp_solver(
            sequences[idx],
            idx,
            unit_costs,
            cost_objective=cost_objective
        )
        all_chains = [(idx, dp_chain)]
        total_cost = best_cost

    # Multi-sequence: compute DP bounds then CP
    else:
        print("Running CP solver on multiple sequences")
        upper_bound = 0
        lower_bound = 0

        # DP pass for bounds
        #   - Upper bound (horizon) = sum of each single‑model’s best-search cost.
        #     This serves as a conservative maximum cost when scheduling them all together.
        #   - Lower bound = the maximum best-search cost among the individual models.
        #     This is the tightest possible minimum cost if one network dominates.
        for idx in non_empty:
            cost_i, metrics, _, dp_chain = dp_solver(
                sequences[idx],
                idx,
                unit_costs,
                cost_objective=cost_objective
            )
            upper_bound += int(cost_i)
            lower_bound = int(max(lower_bound, cost_i))

        # CP solver
        all_chains, total_cost = cp_solver(
            sequences,
            total_resources,
            total_mem_tiles,
            unit_costs,
            cost_objective,
            lower_bound,
            horizon=upper_bound
        )
        metrics = compute_cp_metrics(all_chains, unit_costs)

    # Build final outputs
    events, action_strs, action_nums = build_events_and_actions(all_chains)
    filtered_output = build_filtered_output(all_chains, sequences)

    return {
        "filtered_output": filtered_output,
        "events": events,
        "action_list_str": action_strs,
        "action_list_num": action_nums,
        "total_cost": total_cost,
        "total_rel_energy": metrics["energy"],
        "total_rel_edp": metrics["edp"],
        "total_rel_acc": metrics["accesses"],
    }



# ----------------------------------------------------------
# OASIS Solver
# ----------------------------------------------------------
def OasisSolver(segmented_networks, cost_objective, tot_acc, tot_mem, test_type):
    """
    Runs the Oasis solver over a series of windows of segmented networks.

    Parameters:
        segmented_networks (List[List[Dict]]): windwos of segment-mappings.
        cost_objective (str): Cost metric to optimize ('EDP', etc.).
        tot_acc (int): Total number of accelerators available.
        tot_mem (int): Total number of memory tiles available.
        test_type (str): Identifier for the type of test
                         ('sm', 'mm', 'mm_set', 'sm_tangram').

    Returns:
        segmentation_stats (np.ndarray): Aggregated segmentation statistics.
        nets_results (dict): Metrics computed across all windows, e.g.:
            {
                'latency': float,
                'energy': float,
                'edp': float,            # if test_type in {'sm','mm','mm_set'}
                'dram_acc': float,       # if test_type == 'sm_tangram'
                'solver_runtime': float  # only if test_type == 'mm'
            }
    """
    # --- Initialize accumulators ---
    total_latency = 0.0
    total_cost = 0.0
    total_energy = 0.0
    total_edp = 0.0
    total_dram_access = 0.0
    segmentation_stats = np.zeros(3, dtype=float)

    start_time = time.perf_counter()

    # --- Solve each window ---
    for window_idx, window in enumerate(segmented_networks):
        # Skip empty windows
        window_length = sum(tenant['length'] for tenant in window)
        if window_length == 0:
            continue

        result = SolverEngine(window, tot_acc, tot_mem, cost_objective)
        allocation = result['filtered_output']
        action_order = result['action_list_num']
        exec_time, _, _, _ = result['events'][-1]

        # Print per‐window summary
        print(f"[Window {window_idx}] Execution time: {exec_time:.3f}s")

        # Accumulate metrics
        total_latency += exec_time
        total_cost += result['total_cost']
        total_energy += result['total_rel_energy']
        total_edp += result['total_rel_edp']
        total_dram_access += result['total_rel_acc']

        print("-------- NSM RESULTS --------")
        print(f"Accumulated Latency = {total_latency:.3f}s")
        print(f"Accumulated Cost ({cost_objective}) = {total_cost:.3f}")

        # Gather segmentation stats
        segmentation_stats += collect_segmentation_statistics(allocation)

    # --- Finalize ---
    solver_runtime = time.perf_counter() - start_time

    nets_results = {
        'latency': total_latency,
        'energy': total_energy
    }

    if test_type in {'sm', 'mm', 'mm_set'}:
        nets_results['edp'] = total_edp
    elif test_type == 'sm_tangram':
        nets_results['dram_acc'] = total_dram_access

    if test_type == 'mm':
        nets_results['solver_runtime'] = solver_runtime

    return segmentation_stats, nets_results


# ----------------------------------------------------------
# Example Usage
# ----------------------------------------------------------
if __name__ == "__main__":
    sched_horiz=1000000
    sequences_example = [
        {
            'length': 4,
            'modes': {
                0: [
                    {'length': 2, 'resources': 2, 'latency': 5, 'mem_tiles': 4, 'mapping': 3, 'off_chip_accesses':0, 'batch':1},
                    {'length': 1, 'resources': 1, 'latency': 3, 'mem_tiles': 2, 'mapping': 4, 'off_chip_accesses':0, 'batch':1},
                    {'length': 2, 'resources': 3, 'latency': 8, 'mem_tiles': 5, 'mapping': 5, 'off_chip_accesses':0, 'batch':1}
                ],
                1: [{'length': 1, 'resources': 2, 'latency': 4, 'mem_tiles': 3, 'mapping': 3, 'off_chip_accesses':0, 'batch':1}],
                2: [{'length': 1, 'resources': 1, 'latency': 3, 'mem_tiles': 2, 'mapping': 2, 'off_chip_accesses':0, 'batch':1}],
                3: [{'length': 1, 'resources': 2, 'latency': 6, 'mem_tiles': 3, 'mapping': 3, 'off_chip_accesses':0, 'batch':1}]
            }
        },
        {
            'length': 3,
            'modes': {
                0: [
                    {'length': 1, 'resources': 1, 'latency': 4, 'mem_tiles': 1, 'mapping': 1, 'off_chip_accesses':0, 'batch':1},
                    {'length': 2, 'resources': 2, 'latency': 7, 'mem_tiles': 3, 'mapping': 3, 'off_chip_accesses':0, 'batch':1}
                ],
                1: [{'length': 1, 'resources': 1, 'latency': 3, 'mem_tiles': 1, 'mapping': 1, 'off_chip_accesses':0, 'batch':1}],
                2: [{'length': 1, 'resources': 1, 'latency': 5, 'mem_tiles': 2, 'mapping': 5, 'off_chip_accesses':0, 'batch':1}]
            }
        }
    ]
    total_resources = 10
    total_mem_tiles = 8
    print("\n=== Solving with cost_objective='latency' ===")
    result1 = SolverEngine(sequences_example, total_resources, total_mem_tiles, cost_objective="latency")
    print("\n=== Final Filtered Output ===")
    for tenant, out in result1["filtered_output"].items():
        print(f"Sequence {tenant}: {out}")

    print("\n=== Solving with cost_objective='area-latency' ===")
    result2 = SolverEngine(sequences_example, total_resources, total_mem_tiles, cost_objective="energy")
    print("\n=== Final Filtered Output ===")
    for tenant, out in result2["filtered_output"].items():
        print(f"Sequence {tenant}: {out}")
