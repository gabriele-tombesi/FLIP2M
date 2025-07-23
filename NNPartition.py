# Copyright (c) 2011-2025 Columbia University, System Level Design Group
# SPDX-License-Identifier: Apache-2.0

from test_FLIP2M import *

def NNPartitionEngine(networks, epochs_num):
    """
    Splits each network into 'epochs_num' epochs by a naive ops threshold,
    then merges them into final epochs. Next, it applies the rule:
      "If epoch e is the last epoch for some network X (i.e. X has layers in e but none in e+1..),
       and another network Y also has layers in e, then push Y's entire layer set from epoch e
       into a new epoch e+1, and shift Y's subsequent epochs forward by one."
    """
    all_network_epochs = []  # For each network, a list of epoch dictionaries.
    operations = []          # For debugging: total ops per network

    # ----------------------------------------------------------
    # 1. Build per-network epochs with your original approach
    # ----------------------------------------------------------
    for net in networks:
        num_layers = len(net['n_channels'])
        batch_size = net.get('batch_size', 1)
        layer_ops = []
        tot_ops = 0
        # compute ops
        for i in range(num_layers):
            int_batch   = net['int_batch'][i]
            input_dim   = net['feature_map'][i]
            fdim        = net['filter_dim'][i]
            is_padded   = net['is_padded'][i]
            stride      = net['stride'][i]
            pool_type   = net['pool_type'][i]
            n_filters   = net['n_filters'][i]
            n_channels  = net['n_channels'][i]

            pad = (fdim >> 1) if is_padded else 0
            out_fmap = (input_dim + 2*pad - fdim)//stride + 1
            if pool_type:
                out_fmap >>= 1
            ops = (out_fmap**2) * n_filters * (fdim**2) * n_channels * batch_size * int_batch
            layer_ops.append(ops)
            tot_ops += ops
        operations.append(tot_ops)

        # group layers by out_add
        segments = []
        current_seg = []
        for i in range(num_layers):
            current_seg.append(i)
            if net['out_add'][i] != 0:
                segments.append(current_seg)
                current_seg = []
        if current_seg:
            segments.append(current_seg)

        ops_per_epoch = tot_ops / float(epochs_num) if epochs_num>0 else tot_ops
        network_epochs_segments = []
        current_epoch_segments = []
        current_epoch_ops = 0

        for seg in segments:
            seg_ops = sum(layer_ops[i] for i in seg)
            if current_epoch_segments and (current_epoch_ops + seg_ops > ops_per_epoch):
                network_epochs_segments.append(current_epoch_segments)
                current_epoch_segments = [seg]
                current_epoch_ops = seg_ops
            else:
                current_epoch_segments.append(seg)
                current_epoch_ops += seg_ops
        if current_epoch_segments:
            network_epochs_segments.append(current_epoch_segments)

        # Pad or merge to get exactly epochs_num
        while len(network_epochs_segments)<epochs_num:
            network_epochs_segments.append([])
        if len(network_epochs_segments)>epochs_num:
            extra = network_epochs_segments[epochs_num-1:]
            merged = []
            for seglist in extra:
                merged.extend(seglist)
            network_epochs_segments = network_epochs_segments[:epochs_num-1] + [merged]

        # Convert each epoch to a dictionary
        net_epoch_dicts = []
        for epoch_segments in network_epochs_segments:
            layer_indices = [i for seg in epoch_segments for i in seg]
            epoch_dict = {}
            for key, val in net.items():
                if key in ('batch_size','network_name'):
                    epoch_dict[key] = val
                else:
                    epoch_dict[key] = [val[idx] for idx in layer_indices]
            net_epoch_dicts.append(epoch_dict)

        all_network_epochs.append(net_epoch_dicts)

    # ----------------------------------------------------------
    # 2. Build an initial final_epochs ignoring the finishing rule
    # ----------------------------------------------------------
    final_epochs = []
    for epoch_idx in range(epochs_num):
        epoch_networks = []
        for net_idx, net_epochs in enumerate(all_network_epochs):
            if epoch_idx < len(net_epochs):
                epoch_networks.append(net_epochs[epoch_idx])
            else:
                epoch_networks.append({})
        final_epochs.append(epoch_networks)

    # ----------------------------------------------------------
    # 3. Finishing rule
    # ----------------------------------------------------------

    def count_layers(epoch_dict):
        return len(epoch_dict['n_channels']) if 'n_channels' in epoch_dict else 0

    def ensure_epoch_exists(net_j, eplus):
        while eplus>=len(all_network_epochs[net_j]):
            new_ep = {}
            if len(all_network_epochs[net_j])>0:
                first_ep = all_network_epochs[net_j][0]
                for mk in ('batch_size','network_name'):
                    if mk in first_ep:
                        new_ep[mk] = first_ep[mk]
            all_network_epochs[net_j].append(new_ep)

    for net_idx, net_epochs in enumerate(all_network_epochs):
        net_len = len(net_epochs)
        e_idx = 0
        while e_idx < net_len:
            layers_e = count_layers(net_epochs[e_idx])
            if layers_e==0:
                e_idx+=1
                continue
            finishing = True
            for ee in range(e_idx+1, net_len):
                if count_layers(net_epochs[ee])>0:
                    finishing=False
                    break
            if not finishing:
                e_idx+=1
                continue

            x = layers_e
            for k_idx, k_epochs in enumerate(all_network_epochs):
                if k_idx==net_idx:
                    continue
                if e_idx>=len(k_epochs):
                    continue
                y = count_layers(k_epochs[e_idx])
                if y<=0:
                    continue
                eplus = e_idx+1
                if eplus< len(k_epochs):
                    k_epochs.insert(eplus, {})
                else:
                    while eplus> len(k_epochs):
                        k_epochs.append({})
                    k_epochs.append({})
                for meta_key in ('batch_size','network_name'):
                    if meta_key in k_epochs[e_idx]:
                        k_epochs[eplus][meta_key] = k_epochs[e_idx][meta_key]
                for key, val in list(k_epochs[e_idx].items()):
                    if key in ('batch_size','network_name'):
                        continue
                    k_epochs[eplus][key] = val
                    k_epochs[e_idx][key] = []
            e_idx+=1

        net_len = len(net_epochs)

    max_ep_count = max(len(x) for x in all_network_epochs)
    final_epochs = []
    for e in range(max_ep_count):
        epoch_networks = []
        for net_idx, net_epochs in enumerate(all_network_epochs):
            if e<len(net_epochs):
                epoch_networks.append(net_epochs[e])
            else:
                epoch_networks.append({})
        final_epochs.append(epoch_networks)

    return operations, final_epochs


def NNMappingWindows(Net_epochs, unsegmented_networks):
    """
    Builds a segmented data structure from:
      1) Net_epochs: a list of epochs. Each epoch is a list of integers indicating
         how many layers each network has at that epoch.
      2) unsegmented_networks: a list of length Y (number of networks), where each
         item is shaped like [[{ 'length': X, 'modes': { '0': [...], '1': [...], ... } }]].
         The keys under 'modes' are strings (e.g. '0', '1', '2', ...).

    Returns a new structure with shape:
       [
         [dict_for_network0_at_epoch0, dict_for_network1_at_epoch0, ...],
         [dict_for_network0_at_epoch1, dict_for_network1_at_epoch1, ...],
         ...
       ]
    where each dict looks like:
       {
         'length': <number_of_layers_for_this_epoch>,
         'modes': {
            0: [...filtered subdicts...],
            1: [...filtered subdicts...],
            ...
         }
       }
    and the filtering removes any sub-dictionary whose 'length' crosses the boundary
    for this epoch (i.e., local_key + item['length'] > epoch_layer_count).

    """

    n_epochs = len(Net_epochs)
    if n_epochs == 0:
        return []

    Y = len(Net_epochs[0]) if n_epochs > 0 else 0
    network_dicts = []
    for net in unsegmented_networks:
        d = net[0][0]
        network_dicts.append(d)

    offsets = [0]*Y
    output = []

    for e in range(n_epochs):
        # Net_epochs[e] is a list of layer counts for each network at epoch e
        epoch_list = []

        for n in range(Y):
            layer_count = Net_epochs[e][n]

            if layer_count < 1:
                epoch_list.append({
                    'length': 0,
                    'modes': {}
                })
                continue

            unseg = network_dicts[n]           # e.g. { 'length': X, 'modes': { '0': [...], ...} }
            modes_full = unseg.get('modes', {})  # dict with string keys: '0', '1', ..., 'X-1'

            start = offsets[n]
            end = start + layer_count

            new_dict = {
                'length': layer_count,
                'modes': {}
            }

            for global_key_str, sublist in modes_full.items():
                global_key = int(global_key_str)

                if not (start <= global_key < end):
                    continue

                local_key = global_key - start

                filtered_sublist = []
                for item in sublist:
                    item_length = item.get('length', 1)
                    if local_key + item_length <= layer_count:
                        filtered_sublist.append(copy.deepcopy(item))

                new_dict['modes'][local_key] = filtered_sublist

            # Add to this epoch's list
            epoch_list.append(new_dict)

            # Advance offset for network n
            offsets[n] += layer_count

        output.append(epoch_list)

    return output


def NNPartition(
    network_ids,
    epochs_num,
    total_acc,
    total_mem,
    deployment_config,
    bench_batch_sizes,
):
    """
    Split networks into epochs, apply windowed segmentation, and prune modes
    based on the test configuration.

    Parameters:
        network_ids (List[int]): IDs of the networks to partition.
        epochs_num (int): Number of epochs to split each network into.
        total_acc (int): Total accelerator resources available.
        total_mem (int): Total memory tiles available.
        deployment_config (Tuple):
            (
                prune_segment_depths,
                prune_nacc_per_seg,
                prune_nmem_per_seg,
                prune_parallelism,
                const_acc_flag,
                const_mem_flag
            )
        bench_batch_sizes (List[int]): Batch sizes for each network for latency scaling.

    Returns:
        List[Dict]: The segmented and pruned networks, ready for mapping.
    """
    # Unpack deployment configuration
    (
        prune_segment_depths,
        prune_nacc_per_seg,
        prune_nmem_per_seg,
        prune_parallelism,
        const_acc_flag,
        const_mem_flag,
    ) = deployment_config

    # Possible discrete resource values
    possible_accs = list(range(2, 33, 2))
    possible_mems = [1, 2, 4]

    # Load network definitions
    networks = [NNList[n_id] for n_id in network_ids]

    # --- Split into epochs ---
    operations, epochs = NNPartitionEngine(networks, epochs_num)

    # Compute per-network constant allocation targets
    total_ops = sum(operations)
    num_acc_const = []
    num_mem_const = []
    for ops in operations:
        target_acc = round_down_to_multiple_of_two(total_acc * ops / total_ops)
        if target_acc < 2:
            target_acc == 2
        num_acc_const.append(target_acc)
        num_mem_const.append(
            round_down_to_power_of_two(total_mem * ops / total_ops)
        )

    # Build list of layer counts per epoch
    net_epochs = [
        [len(slice_.get("n_channels", [])) for slice_ in epoch]
        for epoch in epochs
    ]

    # --- Load raw network mappings ---
    unsegmented_networks = []
    for n_id in network_ids:
        path = f"./NNMappings/{n_id}.json"
        with open(path, "r") as f:
            unsegmented_networks.append(json.load(f))

    # Apply windowed segmentation
    segmented_networks = NNMappingWindows(net_epochs, unsegmented_networks)

    # --- Prune modes by configuration ---
    for sd in prune_segment_depths:
        segmented_networks = remove_modes_with_length(segmented_networks, sd)

    for nacc in prune_nacc_per_seg:
        segmented_networks = remove_modes_with_resource(segmented_networks, nacc)

    for nmem in prune_nmem_per_seg:
        segmented_networks = remove_modes_with_mem(segmented_networks, nmem)

    for paral in prune_parallelism:
        segmented_networks = remove_modes_with_paral(segmented_networks, paral)

    # Scale latencies for faster CP search
    segmented_networks = scale_latency(segmented_networks, bench_batch_sizes)

    # --- Enforce constant resource constraints if requested ---
    if const_acc_flag == "true":
        for idx in range(len(network_ids)):
            for acc in possible_accs:
                if acc > num_acc_const[idx]:
                    segmented_networks = remove_modes_with_resource(
                        segmented_networks, acc, idx
                    )

    if const_mem_flag == "true":
        for idx in range(len(network_ids)):
            for mem in possible_mems:
                if mem > num_mem_const[idx]:
                    segmented_networks = remove_modes_with_mem(
                        segmented_networks, mem, idx
                    )

    return segmented_networks
