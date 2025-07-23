# Copyright (c) 2011-2025 Columbia University, System Level Design Group
# SPDX-License-Identifier: Apache-2.0

from test_FLIP2M import *

test_suite_st={
  "Baseline": [ [2,3], [8,16], [1,2], [],'false', 'false'],
  "FLIP2M-FlexIntra": [ [2,3], [], [], [], 'false', 'false'],
  "FLIP2M-Full":[ [], [], [], [], 'false', 'false']
}

test_suite_st_tangram=[
    [ [], [4,6,8,10,12,14,16,18,20,22,24,26,28,30,32], [], [], 'false', 'false'],
    [ [], [6,8,10,12,14,16,18,20,22,24,26,28,30,32], [], [], 'false', 'false'],
    [ [], [10,12,14,16,18,20,22,24,26,28,30,32], [], [], 'false', 'false'],
    [ [], [18,20,22,24,26,28,30,32], [], [], 'false', 'false']
]

# Baseline Configuration Note for Multi-Model Experiments:
# In the original results presented in the paper, the Baseline configuration was evaluated with n_acc = 4 allowed for all layers.
# This was done to ensure a feasible mapping for layers that could not be scheduled on configurations with more than 4 accelerators.
# For simplicity, this constraint was uniformly applied to all layers, including those that could
# have been mapped on >4 accelerators.
#
# We have since added support to automatically detect layers that are not mappable with more than 4 accelerators.
# In this updated version, pruning is selectively applied only to those layers.
#
# To reproduce the updated results with this refined behavior, replace the Baseline deployment configuration below:
# From: [ [2,3], [2,8,16], [], [], 'false', 'true' ]
# To:   [ [2,3], [2,4,8,16], [], [], 'false', 'true' ]
#
# This leads to even more degraded Baseline performance compared to the other configurations that progressively
# enable additional FLIP2M features.
test_suite_mt={
    "Baseline" : [ [2, 3], [2, 8, 16], [], [],'false', 'true'],
    "FLIP2M-FlexIntra1" : [ [2, 3], [], [], [], 'false', 'true'],
    "FLIP2M-FlexIntra2" :[ [2, 3], [], [], [], 'false', 'false'],
    "FLIP2M-Full" : [ [], [], [], [], 'false', 'false'],
}

test_suite_mt_set_comp=[
  [ [], [6,8,10,12,14,16,18,20,22,24,26,28,30,32], [], [], 'false', 'false'],
  [ [], [10,12,14,16,18,20,22,24,26,28,30,32], [], [], 'false', 'false'],
  [ [], [18,20,22,24,26,28,30,32], [], [], 'false', 'false'],
  [ [], [26,28,30,32], [], [], 'false', 'false'],
  [ [], [], [], [], 'false', 'false'],
]



def round_down_to_multiple_of_two(value):
    return ((value - 1) // 2) * 2

def round_down_to_power_of_two(n):
    if n < 1:
        return 1
    exponent = math.floor(math.log2(n))
    return 2 ** exponent


def scale_latency(data,batch_size):
    cleaned_data = []
    scaling_factor=100000

    for sublist in data:
        new_sublist = []
        for index,item in enumerate(sublist):
          if 'modes' in item and isinstance(item['modes'], dict):
                new_modes = {}
                for k, mode_list in item['modes'].items():
                  for mode_entry in mode_list:
                    mode_entry['latency'] = int(mode_entry['latency'] * batch_size[index] / scaling_factor)
                  if isinstance(mode_list, list):
                    filtered_list = [
                      mode_entry
                      for mode_entry in mode_list
                    ]
                    if filtered_list:
                      new_modes[k] = filtered_list
                  else:
                    new_modes[k] = mode_list
                new_item = {
                    **item,
                    'modes': new_modes
                }
          else:
            new_item = {**item}
          new_sublist.append(new_item)
        cleaned_data.append(new_sublist)
    return cleaned_data


def remove_modes_with_resource(data, resource_to_remove, model_id=None):
    cleaned_data = []

    if model_id==None:
      target_model=-1
    else:
      target_model=model_id

    for sublist in data:
        new_sublist = []
        for index,item in enumerate(sublist):
          if 'modes' in item and isinstance(item['modes'], dict):
                new_modes = {}
                for k, mode_list in item['modes'].items():
                  if target_model==-1:
                    if isinstance(mode_list, list) and not (all(d.get("resources") == resource_to_remove for d in mode_list)):
                        filtered_list = [
                            mode_entry
                            for mode_entry in mode_list
                            if (mode_entry.get('resources') != resource_to_remove)
                        ]
                        if filtered_list:
                            new_modes[k] = filtered_list
                    else:
                        new_modes[k] = mode_list
                  else:
                    if isinstance(mode_list, list) and not (all(d.get("resources") == resource_to_remove for d in mode_list)):
                        filtered_list = [
                            mode_entry
                            for mode_entry in mode_list
                            if (index!=target_model or mode_entry.get('resources') != resource_to_remove)
                        ]
                        if filtered_list:
                            new_modes[k] = filtered_list
                    else:
                        new_modes[k] = mode_list
                new_item = {
                    **item,
                    'modes': new_modes
                }
          else:
            new_item = {**item}
          new_sublist.append(new_item)
        cleaned_data.append(new_sublist)
    return cleaned_data

def remove_modes_with_mem(data, nmem_to_remove, model_id=-1):
    cleaned_data = []

    if model_id==None:
      target_model=-1
    else:
      target_model=model_id


    for sublist in data:
        new_sublist = []
        for index,item in enumerate(sublist):
            if 'modes' in item and isinstance(item['modes'], dict):
                new_modes = {}
                for k, mode_list in item['modes'].items():
                  if target_model==-1:
                    if isinstance(mode_list, list):
                        filtered_list = [
                            mode_entry
                            for mode_entry in mode_list
                            if mode_entry.get('mem_tiles') != nmem_to_remove
                        ]
                        if filtered_list:
                            new_modes[k] = filtered_list
                    else:
                        new_modes[k] = mode_list
                  else:
                    if isinstance(mode_list, list):
                        filtered_list = [
                            mode_entry
                            for mode_entry in mode_list
                            if (index!=target_model or mode_entry.get('mem_tiles') != nmem_to_remove)
                        ]
                        if filtered_list:
                            new_modes[k] = filtered_list
                    else:
                        new_modes[k] = mode_list

                new_item = {
                    **item,
                    'modes': new_modes
                }
            else:
                new_item = {**item}
            new_sublist.append(new_item)
        cleaned_data.append(new_sublist)
    return cleaned_data

def remove_modes_with_paral(data, paral_to_remove, model_id=None):
    cleaned_data = []

    if model_id==None:
      target_model=-1
    else:
      target_model=model_id

    for sublist in data:
        new_sublist = []
        for index,item in enumerate(sublist):
          if 'modes' in item and isinstance(item['modes'], dict):
                new_modes = {}
                for k, mode_list in item['modes'].items():
                  if target_model==-1:
                    if isinstance(mode_list, list):
                        filtered_list = [
                            mode_entry
                            for mode_entry in mode_list
                            if (mode_entry.get('paral') != paral_to_remove)
                        ]
                        if filtered_list:
                            new_modes[k] = filtered_list
                    else:
                        new_modes[k] = mode_list
                  else:
                    if isinstance(mode_list, list):
                        filtered_list = [
                            mode_entry
                            for mode_entry in mode_list
                            if (index!=target_model or mode_entry.get('paral') != paral_to_remove)
                        ]
                        if filtered_list:
                            new_modes[k] = filtered_list
                    else:
                        new_modes[k] = mode_list
                new_item = {
                    **item,
                    'modes': new_modes
                }
          else:
            new_item = {**item}
          new_sublist.append(new_item)
        cleaned_data.append(new_sublist)
    return cleaned_data


def remove_modes_with_length(data, length_to_remove):
    cleaned_data = []

    for sublist in data:
        new_sublist = []
        for item in sublist:
            if 'modes' in item and isinstance(item['modes'], dict):
                new_modes = {}
                for k, mode_list in item['modes'].items():
                    if isinstance(mode_list, list):
                        filtered_list = [
                            mode_entry
                            for mode_entry in mode_list
                            if mode_entry.get('length') != length_to_remove
                        ]
                        if filtered_list:
                            new_modes[k] = filtered_list
                    else:
                        new_modes[k] = mode_list

                new_item = {
                    **item,
                    'modes': new_modes
                }
            else:
                new_item = {**item}
            new_sublist.append(new_item)
        cleaned_data.append(new_sublist)
    return cleaned_data
