# Copyright (c) 2011-2025 Columbia University, System Level Design Group
# SPDX-License-Identifier: Apache-2.0

import os
import time
import csv
import copy
import json
import math
import pprint
import random
import random
import argparse
import unittest
import numpy as np

from collections import defaultdict
from ortools.sat.python import cp_model

from NN import NNList
from TestConfigUtils import *
from NNPartition import *
from TestConfigUtils import *
from OASIS import *
from utils import *

class TestFLIP2M(unittest.TestCase):

  def test_single_model(self):
    # Constants & configuration
    cost_objective   = os.getenv('COST_OBJECTIVE')
    self.assertIsNotNone(cost_objective, "You must set COST_OBJECTIVE")

    test_type        = "sm"
    target_models    = [
      "resnet18", "resnet34", "resnet50",
      "mobilenet", "unet",     "squeezenet",
      "vgg16",     "mobile_bert"
    ]
    test_configs    = ["Baseline", "FLIP2M-FlexIntra", "FLIP2M-Full"]
    bench_batch_sizes = [1]
    tot_acc_test     = 36
    tot_mem_test     = 7
    epochs_num       = 1

    # Run the solver & collect results
    results = defaultdict(dict)
    for model_id in target_models:
      for config in test_configs:
        cfg       = test_suite_st[config]
        networks  = [model_id]
        segmented = NNPartition(
          networks,
          epochs_num,
          tot_acc_test,
          tot_mem_test,
          cfg,
          bench_batch_sizes
        )
        # OasisSolver returns (segs_stats, results)
        _, results[model_id][config] = OasisSolver(
          segmented,
          cost_objective,
          tot_acc_test,
          tot_mem_test,
          test_type
        )

    # Dump Results
    pprint.pprint(results, width=200, indent=2, sort_dicts=False)
    generate_sm_csv(results,f"./output/sm_{cost_objective}_table.csv")

  def test_single_model_TANGRAMcomp(self):
    # Constants & configuration
    test_type         = "sm_tangram"
    target_models     = ["resnet50", "vgg16"]
    tot_acc_values    = [2, 4, 8, 16]
    bench_batch_sizes = [1]
    epochs_num        = 1
    tot_mem_test      = 7
    cost_objective    = 'latency'

    # Run the solver & collect results
    results = defaultdict(dict)
    for idx, acc in enumerate(tot_acc_values):
      cfg = test_suite_st_tangram[idx]
      for model in target_models:
        segmented = NNPartition(
          [model],
          epochs_num,
          acc,
          tot_mem_test,
          cfg,
          bench_batch_sizes
        )
        # OasisSolver returns (segs_stats, results)
        _, results[model][str(acc)] = OasisSolver(
          segmented,
          cost_objective,
          acc,
          tot_mem_test,
          test_type
        )

    # Dump results
    pprint.pprint(results, width=200, indent=2, sort_dicts=False)
    generate_sm_TANGRAMcomp_csv(results, "./output/sm_TANGRAMcomp.csv")


  def test_multi_model(self):
    # Constants & configuration
    cost_objective = os.getenv('COST_OBJECTIVE')
    self.assertIsNotNone(cost_objective, "You must set COST_OBJECTIVE")

    test_type     = "mm"
    epochs_num    = 10
    tot_acc_test  = 36
    tot_mem_test  = 7

    scenarios = ["ARVR1", "ARVR2", "ARVR3"]
    networks_per_scenario = {
      "ARVR1": ["resnet18", "squeezenet", "mobilenet"],
      "ARVR2": ["resnet34", "mobile_bert", "resnet50"],
      "ARVR3": ["vgg16", "mobilenet", "resnet18", "mobile_bert"],
    }
    batch_sizes_per_scenario = {
      "ARVR1": [2, 1, 2],
      "ARVR2": [4, 1, 2],
      "ARVR3": [2, 2, 4, 1],
    }
    test_configs = [
      "Baseline",
      "FLIP2M-FlexIntra1",
      "FLIP2M-FlexIntra2",
      "FLIP2M-Full",
    ]

    # Run the solver & collect results
    results = defaultdict(dict)
    for scenario in scenarios:
      networks    = networks_per_scenario[scenario]
      batch_sizes = batch_sizes_per_scenario[scenario]
      for config in test_configs:
        cfg = test_suite_mt[config]
        segmented = NNPartition(
          networks,
          epochs_num,
          tot_acc_test,
          tot_mem_test,
          cfg,
          batch_sizes
        )
        # OasisSolver returns (segs_stats, results)
        _, results[scenario][config] = OasisSolver(
          segmented,
          cost_objective,
          tot_acc_test,
          tot_mem_test,
          test_type
        )

    # Dump Results
    pprint.pprint(results, width=200, indent=2, sort_dicts=False)
    generate_mm_csv(results, f"./output/mm_{cost_objective}_search.csv")


  def test_multi_model_SETcomp(self):
    # Constants & configuration
    epochs_num = 8
    networks_id_vec = ["resnet34", "resnet50", "vgg16"]
    bench_batch_sizes_vec = [1, 1, 1]
    tot_acc_test = [4, 8, 16, 24, 32]
    tot_mem_test = 7
    cost_objective = "EDP"
    test_type = "mm_set"

    # Run the solver & collect results
    results = defaultdict(dict)
    for idx, acc in enumerate(tot_acc_test):
      cfg = test_suite_mt_set_comp[idx]
      segmented_networks = NNPartition(
        networks_id_vec,
        epochs_num,
        acc,
        tot_mem_test,
        cfg,
        bench_batch_sizes_vec
      )
      # OasisSolver returns (segs_stats, results)
      _, results[f"{acc}"] = OasisSolver(
        segmented_networks,
        cost_objective,
        acc,
        tot_mem_test,
        test_type
      )

    pprint.pprint(results, width=200, indent=2, sort_dicts=False)
    generate_mm_SETcomp_csv(results, "./output/mm_SETcomp.csv")
