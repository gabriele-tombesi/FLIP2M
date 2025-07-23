# Copyright (c) 2011-2025 Columbia University, System Level Design Group
# SPDX-License-Identifier: Apache-2.0

from test_FLIP2M import *

def pprint_aligned(d, indent=0, indent_step=4):
    keys = list(d.keys())
    max_key_width = max(len(str(k)) for k in keys) if keys else 0

    for k, v in d.items():
        key_str = str(k).ljust(max_key_width)
        prefix = ' ' * indent + key_str + ': '
        if isinstance(v, dict):
            print(prefix)
            pprint_aligned(v, indent + indent_step, indent_step)
        else:
            print(prefix + str(v))


def generate_sm_csv(data, filename):
    metrics  = ['latency', 'energy', 'edp']
    nets     = list(data.keys())
    variants = list(next(iter(data.values())).keys())

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)

        for metric in metrics:
            writer.writerow(['', metric.upper()])
            writer.writerow([''] + variants)
            for n in nets:
                row = [n] + [data[n][v][metric] for v in variants]
                writer.writerow(row)
            writer.writerow([])

def generate_sm_TANGRAMcomp_csv(data, out_path='results.csv'):
    metrics = [
        ('latency',   'latency'),
        ('energy',    'energy'),
        ('dram_acc',  'DRAM Access'),
    ]

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')

        for model_name, by_nacc in data.items():
            writer.writerow([model_name])

            naccs = sorted(by_nacc.keys(), key=int)

            for i, (key, display_name) in enumerate(metrics):
                writer.writerow([display_name])

                if i == 0:
                    writer.writerow([''] + [f'nacc = {n}' for n in naccs])

                writer.writerow([''] + [by_nacc[n][key] for n in naccs])

            writer.writerow([])



def generate_mm_csv(data, path, delimiter=','):
    variants = list(next(iter(data.values())).keys())

    groups = [
        ('latency', 'Latency'),
        ('energy',  'Energy'),
        ('edp',     'EDP'),
    ]

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=delimiter)

        writer.writerow(['', ''] + variants)

        for key, label in groups:
            writer.writerow(['', label])

            for scenario, stats in data.items():
                row = ['', scenario] + [stats[v][key] for v in variants]
                writer.writerow(row)

            writer.writerow([])

def generate_mm_SETcomp_csv(data, out_path):
    naccs = sorted(int(k) for k in data.keys() if k != '24')
    naccs = [str(n) for n in naccs]

    metrics = [
        ('latency', 'delay'),
        ('energy',  'energy'),
        ('edp',     'EDP'),
    ]

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)

        writer.writerow([''] + naccs)

        for key, label in metrics:
            writer.writerow([label] + [data[n][key] for n in naccs])
