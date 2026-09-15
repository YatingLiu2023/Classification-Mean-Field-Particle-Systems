#!/usr/bin/env bash
set -euo pipefail

python scripts/run_d_selection.py --config configs/D_selection_paper.json --output results/paper_untruncated/D_selection_paper
python scripts/run_main_experiment.py --config configs/main_paper.json --output results/paper_untruncated/main_paper
python scripts/run_scenario_b.py --config configs/scenario_b_paper.json --output results/paper_untruncated/scenario_b_paper
python scripts/run_neural_baseline.py --scenario A --config configs/nn_scenario_a_paper.json --output results/nn_empirical_measure_single_finalist/nn_A_paper
python scripts/run_neural_baseline.py --scenario B --config configs/nn_scenario_b_paper.json --output results/nn_empirical_measure_single_finalist/nn_B_paper
