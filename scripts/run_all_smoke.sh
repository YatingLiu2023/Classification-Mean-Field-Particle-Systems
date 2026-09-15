#!/usr/bin/env bash
set -euo pipefail

python scripts/run_main_experiment.py --profile smoke --output results/main_smoke
python scripts/run_d_selection.py --profile smoke --output results/D_selection_smoke
python scripts/run_scenario_b.py --profile smoke --output results/scenario_b_smoke
python scripts/run_neural_baseline.py --scenario A --profile smoke --output results/nn_empirical_measure_A_smoke
python scripts/run_neural_baseline.py --scenario B --profile smoke --output results/nn_empirical_measure_B_smoke
