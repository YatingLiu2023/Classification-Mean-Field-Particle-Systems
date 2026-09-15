# Supervised Classification for Interacting Particle Systems

This repository contains the code accompanying the manuscript *Who Drives the System? Classifying Mean-Field Particle Systems from Trajectory Data.* by Christophe Denis, Charlotte Dion-Blanc, Yating Liu.

It implements the numerical experiments for:

* the two-class setting of Scenario A;
* the five-class setting of Scenario B;
* the B-spline plug-in classifier and drift estimation;
* Hyperparameter selection of the spline dimension \(D\);
* direct neural-network classification baselines;
* the model figures used in the paper.

The repository also contains frozen outputs from the numerical experiments reported in the manuscript.

For a detailed correspondence between paper figures/tables and executable commands, see [`docs/REPRODUCING_PAPER.md`](docs/REPRODUCING_PAPER.md).

## Installation

Python 3.10 or newer is required. 

```bash
python -m pip install -e '.[dev]'
```

Run the test suite with:

```bash
pytest
```

## Quick check

A lightweight smoke run is provided to verify that the main pipelines execute correctly (Its outputs should not be used as paper results):

```bash
bash scripts/run_all_smoke.sh
```

## Reproducing the numerical experiments

To run all paper configurations sequentially:

```bash
bash scripts/run_all_paper.sh
```

Individual experiments can also be run separately. For example, the Scenario A plug-in experiment is launched with:

```bash
python scripts/run_main_experiment.py \
  --config configs/main_paper.json \
  --output results/main_paper
```

The complete experiment-to-command correspondence is given in [`docs/REPRODUCING_PAPER.md`](docs/REPRODUCING_PAPER.md).

## Experimental protocol

The main implementation details relevant for reproducing the numerical results are summarized below.

* `N` denotes the number of particles **per class and per interacting particle system**.
* The SDEs are simulated on a fixed time grid. Coarser observation grids are obtained by regular subsampling of the simulated trajectories.
* Independent sources of randomness are used across repetitions, classes, and simulated systems. In particular, separate systems are used for training, testing, and the evaluation of drift-estimation errors.
* The reported plug-in experiments use the practical untruncated estimator (`clip_estimator=false`). The optional truncated version is retained only for large `N`.
* Classification is performed using log-scores directly.
* Reported Monte Carlo summaries use the sample standard deviation (`ddof=1`), standard error, and a 95% confidence interval.

For the neural-network baseline, each repetition uses independent systems for training, validation, and final testing. Hyperparameter selection is performed only on the validation system. A single selected configuration is then reinitialized and fully trained once before evaluation on the final test system. All pre-specified repetitions are retained, including repetitions exhibiting poor performance or class collapse.

## Frozen reference results

The directory [`reference_results/`](reference_results/) contains results in the reported numerical summaries.

In particular:

```text
reference_results/
├── paper_untruncated_20260909/
│   ├── D_selection_paper/
│   ├── main_paper/
│   └── scenario_b_paper/
└── nn_empirical_measure_single_finalist_20260911/
    ├── nn_A_paper/
    └── nn_B_paper/
```

## Repository structure

```text
configs/             Frozen configurations used for the paper experiments
docs/                Detailed reproduction instructions
reference_results/   Frozen numerical results
scripts/             Command-line entry points and experiment runners
src/                 Python package
tests/               Numerical and protocol tests
results/             Locally generated outputs, ignored by Git
```

The scientific implementation is contained in `src/mkv_classification/`. The scripts in `scripts/` provide thin command-line interfaces to the same package used by the tests.
