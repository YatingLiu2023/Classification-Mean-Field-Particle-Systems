"""Validated experiment configurations and documented profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any, TypeVar


@dataclass(frozen=True)
class MainExperimentConfig:
    profile: str
    base_seed: int
    total_time: float
    simulation_steps: int
    observation_steps: tuple[int, ...]
    train_particles_per_class: tuple[int, ...]
    test_particles_per_class: int
    auxiliary_particles_per_class: int
    repetitions: int
    theta0_values: tuple[float, ...]
    D: int
    degree: int
    domain_half_width: float
    priors: tuple[float, float]
    clip_estimator: bool
    pair_chunk_size: int

    def validate(self) -> None:
        if self.profile not in {"smoke", "paper", "custom"}:
            raise ValueError("profile must be smoke, paper, or custom.")
        if self.repetitions < 1 or self.D < 1:
            raise ValueError("repetitions and D must be positive.")
        if any(self.simulation_steps % value for value in self.observation_steps):
            raise ValueError("Every observation_steps value must divide simulation_steps.")
        if min((*self.train_particles_per_class, self.test_particles_per_class, self.auxiliary_particles_per_class)) < 2:
            raise ValueError("Every particle-system size must be at least two.")


@dataclass(frozen=True)
class DSelectionConfig:
    profile: str
    base_seed: int
    total_time: float
    simulation_steps: int
    observation_steps: tuple[int, ...]
    total_particles_per_class: tuple[int, ...]
    repetitions: int
    theta0_values: tuple[float, ...]
    D_candidates: tuple[int, ...]
    train_fraction: float
    degree: int
    domain_half_width: float
    priors: tuple[float, float]
    clip_estimator: bool
    pair_chunk_size: int

    def validate(self) -> None:
        if self.profile not in {"smoke", "paper", "custom"}:
            raise ValueError("profile must be smoke, paper, or custom.")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be strictly between zero and one.")
        if self.repetitions < 1 or not self.D_candidates:
            raise ValueError("At least one repetition and D candidate are required.")
        if tuple(sorted(set(self.D_candidates))) != self.D_candidates:
            raise ValueError("D_candidates must be sorted and unique.")
        if any(self.simulation_steps % value for value in self.observation_steps):
            raise ValueError("Every observation_steps value must divide simulation_steps.")


@dataclass(frozen=True)
class ScenarioBConfig:
    profile: str
    base_seed: int
    total_time: float
    simulation_steps: int
    observation_steps: tuple[int, ...]
    train_particles_per_class: tuple[int, ...]
    test_particles_per_class: int
    auxiliary_particles_per_class: int
    repetitions: int
    theta_values: tuple[float, ...]
    D: int
    degree: int
    domain_half_width: float
    clip_estimator: bool
    pair_chunk_size: int

    def validate(self) -> None:
        if self.profile not in {"smoke", "paper", "custom"}:
            raise ValueError("profile must be smoke, paper, or custom.")
        if len(self.theta_values) < 2 or len(set(self.theta_values)) != len(self.theta_values):
            raise ValueError("Scenario B requires at least two distinct class parameters.")
        if self.repetitions < 1 or self.D < 1:
            raise ValueError("repetitions and D must be positive.")
        if any(self.simulation_steps % value for value in self.observation_steps):
            raise ValueError("Every observation_steps value must divide simulation_steps.")
        if min((*self.train_particles_per_class, self.test_particles_per_class, self.auxiliary_particles_per_class)) < 2:
            raise ValueError("Every particle-system size must be at least two.")


@dataclass(frozen=True)
class NeuralBaselineConfig:
    profile: str
    scenario: str
    base_seed: int
    total_time: float
    simulation_steps: int
    observation_steps: tuple[int, ...]
    train_particles_per_class: tuple[int, ...]
    test_particles_per_class: int
    repetitions: int
    theta_values: tuple[float, ...]
    trials: int
    tune_epochs: int
    max_epochs: int
    batch_size: int
    patience: int
    validation_protocol: str
    sparsity_budget: int

    def validate(self) -> None:
        if self.profile not in {"smoke", "paper", "custom"}:
            raise ValueError("profile must be smoke, paper, or custom.")
        if self.scenario not in {"A", "B"}:
            raise ValueError("scenario must be A or B.")
        if self.trials < 1 or self.max_epochs < 1 or self.repetitions < 1:
            raise ValueError("trials, epochs, and repetitions must be positive.")
        if self.validation_protocol != "empirical_measure_system":
            raise ValueError("validation_protocol must be empirical_measure_system.")
        if any(self.simulation_steps % value for value in self.observation_steps):
            raise ValueError("Every observation_steps value must divide simulation_steps.")


def main_profile(profile: str) -> MainExperimentConfig:
    if profile == "smoke":
        config = MainExperimentConfig(
            profile="smoke", base_seed=20260902, total_time=1.0, simulation_steps=20,
            observation_steps=(10, 20), train_particles_per_class=(32, 48),
            test_particles_per_class=48, auxiliary_particles_per_class=48,
            repetitions=2, theta0_values=(1.0, 0.5), D=2, degree=2,
            domain_half_width=6.0, priors=(0.5, 0.5), clip_estimator=False,
            pair_chunk_size=64,
        )
    elif profile == "paper":
        config = MainExperimentConfig(
            profile="paper", base_seed=20260902, total_time=1.0, simulation_steps=100,
            observation_steps=(10, 100), train_particles_per_class=(100, 1000),
            test_particles_per_class=1000, auxiliary_particles_per_class=1000,
            repetitions=50, theta0_values=(1.0, 0.5), D=2, degree=2,
            domain_half_width=6.0, priors=(0.5, 0.5), clip_estimator=False,
            pair_chunk_size=128,
        )
    else:
        raise ValueError("Use profile='smoke' or profile='paper'.")
    config.validate()
    return config


def d_selection_profile(profile: str) -> DSelectionConfig:
    if profile == "smoke":
        config = DSelectionConfig(
            profile="smoke", base_seed=20260903, total_time=1.0, simulation_steps=20,
            observation_steps=(10,), total_particles_per_class=(40,), repetitions=2,
            theta0_values=(1.0, 0.5), D_candidates=(2, 5), train_fraction=0.6,
            degree=2, domain_half_width=6.0, priors=(0.5, 0.5),
            clip_estimator=False, pair_chunk_size=64,
        )
    elif profile == "paper":
        config = DSelectionConfig(
            profile="paper", base_seed=20260903, total_time=1.0, simulation_steps=100,
            observation_steps=(10, 100), total_particles_per_class=(100, 1000),
            repetitions=50, theta0_values=(1.0, 0.5), D_candidates=(2, 5, 10, 15, 20),
            train_fraction=0.6, degree=2, domain_half_width=6.0,
            priors=(0.5, 0.5), clip_estimator=False, pair_chunk_size=128,
        )
    else:
        raise ValueError("Use profile='smoke' or profile='paper'.")
    config.validate()
    return config


def scenario_b_profile(profile: str) -> ScenarioBConfig:
    if profile == "smoke":
        config = ScenarioBConfig(
            profile="smoke", base_seed=20260905, total_time=1.0, simulation_steps=20,
            observation_steps=(10,), train_particles_per_class=(24,),
            test_particles_per_class=32, auxiliary_particles_per_class=32,
            repetitions=1, theta_values=(-6.0, -3.0, 0.0, 3.0, 6.0),
            D=2, degree=2, domain_half_width=12.0,
            clip_estimator=False, pair_chunk_size=64,
        )
    elif profile == "paper":
        config = ScenarioBConfig(
            profile="paper", base_seed=20260905, total_time=1.0, simulation_steps=100,
            observation_steps=(10, 100), train_particles_per_class=(100, 1000),
            test_particles_per_class=1000, auxiliary_particles_per_class=1000,
            repetitions=50, theta_values=(-6.0, -3.0, 0.0, 3.0, 6.0),
            D=20, degree=2, domain_half_width=12.0,
            clip_estimator=False, pair_chunk_size=128,
        )
    else:
        raise ValueError("Use profile='smoke' or profile='paper'.")
    config.validate()
    return config


def neural_baseline_profile(profile: str, scenario: str) -> NeuralBaselineConfig:
    scenario = scenario.upper()
    if scenario == "A":
        theta_values = (1.0, 0.5)
    elif scenario == "B":
        theta_values = (-6.0, -3.0, 0.0, 3.0, 6.0)
    else:
        raise ValueError("scenario must be A or B.")
    if profile == "smoke":
        config = NeuralBaselineConfig(
            profile="smoke", scenario=scenario, base_seed=20260906,
            total_time=1.0, simulation_steps=20, observation_steps=(10,),
            train_particles_per_class=(24,), test_particles_per_class=32,
            repetitions=1, theta_values=theta_values, trials=2, tune_epochs=3,
            max_epochs=8, batch_size=64, patience=3,
            validation_protocol="empirical_measure_system", sparsity_budget=500,
        )
    elif profile == "paper":
        config = NeuralBaselineConfig(
            profile="paper", scenario=scenario, base_seed=20260906,
            total_time=1.0, simulation_steps=100, observation_steps=(10, 100),
            train_particles_per_class=(100, 1000), test_particles_per_class=1000,
            repetitions=50, theta_values=theta_values, trials=20, tune_epochs=20,
            max_epochs=500, batch_size=256, patience=25,
            validation_protocol="empirical_measure_system", sparsity_budget=500,
        )
    else:
        raise ValueError("Use profile='smoke' or profile='paper'.")
    config.validate()
    return config


ConfigType = TypeVar(
    "ConfigType", MainExperimentConfig, DSelectionConfig,
    ScenarioBConfig, NeuralBaselineConfig,
)


def save_config(config: ConfigType, path: Path) -> None:
    payload = json.loads(json.dumps(asdict(config)))
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(
                f"Output directory already contains a different configuration: {path}"
            )
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_config(path: Path, config_type: type[ConfigType]) -> ConfigType:
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    tuple_names = {
        item.name for item in fields(config_type) if "tuple" in str(item.type).lower()
    }
    for name in tuple_names:
        if name in raw:
            raw[name] = tuple(raw[name])
    config = config_type(**raw)
    config.validate()
    return config
