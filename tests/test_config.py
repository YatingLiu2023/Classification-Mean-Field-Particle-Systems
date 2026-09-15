from dataclasses import replace

import pytest

from mkv_classification.config import (
    MainExperimentConfig,
    d_selection_profile,
    load_config,
    main_profile,
    neural_baseline_profile,
    save_config,
    scenario_b_profile,
)


def test_profiles_validate() -> None:
    main_profile("smoke").validate()
    d_selection_profile("smoke").validate()
    neural_baseline_profile("smoke", "A").validate()
    neural_baseline_profile("smoke", "B").validate()


def test_neural_validation_uses_empirical_measure_system() -> None:
    assert (
        neural_baseline_profile("paper", "A").validation_protocol
        == "empirical_measure_system"
    )


def test_practical_profiles_are_untruncated() -> None:
    for profile in ("smoke", "paper"):
        assert main_profile(profile).clip_estimator is False
        assert d_selection_profile(profile).clip_estimator is False
        assert scenario_b_profile(profile).clip_estimator is False


def test_invalid_observation_grid_is_rejected() -> None:
    config = replace(main_profile("smoke"), observation_steps=(7,))
    with pytest.raises(ValueError):
        config.validate()


def test_json_loader_preserves_profile_label(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        '{"profile":"paper","base_seed":1,"total_time":1.0,'
        '"simulation_steps":10,"observation_steps":[10],'
        '"train_particles_per_class":[10],"test_particles_per_class":10,'
        '"auxiliary_particles_per_class":10,"repetitions":1,'
        '"theta0_values":[1.0],"D":2,"degree":2,'
        '"domain_half_width":6.0,"priors":[0.5,0.5],'
        '"clip_estimator":true,"pair_chunk_size":8}',
        encoding="utf-8",
    )
    assert load_config(path, MainExperimentConfig).profile == "paper"


def test_checkpoint_rejects_a_different_configuration(tmp_path) -> None:
    path = tmp_path / "config.json"
    save_config(main_profile("smoke"), path)
    with pytest.raises(ValueError, match="different configuration"):
        save_config(replace(main_profile("smoke"), D=5), path)
