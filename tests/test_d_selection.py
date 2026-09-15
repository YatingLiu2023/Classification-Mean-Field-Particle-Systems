from mkv_classification.experiments import _d_analysis


def test_one_standard_error_rule_handles_one_repetition() -> None:
    rows = []
    for strategy in ("within", "independent"):
        for D, accuracy, mse in ((2, 0.70, 0.20), (5, 0.80, 0.10)):
            rows.append(
                {
                    "regime": "A(i)",
                    "M": 10,
                    "N_total": 40,
                    "repetition": 0,
                    "strategy": strategy,
                    "D": D,
                    "validation_accuracy": accuracy,
                    "drift_mse_mean": mse,
                    "fit_seconds": 0.01,
                }
            )

    analysis = _d_analysis(rows)

    assert len(analysis["one_se"]) == 2
    assert all(row["classification_one_se_D"] == 5 for row in analysis["one_se"])
    assert all(row["drift_one_se_D"] == 5 for row in analysis["one_se"])
