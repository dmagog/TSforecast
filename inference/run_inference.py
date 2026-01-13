from __future__ import annotations

import sys
import time
from pathlib import Path

if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parents[1]))


from inference.pipeline import (
    backtest_baselines,
    backtest_mvp,
    build_ci_map,
    build_eligibility,
    classify_quality_overall,
    compare_mvp_baseline,
    compute_metrics,
    forecast_next_baseline,
    forecast_next_mvp,
    load_branch_month,
    load_eda_portrait_flags,
    select_best_baseline,
    select_best_forecast,
    write_json,
)

CONFIG = {
    "paths": {
        "incoming_monthly": Path("eda/data-mod/Поступления (по месячно).xlsx"),
        "invoices_monthly": Path("eda/data-mod/Фактуры (по месячно).xlsx"),
        "eda_portrait": Path("eda/data-mod/Портрет филиалов.xlsx"),
        "output_dir": Path("outputs/inference"),
    },
    "targets": ["stoimost_netto_month", "summa_vo_vv_month"],
    "backtesting": {
        "min_history_months": 12,
        "min_points_per_method": 6,
        "methods": {
            "naive_t12": {"type": "naive_last_year"},
            "ma_3": {"type": "moving_average", "window": 3},
            "ma_6": {"type": "moving_average", "window": 6},
            "ma_12": {"type": "moving_average", "window": 12},
        },
    },
    "mvp": {
        "model": {"type": "catboost", "iterations": 300, "depth": 6, "learning_rate": 0.1},
        "lags": list(range(1, 13)),
        "rolling_windows": [3, 6, 12],
        "min_train_rows": 200,
    },
    "eligibility": {"min_history_months": 12, "min_active_share": 0.5},
    "ci": {"quantiles": (0.1, 0.9), "min_points": 6, "low_confidence_ratio": 0.5},
    "classification": {"threshold_a": 0.10, "threshold_b": 0.15},
}


def run_inference() -> dict:
    t0 = time.time()

    df_branch_month = load_branch_month(
        CONFIG["paths"]["incoming_monthly"],
        CONFIG["paths"]["invoices_monthly"],
    )

    df_portrait = load_eda_portrait_flags(CONFIG["paths"]["eda_portrait"])
    eligible_map, reasons_map = build_eligibility(
        df_branch_month,
        CONFIG["eligibility"]["min_history_months"],
        CONFIG["eligibility"]["min_active_share"],
        df_portrait,
    )

    df_backtest = backtest_baselines(
        df_branch_month,
        CONFIG["targets"],
        CONFIG["backtesting"]["methods"],
        CONFIG["backtesting"]["min_history_months"],
    )
    df_baseline_metrics = compute_metrics(df_backtest)
    df_best_baseline = select_best_baseline(
        df_baseline_metrics,
        df_branch_month,
        CONFIG["targets"],
        CONFIG["backtesting"]["min_points_per_method"],
    )

    df_forecast_next = forecast_next_baseline(
        df_branch_month,
        df_best_baseline,
        CONFIG["backtesting"]["methods"],
    )

    df_mvp_backtest = backtest_mvp(
        df_branch_month,
        CONFIG["targets"],
        CONFIG["mvp"]["lags"],
        CONFIG["mvp"]["rolling_windows"],
        CONFIG["mvp"]["min_train_rows"],
        CONFIG["mvp"]["model"],
    )
    df_mvp_metrics = compute_metrics(df_mvp_backtest)
    compare = compare_mvp_baseline(df_best_baseline, df_mvp_metrics)
    df_quality_final = classify_quality_overall(
        compare,
        eligible_map,
        reasons_map,
        CONFIG["classification"]["threshold_a"],
        CONFIG["classification"]["threshold_b"],
        CONFIG["backtesting"]["min_points_per_method"],
    )

    df_mvp_forecast_next = forecast_next_mvp(
        df_branch_month,
        CONFIG["targets"],
        CONFIG["mvp"]["lags"],
        CONFIG["mvp"]["rolling_windows"],
        CONFIG["mvp"]["model"],
    )

    ci_map = build_ci_map(
        compare,
        df_backtest,
        df_mvp_backtest,
        CONFIG["ci"]["quantiles"],
        CONFIG["ci"]["min_points"],
    )

    df_forecast_next_best = select_best_forecast(
        compare,
        df_forecast_next,
        df_mvp_forecast_next,
        df_quality_final,
        ci_map,
        CONFIG["ci"]["low_confidence_ratio"],
    )

    out_dir = CONFIG["paths"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    df_quality_out = df_quality_final.copy()
    df_quality_out["reasons"] = df_quality_out["reasons"].apply(lambda x: ",".join(x))

    active_share = (
        df_branch_month.groupby("branch_id")
        .apply(
            lambda g: ((g[["summa_vo_vv_month", "stoimost_netto_month"]] > 0).any(axis=1)).mean()
        )
        .rename("active_share")
        .reset_index()
    )
    df_quality_out = df_quality_out.merge(active_share, on="branch_id", how="left")

    if df_portrait is not None and not df_portrait.empty:
        cols = ["branch_id", "months_count"]
        if "Сумма во ВВ" in df_portrait.columns:
            cols.append("Сумма во ВВ")
        if "Дебитор" in df_portrait.columns:
            cols.append("Дебитор")

        df_portrait_out = df_portrait[cols].copy()
        df_portrait_out = df_portrait_out.rename(
            columns={
                "months_count": "eda_months_count",
                "Сумма во ВВ": "eda_summa_vv_total",
                "Дебитор": "eda_debitor",
            }
        )
        df_quality_out = df_quality_out.merge(df_portrait_out, on="branch_id", how="left")

    df_quality_out.to_csv(out_dir / "quality_by_branch.csv", index=False)
    df_forecast_next_best.to_csv(out_dir / "forecast_next_month.csv", index=False)
    compare.to_csv(out_dir / "mvp_vs_baseline.csv", index=False)

    if not df_forecast_next.empty:
        df_forecast_next.to_csv(out_dir / "baseline_forecast_next_month.csv", index=False)
    if not df_mvp_forecast_next.empty:
        df_mvp_forecast_next.to_csv(out_dir / "mvp_forecast_next_month.csv", index=False)

    share_overall = None
    share_by_target = None
    if not compare.empty:
        comp = compare.dropna(subset=["best_mape", "mape"]).copy()
        if not comp.empty:
            comp["mvp_better"] = comp["mape"] < comp["best_mape"]
            share_overall = float(comp["mvp_better"].mean())
            share_by_target = (
                comp.groupby("target")["mvp_better"].mean().rename("mvp_better_share")
            ).to_dict()

    summary = {
        "branches_total": int(df_branch_month["branch_id"].nunique()),
        "series_total": int(len(df_quality_final)),
        "class_counts": df_quality_final["quality_class"].value_counts().to_dict(),
        "mvp_better_share_overall": share_overall,
        "mvp_better_share_by_target": share_by_target,
        "artifacts": {
            "forecast_next_month": str(out_dir / "forecast_next_month.csv"),
            "quality_by_branch": str(out_dir / "quality_by_branch.csv"),
            "mvp_vs_baseline": str(out_dir / "mvp_vs_baseline.csv"),
        },
        "runtime_sec": round(time.time() - t0, 2),
    }

    write_json(out_dir / "run_summary.json", summary)
    return summary


if __name__ == "__main__":
    summary = run_inference()
    print(summary)
