from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    if path.suffix.lower() in {".parquet"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() in {".csv"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def wide_monthly_to_long(df: pd.DataFrame, branch_col: str, value_name: str) -> pd.DataFrame:
    if branch_col not in df.columns:
        raise KeyError(f"Expected column '{branch_col}' in source for {value_name}")

    month_cols = [c for c in df.columns if c != branch_col]
    parsed = pd.to_datetime(month_cols, errors="coerce")
    month_cols = [c for c, dt in zip(month_cols, parsed) if not pd.isna(dt)]

    long_df = df[[branch_col] + month_cols].melt(
        id_vars=[branch_col], var_name="month", value_name=value_name
    )
    long_df["month"] = pd.to_datetime(long_df["month"]).dt.to_period("M").dt.to_timestamp()
    long_df["branch_id"] = long_df[branch_col].astype(str)
    return long_df[["branch_id", "month", value_name]]


def load_branch_month(incoming_path: Path, invoices_path: Path) -> pd.DataFrame:
    incoming = read_table(incoming_path)
    invoices = read_table(invoices_path)

    incoming_long = wide_monthly_to_long(incoming, "Код филиала", "summa_vo_vv_month")
    invoices_long = wide_monthly_to_long(invoices, "Код филиала", "stoimost_netto_month")

    common_branches = set(incoming_long["branch_id"]).intersection(invoices_long["branch_id"])
    incoming_long = incoming_long[incoming_long["branch_id"].isin(common_branches)]
    invoices_long = invoices_long[invoices_long["branch_id"].isin(common_branches)]

    df_branch_month = incoming_long.merge(invoices_long, on=["branch_id", "month"], how="outer")

    for col in ["summa_vo_vv_month", "stoimost_netto_month"]:
        df_branch_month[col] = pd.to_numeric(df_branch_month[col], errors="coerce")

    df_branch_month = df_branch_month.sort_values(["branch_id", "month"]).reset_index(drop=True)

    if df_branch_month.set_index(["branch_id", "month"]).index.has_duplicates:
        raise ValueError("Duplicate key (branch_id, month) after merge")

    return df_branch_month


def forecast_naive_last_year(history: pd.Series) -> float:
    if len(history) < 12:
        return np.nan
    return float(history.iloc[-12])


def forecast_moving_average(history: pd.Series, window: int) -> float:
    if len(history) < window:
        return np.nan
    return float(history.tail(window).mean())


def make_forecaster(method_cfg: dict):
    method_type = method_cfg.get("type")
    if method_type == "naive_last_year":
        return lambda s: forecast_naive_last_year(s)
    if method_type == "moving_average":
        window = int(method_cfg.get("window", 3))
        return lambda s: forecast_moving_average(s, window)
    raise ValueError(f"Unknown baseline type: {method_type}")


def backtest_baselines(
    df_branch_month: pd.DataFrame,
    targets: list[str],
    methods_cfg: dict[str, dict],
    min_history: int,
) -> pd.DataFrame:
    forecast_funcs = {name: make_forecaster(cfg) for name, cfg in methods_cfg.items()}

    records = []
    for branch_id, df_branch in df_branch_month.groupby("branch_id"):
        df_branch = df_branch.sort_values("month")
        for target in targets:
            series = df_branch[["month", target]].dropna()
            values = series[target].reset_index(drop=True)
            months = series["month"].reset_index(drop=True)
            for idx in range(min_history, len(values)):
                history = values.iloc[:idx]
                test_y = float(values.iloc[idx])
                test_month = months.iloc[idx]
                for method_name, forecaster in forecast_funcs.items():
                    pred = forecaster(history)
                    records.append(
                        {
                            "branch_id": branch_id,
                            "target": target,
                            "month": test_month,
                            "method": method_name,
                            "y_true": test_y,
                            "y_pred": pred,
                        }
                    )
    return pd.DataFrame.from_records(records)


def safe_mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float((y_true[mask] - y_pred[mask]).abs().div(y_true[mask].abs()).mean())


def safe_mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float((y_true - y_pred).abs().mean())


def compute_metrics(df_backtest: pd.DataFrame) -> pd.DataFrame:
    groups = ["branch_id", "target", "method"]
    df = df_backtest.dropna(subset=["y_true", "y_pred"]).copy()
    rows = []
    for keys, g in df.groupby(groups):
        rows.append(
            {
                "branch_id": keys[0],
                "target": keys[1],
                "method": keys[2],
                "mape": safe_mape(g["y_true"], g["y_pred"]),
                "mae": safe_mae(g["y_true"], g["y_pred"]),
                "n_backtest_points": len(g),
            }
        )
    return pd.DataFrame(rows)


def select_best_baseline(
    metrics: pd.DataFrame,
    df_branch_month: pd.DataFrame,
    targets: list[str],
    min_points: int,
) -> pd.DataFrame:
    metrics_filtered = metrics[metrics["n_backtest_points"] >= min_points].copy()

    best_rows = []
    for (branch_id, target), df_grp in metrics_filtered.groupby(["branch_id", "target"]):
        best = df_grp.sort_values(["mape", "mae"]).head(1).iloc[0]
        best_rows.append(
            {
                "branch_id": branch_id,
                "target": target,
                "best_method": best["method"],
                "best_mape": best["mape"],
                "best_mae": best["mae"],
                "n_backtest_points": best["n_backtest_points"],
            }
        )

    df_best = pd.DataFrame(best_rows)

    full_index = (
        df_branch_month[["branch_id"]]
        .assign(key=1)
        .merge(pd.DataFrame({"target": targets, "key": 1}), on="key")[["branch_id", "target"]]
        .drop_duplicates()
    )

    return full_index.merge(df_best, on=["branch_id", "target"], how="left")


def load_eda_portrait_flags(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.exists():
        return None
    df = pd.read_excel(path)
    if "Код филиала" not in df.columns:
        return None
    df = df.rename(columns={"Код филиала": "branch_id", "месяцев": "months_count"})
    df["branch_id"] = df["branch_id"].astype(str)
    return df


def build_eligibility(
    df_branch_month: pd.DataFrame,
    min_history_months: int,
    min_active_share: float,
    df_portrait: pd.DataFrame | None,
) -> tuple[dict[str, bool], dict[str, list[str]]]:
    eligible_map: dict[str, bool] = {}
    reasons_map: dict[str, list[str]] = {}

    portrait_map: dict[str, dict] = {}
    if df_portrait is not None and not df_portrait.empty:
        portrait_map = df_portrait.set_index("branch_id").to_dict(orient="index")

    for branch_id, df_branch in df_branch_month.groupby("branch_id"):
        branch_key = str(branch_id)
        df_branch = df_branch.sort_values("month")

        portrait_row = portrait_map.get(branch_key)
        if portrait_row and "months_count" in portrait_row:
            history_len = int(portrait_row["months_count"])
        else:
            history_len = int(df_branch["month"].nunique())
        active_mask = (df_branch[["summa_vo_vv_month", "stoimost_netto_month"]] > 0).any(axis=1)
        active_share = float(active_mask.mean()) if history_len > 0 else 0.0

        reasons: list[str] = []
        eligible = True

        if history_len < min_history_months:
            eligible = False
            reasons.append("history_lt_min")
        if active_share < min_active_share:
            eligible = False
            reasons.append("low_active_share")
        if portrait_row is None:
            reasons.append("no_eda_portrait")

        eligible_map[branch_key] = eligible
        reasons_map[branch_key] = reasons

    return eligible_map, reasons_map


def classify_quality_overall(
    compare: pd.DataFrame,
    eligible_map: dict[str, bool],
    reasons_map: dict[str, list[str]],
    threshold_a: float,
    threshold_b: float,
    min_points: int,
) -> pd.DataFrame:
    quality_records = []
    for _, row in compare.iterrows():
        branch = row["branch_id"]
        target = row["target"]
        eligible = eligible_map.get(branch, True)
        reasons = list(reasons_map.get(branch, []))

        method = row["best_overall_method"]
        mape = row["best_overall_mape"]

        if isinstance(method, str) and method.startswith("mvp_"):
            n_points = row.get("n_backtest_points_mvp")
        else:
            n_points = row.get("n_backtest_points_baseline")

        if not eligible:
            quality = "C"
            reasons.append("eligible_for_forecast=False")
        elif pd.isna(method) or pd.isna(mape):
            quality = "C"
            reasons.append("no_valid_method")
        elif mape <= threshold_a:
            quality = "A"
        elif mape <= threshold_b:
            quality = "B"
        else:
            quality = "C"
            reasons.append("mape_gt_threshold")

        if n_points is not None and n_points < min_points:
            reasons.append("insufficient_backtest_points")

        quality_records.append(
            {
                "branch_id": branch,
                "target": target,
                "best_overall_method": method,
                "mape": mape,
                "n_backtest_points": n_points,
                "eligible_for_forecast": eligible,
                "quality_class": quality,
                "reasons": reasons,
            }
        )

    return pd.DataFrame(quality_records)


def build_feature_frame(
    df: pd.DataFrame, target: str, lags: list[int], rolling_windows: list[int]
) -> pd.DataFrame:
    df = df[["branch_id", "month", target]].copy()
    df = df.sort_values(["branch_id", "month"])

    grp = df.groupby("branch_id", sort=False)
    for lag in lags:
        df[f"lag_{lag}"] = grp[target].shift(lag)

    shifted = grp[target].shift(1)
    for w in rolling_windows:
        df[f"roll_mean_{w}"] = (
            shifted.groupby(df["branch_id"]).rolling(w).mean().reset_index(level=0, drop=True)
        )
        df[f"roll_std_{w}"] = (
            shifted.groupby(df["branch_id"]).rolling(w).std().reset_index(level=0, drop=True)
        )

    df["month_num"] = df["month"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)
    df["quarter"] = df["month"].dt.quarter

    df["active_last_3"] = (
        shifted.groupby(df["branch_id"])
        .rolling(3)
        .apply(lambda x: float((x > 0).any()), raw=False)
        .reset_index(level=0, drop=True)
    )

    return df


def fit_ridge(X: pd.DataFrame, y: pd.Series, alpha: float = 1.0) -> dict:
    Xn = X.to_numpy()
    yn = y.to_numpy()
    Xn = np.hstack([np.ones((Xn.shape[0], 1)), Xn])
    eye = np.eye(Xn.shape[1])
    eye[0, 0] = 0.0
    coef = np.linalg.pinv(Xn.T @ Xn + alpha * eye) @ (Xn.T @ yn)
    return {"coef": coef}


def predict_ridge(model: dict, X: pd.DataFrame) -> np.ndarray:
    Xn = X.to_numpy()
    Xn = np.hstack([np.ones((Xn.shape[0], 1)), Xn])
    return Xn @ model["coef"]


def backtest_mvp(
    df_branch_month: pd.DataFrame,
    targets: list[str],
    lags: list[int],
    rolling_windows: list[int],
    min_train_rows: int,
    model_cfg: dict,
) -> pd.DataFrame:
    model_type = model_cfg.get("type", "catboost")

    catboost_available = False
    CatBoostRegressor = None
    if model_type == "catboost":
        try:
            from catboost import CatBoostRegressor  # type: ignore

            catboost_available = True
        except Exception:
            model_type = "ridge"

    records = []
    for target in targets:
        feat_df = build_feature_frame(df_branch_month, target, lags, rolling_windows)

        base_cols = [c for c in feat_df.columns if c not in {"branch_id", "month", target}]

        if model_type == "catboost" and catboost_available:
            feature_cols = base_cols + ["branch_id"]
            feat_df["branch_id"] = feat_df["branch_id"].astype(str)
            cat_feature_indices = [feature_cols.index("branch_id")]
        else:
            feat_df = feat_df.join(
                pd.get_dummies(feat_df["branch_id"], prefix="branch", drop_first=True)
            )
            feature_cols = [c for c in feat_df.columns if c not in {"branch_id", "month", target}]
            cat_feature_indices = None

        feat_df = feat_df.dropna(subset=feature_cols + [target]).reset_index(drop=True)

        months_sorted = sorted(feat_df["month"].unique())
        for test_month in months_sorted:
            train = feat_df[feat_df["month"] < test_month]
            test = feat_df[feat_df["month"] == test_month]
            if train.shape[0] < min_train_rows or test.empty:
                continue

            if model_type == "catboost" and catboost_available:
                model = CatBoostRegressor(
                    iterations=model_cfg.get("iterations", 300),
                    depth=model_cfg.get("depth", 6),
                    learning_rate=model_cfg.get("learning_rate", 0.1),
                    loss_function="MAE",
                    verbose=False,
                    random_seed=42,
                )
                model.fit(train[feature_cols], train[target], cat_features=cat_feature_indices)
                preds = model.predict(test[feature_cols])
            else:
                model = fit_ridge(
                    train[feature_cols], train[target], alpha=model_cfg.get("alpha", 1.0)
                )
                preds = predict_ridge(model, test[feature_cols])

            for row, pred in zip(test.itertuples(index=False), preds):
                records.append(
                    {
                        "branch_id": row.branch_id,
                        "target": target,
                        "month": row.month,
                        "method": f"mvp_{model_type}",
                        "y_true": float(getattr(row, target)),
                        "y_pred": float(pred),
                    }
                )

    return pd.DataFrame.from_records(records)


def compare_mvp_baseline(
    df_best_baseline: pd.DataFrame, df_mvp_metrics: pd.DataFrame
) -> pd.DataFrame:
    df_best = df_best_baseline.rename(columns={"n_backtest_points": "n_backtest_points_baseline"})
    df_mvp = df_mvp_metrics.rename(columns={"n_backtest_points": "n_backtest_points_mvp"})
    compare = df_best.merge(
        df_mvp[
            [
                "branch_id",
                "target",
                "mape",
                "mae",
                "method",
                "n_backtest_points_mvp",
            ]
        ],
        on=["branch_id", "target"],
        how="left",
        suffixes=("_baseline", "_mvp"),
    )

    compare["best_overall_mape"] = compare[["best_mape", "mape"]].min(axis=1)
    compare["best_overall_method"] = np.where(
        compare["mape"].notna() & (compare["mape"] < compare["best_mape"]),
        compare["method"],
        compare["best_method"],
    )
    return compare


def forecast_next_baseline(
    df_branch_month: pd.DataFrame,
    df_best_baseline: pd.DataFrame,
    methods_cfg: dict[str, dict],
) -> pd.DataFrame:
    forecast_funcs = {name: make_forecaster(cfg) for name, cfg in methods_cfg.items()}

    forecast_rows = []
    for branch_id, df_branch in df_branch_month.groupby("branch_id"):
        for target in df_best_baseline["target"].unique():
            best = df_best_baseline[
                (df_best_baseline["branch_id"] == branch_id)
                & (df_best_baseline["target"] == target)
            ]
            if best.empty:
                continue
            best = best.iloc[0]
            method_name = best["best_method"]
            if pd.isna(method_name):
                continue

            series = (
                df_branch.sort_values("month")[["month", target]].dropna().reset_index(drop=True)
            )
            history = series[target]
            next_month = (series["month"].max() + pd.offsets.MonthBegin(1)).normalize()

            forecaster = forecast_funcs.get(method_name)
            if forecaster is None:
                continue
            pred = forecaster(history)

            forecast_rows.append(
                {
                    "branch_id": str(branch_id),
                    "target": target,
                    "next_month": next_month,
                    "method": method_name,
                    "y_pred": pred,
                }
            )

    return pd.DataFrame(forecast_rows)


def make_next_month_features(
    df_branch: pd.DataFrame, target: str, lags: list[int], rolling_windows: list[int]
) -> dict:
    df_branch = df_branch.sort_values("month")
    series = df_branch[target].dropna()
    if series.empty:
        return {}

    features = {}
    for lag in lags:
        features[f"lag_{lag}"] = float(series.iloc[-lag]) if len(series) >= lag else np.nan

    for w in rolling_windows:
        if len(series) >= w:
            window_vals = series.iloc[-w:]
            features[f"roll_mean_{w}"] = float(window_vals.mean())
            features[f"roll_std_{w}"] = float(window_vals.std())
        else:
            features[f"roll_mean_{w}"] = np.nan
            features[f"roll_std_{w}"] = np.nan

    next_month = (df_branch["month"].max() + pd.offsets.MonthBegin(1)).normalize()
    features["month_num"] = int(next_month.month)
    features["month_sin"] = float(np.sin(2 * np.pi * features["month_num"] / 12))
    features["month_cos"] = float(np.cos(2 * np.pi * features["month_num"] / 12))
    features["quarter"] = int(((features["month_num"] - 1) // 3) + 1)
    features["active_last_3"] = float((series.tail(3) > 0).any()) if len(series) >= 1 else np.nan

    return features


def forecast_next_mvp(
    df_branch_month: pd.DataFrame,
    targets: list[str],
    lags: list[int],
    rolling_windows: list[int],
    model_cfg: dict,
) -> pd.DataFrame:
    model_type = model_cfg.get("type", "catboost")

    catboost_available = False
    CatBoostRegressor = None
    if model_type == "catboost":
        try:
            from catboost import CatBoostRegressor  # type: ignore

            catboost_available = True
        except Exception:
            model_type = "ridge"

    rows = []
    for target in targets:
        feat_df = build_feature_frame(df_branch_month, target, lags, rolling_windows)

        base_cols = [c for c in feat_df.columns if c not in {"branch_id", "month", target}]

        if model_type == "catboost" and catboost_available:
            feature_cols = base_cols + ["branch_id"]
            feat_df["branch_id"] = feat_df["branch_id"].astype(str)
            cat_feature_indices = [feature_cols.index("branch_id")]
        else:
            feat_df = feat_df.join(
                pd.get_dummies(feat_df["branch_id"], prefix="branch", drop_first=True)
            )
            feature_cols = [c for c in feat_df.columns if c not in {"branch_id", "month", target}]
            cat_feature_indices = None

        feat_df = feat_df.dropna(subset=feature_cols + [target]).reset_index(drop=True)
        if feat_df.empty:
            continue

        if model_type == "catboost" and catboost_available:
            model = CatBoostRegressor(
                iterations=model_cfg.get("iterations", 300),
                depth=model_cfg.get("depth", 6),
                learning_rate=model_cfg.get("learning_rate", 0.1),
                loss_function="MAE",
                verbose=False,
                random_seed=42,
            )
            model.fit(feat_df[feature_cols], feat_df[target], cat_features=cat_feature_indices)
        else:
            model = fit_ridge(
                feat_df[feature_cols], feat_df[target], alpha=model_cfg.get("alpha", 1.0)
            )

        for branch_id, df_branch in df_branch_month.groupby("branch_id"):
            features = make_next_month_features(
                df_branch[["month", target]].dropna(), target, lags, rolling_windows
            )
            if not features:
                continue

            next_month = (df_branch["month"].max() + pd.offsets.MonthBegin(1)).normalize()

            row = {"branch_id": str(branch_id), "month": next_month}
            row.update(features)

            if model_type == "catboost" and catboost_available:
                X = pd.DataFrame([row])[feature_cols]
                pred = float(model.predict(X)[0])
            else:
                X = pd.DataFrame([row])
                X = X.join(pd.get_dummies(X["branch_id"], prefix="branch", drop_first=True))
                X = X.reindex(columns=feature_cols, fill_value=0.0)
                pred = float(predict_ridge(model, X)[0])

            rows.append(
                {
                    "branch_id": str(branch_id),
                    "target": target,
                    "next_month": next_month,
                    "method": f"mvp_{model_type}",
                    "y_pred": pred,
                }
            )

    return pd.DataFrame(rows)


def build_ci_map(
    compare: pd.DataFrame,
    df_backtest_baseline: pd.DataFrame,
    df_backtest_mvp: pd.DataFrame,
    quantiles: tuple[float, float],
    min_points: int,
) -> dict[tuple[str, str], tuple[float, float]]:
    q_low, q_high = quantiles
    residuals = []

    for _, row in compare.iterrows():
        branch_id = row["branch_id"]
        target = row["target"]
        method = row["best_overall_method"]
        if pd.isna(method):
            continue

        if isinstance(method, str) and method.startswith("mvp_"):
            df_bt = df_backtest_mvp
        else:
            df_bt = df_backtest_baseline

        df_sel = df_bt[
            (df_bt["branch_id"] == branch_id)
            & (df_bt["target"] == target)
            & (df_bt["method"] == method)
        ]
        if df_sel.empty or len(df_sel) < min_points:
            continue

        res = (df_sel["y_true"] - df_sel["y_pred"]).astype(float)
        residuals.append(
            {
                "branch_id": branch_id,
                "target": target,
                "q_low": float(res.quantile(q_low)),
                "q_high": float(res.quantile(q_high)),
            }
        )

    ci_map = {(r["branch_id"], r["target"]): (r["q_low"], r["q_high"]) for r in residuals}

    if not residuals:
        return ci_map

    # Fallback: global residuals per target for missing branches
    for target in compare["target"].unique():
        res_all = []
        for df_bt in [df_backtest_baseline, df_backtest_mvp]:
            df_t = df_bt[df_bt["target"] == target]
            res_all.extend((df_t["y_true"] - df_t["y_pred"]).dropna().tolist())
        if not res_all:
            continue
        ql = float(pd.Series(res_all).quantile(q_low))
        qh = float(pd.Series(res_all).quantile(q_high))
        for branch_id in compare[compare["target"] == target]["branch_id"].unique():
            key = (branch_id, target)
            if key not in ci_map:
                ci_map[key] = (ql, qh)

    return ci_map


def select_best_forecast(
    compare: pd.DataFrame,
    df_forecast_next: pd.DataFrame,
    df_mvp_forecast_next: pd.DataFrame,
    df_quality_final: pd.DataFrame,
    ci_map: dict[tuple[str, str], tuple[float, float]],
    low_confidence_ratio: float,
    include_c: bool = False,
) -> pd.DataFrame:
    best_rows = []

    for _, row in compare.iterrows():
        branch_id = row["branch_id"]
        target = row["target"]
        best_method = row["best_overall_method"]

        quality_row = df_quality_final[
            (df_quality_final["branch_id"] == branch_id) & (df_quality_final["target"] == target)
        ]
        if quality_row.empty:
            continue
        quality_row = quality_row.iloc[0]
        if quality_row["quality_class"] == "C" and not include_c:
            continue

        if isinstance(best_method, str) and best_method.startswith("mvp_"):
            pred_row = df_mvp_forecast_next[
                (df_mvp_forecast_next["branch_id"] == branch_id)
                & (df_mvp_forecast_next["target"] == target)
            ]
        else:
            pred_row = df_forecast_next[
                (df_forecast_next["branch_id"] == branch_id)
                & (df_forecast_next["target"] == target)
            ]

        if pred_row.empty:
            continue

        pred_row = pred_row.iloc[0]
        ci_key = (branch_id, target)
        ci_vals = ci_map.get(ci_key)
        if ci_vals:
            ci_low, ci_high = ci_vals
        else:
            ci_low, ci_high = (np.nan, np.nan)

        pred_val = float(pred_row["y_pred"])
        if pd.isna(pred_val) or pd.isna(ci_low) or pd.isna(ci_high):
            low_confidence = True
        else:
            denom = max(abs(pred_val), 1.0)
            low_confidence = ((ci_high - ci_low) / denom) > low_confidence_ratio

        best_rows.append(
            {
                "branch_id": branch_id,
                "target": target,
                "next_month": pred_row["next_month"],
                "method": best_method,
                "y_pred": pred_row["y_pred"],
                "ci_lower": pred_val + ci_low if not pd.isna(ci_low) else np.nan,
                "ci_upper": pred_val + ci_high if not pd.isna(ci_high) else np.nan,
                "low_confidence": low_confidence,
                "quality_class": quality_row["quality_class"],
            }
        )

    return pd.DataFrame(best_rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
