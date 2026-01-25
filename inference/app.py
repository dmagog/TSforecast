from __future__ import annotations

import base64
import io
from functools import lru_cache
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from inference.pipeline import load_branch_month
from inference.run_inference import run_inference

app = FastAPI(title="TS Forecast Inference", version="0.1.0")

OUTPUT_DIR = Path("outputs/inference")
FORECAST_PATH = OUTPUT_DIR / "forecast_next_month_demo.csv"
QUALITY_PATH = OUTPUT_DIR / "quality_by_branch.csv"
SUMMARY_PATH = OUTPUT_DIR / "run_summary.json"
TEMPLATES = Jinja2Templates(directory="templates")

EDA_INCOMING = Path("eda/data-mod/Поступления (по месячно).xlsx")
EDA_INVOICES = Path("eda/data-mod/Фактуры (по месячно).xlsx")


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def _load_branch_month() -> pd.DataFrame:
    return load_branch_month(EDA_INCOMING, EDA_INVOICES)


@lru_cache(maxsize=1)
def _load_forecast_df() -> pd.DataFrame:
    return _load_csv(FORECAST_PATH)


@lru_cache(maxsize=1)
def _load_quality_df() -> pd.DataFrame:
    return _load_csv(QUALITY_PATH)


def _make_plot(
    history: pd.DataFrame,
    next_month: pd.Timestamp | None,
    model_pred: float | None,
    ci_lower: float | None,
    ci_upper: float | None,
    manager_forecast: float | None,
) -> str:
    matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(7, 3))

    ax.plot(history["month"], history["summa_vo_vv_month"], label="Поступления", marker="o")
    ax.plot(history["month"], history["stoimost_netto_month"], label="Отгрузки", marker="o")

    if next_month is not None and model_pred is not None:
        ax.plot(
            [history["month"].iloc[-1], next_month],
            [history["summa_vo_vv_month"].iloc[-1], model_pred],
            "--",
            color="C4",
        )
        ax.scatter([next_month], [model_pred], color="C4", label="Прогноз модели")

    if next_month is not None and ci_lower is not None and ci_upper is not None:
        ax.vlines(
            x=[next_month],
            ymin=[ci_lower],
            ymax=[ci_upper],
            color="C4",
            linestyles="--",
            linewidth=2,
            label="CI",
        )
        ax.scatter([next_month], [ci_lower], color="C4", marker="_")
        ax.scatter([next_month], [ci_upper], color="C4", marker="_")

    if next_month is not None and manager_forecast is not None:
        ax.scatter([next_month], [manager_forecast], color="C3", label="Прогноз менеджера")

    ax.set_title("Факт + прогноз + доверительный интервал")
    ax.set_xlabel("Месяц")
    ax.set_ylabel("Сумма")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.subplots_adjust(right=0.78)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


@app.get("/", response_class=HTMLResponse)
def ui(
    request: Request,
    branch_id: str | None = None,
    target: str | None = None,
    manager_forecast: str | None = None,
) -> HTMLResponse:
    try:
        df_forecast = _load_forecast_df()
        df_quality = _load_quality_df()
        df_branch_month = _load_branch_month()
    except FileNotFoundError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err

    branches = sorted(df_quality["branch_id"].astype(str).unique().tolist())
    targets = sorted(df_forecast["target"].unique().tolist())

    if branch_id is None and branches:
        branch_id = branches[0]
    if target is None and targets:
        target = targets[0]

    df_hist = df_branch_month[df_branch_month["branch_id"].astype(str) == str(branch_id)]
    df_hist = df_hist.sort_values("month")
    history = pd.DataFrame()
    if target and not df_hist.empty:
        history = pd.DataFrame(
            {
                "month": df_hist["month"],
                "summa_vo_vv_month": df_hist["summa_vo_vv_month"],
                "stoimost_netto_month": df_hist["stoimost_netto_month"],
            }
        ).dropna()

    forecast_row = df_forecast[
        (df_forecast["branch_id"].astype(str) == str(branch_id)) & (df_forecast["target"] == target)
    ]
    forecast_row = forecast_row.iloc[0] if not forecast_row.empty else None

    quality_row = df_quality[
        (df_quality["branch_id"].astype(str) == str(branch_id)) & (df_quality["target"] == target)
    ]
    quality_row = quality_row.iloc[0] if not quality_row.empty else None

    df_quality_target = df_quality[df_quality["target"] == target]
    class_counts = df_quality_target["quality_class"].value_counts().to_dict()
    class_share = df_quality_target["quality_class"].value_counts(normalize=True).round(3).to_dict()
    class_counts_all = df_quality["quality_class"].value_counts().to_dict()
    class_share_all = df_quality["quality_class"].value_counts(normalize=True).round(3).to_dict()
    mape_stats = df_quality["mape"].describe().round(3).to_dict()
    mape_by_target = df_quality.groupby("target")["mape"].describe().round(3).reset_index()
    mape_hist_b64 = ""
    if not df_quality["mape"].dropna().empty:
        matplotlib.use("Agg")
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.hist(df_quality["mape"].dropna(), bins=20, alpha=0.8, color="C0")
        ax.set_title("Распределение MAPE")
        ax.set_xlabel("MAPE")
        ax.set_ylabel("Count")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        mape_hist_b64 = base64.b64encode(buf.read()).decode("utf-8")
    class_by_branch = df_quality_target.set_index(df_quality_target["branch_id"].astype(str))[
        "quality_class"
    ].to_dict()

    next_month = None
    next_month_label = None
    excluded_from_prod = None
    model_pred = None
    ci_lower = None
    ci_upper = None
    low_confidence = None

    if forecast_row is not None:
        next_month = pd.to_datetime(forecast_row.get("next_month"))
        next_month_label = next_month.date().isoformat()
        model_pred = forecast_row.get("y_pred")
        ci_lower = forecast_row.get("ci_lower")
        ci_upper = forecast_row.get("ci_upper")
        low_confidence = forecast_row.get("low_confidence")
        excluded_from_prod = forecast_row.get("excluded_from_prod")
    elif not history.empty:
        next_month = history["month"].max() + pd.offsets.MonthBegin(1)
        next_month_label = next_month.date().isoformat()
        excluded_from_prod = None

    manager_value = None
    if manager_forecast not in (None, ""):
        try:
            manager_value = float(manager_forecast)
        except ValueError:
            manager_value = None

    plot_b64 = ""
    if not history.empty and next_month is not None:
        plot_b64 = _make_plot(
            history,
            next_month,
            model_pred,
            ci_lower,
            ci_upper,
            manager_value,
        )

    return TEMPLATES.TemplateResponse(
        "index.html",
        {
            "request": request,
            "branches": branches,
            "targets": targets,
            "branch_id": branch_id,
            "target": target,
            "manager_forecast": manager_forecast,
            "forecast_row": forecast_row,
            "quality_row": quality_row,
            "plot_b64": plot_b64,
            "low_confidence": low_confidence,
            "next_month": next_month_label,
            "excluded_from_prod": excluded_from_prod,
            "class_counts": class_counts,
            "class_share": class_share,
            "class_counts_all": class_counts_all,
            "class_share_all": class_share_all,
            "class_by_branch": class_by_branch,
            "mape_stats": mape_stats,
            "mape_by_target": mape_by_target.to_dict(orient="records"),
            "mape_hist_b64": mape_hist_b64,
        },
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/run")
def run_batch() -> dict:
    summary = run_inference()
    return {"status": "ok", "summary": summary}


@app.get("/forecast/{branch_id}")
def get_forecast(branch_id: str, target: str | None = None) -> dict:
    try:
        df = _load_csv(FORECAST_PATH)
    except FileNotFoundError as err:
        raise HTTPException(
            status_code=404,
            detail="Forecast file not found. Run /run first.",
        ) from err

    df_branch = df[df["branch_id"].astype(str) == str(branch_id)]
    if target:
        df_branch = df_branch[df_branch["target"] == target]

    if df_branch.empty:
        raise HTTPException(status_code=404, detail="No forecast for branch_id")

    return {"branch_id": branch_id, "items": df_branch.to_dict(orient="records")}


@app.get("/quality/{branch_id}")
def get_quality(branch_id: str) -> dict:
    try:
        df = _load_csv(QUALITY_PATH)
    except FileNotFoundError as err:
        raise HTTPException(
            status_code=404,
            detail="Quality file not found. Run /run first.",
        ) from err

    df_branch = df[df["branch_id"].astype(str) == str(branch_id)]
    if df_branch.empty:
        raise HTTPException(status_code=404, detail="No quality data for branch_id")

    return {"branch_id": branch_id, "items": df_branch.to_dict(orient="records")}


@app.get("/summary")
def get_summary() -> dict:
    if not SUMMARY_PATH.exists():
        raise HTTPException(status_code=404, detail="Summary file not found. Run /run first.")
    return pd.read_json(SUMMARY_PATH, typ="series").to_dict()


@app.get("/forecast")
def list_forecasts(
    target: str | None = None,
    limit: int = Query(100, ge=1, le=1000),
) -> dict:
    try:
        df = _load_csv(FORECAST_PATH)
    except FileNotFoundError as err:
        raise HTTPException(
            status_code=404,
            detail="Forecast file not found. Run /run first.",
        ) from err

    if target:
        df = df[df["target"] == target]

    return {"items": df.head(limit).to_dict(orient="records")}


@app.get("/branches")
def list_branches(limit: int = Query(200, ge=1, le=5000)) -> dict:
    try:
        df = _load_csv(QUALITY_PATH)
    except FileNotFoundError as err:
        raise HTTPException(
            status_code=404,
            detail="Quality file not found. Run /run first.",
        ) from err

    branches = sorted(df["branch_id"].astype(str).unique().tolist())
    return {"branches": branches[:limit], "total": len(branches)}
