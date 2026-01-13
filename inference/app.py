from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from inference.run_inference import run_inference

app = FastAPI(title="TS Forecast Inference", version="0.1.0")

OUTPUT_DIR = Path("outputs/inference")
FORECAST_PATH = OUTPUT_DIR / "forecast_next_month.csv"
QUALITY_PATH = OUTPUT_DIR / "quality_by_branch.csv"
SUMMARY_PATH = OUTPUT_DIR / "run_summary.json"


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


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
