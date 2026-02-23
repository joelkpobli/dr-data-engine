import io, os, base64
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ================================
# INITIALISATION
# ================================
app = FastAPI(title="Dr Data 2.0 - Analysis Engine", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# UTILITIES
# ================================
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().dropna(how='all')
    df = df.dropna(axis=1, how='all')
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    df = df.replace(["-", "NaN", "null", ""], np.nan)

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass

    return df


def fig_to_data_uri(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def profile_df(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "missing_pct": float(df.isna().mean().mean() * 100.0)
    }

# ================================
# REQUEST MODELS
# ================================
class AnalyzeRequest(BaseModel):
    dataset: List[Dict]
    profile_type: str
    level: str
    domain: Optional[str] = None
    analysis_types: List[str]
    dependent_var: Optional[str] = None
    independent_vars: Optional[List[str]] = None
    group_var: Optional[str] = None
    arima_col: Optional[str] = None
    preferred_software: Optional[str] = None


class SmartCheckRequest(BaseModel):
    dataset: List[Dict]


class InterpretRequest(BaseModel):
    results: Dict[str, Any]


class ExportRequest(BaseModel):
    results: Dict[str, Any]
    format: str  # "pdf" or "word"

# ================================
# ANALYSIS FUNCTIONS
# ================================
def descriptive_analysis(df):
    num = df.select_dtypes(include=np.number)
    if num.empty:
        return {"error": "No numeric columns available."}
    desc = num.describe().to_dict()
    fig, ax = plt.subplots()
    first = num.columns[0]
    ax.hist(num[first].dropna(), bins=20)
    uri = fig_to_data_uri(fig)
    return {"summary": desc, "histogram": uri}


def correlation_analysis(df):
    num = df.select_dtypes(include=np.number)
    return {"correlation_matrix": num.corr().to_dict()}


def regression_analysis(df, dep, indep):
    X = sm.add_constant(df[indep])
    Y = df[dep]
    model = sm.OLS(Y, X, missing='drop').fit()
    return {"coefficients": model.params.to_dict(), "r2": float(model.rsquared)}

# ================================
# SMARTCHECK
# ================================
@app.post("/smartcheck")
async def smartcheck(request: SmartCheckRequest):
    try:
        df = pd.DataFrame(request.dataset)
        if df.empty:
            return {"success": False, "error": "Dataset is empty"}
        df = clean_dataframe(df)
        return {"success": True, "metadata": profile_df(df)}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ================================
# ANALYZE
# ================================
@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    try:
        df = pd.DataFrame(request.dataset)
        if df.empty:
            raise HTTPException(status_code=400, detail="Dataset empty")

        df = clean_dataframe(df)
        results = {}

        if "descriptive" in request.analysis_types:
            results["descriptive"] = descriptive_analysis(df)

        if "correlation" in request.analysis_types:
            results["correlation"] = correlation_analysis(df)

        if "regression" in request.analysis_types:
            results["regression"] = regression_analysis(
                df,
                request.dependent_var,
                request.independent_vars
            )

        return {
            "success": True,
            "results": results,
            "metadata": profile_df(df)
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

# ================================
# INTERPRET RESULTS
# ================================
@app.post("/interpret-results")
async def interpret_results(request: InterpretRequest):
    try:
        # Ici tu pourras brancher GPT plus tard
        return {
            "success": True,
            "interpretation": "The statistical results show meaningful patterns. Please review coefficients and significance levels."
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ================================
# EXPORT ANALYSIS
# ================================
@app.post("/export-analysis")
async def export_analysis(request: ExportRequest):
    try:
        # Ici tu brancheras génération PDF réelle plus tard
        return {
            "success": True,
            "download_url": "https://your-production-storage-link.com/file"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ================================
# PAYMENT WEBHOOK
# ================================
@app.post("/payment-webhook")
async def payment_webhook(request: Request):
    payload = await request.json()
    print("Payment received:", payload)
    return {"success": True}

# ================================
# HEALTH
# ================================
@app.get("/")
def root():
    return {"DrData2.0": "API running 🚀"}

@app.get("/health")
def health():
    return {"ok": True}
