import io, os, json, base64, math
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import chi2_contingency, shapiro, levene, mannwhitneyu, wilcoxon, kruskal
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ================================
# INITIALISATION
# ================================
SERVICE_API_KEY = os.getenv("SERVICE_API_KEY", "")

app = FastAPI(title="Dr Data 2.0 - Analysis Engine", version="2.0")

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
    df.columns = [str(c).strip().replace(" ", "").replace("-", "").replace("/", "_") for c in df.columns]
    df = df.replace(["-", "–", "--", "NaN", "nan", "null", "NULL", ""], np.nan)

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass

    df = df.dropna(axis=1, how='all')
    return df


def fig_to_data_uri(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def profile_df(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "missing_pct": float(df.isna().mean().mean() * 100.0)
    }

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
    ax.hist(num[first].dropna(), bins=25)
    ax.set_title(f"Distribution of {first}")
    uri = fig_to_data_uri(fig)
    return {"summary": desc, "histogram": uri}


def correlation_analysis(df):
    num = df.select_dtypes(include=np.number)
    if num.empty:
        return {"error": "No numeric columns available."}
    return {"correlation_matrix": num.corr().to_dict()}


def regression_analysis(df, dep, indep):
    try:
        X = sm.add_constant(df[indep])
        Y = df[dep]
    except KeyError:
        return {"error": "Variables not found."}
    model = sm.OLS(Y, X, missing='drop').fit()
    return {"coefficients": model.params.to_dict(), "r2": float(model.rsquared)}


def anova_analysis(df, y, group):
    try:
        df = df[[y, group]].dropna()
    except KeyError:
        return {"error": "Variables not found."}
    df[group] = df[group].astype("category")
    model = smf.ols(f"{y} ~ C({group})", data=df).fit()
    table = sm.stats.anova_lm(model, typ=2)
    return {"anova": table.to_dict()}


def pca_analysis(df):
    num = df.select_dtypes(include=np.number).dropna()
    if num.shape[1] < 2:
        return {"error": "Not enough variables for PCA."}
    pca = PCA(n_components=min(3, num.shape[1]))
    pca.fit(num)
    return {"explained_variance_ratio": pca.explained_variance_ratio_.tolist()}


def clustering_analysis(df):
    num = df.select_dtypes(include=np.number).dropna()
    if num.shape[0] < 3:
        return {"error": "Not enough rows for clustering."}
    km = KMeans(n_clusters=3, n_init=10, random_state=42)
    labels = km.fit_predict(num)
    return {"cluster_counts": pd.Series(labels).value_counts().to_dict()}


def timeseries_analysis(df, col):
    s = pd.to_numeric(df[col], errors='coerce').dropna()
    if s.shape[0] < 20:
        return {"error": "Not enough data for forecasting."}
    model = ARIMA(s, order=(1, 0, 1)).fit()
    forecast = model.forecast(steps=12)
    return {"forecast": forecast.tolist()}

# ================================
# CREDIT ESTIMATION
# ================================
def compute_credit_estimate(profile_type, level, analysis_list):
    if profile_type.lower() != "student":
        return 0

    base = 0
    analysis_set = set(a.lower() for a in analysis_list or [])

    if "descriptive" in analysis_set:
        base += 100

    if {"analytic", "correlation", "regression", "anova"} & analysis_set:
        base += 150

    if {"pca", "clustering", "timeseries"} & analysis_set:
        base += 200

    if level.lower() == "master":
        base += 50
    elif level.lower() in ["doctorate", "phd"]:
        base += 100

    if base == 0 and analysis_set:
        base = 50

    return base

# ================================
# REQUEST MODEL
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

# ================================
# MAIN ANALYZE ENDPOINT (JSON ONLY)
# ================================
@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    try:
        df = pd.DataFrame(request.dataset)

        if df.empty:
            raise HTTPException(status_code=400, detail="Dataset is empty")

        df = clean_dataframe(df)

        results: Dict[str, Any] = {}
        analysis_list = request.analysis_types or []

        if "descriptive" in analysis_list:
            results["descriptive"] = descriptive_analysis(df)

        if ("analytic" in analysis_list) or ("correlation" in analysis_list):
            results["correlation"] = correlation_analysis(df)

        if "regression" in analysis_list and request.dependent_var and request.independent_vars:
            results["regression"] = regression_analysis(
                df, request.dependent_var, request.independent_vars
            )

        if "anova" in analysis_list and request.dependent_var and request.group_var:
            results["anova"] = anova_analysis(
                df, request.dependent_var, request.group_var
            )

        if "pca" in analysis_list:
            results["pca"] = pca_analysis(df)

        if "clustering" in analysis_list:
            results["clustering"] = clustering_analysis(df)

        if "timeseries" in analysis_list and request.arima_col:
            results["timeseries"] = timeseries_analysis(df, request.arima_col)

        credit_estimate = compute_credit_estimate(
            request.profile_type,
            request.level,
            analysis_list
        )

        return {
            "success": True,
            "credit_estimate": credit_estimate,
            "software_style": request.preferred_software or "generic",
            "results": results,
            "metadata": profile_df(df)
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/")
def root():
    return {"DrData2.0": "API running 🚀"}

@app.get("/health")
def health():
    return {"ok": True}
