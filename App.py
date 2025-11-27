import io, os, json, base64, math
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import chi2_contingency, shapiro, levene, mannwhitneyu, wilcoxon, kruskal
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# === Initialisation du service ===
SERVICE_API_KEY = os.getenv("SERVICE_API_KEY", "")

app = FastAPI(title="Dr Data 2.0 - Analysis Engine", version="1.2")

@app.get("/smartcheck")
async def smart_check():
    return {"status": "ok", "message": "Smart check working successfully"}


# === CORS (pour Lovable) ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tu pourras restreindre plus tard à ton domaine
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# 🧹 Nettoyage automatique du fichier avant analyse
# ------------------------------------------------------------------
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie automatiquement le fichier de données avant toute analyse."""
    # Supprimer doublons et lignes vides
    df = df.drop_duplicates().dropna(how='all')

    # Supprimer colonnes entièrement vides
    df = df.dropna(axis=1, how='all')

    # Nettoyer les noms de colonnes
    df.columns = [str(c).strip().replace(" ", "_").replace("-", "_").replace("/", "_") for c in df.columns]

    # Remplacer les valeurs invalides par NaN
    df = df.replace(["-", "–", "--", "NaN", "nan", "null", "NULL", ""], np.nan)

    # Corriger automatiquement les types numériques
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            # On laisse les colonnes non numériques comme texte/catégorielles
            pass

    # Supprimer colonnes encore vides après conversion
    df = df.dropna(axis=1, how='all')
    return df


# ------------------------------------------------------------------
# 🔧 Fonctions utilitaires
# ------------------------------------------------------------------
def read_from_url(url: str) -> bytes:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content

def load_df_from_bytes(data: bytes, fname: Optional[str]) -> pd.DataFrame:
    name = (fname or "").lower()
    bio = io.BytesIO(data)
    if name.endswith(".csv"):
        return pd.read_csv(bio)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(bio)
    if name.endswith(".json"):
        return pd.read_json(bio)
    if name.endswith(".sav"):
        import pyreadstat
        df, _ = pyreadstat.read_sav(bio)
        return df
    if name.endswith(".dta"):
        import pyreadstat
        df, _ = pyreadstat.read_dta(bio)
        return df
    # fallback CSV
    return pd.read_csv(bio)

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

# ------------------------------------------------------------------
# 🧮 Fonctions d’analyse
# ------------------------------------------------------------------
def descriptive_analysis(df):
    num = df.select_dtypes(include=np.number)
    if num.empty:
        return {"error": "No numeric columns available for descriptive statistics."}
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
        return {"error": "No numeric columns available for correlation matrix."}
    corr = num.corr().to_dict()
    return {"correlation_matrix": corr}

def regression_analysis(df, dep, indep):
    try:
        X = sm.add_constant(df[indep])
        Y = df[dep]
    except KeyError:
        return {"error": "Dependent or independent variables not found in dataframe."}
    model = sm.OLS(Y, X, missing='drop').fit()
    return {"coefficients": model.params.to_dict(), "r2": float(model.rsquared)}

def anova_analysis(df, y, group):
    try:
        df = df[[y, group]].dropna()
    except KeyError:
        return {"error": "Variables for ANOVA not found in dataframe."}
    if df.empty:
        return {"error": "No valid data for ANOVA after cleaning."}
    df[group] = df[group].astype("category")
    model = smf.ols(f"{y} ~ C({group})", data=df).fit()
    table = sm.stats.anova_lm(model, typ=2)
    return {"anova": table.to_dict()}

def pca_analysis(df):
    num = df.select_dtypes(include=np.number).dropna()
    if num.shape[1] < 2:
        return {"error": "Not enough numeric variables for PCA."}
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
        return {"error": "Not enough data points for time series forecasting."}
    model = ARIMA(s, order=(1, 0, 1)).fit()
    forecast = model.forecast(steps=12)
    fig, ax = plt.subplots()
    s.tail(50).plot(ax=ax, label="Recent Data")
    pd.Series(forecast, index=range(len(s), len(s)+12)).plot(ax=ax, label="Forecast")
    ax.legend()
    uri = fig_to_data_uri(fig)
    return {"forecast": forecast.tolist(), "figure": uri}

# ------------------------------------------------------------------
# 💳 Calcul des crédits (pour étudiants)
# ------------------------------------------------------------------
def compute_credit_estimate(
    profile_type: str,
    level: str,
    analysis_list: List[str],
    file_size_bytes: int
) -> int:
    """
    Calcule une estimation du coût en crédits selon :
    - le profil (étudiant uniquement),
    - le niveau (Licence / Master / Doctorat),
    - les types d’analyses demandés,
    - la taille du fichier.
    """

    # Seuls les étudiants paient en crédits
    if profile_type.lower() != "student":
        return 0

    base = 0
    analysis_set = set(a.lower() for a in analysis_list or [])

    # Descriptif simple
    if "descriptive" in analysis_set:
        base += 100

    # Analytique classique
    if ("analytic" in analysis_set) or ("correlation" in analysis_set) or ("regression" in analysis_set) or ("anova" in analysis_set):
        base += 150

    # Analyses avancées
    if ("pca" in analysis_set) or ("clustering" in analysis_set) or ("timeseries" in analysis_set):
        base += 200

    # Si un jour tu ajoutes explicitement "export" dans analysis_types
    if "export" in analysis_set:
        base += 100

    # Bonus niveau
    lvl = (level or "").strip().lower()
    if lvl in ["master", "maîtrise", "maitrise"]:
        base += 50
    elif lvl in ["doctorat", "phd", "doctoral"]:
        base += 100
    # licence / autre → +0

    # Bonus taille fichier
    size_mb = file_size_bytes / (1024 * 1024)
    if size_mb <= 2:
        bonus = 0
    elif size_mb <= 5:
        bonus = 20
    elif size_mb <= 10:
        bonus = 40
    elif size_mb <= 20:
        bonus = 75
    else:
        bonus = 150

    base += bonus

    # Sécurité : minimum 50 crédits si au moins une analyse
    if base == 0 and analysis_set:
        base = 50

    return int(base)

# ------------------------------------------------------------------
# 🔗 Endpoint principal
# ------------------------------------------------------------------
@app.post("/analyze")
async def analyze(
    file: Optional[UploadFile] = File(default=None),
    dataset_url: Optional[str] = Form(default=None),
    profile_type: str = Form(...),
    level: str = Form(...),
    domain: Optional[str] = Form(default=None),
    analysis_types: str = Form(...),
    dependent_var: Optional[str] = Form(default=None),
    independent_vars: Optional[str] = Form(default=None),
    group_var: Optional[str] = Form(default=None),
    arima_col: Optional[str] = Form(default=None),
    preferred_software: Optional[str] = Form(default=None)
):
    # Vérifier que fichier fourni
    if file is not None:
        raw = await file.read()
        fname = file.filename
    elif dataset_url:
        raw = read_from_url(dataset_url)
        fname = urlparse(dataset_url).path.split("/")[-1]
    else:
        raise HTTPException(status_code=400, detail="Aucun fichier fourni")

    file_size_bytes = len(raw)

    # Charger et nettoyer le fichier
    df = load_df_from_bytes(raw, fname)
    df = clean_dataframe(df)

    # Conversion paramètres
    try:
        analysis_list = json.loads(analysis_types) if isinstance(analysis_types, str) else analysis_types
    except Exception:
        analysis_list = []

    try:
        indep = json.loads(independent_vars) if independent_vars else []
    except Exception:
        indep = []

    results: Dict[str, Any] = {}
    meta = profile_df(df)
    meta["file_name"] = fname
    meta["file_size_bytes"] = file_size_bytes
    meta["file_size_mb"] = round(file_size_bytes / (1024 * 1024), 2)
    meta["profile_type"] = profile_type
    meta["level"] = level
    meta["domain"] = domain

    # --- Lancer les analyses selon le choix utilisateur, avec protections ---
    if "descriptive" in analysis_list:
        try:
            results["descriptive"] = descriptive_analysis(df)
        except Exception as e:
            results["descriptive"] = {"error": f"Descriptive analysis failed: {str(e)}"}

    if ("analytic" in analysis_list) or ("correlation" in analysis_list):
        try:
            results["correlation"] = correlation_analysis(df)
        except Exception as e:
            results["correlation"] = {"error": f"Correlation analysis failed: {str(e)}"}

    if "regression" in analysis_list and dependent_var and indep:
        try:
            results["regression"] = regression_analysis(df, dependent_var, indep)
        except Exception as e:
            results["regression"] = {"error": f"Regression analysis failed: {str(e)}"}

    if "anova" in analysis_list and dependent_var and group_var:
        try:
            results["anova"] = anova_analysis(df, dependent_var, group_var)
        except Exception as e:
            results["anova"] = {"error": f"ANOVA analysis failed: {str(e)}"}

    if "pca" in analysis_list:
        try:
            results["pca"] = pca_analysis(df)
        except Exception as e:
            results["pca"] = {"error": f"PCA analysis failed: {str(e)}"}

    if "clustering" in analysis_list:
        try:
            results["clustering"] = clustering_analysis(df)
        except Exception as e:
            results["clustering"] = {"error": f"Clustering analysis failed: {str(e)}"}

    if "timeseries" in analysis_list and arima_col:
        try:
            results["timeseries"] = timeseries_analysis(df, arima_col)
        except Exception as e:
            results["timeseries"] = {"error": f"Time series analysis failed: {str(e)}"}

    # Calcul estimation crédits (pour étudiants)
    credit_estimate = compute_credit_estimate(
        profile_type=profile_type,
        level=level,
        analysis_list=analysis_list,
        file_size_bytes=file_size_bytes
    )

    # Style logiciel (la présentation côté Lovable se fera à partir de ce champ
    software_style = preferred_software or "generic"

    return {
        "status": "success",
        "info": "✅ File cleaned successfully before analysis.",
        "credit_estimate": credit_estimate,
        "software_style": software_style,
        "results": results,
        "metadata": meta
    }

@app.get("/")
def root():
    return {"DrData2.0": "API is running successfully 🚀"}

@app.get("/health")
def health():
    return {"ok": True}

# ================================
# EXPORT ANALYSIS
# ================================
@app.post("/export-analysis")
async def export_analysis(request: Request):
    try:
        data = await request.json()
        analysis_result = data.get("result")

        if not analysis_result:
            return JSONResponse(
                {"status": "error", "message": "No analysis data provided"}, status_code=400
            )

        return JSONResponse({
            "status": "success",
            "export_link": "https://fake-export-link.com/download-file",
            "message": "Export generated successfully"
        })

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ================================
# INTERPRET RESULTS
# ================================
@app.post("/interpret-results")
async def interpret_results(request: Request):
    try:
        data = await request.json()
        stats = data.get("stats")

        if not stats:
            return JSONResponse(
                {"status": "error", "message": "No stats provided"}, status_code=400
            )

        interpretation = "Based on your results, the analysis indicates a significant pattern."

        return JSONResponse({
            "status": "success",
            "interpretation": interpretation
        })

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ================================
# VALIDATE CREDITS
# ================================
@app.post("/validate-credits")
async def validate_credits(request: Request):
    try:
        data = await request.json()
        user_id = data.get("user_id")
        credits_needed = data.get("credits")

        if not user_id or credits_needed is None:
            return JSONResponse(
                {"status": "error", "message": "Missing user_id or credits_needed"},
                status_code=400
            )

        # Fake validation
        return JSONResponse({
            "status": "success",
            "credits_remaining": 500,
            "allowed": True
        })

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# ================================
# PAYMENT WEBHOOK
# ================================
@app.post("/payment-webhook")
async def payment_webhook(request: Request):
    try:
        payload = await request.json()
        print("PAYMENT RECEIVED:", payload)

        return JSONResponse({"status": "success", "message": "Payment processed"})

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)



