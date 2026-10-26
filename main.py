"""
main.py: FastAPI Unified Backend Server for Auto-analysis-hub.
Serves REST API endpoints for dynamic visual profiling, high-speed multimodal AI analysis,
data cleaning, interactive chat, and mounts the modern static frontend UI.
Everything is 100% dynamic: zero static mock data or sample dataset buttons.
"""

from __future__ import annotations
import io
import os
import uuid
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, Form, Header, HTTPException, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
from pydantic import BaseModel

from auto_analysis_hub_skill import (
    DataProfiler,
    VisualPlotter,
    AutoDataCleaner,
    GemmaMultimodalHarness,
    profile_dataset,
    generate_diagnostic_charts,
    clean_dataset,
    analyze_charts_with_gemma,
    ask_gemma_chat,
)

app = FastAPI(
    title="Auto-analysis-hub",
    description="Autonomous Visual Data Diagnostic & Storytelling Agent Skill & Open-Source Model Harness",
    version="1.0.0"
)

# CORS middleware for open accessibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session cache for fast interactive chat and downloads
# Keys: session_id -> { df, profile_data, charts_data, clean_result, ai_insights, ... }
SESSION_CACHE: Dict[str, Dict[str, Any]] = {}

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Mount static folder
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# Pydantic models for request bodies
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    question: str
    persona: Optional[str] = "executive"


class CleanRequest(BaseModel):
    session_id: Optional[str] = None
    handle_outliers: Optional[bool] = True
    impute_numeric_strategy: Optional[str] = "auto"
    impute_categorical_strategy: Optional[str] = "mode"


# Root endpoint: Serve index.html
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h1>Auto-analysis-hub UI Loading...</h1>")


@app.get("/api/health")
async def health_check():
    """System health check and backend configuration status."""
    backend_key_set = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    return {
        "status": "healthy",
        "service": "Auto-analysis-hub",
        "version": "1.0.0",
        "gemini_api_configured": backend_key_set,
        "default_model": os.getenv("DEFAULT_MODEL", "gemini-2.5-flash"),
        "cached_sessions": len(SESSION_CACHE)
    }


@app.post("/api/analyze")
async def analyze_dataset_endpoint(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    persona: Optional[str] = Form("executive")
):
    """Core 100% dynamic workflow: Upload CSV -> Profile -> 4 Dark-Theme Charts -> Multimodal Vision -> Auto-Clean."""
    backend_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="Please upload a valid CSV dataset to begin dynamic analysis.")

    source_name = file.filename
    csv_bytes = await file.read()

    if not csv_bytes or len(csv_bytes.strip()) == 0:
        raise HTTPException(status_code=400, detail="The uploaded CSV file is empty. Please provide a file with rows of data.")

    try:
        # Step 1: Profiler
        profiler = DataProfiler.from_csv_bytes(csv_bytes, source_name=source_name)
        profile_data = profiler.profile()
        df = profiler.df

        if df.empty or df.shape[0] == 0:
            raise HTTPException(status_code=400, detail="The uploaded CSV contains no valid data rows.")

        # Step 2: Plotter (4 In-Memory Dark-Themed Charts)
        plotter = VisualPlotter(df)
        charts_data = plotter.generate_all_charts()

        # Step 3: Auto-Clean Studio Preview
        cleaner = AutoDataCleaner(df)
        clean_result = cleaner.clean()

        # Step 4: High-speed Multimodal Vision Diagnostic Analysis
        harness = GemmaMultimodalHarness(api_key=backend_api_key)
        ai_insights = harness.analyze_dataset(
            chart_images_bytes=charts_data["all_bytes"],
            statistical_profile_text=profile_data["text_summary"],
            user_prompt=prompt,
            persona=persona or "executive"
        )

        # Cache session in memory for interactive chat and downloads
        session_id = str(uuid.uuid4())
        SESSION_CACHE[session_id] = {
            "df": df,
            "profile_data": profile_data,
            "charts_data": charts_data,
            "clean_result": clean_result,
            "ai_insights": ai_insights,
            "source_name": source_name,
            "persona": persona or "executive"
        }

        # Keep cache manageable (last 25 sessions)
        if len(SESSION_CACHE) > 25:
            oldest_key = next(iter(SESSION_CACHE))
            del SESSION_CACHE[oldest_key]

        # Format client charts without raw binary bytes
        def format_chart_for_json(c):
            return {
                "title": c.get("title", ""),
                "description": c.get("description", ""),
                "base64": c.get("base64", ""),
                "type": c.get("type", "")
            }

        client_charts = {
            "chart_1": format_chart_for_json(charts_data["chart_1"]),
            "chart_2": format_chart_for_json(charts_data["chart_2"]),
            "chart_3": format_chart_for_json(charts_data["chart_3"]),
            "chart_4": format_chart_for_json(charts_data["chart_4"]),
            "gallery": [format_chart_for_json(c) for c in charts_data.get("gallery", [])]
        }

        return {
            "status": "success",
            "session_id": session_id,
            "filename": source_name,
            "dimensions": profile_data["dimensions"],
            "health": profile_data["health"],
            "missing_breakdown": profile_data["missing_breakdown"],
            "numerical_metrics": profile_data["numerical_metrics"],
            "categorical_metrics": profile_data["categorical_metrics"],
            "top_correlations": profile_data["top_correlations"],
            "preview_rows": profile_data["preview_rows"],
            "charts": client_charts,
            "ai_insights": ai_insights,
            "cleaner_preview": {
                "metrics": clean_result["metrics"],
                "action_logs": clean_result["action_logs"],
                "preview_cleaned_rows": clean_result["preview_cleaned_rows"]
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Dynamic diagnostic error: {str(e)}")


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """High-speed 'Ask the Charts' interactive Q&A endpoint with multimodal grounding."""
    session = SESSION_CACHE.get(request.session_id or "") if request.session_id else None

    chart_bytes = session["charts_data"]["all_bytes"] if session else []
    profile_text = session["profile_data"]["text_summary"] if session else "Dataset general summary."
    backend_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    harness = GemmaMultimodalHarness(api_key=backend_api_key)
    chat_result = harness.chat_with_charts(
        question=request.question,
        chart_images_bytes=chart_bytes,
        profile_text=profile_text,
        persona=request.persona or (session["persona"] if session else "executive")
    )

    return chat_result


@app.post("/api/clean")
async def clean_endpoint(
    request: Optional[CleanRequest] = None,
    file: Optional[UploadFile] = File(None),
    download: bool = Query(False)
):
    """Execute data cleaning pipeline on active session dataset or uploaded CSV."""
    df: Optional[pd.DataFrame] = None
    handle_outliers = True
    impute_numeric = "auto"
    impute_cat = "mode"

    if request and request.session_id and request.session_id in SESSION_CACHE:
        df = SESSION_CACHE[request.session_id]["df"].copy()
        handle_outliers = request.handle_outliers if request.handle_outliers is not None else True
        impute_numeric = request.impute_numeric_strategy or "auto"
        impute_cat = request.impute_categorical_strategy or "mode"
    elif file and file.filename:
        csv_bytes = await file.read()
        profiler = DataProfiler.from_csv_bytes(csv_bytes)
        df = profiler.df

    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="No active dataset session or CSV file found for cleaning.")

    cleaner = AutoDataCleaner(
        df=df,
        handle_outliers=handle_outliers,
        impute_numeric_strategy=impute_numeric,
        impute_categorical_strategy=impute_cat
    )
    result = cleaner.clean()

    # Update cache if session exists
    if request and request.session_id and request.session_id in SESSION_CACHE:
        SESSION_CACHE[request.session_id]["clean_result"] = result

    if download:
        return Response(
            content=result["csv_bytes"],
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="cleaned_dataset.csv"'}
        )

    return {
        "status": "success",
        "metrics": result["metrics"],
        "action_logs": result["action_logs"],
        "preview_cleaned_rows": result["preview_cleaned_rows"]
    }


@app.get("/api/clean/download/{session_id}")
async def download_cleaned_csv(session_id: str):
    """Download cleaned CSV by session ID."""
    if session_id not in SESSION_CACHE:
        raise HTTPException(status_code=404, detail="Session not found or expired. Please upload and analyze a dataset first.")

    clean_result = SESSION_CACHE[session_id]["clean_result"]
    csv_bytes = clean_result["csv_bytes"]
    raw_name = SESSION_CACHE[session_id].get("source_name", "dataset.csv")
    filename = f"cleaned_{raw_name}"

    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
