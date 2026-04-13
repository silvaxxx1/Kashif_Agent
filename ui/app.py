"""
ui/app.py — Streamlit UI for Kashif

Run with:
  uv run streamlit run ui/app.py

Start the API first:
  uv run uvicorn api.server:app --reload --port 8000
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Kashif — ML Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _api_healthy() -> bool:
    try:
        return requests.get(f"{API_URL}/health", timeout=2).status_code == 200
    except Exception:
        return False


def _read_artifact(path: str | None) -> bytes | None:
    if not path:
        return None
    try:
        return Path(path).read_bytes()
    except Exception:
        return None


def _run_streaming(file, target, rounds, provider, no_agent):
    """
    POST /run/stream and yield progress dicts line-by-line.
    Last event has stage='complete' with a 'result' key, or stage='error'.
    """
    file.seek(0)
    with requests.post(
        f"{API_URL}/run/stream",
        files={"file": (file.name, file, "text/csv")},
        data={
            "target": target,
            "rounds": str(rounds),
            "provider": provider,
            "no_agent": str(no_agent).lower(),
        },
        stream=True,
        timeout=660,
    ) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if raw:
                yield json.loads(raw)


# ---------------------------------------------------------------------------
# EDA helpers (pure pandas — no API call needed)
# ---------------------------------------------------------------------------

def _eda_missing(df: pd.DataFrame) -> pd.DataFrame:
    null_pct = (df.isnull().mean() * 100).round(1)
    return null_pct[null_pct > 0].sort_values(ascending=False).rename("Null %")


def _eda_corr(df: pd.DataFrame, target: str) -> pd.Series | None:
    num = df.select_dtypes(include="number")
    if target not in num.columns or num.shape[1] < 2:
        return None
    return num.corr()[target].drop(target).sort_values(key=abs, ascending=False)


def _eda_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame | None:
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        return None
    return num.corr().round(2)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    title_col, status_col = st.columns([3, 1])
    with title_col:
        st.markdown("## 🤖 Kashif")
        st.caption("LLM-powered AutoML")
    with status_col:
        st.markdown("<br>", unsafe_allow_html=True)
        healthy = _api_healthy()
        st.markdown(
            "🟢 API" if healthy else "🔴 API",
            help="API online" if healthy else
                 "API offline — run:\nuv run uvicorn api.server:app --port 8000",
        )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload CSV", type=["csv"],
        help="Any tabular CSV with a clear target column",
    )

    target_col = None
    df_preview = None

    if uploaded_file:
        try:
            df_preview = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)
            cols = list(df_preview.columns)
            target_col = st.selectbox("Target column", cols,
                                      help="The column you want to predict")
            n_rows, n_cols = df_preview.shape
            st.caption(f"{n_rows:,} rows · {n_cols} cols · "
                       f"{df_preview.memory_usage(deep=True).sum() / 1024:.0f} KB")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    st.divider()

    with st.expander("⚙️ Run options", expanded=True):
        no_agent = st.toggle(
            "Static pipeline only",
            value=False,
            help="Skip LLM loop — faster, no API key needed",
        )
        provider = st.selectbox(
            "LLM provider", ["groq", "anthropic", "ollama"],
            disabled=no_agent,
            help="groq: fastest  |  anthropic: claude  |  ollama: local",
        )
        rounds = st.slider(
            "Max FE rounds", 1, 10, 3,
            disabled=no_agent,
            help="Each round: LLM writes code → executor runs it → CV scores it",
        )

    st.divider()

    can_run = uploaded_file is not None and target_col is not None and healthy
    run_btn = st.button(
        "▶  Run Kashif", type="primary",
        disabled=not can_run, width="stretch",
    )
    if not healthy:
        st.warning("Start the API to enable.", icon="⚠️")
    elif uploaded_file is None:
        st.caption("Upload a CSV to enable.")

    if "result" in st.session_state:
        if st.button("✕  Clear results", width="stretch"):
            for k in ("result", "run_meta"):
                st.session_state.pop(k, None)
            st.rerun()

# ---------------------------------------------------------------------------
# Run — streaming progress
# ---------------------------------------------------------------------------

if run_btn and can_run:
    for k in ("result", "run_meta", "error"):
        st.session_state.pop(k, None)

    st.markdown(f"### Running on `{uploaded_file.name}` → `{target_col}`")
    progress_bar = st.progress(0, text="Starting…")
    stage_box    = st.empty()

    try:
        for event in _run_streaming(uploaded_file, target_col, rounds, provider, no_agent):
            pct  = int(event.get("progress", 0) * 100)
            msg  = event.get("message", "")
            stage = event.get("stage", "")

            progress_bar.progress(pct, text=msg)
            stage_icon = {
                "loading":   "📂", "profiling": "🔍", "training":  "🏋️",
                "baseline":  "📏", "fe_loop":   "🔄", "setup":     "⚙️",
                "report":    "📄", "narrating": "✍️",  "complete":  "✅",
                "error":     "❌",
            }.get(stage, "⏳")
            stage_box.markdown(f"{stage_icon} **{msg}**")

            if stage == "complete":
                st.session_state["result"]   = event["result"]
                st.session_state["run_meta"] = {
                    "file": uploaded_file.name, "target": target_col,
                    "provider": provider if not no_agent else "static",
                    "rounds": rounds,
                }
                break
            elif stage == "error":
                st.session_state["error"] = event.get("error", msg)
                break

        progress_bar.progress(100, text="Done!")

    except requests.exceptions.ConnectionError:
        st.session_state["error"] = (
            "**Cannot connect to the API.**\n\n"
            "```bash\nuv run uvicorn api.server:app --port 8000\n```"
        )
    except Exception as exc:
        st.session_state["error"] = f"**Unexpected error:** {exc}"

    st.rerun()

# ---------------------------------------------------------------------------
# Error banner
# ---------------------------------------------------------------------------

if "error" in st.session_state:
    st.error(st.session_state["error"])
    if st.button("Dismiss"):
        st.session_state.pop("error", None)
        st.rerun()
    st.stop()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_eda, tab_results = st.tabs(["🏠 Overview", "📊 EDA", "📈 Results"])

# ============================================================
# TAB: Overview
# ============================================================

with tab_overview:
    if "result" not in st.session_state:
        st.markdown("## What is Kashif?")
        st.markdown(
            "Kashif is an ML agent that wraps a battle-tested AutoML pipeline "
            "with an LLM-powered feature engineering loop. Upload a CSV, pick a "
            "target column, and let the agent discover better features automatically."
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AutoML backend", "sklearn", "LightGBM · XGBoost · RF")
        c2.metric("FE loop", "Karpathy-style", "reflect → improve → score")
        c3.metric("LLM providers", "3", "Groq · Anthropic · Ollama")
        c4.metric("Output", "JSON + Report", "model · EDA · markdown")
        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### How it works")
            st.markdown("""
1. **Profile** — detect task type, dtypes, null rates, class balance
2. **Baseline** — run AutoML, score with 5-fold CV
3. **FE loop** — LLM writes `engineer_features(df)`, executor sandboxes it, CV keeps improvements
4. **Report** — score progression, top SHAP features, plain-English summary
            """)
        with col_b:
            st.markdown("#### Quick start")
            st.code(
                "# Static pipeline (no LLM key needed)\n"
                "Upload CSV → pick target → toggle 'Static pipeline' → Run\n\n"
                "# Full agent loop\n"
                "export GROQ_API_KEY=gsk_...\n"
                "Upload CSV → pick target → Run",
                language="bash",
            )
    else:
        result   = st.session_state["result"]
        meta     = st.session_state.get("run_meta", {})
        baseline = result.get("baseline_score", 0)
        best     = result.get("cv_score", 0)
        delta    = result.get("delta", 0)
        n_rounds = len(result.get("rounds", []))

        st.markdown(f"### Last run — `{meta.get('file','')}` → `{meta.get('target','')}`")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Baseline CV", f"{baseline:.4f}")
        c2.metric("Best CV", f"{best:.4f}",
                  delta=f"{delta:+.4f}" if delta else None)
        c3.metric("Best round", result.get("best_round", 0))
        c4.metric("FE rounds run", n_rounds)

        if delta > 0:
            st.success(f"Feature engineering improved CV by **{delta:+.4f}** ({delta/baseline*100:+.1f}%)")
        elif n_rounds > 0:
            st.info("Feature engineering did not improve upon the static baseline.")
        else:
            st.info("Static mode — no feature engineering applied.")

        if n_rounds > 0:
            st.markdown("#### Score progression")
            rounds_data = result.get("rounds", [])
            chart_df = pd.DataFrame(
                [{"Round": "baseline", "CV Score": baseline}]
                + [{"Round": f"r{r['round']}", "CV Score": r["cv_score"]}
                   for r in rounds_data]
            ).set_index("Round")
            st.line_chart(chart_df, height=200)

        narr = result.get("narration")
        if narr:
            st.divider()
            st.markdown("#### Summary")
            st.write(narr.get("executive_summary", ""))
            st.info(narr.get("accuracy_statement", ""))
            if narr.get("what_improved"):
                st.success(narr["what_improved"])

# ============================================================
# TAB: EDA
# ============================================================

with tab_eda:
    if df_preview is None:
        st.info("Upload a CSV in the sidebar to see EDA here.")
        st.stop()

    st.markdown(f"### {uploaded_file.name}")

    # ── Shape row ──
    n_num = df_preview.select_dtypes(include="number").shape[1]
    n_cat = df_preview.select_dtypes(exclude="number").shape[1]
    n_null_cells = int(df_preview.isnull().sum().sum())
    n_dupes = int(df_preview.duplicated().sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{df_preview.shape[0]:,}")
    c2.metric("Columns", df_preview.shape[1])
    c3.metric("Numeric", n_num)
    c4.metric("Categorical", n_cat)
    c5.metric("Missing cells", n_null_cells, delta=f"{n_dupes} dupes" if n_dupes else None,
              delta_color="off")

    st.divider()

    # ── Target distribution ──
    if target_col and target_col in df_preview.columns:
        left, right = st.columns([1, 2])
        with left:
            st.markdown(f"#### Target: `{target_col}`")
            series = df_preview[target_col].dropna()
            if series.dtype == "object" or series.nunique() <= 20:
                counts = series.value_counts()
                total  = len(series)
                st.bar_chart(counts, height=180)
                if series.nunique() == 2:
                    majority = counts.iloc[0] / total
                    if majority > 0.75:
                        st.warning(f"Imbalanced: {majority:.0%} in majority class", icon="⚠️")
                    else:
                        st.success(f"Balanced: {majority:.0%} / {1-majority:.0%}", icon="✅")
                st.caption(f"{series.nunique()} classes · {series.isnull().sum()} nulls")
            else:
                hist = series.value_counts(bins=20, sort=False).sort_index()
                hist.index = [f"{iv.left:.2g}–{iv.right:.2g}" for iv in hist.index]
                st.bar_chart(hist, height=180)
                st.caption(f"min {series.min():.2f} · mean {series.mean():.2f} · max {series.max():.2f}")

        with right:
            st.markdown("#### Feature → target correlation")
            corr = _eda_corr(df_preview, target_col)
            if corr is not None and len(corr) > 0:
                corr_df = corr.reset_index()
                corr_df.columns = ["Feature", "Correlation"]
                corr_df = corr_df.set_index("Feature")
                st.bar_chart(corr_df, y="Correlation", height=220,
                             color="#4c78a8")
                st.caption("Pearson correlation with target (numeric features only)")
            else:
                st.caption("No numeric features to correlate.")

        st.divider()

    # ── Missing values ──
    missing = _eda_missing(df_preview)
    if len(missing) > 0:
        st.markdown("#### Missing values")
        miss_df = missing.reset_index()
        miss_df.columns = ["Column", "Null %"]
        miss_df = miss_df.set_index("Column")
        st.bar_chart(miss_df, y="Null %", height=max(150, len(missing) * 28),
                     color="#e45756")
        st.caption(f"{len(missing)} columns have missing values out of {df_preview.shape[1]}")
        st.divider()
    else:
        st.success("No missing values.", icon="✅")
        st.divider()

    # ── Column summary ──
    st.markdown("#### Column summary")
    summary = pd.DataFrame({
        "dtype":    df_preview.dtypes.astype(str),
        "non-null": df_preview.count(),
        "null %":   (df_preview.isnull().mean() * 100).round(1),
        "unique":   df_preview.nunique(),
        "min":      df_preview.min(numeric_only=True),
        "mean":     df_preview.mean(numeric_only=True).round(3),
        "max":      df_preview.max(numeric_only=True),
    })
    st.dataframe(summary, use_container_width=True)

    # ── Correlation matrix ──
    corr_matrix = _eda_correlation_matrix(df_preview)
    if corr_matrix is not None:
        with st.expander("Correlation matrix (numeric features)", expanded=False):
            st.dataframe(
                corr_matrix.style.background_gradient(cmap="RdYlGn", vmin=-1, vmax=1),
                use_container_width=True,
                height=min(400, corr_matrix.shape[0] * 35 + 40),
            )

    # ── Numeric distributions ──
    num_cols = df_preview.select_dtypes(include="number").columns.tolist()
    if target_col in num_cols:
        num_cols = [c for c in num_cols if c != target_col]
    if num_cols:
        with st.expander(f"Numeric distributions ({len(num_cols)} features)", expanded=False):
            n_cols_grid = 3
            rows = [num_cols[i:i+n_cols_grid] for i in range(0, len(num_cols), n_cols_grid)]
            for row in rows:
                grid = st.columns(len(row))
                for col_widget, feat in zip(grid, row):
                    with col_widget:
                        st.caption(f"`{feat}`")
                        s = df_preview[feat].dropna()
                        hist = s.value_counts(bins=15, sort=False).sort_index()
                        # Convert Interval index → string so Altair can render it
                        hist.index = [f"{iv.left:.2g}–{iv.right:.2g}" for iv in hist.index]
                        st.bar_chart(hist, height=120, use_container_width=True)

    # ── Categorical value counts ──
    cat_cols = df_preview.select_dtypes(exclude="number").columns.tolist()
    if target_col in cat_cols:
        cat_cols = [c for c in cat_cols if c != target_col]
    if cat_cols:
        with st.expander(f"Categorical columns ({len(cat_cols)})", expanded=False):
            for feat in cat_cols:
                counts = df_preview[feat].value_counts().head(10)
                st.caption(f"`{feat}` — {df_preview[feat].nunique()} unique")
                st.bar_chart(counts, height=120, use_container_width=True)

# ============================================================
# TAB: Results
# ============================================================

with tab_results:
    if "result" not in st.session_state:
        st.info("Run the pipeline first — upload a CSV and click **▶ Run Kashif**.")
        st.stop()

    result      = st.session_state["result"]
    meta        = st.session_state.get("run_meta", {})
    baseline    = result.get("baseline_score", 0)
    best        = result.get("cv_score", 0)
    delta       = result.get("delta", 0)
    best_round  = result.get("best_round", 0)
    rounds_data = result.get("rounds", [])
    top_feats   = result.get("top_features", [])
    dead_feats  = result.get("dead_features", [])
    run_dir     = result.get("output_dir", "")

    # ── Score cards ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline CV", f"{baseline:.4f}")
    c2.metric("Best CV", f"{best:.4f}",
              delta=f"{delta:+.4f}" if delta else None)
    c3.metric("Best round", best_round)
    c4.metric("Rounds run", len(rounds_data))

    st.divider()

    left, right = st.columns([2, 3])

    # ── SHAP features ──
    with left:
        if top_feats:
            st.markdown("#### Top features (SHAP)")
            feat_df = pd.DataFrame({
                "Feature": top_feats,
                "Rank":    list(range(len(top_feats), 0, -1)),
            }).set_index("Feature")
            st.bar_chart(feat_df, y="Rank", horizontal=True, height=300)
        if dead_feats:
            with st.expander(f"Dead features ({len(dead_feats)}) — low SHAP"):
                for f in dead_feats:
                    st.markdown(f"- `{f}`")

    # ── AI narration ──
    with right:
        narr = result.get("narration")
        if narr:
            st.markdown("#### AI summary")
            st.write(narr.get("executive_summary", ""))
            st.info(narr.get("accuracy_statement", ""))
            if narr.get("what_improved"):
                st.success(narr["what_improved"])
            if narr.get("key_factors"):
                st.markdown("**Key factors:**")
                for f in narr["key_factors"]:
                    st.markdown(f"- {f}")
            if narr.get("next_steps"):
                with st.expander("Recommended next steps"):
                    for s in narr["next_steps"]:
                        st.markdown(f"- {s}")
        else:
            st.markdown("#### Result")
            st.markdown(
                f"Static pipeline complete. Best model scored **{best:.4f}** CV.\n\n"
                "Enable the LLM loop for a plain-English summary and feature engineering."
            )

    st.divider()

    # ── Score progression chart ──
    if rounds_data:
        st.markdown("#### Score progression")
        prog_df = pd.DataFrame(
            [{"Round": "baseline", "CV Score": baseline}]
            + [{"Round": f"r{r['round']}", "CV Score": r["cv_score"]}
               for r in rounds_data]
        ).set_index("Round")
        st.line_chart(prog_df, height=200)

    # ── Rounds table ──
    if rounds_data:
        st.markdown("#### Feature engineering rounds")
        rows = [{
            "Round":        r["round"],
            "CV Score":     round(r["cv_score"], 4),
            "Delta":        f"{r['delta']:+.4f}",
            "Improved":     "✓" if r.get("improved") else "✗",
            "Top features": ", ".join(r.get("top_features", [])[:3]),
            "Error":        r.get("executor_error", "") or "",
        } for r in rounds_data]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with st.expander("Feature engineering code per round"):
            for r in rounds_data:
                icon = "✓" if r.get("improved") else "✗"
                st.markdown(f"**Round {r['round']}** {icon} — CV `{r['cv_score']}` delta `{r['delta']:+.4f}`")
                if r.get("fe_code"):
                    st.code(r["fe_code"], language="python")
                if r.get("executor_error"):
                    st.error(r["executor_error"])
                st.markdown("---")

    st.divider()

    # ── Downloads ──
    st.markdown("#### Download artifacts")
    dl1, dl2, dl3 = st.columns(3)

    with dl1:
        b = _read_artifact(str(Path(run_dir) / "best_model.pkl") if run_dir else None)
        if b:
            st.download_button("⬇ best_model.pkl", b, "best_model.pkl",
                               "application/octet-stream", width="stretch")
        else:
            st.button("⬇ best_model.pkl", disabled=True, width="stretch")

    with dl2:
        b = _read_artifact(result.get("report_path"))
        if b:
            st.download_button("⬇ report.md", b, "report.md",
                               "text/markdown", width="stretch")
        else:
            st.button("⬇ report.md", disabled=True, width="stretch")

    with dl3:
        b = _read_artifact(str(Path(run_dir) / "eda_report.html") if run_dir else None)
        if b:
            st.download_button("⬇ eda_report.html", b, "eda_report.html",
                               "text/html", width="stretch")
        else:
            st.button("⬇ eda_report.html", disabled=True, width="stretch")

    if run_dir:
        st.caption(f"Artifacts saved to: `{run_dir}`")

    with st.expander("Raw JSON output"):
        st.json(result)
