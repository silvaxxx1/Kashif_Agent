"""
test_api.py — Step 5 tests for api/server.py

Strategy:
  - Uses FastAPI TestClient (httpx under the hood) — no real server needed
  - POST /run with no_agent=True uses a real small CSV — no LLM key required
  - LLM-dependent paths are skipped (marked with no_agent=False) in unit tests
  - Covers: health, providers, run success, run error cases
"""

from __future__ import annotations

import io
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.server import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_csv(n_rows: int = 50) -> bytes:
    """Generate a small binary CSV for upload tests."""
    import numpy as np
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "age":    rng.integers(18, 70, n_rows).astype(float),
        "income": rng.integers(20000, 120000, n_rows).astype(float),
        "score":  rng.uniform(0, 1, n_rows),
        "target": rng.integers(0, 2, n_rows),
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


_CSV_BYTES = _make_csv()


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_status_ok(self):
        r = client.get("/health")
        assert r.json()["status"] == "ok"

    def test_has_version(self):
        r = client.get("/health")
        assert "version" in r.json()


# ---------------------------------------------------------------------------
# GET /providers
# ---------------------------------------------------------------------------

class TestProviders:
    def test_returns_200(self):
        r = client.get("/providers")
        assert r.status_code == 200

    def test_providers_list(self):
        r = client.get("/providers")
        data = r.json()
        assert "providers" in data
        assert set(data["providers"]) == {"groq", "anthropic", "ollama"}

    def test_has_default(self):
        r = client.get("/providers")
        assert r.json()["default"] == "groq"

    def test_has_notes(self):
        r = client.get("/providers")
        assert "notes" in r.json()


# ---------------------------------------------------------------------------
# POST /run — error cases (no CSV needed)
# ---------------------------------------------------------------------------

class TestRunErrors:
    def test_missing_file_returns_422(self):
        r = client.post("/run", data={"target": "target"})
        assert r.status_code == 422

    def test_missing_target_returns_422(self):
        r = client.post(
            "/run",
            files={"file": ("data.csv", _CSV_BYTES, "text/csv")},
        )
        assert r.status_code == 422

    def test_wrong_target_column_returns_422(self):
        r = client.post(
            "/run",
            files={"file": ("data.csv", _CSV_BYTES, "text/csv")},
            data={"target": "nonexistent_column", "no_agent": "true"},
        )
        assert r.status_code == 422

    def test_wrong_target_error_message_mentions_column(self):
        r = client.post(
            "/run",
            files={"file": ("data.csv", _CSV_BYTES, "text/csv")},
            data={"target": "nonexistent_column", "no_agent": "true"},
        )
        assert "nonexistent_column" in r.json()["detail"]

    def test_invalid_csv_returns_400(self):
        r = client.post(
            "/run",
            files={"file": ("data.csv", b"not,,valid\ncsv data\xff\xfe", "text/csv")},
            data={"target": "target", "no_agent": "true"},
        )
        # Either 400 (bad CSV) or 422 (column not found after parse attempt)
        assert r.status_code in (400, 422)

    def test_unknown_provider_returns_422(self):
        r = client.post(
            "/run",
            files={"file": ("data.csv", _CSV_BYTES, "text/csv")},
            data={"target": "target", "provider": "unknown_llm", "no_agent": "false"},
        )
        assert r.status_code == 422

    def test_unknown_provider_error_message(self):
        r = client.post(
            "/run",
            files={"file": ("data.csv", _CSV_BYTES, "text/csv")},
            data={"target": "target", "provider": "unknown_llm", "no_agent": "false"},
        )
        assert "unknown_llm" in r.json()["detail"]


# ---------------------------------------------------------------------------
# POST /run — static pipeline (no_agent=True, no LLM key needed)
# ---------------------------------------------------------------------------

class TestRunNoAgent:
    def test_returns_200(self):
        r = client.post(
            "/run",
            files={"file": ("data.csv", _CSV_BYTES, "text/csv")},
            data={"target": "target", "no_agent": "true"},
            timeout=120,
        )
        assert r.status_code == 200

    def test_status_complete(self):
        r = client.post(
            "/run",
            files={"file": ("data.csv", _CSV_BYTES, "text/csv")},
            data={"target": "target", "no_agent": "true"},
            timeout=120,
        )
        assert r.json()["status"] == "complete"

    def test_output_has_required_keys(self):
        r = client.post(
            "/run",
            files={"file": ("data.csv", _CSV_BYTES, "text/csv")},
            data={"target": "target", "no_agent": "true"},
            timeout=120,
        )
        data = r.json()
        for key in ("status", "cv_score", "baseline_score", "delta", "rounds"):
            assert key in data, f"Missing key: {key}"

    def test_baseline_score_is_nonzero(self):
        r = client.post(
            "/run",
            files={"file": ("data.csv", _CSV_BYTES, "text/csv")},
            data={"target": "target", "no_agent": "true"},
            timeout=120,
        )
        assert r.json()["baseline_score"] > 0

    def test_no_agent_rounds_empty(self):
        r = client.post(
            "/run",
            files={"file": ("data.csv", _CSV_BYTES, "text/csv")},
            data={"target": "target", "no_agent": "true"},
            timeout=120,
        )
        assert r.json()["rounds"] == []

    def test_no_agent_best_round_is_zero(self):
        r = client.post(
            "/run",
            files={"file": ("data.csv", _CSV_BYTES, "text/csv")},
            data={"target": "target", "no_agent": "true"},
            timeout=120,
        )
        assert r.json()["best_round"] == 0

    def test_response_is_valid_json(self):
        r = client.post(
            "/run",
            files={"file": ("data.csv", _CSV_BYTES, "text/csv")},
            data={"target": "target", "no_agent": "true"},
            timeout=120,
        )
        # If this doesn't raise, it's valid JSON
        json.dumps(r.json())

    def test_rounds_flag_respected(self):
        """rounds param is accepted without error even in no_agent mode."""
        r = client.post(
            "/run",
            files={"file": ("data.csv", _CSV_BYTES, "text/csv")},
            data={"target": "target", "no_agent": "true", "rounds": "2"},
            timeout=120,
        )
        assert r.status_code == 200

    def test_custom_provider_accepted_in_no_agent_mode(self):
        """Provider param is accepted but ignored when no_agent=True."""
        r = client.post(
            "/run",
            files={"file": ("data.csv", _CSV_BYTES, "text/csv")},
            data={"target": "target", "no_agent": "true", "provider": "anthropic"},
            timeout=120,
        )
        assert r.status_code == 200

    def test_titanic_csv(self):
        """Run the real titanic CSV through the static pipeline."""
        import pathlib
        titanic = pathlib.Path(__file__).parent.parent / "data" / "titanic.csv"
        if not titanic.exists():
            pytest.skip("titanic.csv not found")
        with open(titanic, "rb") as f:
            csv_bytes = f.read()
        r = client.post(
            "/run",
            files={"file": ("titanic.csv", csv_bytes, "text/csv")},
            data={"target": "survived", "no_agent": "true"},
            timeout=120,
        )
        assert r.status_code == 200
        assert r.json()["baseline_score"] > 0.7
