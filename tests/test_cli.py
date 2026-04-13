"""
test_cli.py — Step 4g tests for cli/main.py

Strategy:
  - All heavy work (profiler, trainer, fe_agent, reporter) is mocked
  - CSV files are synthetic tmp files
  - Tests cover: happy path (agent + no-agent), flag parsing, error handling,
    JSON output contract, provider resolution, info command
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from cli.main import app, _load_config, _resolve_llm, _build_output

runner = CliRunner()


def _extract_json(output: str) -> dict:
    """Extract the JSON object from CLI output that may contain [kashif] status lines."""
    # Find the first line that starts a JSON object
    lines = output.splitlines()
    start = next((i for i, l in enumerate(lines) if l.strip().startswith("{")), None)
    if start is None:
        raise ValueError(f"No JSON object found in output:\n{output}")
    return json.loads("\n".join(lines[start:]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_csv(tmp_path: Path, n: int = 50) -> Path:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "age": rng.integers(18, 80, n).astype(float),
        "fare": rng.uniform(5, 200, n),
        "target": rng.choice([0, 1], n),
    })
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)
    return p


def make_profile() -> dict:
    return {
        "task_type": "classification",
        "task_confidence": 0.92,
        "target_col": "target",
        "n_rows": 50,
        "n_cols": 2,
        "columns": {},
        "target": {"is_imbalanced": False},
        "warnings": [],
    }


def make_round_result(round_num: int = 0, cv_score: float = 0.75):
    from core.fe_agent import RoundResult
    tr = MagicMock()
    tr.cv_score = cv_score
    tr.cv_loss = 1.0 - cv_score
    tr.best_model_name = "Random Forest"
    tr.model_path = "/tmp/model.pkl"
    tr.status = "complete"
    return RoundResult(
        round_num=round_num,
        fe_code=None if round_num == 0 else "def engineer_features(df):\n    return df.copy()",
        cv_score=cv_score,
        cv_loss=1.0 - cv_score,
        delta=0.0 if round_num == 0 else cv_score - 0.70,
        improved=round_num > 0 and cv_score > 0.70,
        executor_error=None,
        shap_top=["age", "fare"],
        shap_dead=[],
        train_result=tr,
    )


def _mock_profiler_run(df, target, output_dir="./outputs", save_eda=True):
    return make_profile(), None


def _mock_agent_run(**kwargs):
    return [make_round_result(0, 0.70), make_round_result(1, 0.78)]


def _mock_reporter_run(log, profile_json=None, output_dir="./outputs", save_report=True):
    return "# Report\n\nTest.", "/tmp/report.md" if save_report else None


# ---------------------------------------------------------------------------
# _load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_returns_dict(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("llm:\n  provider: groq\n")
        result = _load_config(cfg_file)
        assert isinstance(result, dict)
        assert result["llm"]["provider"] == "groq"

    def test_missing_file_returns_empty(self, tmp_path):
        result = _load_config(tmp_path / "nonexistent.yaml")
        assert result == {}


# ---------------------------------------------------------------------------
# _resolve_llm
# ---------------------------------------------------------------------------

class TestResolveLlm:
    def test_groq_returns_groq_llm(self):
        from core.llm.groq import GroqLLM
        llm = _resolve_llm("groq", {"llm": {"model": "llama-3.3-70b-versatile"}})
        assert isinstance(llm, GroqLLM)

    def test_anthropic_returns_anthropic_llm(self):
        from core.llm.anthropic import AnthropicLLM
        llm = _resolve_llm("anthropic", {"llm": {"model": "claude-sonnet-4-6"}})
        assert isinstance(llm, AnthropicLLM)

    def test_unknown_provider_exits(self):
        result = runner.invoke(app, ["run", "--csv", "x.csv", "--target", "y", "--llm", "unknown"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# _build_output
# ---------------------------------------------------------------------------

class TestBuildOutput:
    def test_has_required_keys(self):
        log = [make_round_result(0, 0.70), make_round_result(1, 0.78)]
        out = _build_output(log, make_profile(), "/tmp/report.md", "./outputs")
        for key in ["status", "best_round", "cv_score", "baseline_score",
                    "delta", "rounds", "report_path", "output_dir"]:
            assert key in out

    def test_status_complete(self):
        log = [make_round_result(0, 0.70)]
        out = _build_output(log, make_profile(), None, "./outputs")
        assert out["status"] in ("complete", "failed")

    def test_baseline_score_correct(self):
        log = [make_round_result(0, 0.70), make_round_result(1, 0.80)]
        out = _build_output(log, make_profile(), None, "./outputs")
        assert out["baseline_score"] == 0.70

    def test_best_cv_score_correct(self):
        log = [make_round_result(0, 0.70), make_round_result(1, 0.80)]
        out = _build_output(log, make_profile(), None, "./outputs")
        assert out["cv_score"] == 0.80

    def test_delta_correct(self):
        log = [make_round_result(0, 0.70), make_round_result(1, 0.80)]
        out = _build_output(log, make_profile(), None, "./outputs")
        assert abs(out["delta"] - 0.10) < 0.001

    def test_rounds_list_excludes_baseline(self):
        log = [make_round_result(0, 0.70), make_round_result(1, 0.78)]
        out = _build_output(log, make_profile(), None, "./outputs")
        assert all(r["round"] > 0 for r in out["rounds"])

    def test_baseline_only_has_empty_rounds(self):
        log = [make_round_result(0, 0.70)]
        out = _build_output(log, make_profile(), None, "./outputs")
        assert out["rounds"] == []


# ---------------------------------------------------------------------------
# run command — happy path (agent)
# ---------------------------------------------------------------------------

class TestRunCommandAgent:
    def test_exit_code_zero(self, tmp_path):
        csv_path = make_csv(tmp_path)
        with patch("core.profiler.run", side_effect=_mock_profiler_run), \
             patch("core.fe_agent.run", return_value=_mock_agent_run()), \
             patch("core.reporter.run", side_effect=_mock_reporter_run):
            result = runner.invoke(app, [
                "run", "--csv", str(csv_path), "--target", "target",
                "--output-dir", str(tmp_path),
            ])
        assert result.exit_code == 0, result.output

    def test_stdout_is_valid_json(self, tmp_path):
        csv_path = make_csv(tmp_path)
        with patch("core.profiler.run", side_effect=_mock_profiler_run), \
             patch("core.fe_agent.run", return_value=_mock_agent_run()), \
             patch("core.reporter.run", side_effect=_mock_reporter_run):
            result = runner.invoke(app, [
                "run", "--csv", str(csv_path), "--target", "target",
                "--output-dir", str(tmp_path),
            ])
        parsed = _extract_json(result.output)
        assert isinstance(parsed, dict)

    def test_output_has_required_keys(self, tmp_path):
        csv_path = make_csv(tmp_path)
        with patch("core.profiler.run", side_effect=_mock_profiler_run), \
             patch("core.fe_agent.run", return_value=_mock_agent_run()), \
             patch("core.reporter.run", side_effect=_mock_reporter_run):
            result = runner.invoke(app, [
                "run", "--csv", str(csv_path), "--target", "target",
                "--output-dir", str(tmp_path),
            ])
        out = _extract_json(result.output)
        for key in ["status", "cv_score", "baseline_score", "delta", "rounds", "report_path"]:
            assert key in out

    def test_rounds_flag_passed_to_agent(self, tmp_path):
        csv_path = make_csv(tmp_path)
        captured = {}

        def capture_agent_run(**kwargs):
            captured["config"] = kwargs.get("config", {})
            return _mock_agent_run()

        with patch("core.profiler.run", side_effect=_mock_profiler_run), \
             patch("core.fe_agent.run", side_effect=capture_agent_run), \
             patch("core.reporter.run", side_effect=_mock_reporter_run):
            runner.invoke(app, [
                "run", "--csv", str(csv_path), "--target", "target",
                "--rounds", "3", "--output-dir", str(tmp_path),
            ])
        assert captured.get("config", {}).get("max_rounds") == 3

    def test_llm_flag_selects_provider(self, tmp_path):
        csv_path = make_csv(tmp_path)
        resolved = {}

        def capture_resolve(provider, cfg):
            resolved["provider"] = provider
            return MagicMock()

        with patch("core.profiler.run", side_effect=_mock_profiler_run), \
             patch("cli.main._resolve_llm", side_effect=capture_resolve), \
             patch("core.fe_agent.run", return_value=_mock_agent_run()), \
             patch("core.reporter.run", side_effect=_mock_reporter_run):
            runner.invoke(app, [
                "run", "--csv", str(csv_path), "--target", "target",
                "--llm", "anthropic", "--output-dir", str(tmp_path),
            ])
        assert resolved.get("provider") == "anthropic"


# ---------------------------------------------------------------------------
# run command — --no-agent flag
# ---------------------------------------------------------------------------

class TestRunCommandNoAgent:
    def test_no_agent_skips_fe_agent(self, tmp_path):
        csv_path = make_csv(tmp_path)
        agent_called = {"called": False}

        def spy_agent(**kwargs):
            agent_called["called"] = True
            return _mock_agent_run()

        mock_train_result = MagicMock()
        mock_train_result.cv_score = 0.75
        mock_train_result.cv_loss = 0.25
        mock_train_result.status = "complete"
        mock_train_result.model_path = "/tmp/model.pkl"
        mock_train_result.best_model_name = "RF"

        with patch("core.profiler.run", side_effect=_mock_profiler_run), \
             patch("core.fe_agent.run", side_effect=spy_agent), \
             patch("core.trainer.train", return_value=mock_train_result), \
             patch("core.reporter.run", side_effect=_mock_reporter_run):
            result = runner.invoke(app, [
                "run", "--csv", str(csv_path), "--target", "target",
                "--no-agent", "--output-dir", str(tmp_path),
            ])

        assert not agent_called["called"]
        assert result.exit_code == 0

    def test_no_agent_output_is_valid_json(self, tmp_path):
        csv_path = make_csv(tmp_path)
        mock_train_result = MagicMock()
        mock_train_result.cv_score = 0.75
        mock_train_result.cv_loss = 0.25
        mock_train_result.status = "complete"
        mock_train_result.model_path = None
        mock_train_result.best_model_name = "RF"

        with patch("core.profiler.run", side_effect=_mock_profiler_run), \
             patch("core.trainer.train", return_value=mock_train_result), \
             patch("core.reporter.run", side_effect=_mock_reporter_run):
            result = runner.invoke(app, [
                "run", "--csv", str(csv_path), "--target", "target",
                "--no-agent", "--output-dir", str(tmp_path),
            ])
        parsed = _extract_json(result.output)
        assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# run command — error handling
# ---------------------------------------------------------------------------

class TestRunCommandErrors:
    def test_missing_csv_exits_nonzero(self, tmp_path):
        result = runner.invoke(app, [
            "run", "--csv", str(tmp_path / "nonexistent.csv"), "--target", "target",
        ])
        assert result.exit_code != 0

    def test_wrong_target_exits_nonzero(self, tmp_path):
        csv_path = make_csv(tmp_path)
        result = runner.invoke(app, [
            "run", "--csv", str(csv_path), "--target", "nonexistent_col",
        ])
        assert result.exit_code != 0

    def test_missing_csv_flag_exits_nonzero(self):
        result = runner.invoke(app, ["run", "--target", "target"])
        assert result.exit_code != 0

    def test_missing_target_flag_exits_nonzero(self, tmp_path):
        csv_path = make_csv(tmp_path)
        result = runner.invoke(app, ["run", "--csv", str(csv_path)])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# info command
# ---------------------------------------------------------------------------

class TestInfoCommand:
    def test_info_exits_zero(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("llm:\n  provider: groq\n")
        result = runner.invoke(app, ["info", "--config", str(cfg_file)])
        assert result.exit_code == 0

    def test_info_shows_provider(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("llm:\n  provider: groq\n")
        result = runner.invoke(app, ["info", "--config", str(cfg_file)])
        assert "groq" in result.output


# ---------------------------------------------------------------------------
# _sanitize and _build_output — NaN / Inf regression tests
# ---------------------------------------------------------------------------

class TestBuildOutputSanitize:
    """Regression tests: json.dumps must never emit NaN or Infinity."""

    def _make_round(self, round_num, cv_score, cv_loss, delta, improved=False):
        from core.fe_agent import RoundResult
        r = MagicMock(spec=RoundResult)
        r.round_num = round_num
        r.cv_score = cv_score
        r.cv_loss = cv_loss
        r.delta = delta
        r.improved = improved
        r.fe_code = None
        r.executor_error = None
        r.shap_top = []
        r.shap_dead = []
        r.train_result = None
        return r

    def test_build_output_no_nan_in_json(self):
        """NaN scores must not appear in json.dumps output (invalid JSON spec)."""
        log = [self._make_round(0, float("nan"), float("inf"), 0.0)]
        from cli.main import _build_output
        output = _build_output(log, {"task_type": "classification"}, None, "/tmp")
        # Verify json.dumps does not raise and produces valid JSON
        serialised = json.dumps(output)
        parsed = json.loads(serialised)
        # NaN/inf fields should be None, not NaN
        assert parsed["cv_score"] is None or isinstance(parsed["cv_score"], (int, float))
        assert "NaN" not in serialised
        assert "Infinity" not in serialised

    def test_build_output_inf_score_becomes_none(self):
        """Infinity in cv_score must serialize as null."""
        log = [self._make_round(0, float("inf"), float("inf"), 0.0)]
        from cli.main import _build_output
        output = _build_output(log, {"task_type": "classification"}, None, "/tmp")
        serialised = json.dumps(output)
        assert "Infinity" not in serialised
        parsed = json.loads(serialised)
        assert parsed["cv_score"] is None

    def test_build_output_normal_scores_preserved(self):
        """Valid float scores must survive sanitization unchanged."""
        log = [self._make_round(0, 0.85, 0.15, 0.0)]
        from cli.main import _build_output
        output = _build_output(log, {"task_type": "classification"}, None, "/tmp")
        assert output["cv_score"] == round(0.85, 4)
        assert output["baseline_score"] == round(0.85, 4)
