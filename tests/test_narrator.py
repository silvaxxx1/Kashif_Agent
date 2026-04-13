"""
test_narrator.py — Tests for core/narrator.py

Strategy:
  - LLM always mocked — no real API calls
  - Covers: happy path, parse fallback, LLMError fallback, JSON extraction,
    key_factors list, what_improved None vs string, next_steps
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from core.narrator import NarratorResult, _build_narrator_prompt, _parse_narrator_response, narrate
from core.fe_agent import RoundResult
from core.llm.base import BaseLLM, LLMError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_round(
    round_num: int,
    cv_score: float,
    improved: bool = False,
    fe_code: str | None = None,
    shap_top: list | None = None,
) -> RoundResult:
    r = MagicMock(spec=RoundResult)
    r.round_num = round_num
    r.cv_score = cv_score
    r.cv_loss = 1.0 - cv_score
    r.delta = 0.0
    r.improved = improved
    r.fe_code = fe_code
    r.executor_error = None
    r.shap_top = shap_top or []
    r.shap_dead = []
    r.train_result = None
    return r


def make_profile(task_type: str = "classification") -> dict:
    return {
        "task_type": task_type,
        "target_col": "survived",
        "n_rows": 891,
        "columns": {},
        "warnings": [],
        "target": {"distribution": {"0": 549, "1": 342}},
        "task_confidence": 0.9,
    }


def make_mock_llm(response: str) -> MagicMock:
    mock = MagicMock(spec=BaseLLM)
    mock.complete_with_system.return_value = response
    return mock


_VALID_RESPONSE = json.dumps({
    "executive_summary": "The model predicts survival with high confidence.",
    "accuracy_statement": "It correctly predicts 9 out of 10 cases.",
    "key_factors": [
        "Passenger class was the strongest predictor.",
        "Age had a moderate effect.",
    ],
    "what_improved": "The AI discovered a useful fare-to-age ratio.",
    "next_steps": [
        "Collect more recent data.",
        "Review edge cases where the model is uncertain.",
    ],
})


# ---------------------------------------------------------------------------
# _build_narrator_prompt
# ---------------------------------------------------------------------------

class TestBuildNarratorPrompt:
    def test_contains_accuracy_percent(self):
        log = [make_round(0, 0.9525), make_round(1, 0.9613, improved=True)]
        prompt = _build_narrator_prompt(log, make_profile())
        assert "96.1" in prompt  # best round score as percent

    def test_contains_target_column(self):
        log = [make_round(0, 0.80)]
        prompt = _build_narrator_prompt(log, make_profile())
        assert "survived" in prompt

    def test_contains_task_type(self):
        log = [make_round(0, 0.80)]
        prompt = _build_narrator_prompt(log, make_profile("regression"))
        assert "regression" in prompt

    def test_shap_features_included(self):
        log = [make_round(0, 0.80, shap_top=["num__age", "num__fare"])]
        prompt = _build_narrator_prompt(log, make_profile())
        assert "age" in prompt
        assert "fare" in prompt

    def test_fe_improvement_included_when_present(self):
        fe_code = "def engineer_features(df):\n    df = df.copy()\n    df['fare_age_ratio'] = df['fare'] / df['age']\n    return df"
        log = [
            make_round(0, 0.80),
            make_round(1, 0.85, improved=True, fe_code=fe_code),
        ]
        prompt = _build_narrator_prompt(log, make_profile())
        assert "fare_age_ratio" in prompt or "improvement" in prompt.lower()

    def test_schema_json_present(self):
        log = [make_round(0, 0.80)]
        prompt = _build_narrator_prompt(log, make_profile())
        assert "executive_summary" in prompt
        assert "key_factors" in prompt
        assert "next_steps" in prompt


# ---------------------------------------------------------------------------
# _parse_narrator_response
# ---------------------------------------------------------------------------

class TestParseNarratorResponse:
    def test_valid_json_parsed(self):
        result = _parse_narrator_response(_VALID_RESPONSE)
        assert result is not None
        assert result["executive_summary"] == "The model predicts survival with high confidence."

    def test_json_in_markdown_fence_parsed(self):
        fenced = f"```json\n{_VALID_RESPONSE}\n```"
        result = _parse_narrator_response(fenced)
        assert result is not None
        assert "executive_summary" in result

    def test_missing_required_keys_returns_none(self):
        bad = json.dumps({"foo": "bar"})
        result = _parse_narrator_response(bad)
        assert result is None

    def test_invalid_json_returns_none(self):
        result = _parse_narrator_response("not json at all")
        assert result is None

    def test_json_embedded_in_text_extracted(self):
        wrapped = f"Here is my analysis:\n{_VALID_RESPONSE}\nHope that helps!"
        result = _parse_narrator_response(wrapped)
        assert result is not None

    def test_what_improved_null_preserved(self):
        response = json.dumps({
            "executive_summary": "Good model.",
            "accuracy_statement": "95% accurate.",
            "key_factors": [],
            "what_improved": None,
            "next_steps": [],
        })
        result = _parse_narrator_response(response)
        assert result is not None
        assert result["what_improved"] is None


# ---------------------------------------------------------------------------
# narrate() — happy path
# ---------------------------------------------------------------------------

class TestNarrateHappyPath:
    def test_returns_narrator_result(self):
        log = [make_round(0, 0.9525)]
        llm = make_mock_llm(_VALID_RESPONSE)
        result = narrate(log, make_profile(), llm)
        assert isinstance(result, NarratorResult)

    def test_executive_summary_populated(self):
        log = [make_round(0, 0.9525)]
        llm = make_mock_llm(_VALID_RESPONSE)
        result = narrate(log, make_profile(), llm)
        assert "survival" in result.executive_summary.lower() or len(result.executive_summary) > 10

    def test_key_factors_is_list(self):
        log = [make_round(0, 0.9525)]
        llm = make_mock_llm(_VALID_RESPONSE)
        result = narrate(log, make_profile(), llm)
        assert isinstance(result.key_factors, list)
        assert len(result.key_factors) >= 1

    def test_what_improved_populated_when_fe_helped(self):
        log = [
            make_round(0, 0.90),
            make_round(1, 0.95, improved=True),
        ]
        llm = make_mock_llm(_VALID_RESPONSE)
        result = narrate(log, make_profile(), llm)
        assert result.what_improved is not None

    def test_what_improved_none_when_no_improvement(self):
        no_improvement_response = json.dumps({
            "executive_summary": "Good model.",
            "accuracy_statement": "95% accurate.",
            "key_factors": ["Age matters."],
            "what_improved": None,
            "next_steps": ["Collect more data."],
        })
        log = [make_round(0, 0.9525)]
        llm = make_mock_llm(no_improvement_response)
        result = narrate(log, make_profile(), llm)
        assert result.what_improved is None

    def test_next_steps_is_list(self):
        log = [make_round(0, 0.9525)]
        llm = make_mock_llm(_VALID_RESPONSE)
        result = narrate(log, make_profile(), llm)
        assert isinstance(result.next_steps, list)

    def test_uses_complete_with_system(self):
        log = [make_round(0, 0.9525)]
        llm = make_mock_llm(_VALID_RESPONSE)
        narrate(log, make_profile(), llm)
        llm.complete_with_system.assert_called_once()


# ---------------------------------------------------------------------------
# narrate() — fallback paths
# ---------------------------------------------------------------------------

class TestNarrateFallbacks:
    def test_llm_error_returns_fallback_not_raises(self):
        log = [make_round(0, 0.9525)]
        llm = MagicMock(spec=BaseLLM)
        llm.complete_with_system.side_effect = LLMError("API key missing")
        result = narrate(log, make_profile(), llm)
        assert isinstance(result, NarratorResult)
        assert len(result.executive_summary) > 0

    def test_unparseable_response_returns_fallback(self):
        log = [make_round(0, 0.9525)]
        llm = make_mock_llm("I cannot provide results in that format.")
        result = narrate(log, make_profile(), llm)
        assert isinstance(result, NarratorResult)
        assert len(result.executive_summary) > 0

    def test_empty_log_returns_fallback(self):
        llm = make_mock_llm(_VALID_RESPONSE)
        result = narrate([], make_profile(), llm)
        assert isinstance(result, NarratorResult)

    def test_raw_response_stored_on_parse_failure(self):
        log = [make_round(0, 0.9525)]
        bad_response = "This is not JSON at all."
        llm = make_mock_llm(bad_response)
        result = narrate(log, make_profile(), llm)
        assert result.raw_response == bad_response
