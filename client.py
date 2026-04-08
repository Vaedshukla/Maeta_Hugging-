"""Typed WebSocket client for the DataClean-Env OpenEnv environment.

Connects to a running DataClean environment server via WebSocket and
provides a type-safe interface for stepping through data-cleaning episodes.

Usage::

    from dataclean_env.client import DataCleanEnv

    with DataCleanEnv(base_url="http://localhost:8000").sync() as env:
        result = env.reset()
        obs = result.observation
        done = False
        while not done:
            action = DataCleanAction(
                action_type="fix_value",
                params={"row_id": 0, "column": "name", "new_value": "Alice"},
            )
            result = env.step(action)
            obs = result.observation
            done = result.done
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from dataclean_env.models import (
    ActionResult,
    DataCleanAction,
    DataCleanObservation,
    DataCleanState,
    DataSummary,
    IssueGroup,
    QualityIssue,
)


class DataCleanEnv(EnvClient[DataCleanAction, DataCleanObservation, DataCleanState]):
    """WebSocket client for the DataClean environment.

    Subclasses :class:`EnvClient` and implements the three required
    serialisation hooks so the generic base class can handle the
    WebSocket transport transparently.

    Parameters
    ----------
    base_url:
        Root URL of the DataClean environment server
        (e.g. ``"http://localhost:8000"``).
    """

    # ------------------------------------------------------------------
    # Abstract-method implementations
    # ------------------------------------------------------------------

    def _step_payload(self, action: DataCleanAction) -> Dict[str, Any]:
        """Serialise a :class:`DataCleanAction` to a JSON-safe dict.

        The dict is sent over the WebSocket to the server's ``step``
        endpoint.
        """
        return {
            "action_type": action.action_type,
            "params": action.params,
        }

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[DataCleanObservation]:
        """Parse the server's step response into a typed :class:`StepResult`.

        The *payload* dict is the JSON body returned by the server after
        a ``step`` or ``reset`` call.
        """
        obs_data: Dict[str, Any] = payload.get("observation", {})

        observation = _build_observation(obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> DataCleanState:
        """Parse the server's state response into :class:`DataCleanState`."""
        return DataCleanState(**payload)


# ------------------------------------------------------------------
# Private helpers — kept outside the class to stay under 50 lines/fn
# ------------------------------------------------------------------


def _parse_quality_issues(
    raw_issues: List[Dict[str, Any]],
) -> List[QualityIssue]:
    """Convert a list of raw dicts into :class:`QualityIssue` models."""
    return [QualityIssue(**item) for item in raw_issues]


def _parse_issue_groups(
    raw_groups: List[Dict[str, Any]],
) -> List[IssueGroup]:
    """Convert a list of raw dicts into :class:`IssueGroup` models."""
    return [IssueGroup(**item) for item in raw_groups]


def _parse_action_result(
    raw: Optional[Dict[str, Any]],
) -> Optional[ActionResult]:
    """Convert a raw dict into an :class:`ActionResult`, or ``None``."""
    if raw is None:
        return None
    return ActionResult(**raw)


def _parse_recent_actions(
    raw_actions: List[Dict[str, Any]],
) -> List[ActionResult]:
    """Convert a list of raw dicts into :class:`ActionResult` models."""
    return [ActionResult(**item) for item in raw_actions]


def _build_observation(obs_data: Dict[str, Any]) -> DataCleanObservation:
    """Construct a fully-typed :class:`DataCleanObservation` from raw JSON.

    Nested models (:class:`DataSummary`, :class:`QualityIssue`, etc.) are
    parsed explicitly so that callers always receive validated Pydantic
    objects rather than raw dicts.
    """
    data_summary_raw: Dict[str, Any] = obs_data.get("data_summary", {})
    data_summary = DataSummary(**data_summary_raw) if data_summary_raw else DataSummary()

    return DataCleanObservation(
        # Issue-first fields
        data_summary=data_summary,
        quality_issues=_parse_quality_issues(obs_data.get("quality_issues", [])),
        issue_groups=_parse_issue_groups(obs_data.get("issue_groups", [])),
        issues_remaining=obs_data.get("issues_remaining", 0),
        # Data fields
        columns=obs_data.get("columns", []),
        rows=obs_data.get("rows", []),
        row_count=obs_data.get("row_count", 0),
        # Schema info
        schema_info=obs_data.get("schema_info", {}),
        # Step context
        step_number=obs_data.get("step_number", 0),
        max_steps=obs_data.get("max_steps", 30),
        steps_remaining=obs_data.get("steps_remaining", 30),
        # History
        last_action_result=_parse_action_result(obs_data.get("last_action_result")),
        recent_actions=_parse_recent_actions(obs_data.get("recent_actions", [])),
        # Task info
        task_id=obs_data.get("task_id", ""),
        task_name=obs_data.get("task_name", ""),
        difficulty=obs_data.get("difficulty", ""),
        # Budget fields
        budget_spent=obs_data.get("budget_spent", 0.0),
        budget_remaining=obs_data.get("budget_remaining", 100.0),
        action_costs=obs_data.get("action_costs", {}),
        # Inherited Observation fields
        done=obs_data.get("done", False),
        reward=obs_data.get("reward"),
        metadata=obs_data.get("metadata", {}),
    )
