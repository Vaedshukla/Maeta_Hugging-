"""Pydantic models for the DataClean-Env environment.

Defines typed Action, Observation, and State models following OpenEnv spec.
Observation uses issue-first design per Codex feedback.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from openenv.core.env_server import Action, Observation, State


# --- Supporting Types ---

class QualityIssue(BaseModel):
    """A detected quality issue in the dataset."""

    row_id: int
    column: str
    issue_type: str = Field(
        description="One of: null, format, duplicate, case, type_violation, cross_field, anomaly"
    )
    description: str
    suggestion: Optional[str] = None


class IssueGroup(BaseModel):
    """Issues grouped by type for compact display."""

    issue_type: str
    count: int
    examples: List[QualityIssue] = Field(default_factory=list)


class DataSummary(BaseModel):
    """Compact summary of the dataset state."""

    row_count: int = 0
    column_count: int = 0
    total_cells: int = 0
    null_count: int = 0
    issue_count: int = 0
    columns: List[str] = Field(default_factory=list)
    dtypes: Dict[str, str] = Field(default_factory=dict)


class ActionResult(BaseModel):
    """Result of executing an action."""

    action: str
    status: str = Field(description="One of: success, error, no_effect")
    message: str
    cells_modified: int = 0


# --- Core Models ---

class DataCleanAction(Action):
    """Agent's action to clean data.

    Actions reference rows by stable `row_id` (integer, unique within
    episode, survives delete/merge operations). The row_id is visible
    in every observation row and does NOT change during the episode.
    """

    action_type: str = Field(
        ...,
        description=(
            "One of: fix_value, delete_row, fill_missing, standardize_format, "
            "merge_duplicates, flag_anomaly, split_column, rename_column, "
            "cast_type, escalate_to_human, mark_complete"
        ),
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Action-specific parameters. Use 'row_id' (not index) to reference rows. "
            "fix_value: {row_id, column, new_value}. "
            "delete_row: {row_id}. "
            "fill_missing: {row_id, column, value}. "
            "standardize_format: {column, format_type}. "
            "merge_duplicates: {row_id1, row_id2, strategy}. "
            "flag_anomaly: {row_id, column, reason}. "
            "split_column: {column, delimiter, new_names}. "
            "rename_column: {old_name, new_name}. "
            "cast_type: {column, target_type}. "
            "escalate_to_human: {row_id, column, confidence, reason}. "
            "mark_complete: {}."
        ),
    )


class DataCleanObservation(Observation):
    """What the agent sees after each step.

    Issue-first design: quality_issues and data_summary are primary.
    Full row data is secondary (truncated for large datasets).
    """

    # --- Issue-first fields (PRIMARY) ---
    data_summary: DataSummary = Field(default_factory=DataSummary)
    quality_issues: List[QualityIssue] = Field(default_factory=list)
    issue_groups: List[IssueGroup] = Field(default_factory=list)
    issues_remaining: int = 0

    # --- Data (SECONDARY, may be truncated) ---
    columns: List[str] = Field(default_factory=list)
    rows: List[List[Any]] = Field(default_factory=list)
    row_count: int = 0

    # --- Schema info ---
    schema_info: Dict[str, Any] = Field(default_factory=dict)

    # --- Step context ---
    step_number: int = 0
    max_steps: int = 30
    steps_remaining: int = 30

    # --- Budget info ---
    budget_spent: float = 0.0
    budget_remaining: float = 100.0
    action_costs: Dict[str, float] = Field(default_factory=dict)

    # --- History ---
    last_action_result: Optional[ActionResult] = None
    recent_actions: List[ActionResult] = Field(default_factory=list)

    # --- Task info ---
    task_id: str = ""
    task_name: str = ""
    difficulty: str = ""

    # --- Inherited from Observation base ---
    # done: bool = False
    # reward: bool | int | float | None = None
    # metadata: Dict[str, Any] = {}


class DataCleanState(State):
    """Internal environment state. Not exposed to agent directly."""

    # Inherited: episode_id (str), step_count (int)
    task_id: str = ""
    difficulty: str = ""
    current_data: List[Dict[str, Any]] = Field(default_factory=list)
    ground_truth: List[Dict[str, Any]] = Field(default_factory=list)
    original_dirty: List[Dict[str, Any]] = Field(default_factory=list)
    schema_def: Dict[str, Any] = Field(default_factory=dict)
    action_log: List[Dict[str, Any]] = Field(default_factory=list)
    flagged_cells: List[Dict[str, str]] = Field(default_factory=list)
    escalated_cells: List[Dict[str, Any]] = Field(default_factory=list)
    max_steps: int = 30
    is_complete: bool = False
    previous_score: float = 0.0  # For delta reward computation (mutates each step)
    initial_raw_score: float = 0.0  # Raw score of dirty data at reset (immutable)

    # Cost-aware intervention budget
    action_budget: float = 100.0      # Total budget for the episode
    budget_spent: float = 0.0         # Cost spent so far
    budget_remaining: float = 100.0   # Budget left
