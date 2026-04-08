"""Baseline inference script for DataClean-Env hackathon.

Runs an LLM-based data-cleaning agent through all three tasks
(easy_contacts, medium_employees, hard_patients), collects scores,
and emits [START]/[STEP]/[END] lines to stdout per OpenEnv spec.

Environment variables required:
    API_BASE_URL  - LLM endpoint URL (default: https://api.openai.com/v1)
    MODEL_NAME    - model identifier (default: gpt-4.1-mini)
    HF_TOKEN      - Hugging Face API token (mandatory, no default)

Usage:
    API_BASE_URL=... MODEL_NAME=... HF_TOKEN=... python inference.py
"""

from __future__ import annotations

import json
import os
import re
import signal
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

from openai import OpenAI

from dataclean_env import DataCleanEnv, DataCleanAction
from dataclean_env.models import DataCleanObservation

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Read environment variables with defaults where required
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")

# Fallback: direct HTTP connection to environment server
ENV_BASE_URL: str = os.getenv("ENV_BASE_URL", "http://localhost:8000")

TASKS: List[str] = ["easy_contacts", "medium_employees", "hard_patients"]
BENCHMARK: str = "dataclean_env"
SEED: int = 42
TEMPERATURE: float = 0.0
MAX_TOKENS: int = 1024
GLOBAL_TIMEOUT_SECONDS: int = 1100  # 18.3 min safety margin
SUCCESS_SCORE_THRESHOLD: float = 0.1

# ---------------------------------------------------------------------------
# Timeout handler
# ---------------------------------------------------------------------------

class GlobalTimeoutError(Exception):
    pass


def _timeout_handler(signum: int, frame: Any) -> None:
    raise GlobalTimeoutError("Global timeout reached")


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = """You are a data-cleaning agent. Your job is to fix quality issues in a tabular dataset.

## Available Actions

Respond with ONLY a JSON object (no markdown, no explanation) containing:
- "action_type": one of the types below
- "params": a dict of parameters for that action

### Action Types

1. **fix_value** - Fix an incorrect value in a cell.
   params: {"row_id": <int>, "column": "<col_name>", "new_value": "<corrected_value>"}

2. **delete_row** - Delete a row (e.g. junk/duplicate that cannot be merged).
   params: {"row_id": <int>}

3. **fill_missing** - Fill a missing (null) value.
   params: {"row_id": <int>, "column": "<col_name>", "value": "<fill_value>"}

4. **standardize_format** - Standardize the format of ALL values in a column (column-level, not row-level).
   params: {"column": "<col_name>", "format_type": "<format>"}
   format_type options: "date:YYYY-MM-DD", "phone:US", "phone:E164", "name:title_case",
                        "email:lowercase", "zip:5digit", "currency:float", "state:abbreviation"

5. **merge_duplicates** - Merge two duplicate rows (keeps the first, deletes the second).
   params: {"row_id1": <int>, "row_id2": <int>, "strategy": "<merge_strategy>"}
   strategy options: "keep_first", "keep_second", "merge_prefer_nonnull", "merge_prefer_row1", "merge_prefer_row2"

6. **flag_anomaly** - Flag a suspicious value for review.
   params: {"row_id": <int>, "column": "<col_name>", "reason": "<why>"}

7. **split_column** - Split a column into multiple columns.
   params: {"column": "<col_name>", "delimiter": "<delim>", "new_names": ["<name1>", "<name2>"]}

8. **rename_column** - Rename a column.
   params: {"old_name": "<old>", "new_name": "<new>"}

9. **cast_type** - Cast a column to a different type.
   params: {"column": "<col_name>", "target_type": "<type>"}
   target_type options: "int", "float", "str", "bool", "date"

10. **escalate_to_human** - Escalate an ambiguous cell to human review when you are uncertain.
   params: {"row_id": <int>, "column": "<col_name>", "confidence": <float 0-1>, "reason": "<why>"}

11. **mark_complete** - Signal that you believe the dataset is clean.
   params: {}

## Important Rules

- Always reference rows by their **row_id** (the first column shown), NOT by row index.
- Examine the quality issues carefully and fix the most impactful ones first.
- When all issues are resolved (or you cannot fix more), use mark_complete.
- Respond with ONLY the JSON object. No extra text."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_user_prompt(obs: DataCleanObservation) -> str:
    """Build the per-step user prompt from the observation."""
    parts: List[str] = []

    # Step info and budget
    parts.append(
        f"## Step {obs.step_number} / {obs.max_steps} "
        f"({obs.steps_remaining} remaining)"
    )
    if hasattr(obs, 'budget_remaining') and obs.budget_remaining is not None:
        parts.append(
            f"- Budget: {obs.budget_remaining:.1f} / "
            f"{obs.budget_remaining + (obs.budget_spent or 0):.1f} remaining"
        )

    # Data summary
    ds = obs.data_summary
    parts.append(
        f"\n## Data Summary\n"
        f"- Rows: {ds.row_count}, Columns: {ds.column_count}\n"
        f"- Total cells: {ds.total_cells}, Null cells: {ds.null_count}\n"
        f"- Quality issues: {ds.issue_count}\n"
        f"- Columns: {', '.join(ds.columns)}"
    )

    # Quality issues (grouped, max 15)
    if obs.issue_groups:
        parts.append("\n## Quality Issues (grouped)")
        shown = 0
        for group in obs.issue_groups:
            parts.append(f"\n### {group.issue_type} ({group.count} issues)")
            for ex in group.examples:
                if shown >= 15:
                    break
                parts.append(
                    f"  - Row {ex.row_id}, col '{ex.column}': "
                    f"{ex.description}"
                    + (f" -> suggestion: {ex.suggestion}" if ex.suggestion else "")
                )
                shown += 1
            if shown >= 15:
                remaining = obs.issues_remaining - shown
                if remaining > 0:
                    parts.append(f"  ... and {remaining} more issues not shown")
                break

    # Last action result
    if obs.last_action_result is not None:
        ar = obs.last_action_result
        parts.append(
            f"\n## Last Action Result\n"
            f"- Action: {ar.action}, Status: {ar.status}\n"
            f"- Message: {ar.message}\n"
            f"- Cells modified: {ar.cells_modified}"
        )

    # Current data table
    if obs.rows:
        parts.append("\n## Current Data (row_id is first column)")
        header_cols = ["row_id"] + obs.columns
        parts.append("| " + " | ".join(str(c) for c in header_cols) + " |")
        parts.append("| " + " | ".join("---" for _ in header_cols) + " |")
        for row in obs.rows:
            parts.append("| " + " | ".join(str(v) for v in row) + " |")

    return "\n".join(parts)


def _normalize_params(action_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize param aliases to canonical names expected by environment.py.

    Handles common LLM mistakes like using 'value' instead of 'new_value'
    for fix_value, or 'row_id_1'/'row_id_2' instead of 'row_id1'/'row_id2'.
    """
    p = dict(params)

    # Universal aliases — LLMs commonly use these instead of canonical names
    if "row" in p and "row_id" not in p:
        p["row_id"] = p.pop("row")
    if "col" in p and "column" not in p:
        p["column"] = p.pop("col")

    if action_type == "fix_value":
        if "value" in p and "new_value" not in p:
            p["new_value"] = p.pop("value")

    elif action_type == "fill_missing":
        # fill_missing uses "value" canonically, but some LLMs send "fill_value"
        if "fill_value" in p and "value" not in p:
            p["value"] = p.pop("fill_value")

    elif action_type == "merge_duplicates":
        if "row_id_1" in p and "row_id1" not in p:
            p["row_id1"] = p.pop("row_id_1")
        if "row_id_2" in p and "row_id2" not in p:
            p["row_id2"] = p.pop("row_id_2")
        if "row1" in p and "row_id1" not in p:
            p["row_id1"] = p.pop("row1")
        if "row2" in p and "row_id2" not in p:
            p["row_id2"] = p.pop("row2")

    return p


def _parse_action(response_text: str) -> DataCleanAction:
    """Parse the LLM response into a DataCleanAction.

    Tries to extract a JSON object with an "action_type" key.
    Falls back to mark_complete if parsing fails.
    """
    text = response_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    # Try direct JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "action_type" in data:
            action_type = data["action_type"]
            params = _normalize_params(action_type, data.get("params", {}))
            return DataCleanAction(
                action_type=action_type,
                params=params,
            )
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object in the text
    match = re.search(r"\{[^{}]*\"action_type\"[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            action_type = data["action_type"]
            params = _normalize_params(action_type, data.get("params", {}))
            return DataCleanAction(
                action_type=action_type,
                params=params,
            )
        except (json.JSONDecodeError, KeyError):
            pass

    # Try nested braces (for params containing dicts)
    match = re.search(r"\{.*\"action_type\".*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict) and "action_type" in data:
                action_type = data["action_type"]
                params = _normalize_params(action_type, data.get("params", {}))
                return DataCleanAction(
                    action_type=action_type,
                    params=params,
                )
        except (json.JSONDecodeError, KeyError):
            pass

    # Fallback
    print(f"  [WARN] Could not parse LLM response, falling back to mark_complete", file=sys.stderr)
    print(f"  [WARN] Raw response: {text[:200]}", file=sys.stderr)
    return DataCleanAction(action_type="mark_complete", params={})


def _call_llm(
    client: OpenAI,
    messages: List[Dict[str, str]],
    retry: bool = True,
) -> str:
    """Call the LLM and return the assistant message content.

    Retries once on failure, then returns a fallback mark_complete JSON.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            seed=SEED,
        )
        content = response.choices[0].message.content
        return content if content is not None else ""
    except Exception as e:
        print(f"  [ERROR] LLM call failed: {e}", file=sys.stderr)
        if retry:
            print("  [INFO] Retrying once...", file=sys.stderr)
            time.sleep(1)
            return _call_llm(client, messages, retry=False)
        print("  [WARN] Retry failed, using fallback action", file=sys.stderr)
        return '{"action_type": "mark_complete", "params": {}}'


# ---------------------------------------------------------------------------
# Stdout logging (mandatory format)
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------


def run_task(
    client: OpenAI,
    env_base_url: str,
    task_id: str,
) -> float:
    """Run a single data-cleaning task and return the final score."""
    print(f"\n  Task: {task_id}", file=sys.stderr)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        with DataCleanEnv(base_url=env_base_url).sync() as env:
            result = env.reset(seed=SEED, task_id=task_id)
            obs: DataCleanObservation = result.observation

            print(f"  Initial issues: {obs.data_summary.issue_count}", file=sys.stderr)
            print(f"  Max steps: {obs.max_steps}", file=sys.stderr)

            messages: List[Dict[str, str]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
            ]

            step = 0
            done = False
            while not done:
                step += 1
                user_msg = _build_user_prompt(obs)
                messages.append({"role": "user", "content": user_msg})

                # Keep conversation manageable: only system + last 6 exchanges
                if len(messages) > 13:
                    messages = [messages[0]] + messages[-12:]

                llm_response = _call_llm(client, messages)
                messages.append({"role": "assistant", "content": llm_response})

                action = _parse_action(llm_response)
                action_str = f"{action.action_type}({json.dumps(action.params) if action.params else ''})"

                error: Optional[str] = None
                try:
                    result = env.step(action)
                    obs = result.observation
                    done = result.done
                    reward = result.reward if result.reward is not None else 0.0
                except Exception as e:
                    print(f"  [ERROR] Environment step failed: {e}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                    reward = 0.0
                    error = str(e)
                    done = True

                rewards.append(float(reward))
                steps_taken = step

                log_step(step=step, action=action_str, reward=float(reward), done=done, error=error)

            # Extract final score from last reward
            score = float(result.reward) if result is not None and result.reward is not None else 0.0
            score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
            success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"  [ERROR] Task failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

    log_end(success=success, steps=steps_taken, rewards=rewards)

    print(f"  Final score: {score:.4f}", file=sys.stderr)
    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def async_main() -> int:
    """Run all tasks and emit [START]/[STEP]/[END] to stdout."""
    if HF_TOKEN is None:
        raise ValueError("HF_TOKEN environment variable is required")

    # Set global timeout (Unix only; no-op on Windows)
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(GLOBAL_TIMEOUT_SECONDS)

    # Initialize OpenAI client per hackathon spec
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    print(f"LLM endpoint: {API_BASE_URL}", file=sys.stderr)
    print(f"Model: {MODEL_NAME}", file=sys.stderr)
    print(f"Environment: {ENV_BASE_URL}", file=sys.stderr)
    print(f"Tasks: {TASKS}", file=sys.stderr)

    scores: Dict[str, float] = {}

    for task_id in TASKS:
        try:
            score = run_task(client, ENV_BASE_URL, task_id)
            scores[task_id] = score
        except GlobalTimeoutError:
            print(f"\n[TIMEOUT] Global timeout reached during task '{task_id}'", file=sys.stderr)
            scores[task_id] = 0.0
            break
        except Exception as e:
            print(f"\n[ERROR] Task '{task_id}' failed: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            scores[task_id] = 0.0

    # Cancel alarm if still active
    if hasattr(signal, "SIGALRM"):
        signal.alarm(0)

    # Summary to stderr
    avg_score = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"\n  RESULTS", file=sys.stderr)
    for task_id, score in scores.items():
        print(f"  {task_id}: {score:.4f}", file=sys.stderr)
    print(f"  Average: {avg_score:.4f}", file=sys.stderr)

    return 0


def main() -> int:
    """Sync entry point — delegates to async_main."""
    import asyncio
    return asyncio.run(async_main())


if __name__ == "__main__":
    sys.exit(main())
