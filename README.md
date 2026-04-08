---
title: DataClean-Env Environment Server
emoji: "\U0001F9F9"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# DataClean-Env: A Trustworthy Data Agent Benchmark for OpenEnv

A deterministic, cost-aware reinforcement learning environment that measures whether AI agents can improve usable data quality without over-editing, collapsing distinct entities, or wasting intervention budget.

## Why This Matters

Data cleaning is where most analytics projects succeed or fail:

- **Bad deduplication** causes duplicate customer outreach, inflated metrics, and CRM pollution.
- **Bad normalization** breaks joins, analytics, and reporting pipelines downstream.
- **Healthcare registry quality** directly affects clinical operations, insurance reimbursement, and patient safety.
- **Data cleaning consumes 60-80% of analyst time** in real companies, yet most agent benchmarks ignore it entirely.

Existing benchmarks test code generation, math, and retrieval. None test the judgment calls that define real data work: when to merge, when to leave ambiguous data alone, and when to escalate to a human.

**This benchmark tests whether agents can improve usable data quality without over-editing or collapsing distinct entities.**

## What Makes This Different

Most data quality benchmarks grade cell-level edits in isolation. DataClean-Env tests the full decision surface:

- **Cost-aware cleaning.** Every action has a budget cost. Being correct is not enough -- agents must be correct with minimal intervention. Cheap spam actions drain budget and reduce the efficiency score.
- **Calibrated abstention.** Agents earn credit for knowing when NOT to act. The `escalate_to_human` action rewards correct uncertainty and penalizes false escalations.
- **Downstream utility probes.** The grader runs aggregate analytics queries (unique counts, distributions, group averages) against the cleaned data. Cell cosmetics that break downstream joins score poorly.
- **False positive traps.** The hard task contains people with identical names who are genuinely different. Agents that merge aggressively are penalized.
- **Gender/name traps.** Valid but unusual data (e.g., Ashley as a male name, Shannon as a male Irish name) must not be "corrected." Agents that impose demographic assumptions are caught.

## Action Space

Actions reference rows by stable `row_id` that persists through delete/merge operations within an episode.

| Action | Cost | Parameters | Description |
|--------|------|-----------|-------------|
| `fix_value` | 1.0 | `row_id`, `column`, `new_value` | Replace a single cell value |
| `delete_row` | 6.0 | `row_id` | Remove a row (expensive -- discourages delete-all strategies) |
| `fill_missing` | 1.0 | `row_id`, `column`, `value` | Fill a null cell |
| `standardize_format` | 2.0 | `column`, `format_type` | Bulk-format a column (date, phone, email, etc.) |
| `merge_duplicates` | 4.0 | `row_id1`, `row_id2`, `strategy` | Merge two duplicate rows |
| `flag_anomaly` | 0.5 | `row_id`, `column`, `reason` | Flag without fixing (partial credit) |
| `split_column` | 3.0 | `column`, `delimiter`, `new_names` | Split a column into multiple columns |
| `rename_column` | 0.5 | `old_name`, `new_name` | Rename a column |
| `cast_type` | 2.0 | `column`, `target_type` | Cast column type |
| `escalate_to_human` | 0.5 | `row_id`, `column`, `confidence`, `reason` | Escalate ambiguous cell to human review |
| `mark_complete` | 0.0 | (none) | Signal episode done |

**Supported format types:** `date:YYYY-MM-DD`, `phone:US`, `phone:E164`, `name:title_case`, `email:lowercase`, `zip:5digit`, `currency:float`, `state:abbreviation`

## Observation Space

Issue-first design. The agent sees quality problems before raw data, reducing context waste on large tables.

| Field | Description |
|-------|-------------|
| `data_summary` | Row count, column count, null count, total issue count |
| `quality_issues` | Individual issues with `row_id`, `column`, `issue_type`, `description`, `suggestion` |
| `issue_groups` | Issues grouped by type (null, format, duplicate, cross_field, anomaly) with counts |
| `rows` / `columns` | Current data table with `_row_id` as first column (truncated for large datasets) |
| `schema_info` | Expected types, format constraints, allowed values |
| `step_number` / `max_steps` / `steps_remaining` | Step context |
| `budget_spent` / `budget_remaining` / `action_costs` | Cost context |
| `last_action_result` / `recent_actions` | Action history with status and results |

## Task Descriptions

| Task | ID | Difficulty | Rows (dirty) | Unique Entities | Max Steps | Budget | Issue Types |
|------|----|-----------|-------------|-----------------|-----------|--------|-------------|
| Customer Contact Cleanup | `easy_contacts` | Easy | 8 | 8 | 30 | 50 | Typos, date formats, null injection, phone format, casing |
| Employee Records Reconciliation | `medium_employees` | Medium | 20 | 17 | 60 | 100 | Near-duplicates (nickname variants), department taxonomy, null injection, salary anomaly |
| Multi-Source Patient Registry | `hard_patients` | Hard | ~40 | 30 | 120 | 200 | Cross-field validation (ZIP/city, insurance prefix), false-positive duplicates, gender/name traps, cascading fixes |

**Difficulty progression:**
- **Easy.** Format-only fixes, null filling, casing. No duplicates. Tests basic data cleaning competence.
- **Medium.** Duplicate detection and merging, department name normalization, contextual missing values. Tests entity resolution judgment.
- **Hard.** ZIP/city mismatches, insurance ID prefix validation, impossible dates, 2 false-positive duplicate pairs (different people with the same name), 3 gender/name traps (valid data that looks wrong), and 1 ambiguous apostrophe case. Tests restraint, calibration, and cross-field reasoning.

## Reward Design

**During the episode:** delta rewards after each action.

```
reward = current_score - previous_score - 0.005
```

The `-0.005` step cost penalizes unnecessary actions.

**At episode end:** weighted composite score in `[0.0, 1.0]`.

| Component | Weight | What It Measures |
|-----------|--------|-----------------|
| Accuracy | 40% | Fraction of dirty cells fixed correctly |
| Completeness | 20% | Improvement in non-null cell correctness over dirty baseline |
| Format Consistency | 10% | Improvement in format compliance over dirty baseline |
| Row Correctness | 10% | Penalty for wrong row count (missing or extra rows) |
| Efficiency | 10% | `1.0 - (budget_spent / budget_allocated)` -- gated on accuracy >= 10% |
| Utility | 10% | Fraction of downstream analytics probes that pass -- gated on accuracy >= 10% |

**Penalties** (deducted from composite, capped at 0.50 total):

| Trigger | Penalty |
|---------|---------|
| Deleting a valid row (exists in ground truth) | -0.10 |
| Introducing an error (changing correct value to wrong) | -0.05 |
| Wrong fix on an ambiguous cell | -0.08 |
| Merging distinct entities (different `_entity_id`) | -0.10 |
| Escalating a clearly fixable cell | -0.02 |

**Bonuses** (added to composite, capped at 0.30 total):

| Trigger | Bonus |
|---------|-------|
| Fully cleaning all issues in a column | +0.10 |
| Correctly flagging a dirty cell | +0.02 |
| Correctly escalating a genuinely ambiguous cell | +0.03 |

## Downstream Utility Probes

Beyond cell-level correctness, the grader runs aggregate analytics queries against the cleaned data. If your agent produces cosmetically clean cells that break downstream analytics, the utility score reflects it.

| Task | Probe | Type | What It Checks |
|------|-------|------|---------------|
| Easy | `unique_email_count` | `unique_count` | Unique non-null email addresses after cleaning |
| Easy | `state_distribution` | `distribution` | Contact count per state |
| Medium | `unique_employee_count` | `unique_count` | Unique employees after deduplication |
| Medium | `department_salary_avg` | `avg_by_group` | Average salary per department |
| Medium | `engineering_headcount` | `count_where` | Number of Engineering employees |
| Hard | `unique_patient_count` | `unique_count` | Unique patients after deduplication |
| Hard | `insurance_provider_distribution` | `distribution` | Patients per insurance provider |
| Hard | `patients_per_city` | `distribution` | Patients per city |
| Hard | `avg_age_by_gender` | `avg_by_group` | Average age (2024 - birth year) by gender |

The utility score is the fraction of probes that pass. Per-probe results (expected vs actual, pass/fail) are included in the episode-end observation metadata.

## Anti-Exploit Properties

| Exploit Strategy | Why It Fails |
|-----------------|-------------|
| **No-op** (do nothing, mark complete) | Score is normalized against the dirty baseline, so doing nothing yields ~0.0. Delta rewards also penalize inaction. |
| **Delete-all** (remove every row) | `row_correctness` drops to 0.0. Each valid row deletion costs -0.10 penalty. Budget drains at 6.0 per delete. |
| **Spam flagging** (flag everything as anomaly) | Bonus capped at 0.30. Incorrect flags provide no credit. Budget drains at 0.5 per flag. |
| **Over-merging** (merge aggressively) | Merging distinct entities (different `_entity_id`) costs -0.10 per merge. Row count penalty compounds. |
| **Cost-unaware spam** | Every action has a budget cost. Cheap actions (0.5 each) still accumulate, reducing the efficiency score component (10% weight). |

## Baseline Scores

Completeness and format consistency are scored as **improvement over the dirty baseline**, not absolute values. Efficiency and utility are gated on a minimum accuracy threshold (doing nothing is laziness, not efficiency).

| Task | Random | No-Op | Format-Only | Heuristic | Oracle |
|------|--------|-------|-------------|-----------|--------|
| Easy (`easy_contacts`) | 0.00 | 0.10 | 0.20 | 0.41 | 1.00 |
| Medium (`medium_employees`) | 0.00 | 0.08 | 0.17 | 0.31 | 1.00 |
| Hard (`hard_patients`) | 0.00 | 0.07 | 0.07 | 0.26 | 1.00 |

**Ordering:** `random(0.00) < no-op(0.08) < format-only(0.15) < heuristic(0.33) < oracle(1.00)`. The no-op score (~0.08) comes entirely from row_correctness (not destroying data is worth 10%). The heuristic agent uses schema-driven fixes (format standardization, department name correction, name casing, duplicate merging) without peeking at ground truth.

## Setup & Usage

### Install Dependencies

```bash
pip install openenv-core[core] fastapi pydantic uvicorn
```

### Local Development

```bash
cd dataclean_env
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
docker build -t dataclean-env -f server/Dockerfile .
docker run -p 8000:8000 dataclean-env
```

### HuggingFace Space

This environment is deployed as a Docker-based HF Space. The `app_port: 8000` frontmatter configures the Space to expose the FastAPI server directly.

**Concurrency:** This environment runs single-session (`SUPPORTS_CONCURRENT_SESSIONS=False`) by design. Each server instance handles one episode at a time, ensuring deterministic grading and reproducible scores. Scale horizontally by running multiple containers behind a load balancer.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/metadata` | GET | Environment metadata (tasks, action space, observation space) |
| `/schema` | GET | Full JSON schemas for Action, Observation, and State models |
| `/reset` | POST | Start a new episode (accepts `seed`, `task_id`) |
| `/step` | POST | Execute an action and receive the next observation |
| `/state` | GET | Current internal state (for debugging, not exposed to agent) |
| `/docs` | GET | Interactive Swagger UI |

## Running Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3-8B-Instruct"
export HF_TOKEN="your-token"
export ENV_BASE_URL="http://localhost:8000"
python inference.py
```

The inference script runs all three tasks sequentially and prints a JSON results block.

## RL Training with GRPO

This environment supports reinforcement learning training via [TRL's GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer) using the `environment_factory` pattern. TRL creates one environment instance per rollout and exposes the environment's `reset()` and `step()` methods as tools the model can call during generation.

```bash
# Install training dependencies
pip install "openenv-dataclean-env[train]"

# Train an agent with GRPO (requires GPU)
accelerate launch scripts/train_grpo.py \
    --model Qwen/Qwen3-0.6B \
    --task easy_contacts \
    --num-episodes 200
```

**Reward decomposition for GRPO** (4 signals for richer gradient):

| Signal | Source | What It Rewards |
|--------|--------|----------------|
| `reward_valid_action` | Completion text | Model outputs parseable JSON with valid action_type |
| `reward_episode_score` | `env.reward` | Final composite grading score at episode end |
| `reward_efficiency` | `env._state` | Frugal budget usage (less cost = higher reward) |
| `reward_no_destruction` | `env._state` | Avoiding destructive actions (delete_row, wrong merges) |

Reward functions read accumulated state from the environment instance provided via `environment_factory`. See `scripts/reward_functions.py` and `scripts/train_grpo.py` for implementation.

## Research Uses

- **Agent calibration benchmarking.** Measure whether agents know when they are uncertain. The escalation mechanism provides a direct signal for calibration quality.
- **Cost-aware RL policy evaluation.** The budget system creates a natural trade-off between exploration and exploitation. Policies must learn to allocate limited intervention budget across heterogeneous issue types.
- **Data quality agent training via GRPO/TRL.** The deterministic reward function and structured action space are compatible with preference optimization and reinforcement learning from environment feedback.
- **Measuring over-intervention and false-confidence rates.** The false-positive traps and ambiguous cells provide ground-truth labels for Type I errors in agent decision-making.

## Failure Modes This Benchmark Measures

| Failure Mode | How It Surfaces |
|-------------|----------------|
| **Over-merging** | Agent collapses distinct entities (e.g., two patients named "James Wilson" who are different people). Detected via `_entity_id` mismatch penalty. |
| **Destructive cleanup** | Agent deletes rows that exist in ground truth. Detected via valid-row deletion penalty and row_correctness score drop. |
| **False confidence** | Agent "fixes" ambiguous cells incorrectly (e.g., changing Ashley's gender from M to F). Detected via elevated penalty (-0.08 vs -0.05) on ambiguous cells. |
| **Failure to preserve downstream utility** | Agent produces clean-looking cells that break aggregate queries. Detected via utility probe failures. |
| **Cost-inefficient strategies** | Agent uses expensive actions (delete at 6.0, merge at 4.0) when cheap fixes (1.0) would suffice. Detected via efficiency score degradation. |
