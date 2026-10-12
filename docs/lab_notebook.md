# Lab notebook

**Notation.** A is SmolVLA with `n_action_steps = 10`, A′ is SmolVLA with the checkpoint
default of 50, B is π0.5 with `n_action_steps = 10` and `compile_model = false`. A and A′ were
exchanged on 2026-10-12 and earlier entries were relabelled. Until that date π0.5 was run
with its default of 50 actions per chunk; those entries say so.

Every number below comes from a file in `results/tables/`, named next to it.

## 2026-10-08 — Step 2: per-episode harness and cost estimate

### What `lerobot-eval` gives (LeRobot 0.6.1)

- `eval_info.json` has per-episode `successes`, `sum_rewards` and `max_rewards` per task. It
  has **no episode length**, and the per-episode seeds are computed but dropped before writing.
- **Initial state:** `LiberoEnv` keeps a counter. Episode *i* of a task starts from LIBERO
  initial state *i* (modulo the number of states, silently) and the counter advances on every
  reset. It is not an input.
- **Seeding:** torch is seeded once per process (`--seed`, default 1000). The flow-matching
  noise of an episode therefore depends on every episode run before it in that process.
- **Step limit** per suite, from `lerobot.envs.libero.TASK_SUITE_MAX_STEPS`: 280 for
  `libero_spatial` and `libero_object`, 300 for `libero_goal`, 520 for `libero_10`.

So a wrapper is needed, as the outline's risk table expected.

### What was built

- `src/evalstats/harness/`: `schema.py` (columns, episode key), `store.py` (parquet shards),
  `grid.py` (YAML configs), `runner.py` (loads a policy, runs episodes through LeRobot's own
  `rollout()` with a single env).
- `scripts/run_grid.py` and `scripts/run_grid.sh`; configs `configs/policies.yaml`,
  `configs/grid_throughput.yaml`, `configs/cost_scenarios.yaml`.
- The initial state is set explicitly before each rollout, and the harness checks afterwards
  that the env consumed exactly that one state.
- Each episode is seeded on its own (torch, numpy, simulator) with `rng_seed`, a hash of
  (suite, task, initial state, seed, perturbation). It does not depend on the policy, so the
  paired design holds, nor on run order, so a resumed run can reproduce a fresh one.
- Schema: the outline's columns plus `rng_seed`, `repo_commit` and `run_id`. LeRobot is
  installed from PyPI, so `lerobot_commit` holds the release version (`0.6.1`).
  `perturbation_level` is part of the key.
- One parquet file per (policy variant, suite, task), rewritten atomically after every
  episode. Crashed episodes go to `failures.jsonl`, never to the rows; one line per process
  goes to `runs.jsonl` (load time, peak GPU memory).
- `config_hash` covers the policy entry and the LeRobot version. The harness refuses to add
  rows for a (policy, variant) already stored under another hash.

### Runs

Grid: `libero_spatial`, 10 tasks × initial states 0 and 1 × seed 0 = 20 episodes per policy.

**Resume test.** The SmolVLA run was launched in tmux and killed with `kill -9` after 7 rows.
The same command then reported "7 already done, 13 to run" and finished with 20 rows, no
duplicated key, no leftover temporary file.

**Reproducibility** (`results/tables/reproducibility.csv`): all 20 keys run a second time into
`results/raw/throughput_rerun/`.

| Policy | Episodes | Identical (success and length) | Same success | Successes, run 1 / run 2 |
|---|---|---|---|---|
| SmolVLA, default | 20 | 20 | 20 | 13 / 13 |
| π0.5, default (torch.compile on) | 20 | 10 | 17 | 16 / 19 |
| π0.5, `compile_model=false` | 20 | 20 | 20 | 18 / 18 |

**Agreement with stock `lerobot-eval`** (`results/tables/stock_comparison.csv`): 2 episodes per
task, hence the same initial states, with LeRobot's own seeds. Successes out of 20, harness vs
stock: SmolVLA 13 vs 14; π0.5 compiled 16 vs 17; π0.5 no-compile 18 vs 17 (stock ran compiled).
No sign that the wrapper changes behaviour.

### Throughput (`results/tables/throughput.csv`)

| | SmolVLA | π0.5 no-compile | π0.5 compiled |
|---|---|---|---|
| Episodes per GPU-hour | 629 | 739 | 692 |
| Mean seconds per episode | 5.6 | 4.8 | 5.0 |
| Mean episode length (steps) | 174 | 127 | 141 |
| Seconds per step | 0.025 | 0.026 | 0.025 |
| Fixed seconds per episode (reset) | 1.3 | 1.5 | 1.6 |
| Model load (s) | 8 | 76 | 79 |
| First-episode warm-up (s) | 1 | 1 | 39 |
| Peak GPU memory (GiB) | 0.9 | 8.9 | 8.8 |

Episode time is close to linear in episode length, so a failed episode (280 steps) costs about
twice a successful one. Both policies cost the same per step: with 50 actions executed per
predicted chunk, simulation and rendering dominate, not inference.

### Cost table (`results/tables/cost_projection.csv`)

Revised on 2026-10-11 to the grid of `docs/plan.md` (see the entry of that date). GPU-hours per
policy on the A5000. "Observed" assumes episodes as long as in the timing run; "worst case"
assumes every episode runs to the step limit.

| Run | Episodes per policy | SmolVLA, 50 actions per chunk (A′) observed / worst | π0.5, 50 actions per chunk observed / worst |
|---|---|---|---|
| Step 3 sanity check (10 tasks × 10 initial states) | 100 | 0.16 / 0.24 | — |
| Main grid (10 tasks × 50 initial states × 4 seeds) | 2,000 | 3.2 / 4.7 | 2.7 / 4.9 |

Not covered by these timings: policy A (`n_action_steps = 10` means five times as many
inference calls per episode, so it will be slower than A′). It is timed in Step 3.

### What surprised us

1. **The study is cheap.** A policy costs a few GPU-hours per 2,000 episodes, so the grid
   can use all 50 initial states and several seeds.
2. **torch.compile makes π0.5 irreproducible.** The checkpoint config sets `compile_model:
   true` (`max-autotune`). With it, a fixed (task, initial state, seed) gave a different
   outcome on 3 of 20 reruns. With it off, reruns are identical and speed is unchanged.
   Decision (Nicholas, 2026-10-08): π0.5 runs with `compile_model=false`, variant `nocompile`.
   Worth a line in the report: a fixed seed does not pin a compiled policy.
3. **Success looks well below published numbers** in these 20 episodes per policy: SmolVLA
   13/20, π0.5 18/20 on `libero_spatial`, and stock `lerobot-eval` gives the same picture, so
   it is not the harness. Twenty episodes prove nothing, but this is the first thing the
   Step 3 sanity check must look at (the third camera SmolVLA expects, the 280-step limit,
   control mode).
4. SmolVLA is deterministic given (task, initial state, `rng_seed`) on this machine, which
   Step 6 needs to know: its between-seed variance comes only from changing the seed.
5. `lerobot.envs.close_envs` does not accept a `SyncVectorEnv`; the harness calls
   `vec_env.close()`.

### Housekeeping

- `results/raw/` is 772 KB for now (`throughput/`, `throughput_rerun/`,
  `stock_lerobot_eval/`); it is gitignored and only on `/Data`.
- Nothing is committed yet.

## 2026-10-11 — Scope narrowed, Steps 1 and 2 checked against the new plan

`docs/plan.md` was written and the outline revised: two research questions, one suite
(`libero_spatial`), 2,000 episodes per policy, no perturbations, no sequential testing.

- `configs/cost_scenarios.yaml` now holds the plan's sanity check and main grid, and
  `results/tables/cost_projection.csv` was regenerated from the same Step 2 timings. The
  earlier rows (1,000-episode grids, `libero_10`, perturbation grids) are gone.
- The harness keeps the `perturbation` and `perturbation_level` columns, always `none` and 0.
- Still open from Step 1: how the third camera that `lerobot/smolvla_libero` expects was
  handled in training. It goes to the Step 3 sanity check.
- Reproducibility for a fixed key was checked on 20 episodes for A′ and for π0.5. It has not been
  checked for A, and the plan relies on it to re-run discordant episodes with video.

## 2026-10-12 — Sequential testing back in scope (RQ3); Steps 1 and 2 re-checked

`docs/plan.md` gained Section 5 (RQ3: STEP unpaired, a sequential McNemar test paired, budget
of 500 rollouts per policy). This replaces the "no sequential testing" of the previous entry.

- No effect on Steps 1 and 2. RQ3 resamples evaluations from the same 2,000-episode grid, so it
  needs no extra rollouts, no new column and no change to the cost table.
- Re-checked without the GPU: the lock file is up to date, the unit tests pass, the four
  tables in `results/tables/` are rebuilt byte for byte from `results/raw/`, both raw
  directories hold 60 rows with no duplicated key, and the stored `config_hash` of A′ and π0.5
  match the current `configs/policies.yaml`.
- New rule (`CLAUDE.md`): nothing runs on the GPU between 08:00 and 18:00 Paris time.

## 2026-10-12 — Step 3 prepared (no GPU run yet)

Written during the day, when the GPU is off limits. The rollouts are in `scripts/step3.sh`.

### Configs

- `configs/policies.yaml`: A added as `smolvla_n10`, variant `n_action_steps_10`. The field
  name was checked against the installed SmolVLA config (chunk size stays 50).
- `configs/grid_main.yaml`: 10 tasks × 50 initial states × seeds 0 to 3 for the three policies. A unit
  test checks that the three policies get the same 2,000 scenarios.
- `configs/grid_sanity.yaml`: A′ on initial states 0 to 9, seed 0. It writes to
  `results/raw/main/`, so the main grid will skip these 100 episodes.
- `configs/grid_throughput.yaml` and `configs/cost_scenarios.yaml` now include A.

### The third camera (open since Step 1): not a problem

From the checkpoint's own `train_config.json` and `policy_preprocessor.json`:

- It was trained on `lerobot/libero`, which has two cameras, with the rename
  `image → camera1`, `image2 → camera2`. The harness applies the same mapping.
- `camera3`, and the state of size 6 in `config.json`, are inherited from `smolvla_base`. The
  stored normalization statistics have a state of size 8, which is what LeRobot's LIBERO
  processor produces. With `empty_cameras = 0` the missing third camera is skipped, as it was
  in training.
- Training used relative control, LeRobot's default, and 360 × 360 env images, also the default.

### What "published" means for these checkpoints (`configs/published.yaml`)

- **SmolVLA.** The model card of `lerobot/smolvla_libero` reports no success rate. The SmolVLA paper
  reports 90% on `libero_spatial` for the 0.45B model, with 10 trials per task, but for another
  training run (100,000 steps at batch size 64; this checkpoint: 25,000 steps at batch size 32)
  and with a new chunk predicted after every executed action, where the checkpoint default executes 50.
- **π0.5.** The LeRobot docs report 97.0% on `libero_spatial` for this checkpoint, with
  `--policy.n_action_steps=10`. The checkpoint default is 50.

So the published numbers were all obtained with frequent replanning, while both checkpoints
execute whole 50-action chunks by default. This is why A and A′ were exchanged and B was moved
to 10 actions per chunk (next entry).

## 2026-10-12 (evening) — Step 3: notation exchanged, B at 10 actions per chunk, sanity check

### Decisions (Nicholas)

- **A and A′ exchanged.** A is now SmolVLA with `n_action_steps = 10`, A′ the checkpoint
  default of 50. Entry names and variant names in `configs/policies.yaml` are unchanged
  (`smolvla_n10` is A, `smolvla` is A′), so the stored rows keep their keys.
- **B uses the published setting**, `n_action_steps = 10`, and keeps `compile_model = false`.
  Its variant is `n_action_steps_10_nocompile`. The 20 timing episodes of π0.5 at 50 actions
  per chunk stay in `results/raw/throughput/` but are no longer a policy of the study.
- The success counts of the timing runs may be looked at.

### Runs (`scripts/step3.sh`, two passes: 18:56 and 19:42)

- Timing grid, 20 episodes each, for A and for B, each run twice.
- Sanity grid, 100 episodes each (10 tasks × initial states 0 to 9 × seed 0), for A, A′ and B,
  stored in `results/raw/main/`. No episode crashed.

### Sanity check (`results/tables/sanity_check.csv`)

| Policy | Successes | Rate, 95% Clopper-Pearson | Published | Same checkpoint and protocol? |
|---|---|---|---|---|
| B (π0.5, 10 actions) | 98 / 100 | 98% [93, 100] | 97.0% | yes, except `compile_model` |
| A (SmolVLA, 10 actions) | 79 / 100 | 79% [70, 87] | 90% | no: paper's own training run, replanning every action |
| A′ (SmolVLA, 50 actions) | 59 / 100 | 59% [49, 69] | none | — |

- **B reproduces its published number.** Same harness, same env settings, same initial states:
  this is the evidence that the setup (control mode, images, step limit, initial states) is
  right.
- **A is below the paper's 90%**, and the paper's number is not about this checkpoint
  (25,000 training steps against 100,000; sources in `configs/published.yaml`). With B
  validating the setup, the gap is attributed to the checkpoint, not to the evaluation.
- **Task 5 fails every time for SmolVLA**: 0/10 for A and 0/10 for A′, against 9/10 for B. The
  other nine tasks give A 79/90. Worth a video in Step 7.

### Reproducibility (`results/tables/reproducibility.csv`)

A and B, rerun on the 20 timing keys: 20/20 identical for each. All three policies of the study
are now checked, which the plan needs to re-run discordant episodes with video.

### Throughput and cost (`results/tables/throughput.csv`, `results/tables/cost_projection.csv`)

| | A | A′ | B |
|---|---|---|---|
| Episodes per GPU-hour | 438 | 629 | 498 |
| Seconds per step | 0.042 | 0.025 | 0.050 |
| Main grid, 2,000 episodes: GPU-hours observed / worst case | 4.6 / 7.2 | 3.2 / 4.7 | 4.0 / 8.7 |

Executing 10 actions per chunk instead of 50 costs about 0.017 s more per step for SmolVLA and
0.025 s for π0.5. The whole grid is about 12 GPU-hours at the observed episode lengths and 21
if every episode ran to the step limit. 300 of the 6,000 episodes are already stored.

### What surprised us

1. The number of executed actions per chunk matters a lot for SmolVLA: 79 against 59 successes
   on the same 100 keys. The plan calls A vs A′ a small-gap comparison; on this evidence it is
   not small.
2. π0.5 gains less from it on the 20 timing keys (19 against 18).

### Plan reworded after the sanity check (post hoc, 2026-10-12)

`docs/plan.md` no longer calls A vs A′ a small-gap comparison and A vs B a large-gap one, since
the sanity check shows two large gaps. The comparisons are now described as within one model
(A vs A′) and between two models (A vs B). Policies, grid, tests and comparisons are unchanged.
The expected pattern of RQ3 and its one-sentence answer were rewritten accordingly. This was
decided after seeing 100 episodes per policy, and the plan says so.
