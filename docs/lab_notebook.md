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

## 2026-10-12 (night) — Repo moved to the shared home, three more machines prepared

Goal: run each policy on its own machine. Only the home directory is shared between lab
machines; `/Data` is local to each.

- **Layout.** The repo and the results now live in `~/vla-eval-stats` (4 MB). The environment,
  weights and assets (17 GB) stay on local `/Data/$USER`, with the environment in
  `/Data/$USER/venvs/vla-eval-stats`. `.venv` in the repo and `/Data/$USER/vla-eval-stats` are
  links, so no path stored in the environment or in the LIBERO config had to change.
- **`scripts/setup_machine.sh <host>`** copies the local part to a machine and checks it
  without the GPU (unit tests, dry run of the main grid). Done for bengali, nandou and quetzal:
  idle RTX A5000 machines with the same driver and OS release as verdier.
- **Several machines writing to one results directory.** Three changes in the harness:
  temporary files carry the host and process id and only stale ones are cleaned up; a lock file
  per policy variant stops two processes from running the same policy, and the stored rows are
  re-read once the lock is held; the run and crash logs are one file per machine
  (`runs.<host>.jsonl`, `failures.<host>.jsonl`).
- Not checked: whether another machine reproduces verdier's episodes exactly. The GPU model
  and driver are the same, but no rollout has been run elsewhere yet.
- The previous copy of the repo is kept as `/Data/$USER/vla-eval-stats.old` on verdier.

## 2026-10-13 — Step 4: main grid run on three machines, complete

### Runs

Launched on 2026-10-12 at 23:42, one policy per machine, each in `tmux` with
`scripts/run_grid.sh --grid configs/grid_main.yaml --policies <name>`. All three wrote to the
shared `results/raw/main/`. The 100 sanity episodes per policy run on verdier in Step 3 were
skipped, as planned. Figures below are from `results/raw/main/runs.<host>.jsonl`.

| | A (`smolvla_n10`) | A′ (`smolvla`) | B (`pi05`) |
|---|---|---|---|
| Machine | bengali | nandou | quetzal |
| Finished | 04:30 | 03:20 | 03:14 |
| Episodes run / already stored / crashed | 1,900 / 100 / 0 | 1,900 / 100 / 0 | 1,900 / 100 / 0 |
| Wall-clock hours for the 1,900 | 4.8 | 3.6 | 3.5 |
| Episodes per hour | 395 | 523 | 541 |
| Peak GPU memory (MiB) | 922 | 922 | 9,084 |

Each log in `logs/` ends with `exit code: 0` and holds no traceback. No `failures.<host>.jsonl`
was written, and no lock or temporary file is left in `results/raw/main/`. One pass was enough.

### Completeness (`scripts/check_completeness.py`, `results/tables/main_completeness.csv`)

**6,000 of 6,000 planned episodes are stored, each exactly once**: for every policy, 2,000
planned, 2,000 stored, 0 missing, 0 duplicated, 0 unplanned, 0 crash records, and the same for
each of the 30 shards (200 rows each). There is no `main_missing.csv` because no key is missing.
The script exits with status 1 and lists the missing keys with a reason when a grid is
incomplete (checked on a copy with one shard removed).

### What surprised us

1. A ran at 395 episodes per hour on bengali against 438 in the Step 3 timing run on verdier,
   and A′ at 523 against 629, while B was faster than projected (541 against 498). Cost only.
2. Nothing crashed in 5,700 episodes on three machines sharing one results directory over NFS.

### Open

- **Provenance.** The rows carry `repo_commit` `c95a938-dirty`: the grid ran with the
  multi-machine harness changes of 2026-10-12 (night) uncommitted. They were committed
  unchanged on 2026-10-13 as `1ef016b`, which is the code that produced the grid.
- **Cross-machine reproducibility is still unchecked.** No key was run on two machines. An
  exact check is to rerun a few of verdier's sanity keys on bengali, nandou or quetzal and
  compare success and episode length (GPU, after 18:00).
- `results/raw/` now sits on the shared home, not on `/Data`, so the pitfall of Step 4 about
  copying it off `/Data` no longer applies. It is gitignored and this is its only copy.
- No success rate of the full grid has been analysed yet (Steps 6 and 7).

## 2026-10-13/14 — Step 5: statistics toolkit, validated on simulated data

No real-data analysis was run. Every method was checked on simulated data with known truth only
(docs/plan.md, Section 6).

### What was built (`src/evalstats/stats/`)

| Module | Contents |
|---|---|
| `variance.py` | `episodes_to_array` (rows → task × state × seed array); `components` (unbiased moment estimators of the task, state and seed components of a variance, or of the covariance between two policies); `rate_variance`, `design_effect` |
| `intervals.py` | Clopper-Pearson; bootstrap that resamples states within each task and keeps the seeds of a state together, the same draw for both policies (`bootstrap_rate`, `bootstrap_gap`, `bootstrap_replicates` for any statistic) |
| `exact.py` | McNemar's exact p-value, vectorized; Barnard's rejection region, computed once per n |
| `power.py` | resampled power for both designs, the sample-size formula and its power, `n_at_power` |
| `sequential.py` | the sequential McNemar test of plan Section 5, `rejection_curve` |
| `step.py` | wrapper around the STEP package, both directions at α/2 |
| `simulate.py` | nested Bernoulli model of two policies with chosen components and correlation per level |

- **Barnard.** scipy's `barnard_exact` is far too slow to call on every table of a power loop
  (n up to 5,000). `barnard_region(n)` computes the rejection region directly: tables are
  ordered by the pooled Wald statistic and the critical value is the smallest whose rejection
  probability stays within α over a fine grid of common rates, refined around the peaks. It
  matches scipy on every table for n = 5, 6, 10, 15, 20 and 30, and on the edge of the region
  at n = 60. It takes about 6 s at n = 5,000.
- **STEP** (`sequentialized_barnard_tests` 0.0.9, added to `pyproject.toml`) is run as the
  package's `MirroredStepTest` with `inference_mode="ranking"`: two one-sided tests at α/2.
  The policy for a budget of 500 at α/2 = 0.025 is not shipped; synthesizing it took 226 s of
  CPU. The package caches it among its own files in the venv, so each machine synthesizes it
  once. Two quirks of the package: its constructor drops the policy it has just loaded (the
  wrapper reloads it), and every reset re-seeds its randomization with the same seed (the
  wrapper gives each evaluation its own seed).
- **Sequential McNemar test, clarification of the plan (2026-10-13, after the grid).** Plan
  Section 5 lets the test look only when a discordant pair arrives. With that rule alone,
  "all the risk at the last step" would reproduce McNemar's exact test only when the last pair
  happens to be discordant, which contradicts item 4 of the same section. The test therefore
  also looks once at the last step of the budget, where it may use all the risk left. Validity
  is unchanged: the total risk is still at most α. This was decided while writing the code,
  after the grid was run but before any comparison was computed.

### Validation (`scripts/validate_stats.py`, about 7 minutes of CPU, seeded)

The scenarios use the main grid's layout (10 tasks × 50 states × 4 seeds): four regimes of one
policy run twice on other seeds (balanced, state-heavy, task-heavy, success rate 95%) and two
pairs of policies 10 and 4 points apart. Rerunning the script gives the same tables.

- **Recovery** (`stats_recovery.csv`): 30 components and covariances over 2,000 grids each.
  Every mean estimate is within 1.7 Monte Carlo standard errors of the truth.
- **Coverage** (`coverage.csv`, 2,000 grids, 2,000 bootstrap replicates):
  - The bootstrap interval covers the rate 94.4% to 95.5% of the time in all six scenarios,
    and the gap 95.0% and 94.8%.
  - Clopper-Pearson goes wrong in the direction of the design effect. It covers 79.0% at a
    design effect of 2.6, 83% to 88% at 1.7 to 2.0, and 98.2% at 0.70.
  - The bootstrap intervals of the components (500 grids) cover 90% to 96%. The state
    component is the weakest: 89.8% at a 95% success rate and 92.0% in the state-heavy
    regime. Worth saying in Step 6.
- **Type-I error** (`type1.csv`, 20,000 evaluations each; all 38 rates have a 95% interval
  that reaches 5% or below):
  - Barnard: 4.2% to 5.0% at n = 50, 200 and 1,000. The exact worst-case size of the region is
    4.84% to 5.00% for n from 50 to 5,000.
  - McNemar: 0.4% to 4.6%. It is conservative at small n, most of all at a 95% success rate,
    where few pairs are discordant.
  - Sequential McNemar, budget 500: 3.2% at 2% discordance, 4.5% to 5.0% from 5% to 50%, and
    4.7% to 5.1% on the four one-policy scenarios.
  - STEP, budget 500: 4.8% to 5.3%. The highest, 5.30% [4.99, 5.61] at both rates 0.2, is
    compatible with 5% but is the closest call of the table. STEP's guarantee rests on a grid
    of 129 worst-case rates.
- **Power against the formula** (`power_check.csv`, 5,000 evaluations per point):
  - Unpaired: Barnard is within 2.5 points of the formula everywhere and within 1.1 points at
    the formula's n for 80%.
  - Paired: exact McNemar is up to 9 points below the formula at small n, 2.6 to 3.9 points
    below at the formula's n for 80% (76.1% and 77.4%), and converges above that.

### What surprised us

1. The normal formula overstates the power of exact McNemar by a few points near 80%, so the
   formula's paired n will be a little optimistic in Step 7. The power curves are the
   reference; the formula is the explanation.
2. The paired fixed-n test is the more conservative of the two at small n (discreteness of the
   discordant count), which works against pairing exactly where few pairs are discordant.

### Housekeeping

- `tests/test_stats.py`: 24 tests, about 20 s, including exhaustive checks against scipy and
  statsmodels and a brute-force check that the sequential boundary is exact and cannot be
  lowered. The whole suite is 33 tests.
- The other machines need `uv sync` before running the stats code (new dependency).
