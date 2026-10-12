# Lab notebook

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

| Run | Episodes per policy | SmolVLA (A) observed / worst | π0.5 (B) observed / worst |
|---|---|---|---|
| Step 3 sanity check (10 tasks × 10 initial states) | 100 | 0.16 / 0.24 | — |
| Main grid (10 tasks × 50 initial states × 4 seeds) | 2,000 | 3.2 / 4.7 | 2.7 / 4.9 |

Not covered by these timings: policy A′ (`n_action_steps = 10` means five times as many
inference calls per episode, so it will be slower than A). It is timed in Step 3.

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
- Reproducibility for a fixed key was checked on 20 episodes for A and for B. It has not been
  checked for A′, and the plan relies on it to re-run discordant episodes with video.

## 2026-10-12 — Sequential testing back in scope (RQ3); Steps 1 and 2 re-checked

`docs/plan.md` gained Section 5 (RQ3: STEP unpaired, a sequential McNemar test paired, budget
of 500 rollouts per policy). This replaces the "no sequential testing" of the previous entry.

- No effect on Steps 1 and 2. RQ3 resamples evaluations from the same 2,000-episode grid, so it
  needs no extra rollouts, no new column and no change to the cost table.
- Re-checked without the GPU: the lock file is up to date, the unit tests pass, the four
  tables in `results/tables/` are rebuilt byte for byte from `results/raw/`, both raw
  directories hold 60 rows with no duplicated key, and the stored `config_hash` of A and B
  match the current `configs/policies.yaml`.
- New rule (`CLAUDE.md`): nothing runs on the GPU between 08:00 and 18:00 Paris time.
