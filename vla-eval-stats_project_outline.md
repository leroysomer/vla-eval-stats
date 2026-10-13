# How Many Rollouts? A Statistically Rigorous Evaluation of VLA Policies on LIBERO

**Working repo name:** `vla-eval-stats`
**Author:** Nicholas Dreesen
**Status:** project outline, to be built step by step with a coding agent

---

## 0. How to use this file with an agent

- Give the agent this whole file at the start of every session, plus the current state of the repo.
- Work **one step at a time**. Each step has a goal, tasks, deliverables and a "done when" check. Do not start the next step until the current one passes its check.
- Rules for the agent (paste these in):
  1. **Never invent numbers.** Every number in a figure, table or README must come from a file in `results/` produced by code in the repo.
  2. **Verify, don't assume.** Hugging Face model IDs, CLI flags and config field names change. Check them against the installed LeRobot version (`--help`, source code, docs) before using them.
  3. **Everything is config-driven and reproducible.** Seeds, model IDs, suites and episode counts go in YAML configs, not hard-coded.
  4. **Runs are resumable.** A job that crashes halfway must be restartable without redoing finished episodes.
  5. **Ask before running anything expensive.** Estimate GPU-hours first (Step 2) and confirm with me.
- Keep a running `docs/lab_notebook.md`: date, what was run, what came out, what surprised us. This becomes the write-up.

---

## 1. Project summary

### The question
VLA papers report success rates on LIBERO, often at 95% or more, and leaderboards rank models that differ by one or two points. **How much of that is noise?** This project measures, on open VLA checkpoints:

1. How uncertain a reported LIBERO success rate really is, and where that uncertainty comes from (tasks, initial states, policy randomness).
2. How many rollouts it takes to reliably tell two policies apart, and how much a **paired** evaluation design (same initial states for both policies) reduces that number.
3. The same question when the evaluator may **stop early**: how soon a **sequential** test decides within a fixed budget, unpaired (STEP) and paired (a sequential McNemar test derived for this project).

The questions are linked: the noise that two policies share (hard tasks, hard initial states) is what RQ1 measures and what pairing removes in RQ2 and RQ3. The exact estimands, tests and figures are fixed in `docs/plan.md`. Perturbations and a leaderboard audit are deliberately left out.

### Why it matters (for the README and interviews)
Robot learning has started treating evaluation as a statistics problem (TRI's best-practices paper, STEP, N-SCORE). This project applies those ideas to **open VLA models on the most-used simulation benchmark**, adds a variance decomposition, a paired-versus-unpaired power analysis and a paired sequential test, and ships a small reusable toolkit.

### Honest positioning
- This is an **evaluation study plus toolkit**, not a new method. Cite prior work clearly (Section 9).
- The original contributions are: empirical numbers on open checkpoints, the variance decomposition, the paired-vs-unpaired power comparison it explains, and a paired sequential test with exact error control.

### What it shows a recruiter
Hands-on VLA models, a robot simulator, long unattended GPU evaluation runs on a remote machine, and the evaluation rigor and uncertainty quantification that set your profile apart.

---

## 2. Tech stack and key resources

| Component | Choice | Notes |
|---|---|---|
| Framework | [LeRobot](https://github.com/huggingface/lerobot) | Policies, LIBERO env wrapper, `lerobot-eval` CLI |
| Benchmark | [LIBERO via LeRobot](https://huggingface.co/docs/lerobot/en/libero) | Suite used: `libero_spatial` |
| Policy A, A′ | [`lerobot/smolvla_libero`](https://huggingface.co/lerobot/smolvla_libero) | 450M parameters, Apache-2.0. A runs it with `n_action_steps = 10`, A′ with the checkpoint default of 50 |
| Policy B | `lerobot/pi05_libero_finetuned` | Larger (about 3B), about 9 GB of GPU memory. Run with `n_action_steps = 10` (the published setting) and `compile_model=false` |
| Stats | `scipy.stats.barnard_exact`, `statsmodels` (`proportion_confint`, `mcnemar`), [`sequentialized_barnard_tests`](https://github.com/TRI-ML/sequentialized_barnard_tests) (STEP) | Plus your own code for variance components, bootstrap, power and the sequential McNemar test. STEP is CC BY-NC 4.0: use it as a dependency, never copy its code into this repo |
| Data | pandas + parquet | One row per episode |
| Plots | matplotlib | Four figures: variance shares and intervals, power curves, rollouts versus gap, sequential stopping curves |
| Environment | [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`) | `source env.sh`, then `uv run ...` |
| Compute | One workstation, `verdier.polytechnique.fr`, reached over SSH: 1× NVIDIA RTX A5000 (24 GB), 16 CPU cores, 62 GB RAM. No Slurm: jobs run directly on the machine inside `tmux` | Headless rendering: `export MUJOCO_GL=egl` (set by `env.sh`) |

**Reference commands from the LeRobot docs** (check against your installed version):

```bash
source env.sh          # caches on /Data, MUJOCO_GL=egl, uv on PATH

uv run lerobot-eval \
  --policy.path="<policy-id>" \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --env.max_parallel_tasks=1
```

---

## 3. Target repo structure

```
vla-eval-stats/
├── README.md                  # headline figure, key findings, how to reproduce
├── pyproject.toml             # dependencies (LeRobot pinned), managed with uv
├── uv.lock                    # exact resolved versions
├── env.sh                     # cache locations on /Data, MUJOCO_GL=egl
├── configs/
│   ├── policies.yaml          # policy IDs + inference overrides
│   └── grid_main.yaml         # suite, tasks, init states, seeds
├── src/evalstats/
│   ├── harness/               # run episodes, log per-episode results, resume logic
│   ├── stats/                 # variance components, intervals, tests, power, sequential
│   └── plots/
├── scripts/                   # thin CLIs: run_grid.py, analyze_*.py, make_figures.py,
│                              # and shell launchers: smoke_test.sh, run_grid.sh (run inside tmux)
├── logs/                      # stdout/stderr of long runs (gitignored)
├── tests/                     # unit tests + coverage / Type-I error simulations
├── results/
│   ├── raw/                   # per-episode parquet files (gitignored if large)
│   └── tables/                # small CSVs used in the README
├── figures/
└── docs/
    ├── plan.md                # estimands and research questions (Step 0)
    ├── lab_notebook.md
    └── report.md              # write-up of about 5 pages (Step 9)
```

---

## 4. The steps

### Step 0: Frame the questions before touching code

**Goal:** decide exactly what is being estimated, so the statistics are right from the start.

**Tasks**
- Read (skim is fine): TRI's *Robot Learning as an Empirical Science* paper, the TRI blog post on statistical thinking, STEP, and the LIBERO-Plus paper (links in Section 9).
- Write `docs/plan.md` with:
  - **Estimands.** For a policy and a suite, define success rate as the average over (a) the fixed 10 tasks and the benchmark's fixed initial states, versus (b) a population of "tasks like these". These lead to different confidence intervals: stratified/fixed-effects for (a), clustered/random-effects for (b). State which one each analysis uses.
  - **The research questions** from Section 1, rewritten in your words.
  - **The success criterion** (LIBERO's built-in task success flag) and the maximum episode length, fixed in advance.
  - **The comparisons planned** (which policy pairs), written down before seeing results, to avoid fishing.

**Deliverables:** `docs/plan.md`.
**Done when:** each research question names its estimand, its test or interval, and the figure that will answer it.

---

### Step 1: Environment setup and smoke test on the workstation

**Goal:** one LIBERO episode runs headless on the verdier GPU and saves a video.

**The machine**
- `verdier.polytechnique.fr`, a single workstation reached over SSH: 1× RTX A5000 (24 GB), 16 CPU cores, 62 GB RAM. It has internet, so weights and assets download directly.
- There is no scheduler. Everything runs as an ordinary process on the one GPU, so runs are sequential and the budget is **wall-clock hours on one GPU**.
- Storage: the home directory is on NFS, shared by every lab machine, with a 30 GB quota that is nearly full. The repo and the results (a few MB) live there, in `~/vla-eval-stats`. The virtual environment, the uv and Hugging Face caches and the LIBERO assets (about 17 GB) live on each machine's local disk under `/Data/$USER`, where `/Data/$USER/vla-eval-stats` is a link to the repo. `scripts/setup_machine.sh <host>` copies the local part to another machine, so that each policy can run on its own machine and write to the same `results/raw/`.

**Tasks**
- Environment (done): uv project with LeRobot pinned in `pyproject.toml` and exact versions in `uv.lock` (LeRobot, MuJoCo, robosuite, torch). `env.sh` sets `HF_HOME`, `UV_CACHE_DIR`, `LIBERO_CONFIG_PATH` and `MUJOCO_GL=egl`.
- Pre-download the policy weights and LIBERO assets (done for both policies). π0.5 also needs the gated `google/paligemma-3b-pt-224` tokenizer: accept its licence on Hugging Face and run `uv run hf auth login`.
- If EGL fails, try `MUJOCO_GL=osmesa` and record the slowdown.
- Write `scripts/smoke_test.sh`: run `lerobot-eval` for the SmolVLA checkpoint on one task of one suite with 2 episodes and save the rollout video. `lerobot/smolvla_libero` names its cameras `camera1`/`camera2`/`camera3`, so the LIBERO env needs `--env.camera_name_mapping='{"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}'`.
- Long runs must survive an SSH disconnect: start them inside `tmux` (`tmux new -s eval`, detach with `Ctrl-b d`, reattach with `tmux attach -t eval`) and write stdout/stderr to `logs/`.
- Before every long run, check that the GPU is free with `nvidia-smi`: the machine sits in a shared computer room and someone else may be using it.

**Deliverables:** `pyproject.toml`, `uv.lock`, `env.sh`, `scripts/smoke_test.sh`, one video in `figures/`.
**Done when:** the smoke test passes when launched from a detached `tmux` session, the video shows the robot attempting the task, and the reported success matches what the video shows.

**Pitfalls:** EGL and driver mismatches; anything written under `~/.cache` eating the home quota (LIBERO hard-codes `~/.cache/libero/assets`, now a symlink to `/Data`); `--env.control_mode` must match how the policy was trained (`relative` is the LeRobot default); the checkpoint expects three cameras and LIBERO provides two, so check how the missing one was handled in training before trusting success rates.

---

### Step 2: Per-episode logging harness and cost estimate

**Goal:** every episode produces one structured record, and you know what the full study will cost in GPU-hours.

**Tasks**
- Find out what `lerobot-eval` writes (for example `eval_info.json`) and whether it includes **per-episode** success, seed and length. If it does not, or if you cannot control which initial state each episode uses, write a thin wrapper around LeRobot's eval loop in `src/evalstats/harness/`.
- Find how the LeRobot LIBERO env picks the **initial state** for each episode (LIBERO ships a fixed set of initial states per task). Make the initial-state index an explicit input. This is essential for the paired design in Step 7.
- Define the per-episode schema, one parquet row per episode:

  | Column | Meaning |
  |---|---|
  | `policy_id`, `policy_variant` | which policy, which inference settings |
  | `suite`, `task_id`, `task_name` | which task |
  | `init_state_id` | which LIBERO initial state |
  | `seed` | seed for policy sampling and the simulator |
  | `perturbation`, `perturbation_level` | `none` for the main grid |
  | `success` | 0 or 1 |
  | `episode_len`, `wall_time_s` | steps and seconds |
  | `lerobot_commit`, `config_hash`, `timestamp` | provenance |

- Make runs **idempotent**: before running an episode, check whether its key `(policy, variant, suite, task, init_state, seed, perturbation)` already exists.
- Time 20 episodes per policy and fill in a cost table: episodes per GPU-hour per policy, and projected GPU-hours for the grids in Steps 3 and 8.

**Deliverables:** harness code, `results/tables/throughput.csv`, a cost table in `docs/lab_notebook.md`.
**Done when:** a 20-episode run produces 20 correct rows, a killed and restarted run does not duplicate rows, and the GPU-hour projection is written down.

---

### Step 3: Policies, grid and sanity check

**Goal:** the three policy entries and the grid of `docs/plan.md` exist as configs, and Policy A behaves as published.

**Policies** (fixed in `docs/plan.md`, Section 1)
- **A:** `lerobot/smolvla_libero` with `n_action_steps = 10`.
- **A′:** the same checkpoint with its default inference settings (`n_action_steps = 50`). A realistic **inference-setting ablation** of A. The values are fixed in advance and are not tuned on results.
- **B:** `lerobot/pi05_libero_finetuned` with `n_action_steps = 10`, the setting of its published evaluation, and `compile_model = false`.

**Grid**
- `libero_spatial`, 10 tasks × 50 initial states × 4 seeds = **2,000 episodes per policy**, 6,000 in total.
- **The same (task, initial state, seed) keys for every policy.** This is the paired design.

**Tasks**
- Add the A entry (`n_action_steps = 10`) to `configs/policies.yaml` and write `configs/grid_main.yaml`.
- Time 20 episodes of A and of B (more inference calls per episode than with 50 actions per chunk) and update the cost table.
- **Sanity check:** run Policy A and Policy B on 10 initial states per task (seed 0), 100 episodes each, and compare with the published success rates (`configs/published.yaml`). The Step 2 timing run gave 13/20 for A′, well below published numbers. If the gap remains, debug the setup (the third camera the checkpoint expects, the 280-step limit, control mode, image preprocessing) before going further. These 100 episodes are part of the grid and are not rerun.

**Deliverables:** `configs/policies.yaml`, `configs/grid_main.yaml`, the sanity-check result and the updated cost table in the notebook.
**Done when:** the sanity check is close to the published numbers (or the difference is understood and documented) and the projected cost of the grid is confirmed with me.

---

### Step 4: Run the main grid

**Goal:** the full per-episode dataset.

**Tasks**
- Split the grid into **shards** (one policy × one task). Each shard writes its own parquet file. `scripts/run_grid.sh` walks through the shards one after another on the single GPU, with one log file per shard in `logs/`.
- A shard that crashes must not stop the queue: log the failure, move on, and let a second pass of the same command pick up what is missing (the harness is idempotent, Step 2).
- Log crashes and timeouts separately from task failures. A crashed episode is missing data, not a task failure.
- No videos are saved during the grid: any episode can be re-run from its key later (Step 7).
- After the run, check completeness: every planned key is present exactly once.

**Deliverables:** `results/raw/main/*.parquet`, `results/tables/main_completeness.csv`.
**Done when:** completeness is 100%, or any missing key is listed with a reason.

**Pitfalls:** do not look at policy comparisons until the run is complete. The machine can be rebooted or used by someone else: rely on resumability, check `nvidia-smi` before launching, and copy `results/raw/` off `/Data` when it finishes.

---

### Step 5: Build and validate the statistics toolkit

**Goal:** a small, tested library in `src/evalstats/stats/` containing only what `docs/plan.md` uses. Every method is checked on simulated data with known truth before it touches real data.

**Methods**
- **Variance components** of the nested design (task, initial state within task, seed within initial state): unbiased moment estimators, for one policy and for the covariance between two policies.
- **Intervals for one success rate:** Clopper-Pearson (`statsmodels.stats.proportion.proportion_confint`, `method="beta"`), and a percentile **bootstrap that resamples initial states within each task**, keeping the seeds of a state together. The same bootstrap gives the interval for a gap between two policies.
- **Tests:** Barnard's exact test (`scipy.stats.barnard_exact`) for the unpaired design, McNemar's exact test (`statsmodels.stats.contingency_tables.mcnemar(exact=True)`) for the paired design.
- **Power by resampling:** draw evaluations of size n from the per-episode pool under each design, run the test, record the rejection rate.
- **Sample-size formula:** n ≈ (z₁₋α/₂ + z₁₋β)² σ² / δ², with the paired and unpaired σ² of `docs/plan.md`.
- **Sequential McNemar test** (`docs/plan.md`, Section 5): the exact stopping boundary on the walk of discordant pairs, with risk spent uniformly over the rollout budget. Written from scratch.
- **STEP wrapper:** add `sequentialized_barnard_tests` to `pyproject.toml`, check its API and Python-version support against the installed package, synthesize the policy for n_max = 500 once and cache it, and expose it in both directions at α/2.

**Simulator for validation:** a nested Bernoulli model with chosen between-task, between-initial-state and between-seed variances, and a chosen correlation between two policies.

**Validation tests in `tests/`**
- **Recovery:** the moment estimators return the chosen components.
- **Coverage:** the bootstrap interval covers the true rate about 95% of the time; the Clopper-Pearson interval departs from 95% in the direction the design effect predicts.
- **Type-I error:** with two identical policies, each test rejects at most 5% of the time.
- **Power:** resampled power agrees with the formula for a known gap.
- **Sequential McNemar:** rejects a true null at most 5% of the time for discordance rates from 2% to 50%; with all the risk placed at the last step it reproduces McNemar's exact test.
- **STEP:** run in both directions at α/2, rejects a true null at most 5% of the time on independent Bernoulli sequences.

**Deliverables:** `src/evalstats/stats/`, `tests/`, `results/tables/coverage.csv`, `results/tables/type1.csv`.
**Done when:** all tests pass and both tables look right.

**Pitfalls:** Barnard's test is slow at large n and is called thousands of times in the power loop. Its outcome depends only on the two success counts, so compute the rejection region once per n and look it up. STEP is not tested by its authors beyond n_max = 500; do not raise the budget. Synthesizing its policy is CPU work, but time it before running it in a loop.

---

### Step 6: Analysis 1: how uncertain is one success rate?

**Goal:** answer RQ1 as specified in `docs/plan.md`, Section 3.

**Tasks**
- For each policy, estimate the three variance components:

  p(1 − p) = σ²_task + σ²_init + σ²_seed

  **between tasks**, **between initial states within a task**, and **between seeds within an initial state**, with bootstrap intervals.
- Compute the **design effect** (true variance of the estimate over the binomial variance) for the grid and for the two protocols in common use, 10 and 50 episodes per task with one seed each.
- For each policy: point estimate, Clopper-Pearson interval and bootstrap interval. Explain the difference in width with the design effect.
- One-sentence answer: "a reported X% on `libero_spatial` at 10 episodes per task is really X ± Y points".

**Figure 1** (two panels): variance shares per policy as stacked bars; success rate per policy with both intervals.
**Table 1:** components, design effect, half-width Y for the two protocols.

**Deliverables:** `results/tables/rq1_*.csv`, `figures/rq1.png`, notebook entry.
**Done when:** the figure and the table are rebuilt by one script, and the one-sentence answer is written.

---

### Step 7: Analysis 2: how many rollouts to tell two policies apart?

**Goal:** answer RQ2 as specified in `docs/plan.md`, Section 4. This is the headline result.

**Tasks**, for A vs A′ and for A vs B
- The observed gap on the full grid with its bootstrap interval.
- The discordance rates (only A succeeds, only the other succeeds) and the covariance between the two policies, split into shared task, shared initial state and shared seed.
- **Power curves** as a function of n, the number of rollouts per policy, for two designs:
  1. **Unpaired** (independent keys for each policy, Barnard).
  2. **Paired** (same keys for both policies, McNemar).
- The **n needed for 80% power** under each design, from the curves and from the formula, and the **percentage saved** by pairing.
- **Null control:** A against itself on the same initial states with different seeds. Both tests must reject at most 5% of the time.
- **Beyond the observed gaps:** a chart of "rollouts needed versus true gap" for gaps of 1, 2, 5 and 10 points, from the formula, holding the discordance rate at the value measured for A vs A′. State the assumption on the chart.
- Re-run 10 **discordant** episodes of A vs A′ with video and describe the failure modes in one paragraph (missed grasp, wrong object, timeout, etc.).

**Figure 2:** power versus n, one panel per comparison, one curve per design, formula overlaid, 80% line marked.
**Figure 3:** rollouts needed versus gap.
**Table 2:** gaps, discordance, covariance split, n at 80% power.

**Deliverables:** `results/tables/rq2_*.csv`, `figures/rq2_*.png`, failure-mode notes.
**Done when:** you can state, with numbers from the data: "Detecting a 2-point gap at 80% power takes about N rollouts per policy unpaired and M paired."

---

### Step 8: Analysis 3: how many rollouts when the evaluator may stop early?

**Goal:** answer RQ3 as specified in `docs/plan.md`, Section 5. No new rollouts: everything is resampled from the main grid.

**Setting:** budget of 500 rollouts per policy, two-sided α = 0.05, risk spent uniformly over the budget, for A vs A′ and for A vs B.

**Tasks**
- Draw 5,000 simulated evaluations per comparison and per design from the grid: independent keys for each policy (unpaired), the same keys for both (paired).
- Run **STEP** on the unpaired sequences and the **sequential McNemar test** on the paired ones. Record the step at which each stops and its decision.
- Compute the probability of having rejected by step n, the **power within the budget**, and the **expected number of rollouts** before stopping. Put them next to the power of Barnard and McNemar at n = 500 from Step 7.
- **Null control:** A against itself, as in Step 7, for both sequential tests.
- State the result whatever it is. Reaching the budget without a decision is not evidence that two policies are equal.

**Figure 4:** probability of having rejected versus n, one panel per comparison, one curve per sequential design, fixed-n power at 500 marked.
**Table 3:** power within budget and expected rollouts for the four designs (unpaired or paired × fixed or sequential).

**Deliverables:** `results/tables/rq3_*.csv`, `figures/rq3.png`, notebook entry.
**Done when:** you can state, with numbers from the data: "With a budget of 500 rollouts per policy, A vs A′ is decided after about N rollouts unpaired and M paired, and A vs B after about N′ and M′."

**Fallback:** if STEP cannot be installed or run at n_max = 500, report the sequential McNemar test against the two fixed-n tests only, and say that the unpaired sequential cell is missing.

---

### Step 9: Package it (README, report, reproducibility)

**Goal:** a repo a screener understands in 30 seconds and an engineer can rerun.

**README structure**
1. **One-line hook** with the headline number from Step 7, for example: "On LIBERO, telling apart two VLA policies 2 points apart takes about N rollouts; pairing initial states cuts that by X%."
2. **Headline figure** (rollouts needed versus gap) and a GIF of a rollout and a failure.
3. **Key findings:** 3 or 4 bullets, each tied to a figure.
4. **Recommendations for practitioners**, drawn from the results.
5. **Reproduce:** environment files, one command per step, expected GPU-hours.
6. **Toolkit usage:** a 10-line example of using `evalstats` on any per-episode CSV.
7. **Limitations:** one suite, simulation only, three policy variants, power figures and stopping times are plug-in estimates, one sequential budget.
8. **References.**

**Also**
- `docs/report.md`: a write-up of about 5 pages built around four figures and three tables (question, setup, methods including the derivation of the sequential McNemar test, results, limitations).
- One script that rebuilds every figure from `results/`.
- Clean commit history, MIT license, CI that runs the unit tests.
- Optionally, contribute back: `lerobot-eval` drops per-episode seeds and episode lengths and does not take the initial state as an input (Step 2). A small PR on one of these would be a strong signal ([issue #4481](https://github.com/huggingface/lerobot/issues/4481) discusses structured evaluation results).

**Deliverables:** README, report, figures, tagged release `v1.0`.
**Done when:** a fresh clone plus the documented commands rebuilds the figures from the saved results.

---

## 5. Scope ladder (if time is short)

| Level | Steps | What you can claim |
|---|---|---|
| **Minimum viable** | 0 to 7 with policies A and A′ | Variance decomposition, paired vs unpaired power within one model |
| **Target** | + Policy B | The same result between two models |
| **Full** | + Step 8 | Sequential testing, unpaired and paired, with a test of your own |
| **Later, as separate work** | perturbations; leaderboard audit | See "Out of scope" in `docs/plan.md` |

---

## 6. Compute and time estimates

Measured in Steps 2 and 3 (`results/tables/throughput.csv`) on `libero_spatial`: about 440 episodes per GPU-hour for A, 630 for A′ and 500 for B.

- Main grid, 2,000 episodes per policy (`results/tables/cost_projection.csv`): about 4.6 GPU-hours for A, 3.2 for A′ and 4.0 for B at the episode lengths observed in the timing runs, 12 in all; 21 if every episode ran to the step limit.
- All compute is one RTX A5000 on verdier, so GPU-hours equal wall-clock hours and the policies run one after the other.
- The statistics (Steps 5 to 8) need no GPU once the parquet files exist. The one unknown is the time to synthesize the STEP policy for n_max = 500.

---

## 7. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Published success not reproduced | Debug the third camera, step limit, control mode and image preprocessing; if it remains, document the gap and continue (the statistics still hold for what you measured) |
| A policy is near 100% (little variance to study) | Report it as is; consider `libero_10` as separate, later work |
| Neither observed gap is small (Step 3 sanity check) | Small gaps are covered by the null control and by the formula-based "rollouts versus gap" chart, which does not depend on the observed gaps |
| STEP cannot be installed, or fails at n_max = 500 | Report the sequential McNemar test against the fixed-n tests only (Step 8 fallback) |
| SSH session drops during a long run | Always launch inside `tmux`; logs in `logs/` |
| Machine rebooted, or GPU taken by another user, mid-run | Resumable harness: rerun the same command; check `nvidia-smi` before launching |
| `/Data` is local to verdier and may be wiped | Code and `results/tables/` in git; copy `results/raw/` elsewhere after each long run |
| Home quota (30 GB) fills up | Keep every cache on `/Data` via `env.sh` |

---

## 8. CV and interview material (fill in at the end)

**CV entry (template; replace the brackets with real results):**

> **Robot Learning Project: Statistically Rigorous Evaluation of VLA Policies** (GitHub)
> - Evaluated open vision-language-action policies (SmolVLA, π0.5) on 6,000 simulated LIBERO manipulation rollouts on a remote GPU workstation; decomposed success-rate variance into task, initial-state and policy-noise components
> - Showed that pairing initial states [cuts the rollouts needed to detect a 2-point gap by X%], as predicted by the variance decomposition
> - Derived a paired sequential test with exact false-positive control and compared it with TRI's STEP: [decides in N rollouts on average against M]; released a tested evaluation toolkit

**Interview story (2 minutes):** the same "diagnose before you build" instinct as NYGC. Robot learning compares policies on noisy, expensive rollouts. You measured where the noise comes from and showed how much of it a paired design removes, using the uncertainty quantification you studied at Polytechnique.

**Questions to be ready for:** why an exact interval at high success rates; fixed versus random effects for tasks; why repeated seeds on one initial state are not independent episodes; what pairing assumes; why peeking breaks fixed-sample tests and how spending the error over time repairs it; why the paired sequential test has no nuisance parameter while STEP does; how this would change on a real robot (initial states cannot be matched exactly, fewer episodes, so sequential testing matters more).

---

## 9. References

- Kress-Gazit et al., *Robot Learning as an Empirical Science: Best Practices for Policy Evaluation* (2024): [arXiv:2409.09491](https://arxiv.org/abs/2409.09491)
- TRI blog, *Statistical Thinking for Robot Policy Evaluation: From Rigorous A/B Testing to Effective Visualization*: [Medium](https://medium.com/toyotaresearch/statistical-thinking-for-robot-policy-evaluation-from-rigorous-a-b-testing-to-effective-0ae886fbd68d)
- Snyder et al., *Is Your Imitation Learning Policy Better than Mine? Policy Comparison with Near-Optimal Stopping* (STEP): [arXiv:2503.10966](https://arxiv.org/abs/2503.10966), [code](https://tri-ml.github.io/step/). Used as the unpaired sequential test in Step 8.
- Lan and DeMets, *Discrete sequential boundaries for clinical trials*, Biometrika 70(3), 1983. The alpha-spending idea behind the sequential McNemar test.
- *Beyond Binary Success: Sample-Efficient and Statistically Rigorous Robot Policy Comparison* (N-SCORE): [arXiv:2603.13616](https://arxiv.org/abs/2603.13616)
- *LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models*: [arXiv:2510.13626](https://arxiv.org/abs/2510.13626). Cited as future work.
- LeRobot LIBERO docs: [huggingface.co/docs/lerobot/en/libero](https://huggingface.co/docs/lerobot/en/libero)
- SmolVLA paper: [arXiv:2506.01844](https://arxiv.org/abs/2506.01844); checkpoint: [lerobot/smolvla_libero](https://huggingface.co/lerobot/smolvla_libero)
