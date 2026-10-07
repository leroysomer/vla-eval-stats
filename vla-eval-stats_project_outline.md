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
2. How many rollouts it takes to reliably tell two policies apart, and how much a **paired** evaluation design (same initial states for both policies) and **sequential** testing reduce that number.
3. How success degrades under realistic perturbations (camera, lighting, language, etc.), with uncertainty and multiple-comparison corrections reported properly.

### Why it matters (for the README and interviews)
Robot learning has started treating evaluation as a statistics problem (TRI's best-practices paper, STEP, N-SCORE). This project applies those ideas to **open VLA models on the most-used simulation benchmark**, adds a variance decomposition and a paired-design analysis, and ships a small reusable toolkit.

### Honest positioning
- This is an **evaluation study plus toolkit**, not a new method. Cite prior work clearly (Section 9).
- The original contributions are: empirical numbers on open checkpoints, the variance decomposition, the paired-vs-unpaired power comparison, and the robustness study with proper uncertainty.

### What it shows a recruiter
Hands-on VLA models, a robot simulator, long unattended GPU evaluation runs on a remote machine, and the evaluation rigor and uncertainty quantification that set your profile apart.

---

## 2. Tech stack and key resources

| Component | Choice | Notes |
|---|---|---|
| Framework | [LeRobot](https://github.com/huggingface/lerobot) | Policies, LIBERO env wrapper, `lerobot-eval` CLI |
| Benchmark | [LIBERO via LeRobot](https://huggingface.co/docs/lerobot/en/libero) | Suites: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90` |
| Perturbations | [LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus), [LeRobot integration](https://github.com/huggingface/lerobot/blob/main/docs/source/libero_plus.mdx) | 7 perturbation dimensions; `--env.type=libero_plus` |
| Policy A | [`lerobot/smolvla_libero`](https://huggingface.co/lerobot/smolvla_libero) | 450M parameters, Apache-2.0. Verify suites and expected success before use |
| Policy B | A π0.5 LIBERO fine-tuned checkpoint (`pi05_libero_finetuned` in the LeRobot docs) | Larger (about 3B). Verify the exact Hub ID and GPU memory needs |
| Stats | `scipy.stats` (`barnard_exact`, `fisher_exact`, `beta`), `statsmodels` (`proportion_confint`, `mcnemar`) | Plus your own code for bootstrap and power |
| Data | pandas + parquet | One row per episode |
| Plots | matplotlib | Forest plots, power curves, heatmaps |
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
│   ├── grid_main.yaml         # suites, tasks, init states, seeds
│   └── grid_robustness.yaml   # LIBERO-Plus perturbation grid
├── envs/
│   └── libero_plus/           # separate uv env (Step 8): LIBERO-Plus replaces vanilla LIBERO
├── src/evalstats/
│   ├── harness/               # run episodes, log per-episode results, resume logic
│   ├── stats/                 # intervals, tests, bootstrap, power, sequential
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
    └── report.md              # 3 to 5 page write-up (Step 10)
```

---

## 4. The steps

### Step 0: Frame the questions before touching code

**Goal:** decide exactly what is being estimated, so the statistics are right from the start.

**Tasks**
- Read (skim is fine): TRI's *Robot Learning as an Empirical Science* paper, the TRI blog post on statistical thinking, STEP, and the LIBERO-Plus paper (links in Section 9).
- Write `docs/plan.md` with:
  - **Estimands.** For a policy and a suite, define success rate as the average over (a) the fixed 10 tasks and the benchmark's fixed initial states, versus (b) a population of "tasks like these". These lead to different confidence intervals: stratified/fixed-effects for (a), clustered/random-effects for (b). State which one each analysis uses.
  - **The three research questions** from Section 1, rewritten in your words.
  - **The success criterion** (LIBERO's built-in task success flag) and the maximum episode length, fixed in advance.
  - **The comparisons planned** (which policy pairs, which perturbations), written down before seeing results, to avoid fishing.

**Deliverables:** `docs/plan.md`.
**Done when:** each research question names its estimand, its test or interval, and the figure that will answer it.

---

### Step 1: Environment setup and smoke test on the workstation

**Goal:** one LIBERO episode runs headless on the verdier GPU and saves a video.

**The machine**
- `verdier.polytechnique.fr`, a single workstation reached over SSH: 1× RTX A5000 (24 GB), 16 CPU cores, 62 GB RAM. It has internet, so weights and assets download directly.
- There is no scheduler. Everything runs as an ordinary process on the one GPU, so runs are sequential and the budget is **wall-clock hours on one GPU**.
- Storage: the home directory is on NFS with a 30 GB quota that is nearly full. The repo, the virtual environment, the uv and Hugging Face caches and the LIBERO assets all live on the local disk under `/Data/$USER`. **`/Data` is local to verdier and may not be backed up**: commit code and small result tables to git, and copy `results/raw/` somewhere safe after each long run.

**Tasks**
- Environment (done): uv project with LeRobot pinned in `pyproject.toml` and exact versions in `uv.lock` (LeRobot, MuJoCo, robosuite, torch). `env.sh` sets `HF_HOME`, `UV_CACHE_DIR`, `LIBERO_CONFIG_PATH` and `MUJOCO_GL=egl`.
- Pre-download the policy weights and LIBERO assets (done for both policies). π0.5 also needs the gated `google/paligemma-3b-pt-224` tokenizer: accept its licence on Hugging Face and run `uv run hf auth login`.
- If EGL fails, try `MUJOCO_GL=osmesa` and record the slowdown.
- Write `scripts/smoke_test.sh`: run `lerobot-eval` for Policy A on one task of one suite with 2 episodes and save the rollout video. `lerobot/smolvla_libero` names its cameras `camera1`/`camera2`/`camera3`, so the LIBERO env needs `--env.camera_name_mapping='{"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}'`.
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

### Step 3: Choose the policies and size the main grid

**Goal:** pick comparisons with interesting gaps (big, small and near-zero) and a grid that fits the compute budget.

**Policies (minimum set)**
- **A:** SmolVLA LIBERO checkpoint, default inference settings.
- **A′:** the same checkpoint with **one inference setting changed**, for example the number of actions executed per predicted chunk (check the exact config field, such as `n_action_steps`, in the installed SmolVLA config). This gives a realistic **small-gap ablation**, the kind of difference papers report.
- **B:** a stronger, larger model (π0.5 LIBERO fine-tune, `lerobot/pi05_libero_finetuned`). It loads in about 10 GB of the A5000's 24 GB.

**Grid design**
- Start with **one suite**: `libero_spatial`, or `libero_10` if you want harder, long-horizon tasks with more room below 100%. Add a second suite if the budget allows.
- Use all 10 tasks × as many initial states as affordable (LIBERO provides 50 per task) × **at least 2 seeds**.
- Run **the same (task, initial state, seed) keys for every policy**. This is the paired design.
- Example size: 10 tasks × 50 initial states × 2 seeds = 1,000 episodes per policy, 3,000 in total. Scale it to the Step 2 cost table.

**Sanity check:** before the big run, do a 10-episodes-per-task evaluation of Policy A and compare it with the success reported on the model card or in the SmolVLA paper. If it is far off, debug the setup (control mode, image preprocessing, versions) before going further.

**Deliverables:** `configs/policies.yaml`, `configs/grid_main.yaml`, and the sanity-check result in the notebook.
**Done when:** the sanity check is close to the published numbers (or the difference is understood and documented) and the grid fits the budget.

---

### Step 4: Run the main grid

**Goal:** the full per-episode dataset for the main analyses.

**Tasks**
- Split the grid into **shards** (for example one policy × one task). Each shard writes its own parquet file, merged afterwards. `scripts/run_grid.sh` walks through the shards one after another on the single GPU, inside `tmux`, with one log file per shard in `logs/`.
- A shard that crashes must not stop the queue: log the failure, move on to the next shard, and let a second pass of the same command pick up what is missing (the harness is idempotent, Step 2).
- Optionally run two or three shards at once if GPU memory and the 16 CPU cores allow it; measure whether that actually raises throughput before relying on it.
- Log failures (crashes, timeouts) separately from task failures. A crashed episode is missing data, not a task failure.
- Save videos for a small random sample (for example 1 in 50 episodes, plus all episodes of one task) for qualitative failure analysis.
- After the run, check completeness: every planned key is present exactly once.

**Deliverables:** `results/raw/main/*.parquet`, `results/tables/main_completeness.csv`, sampled videos.
**Done when:** completeness is at least 99%, and any missing keys are listed with a reason.

**Pitfalls:** do not look at policy comparisons until the run is complete (no peeking, which inflates false positives with fixed-sample tests). The run takes hours to days on one GPU and the machine can be rebooted or used by someone else: rely on resumability, check progress with `tmux attach` and `nvidia-smi`, and copy `results/raw/` off `/Data` when it finishes.

---

### Step 5: Build and validate the statistics toolkit

**Goal:** a small, tested library in `src/evalstats/stats/`. Every method is checked by simulation before it touches real data.

**Methods to implement**
- **Intervals for one success rate:**
  - Wilson and Clopper-Pearson intervals (`statsmodels.stats.proportion.proportion_confint`).
  - Bayesian Beta posterior with a Jeffreys or uniform prior, reported as a credible interval and plotted as a violin.
  - **Stratified** estimate across tasks (fixed-effects estimand) and **cluster bootstrap** (resample tasks, then initial states within tasks) for the random-effects estimand.
- **Comparing two policies:**
  - Unpaired: Barnard's exact test (`scipy.stats.barnard_exact`) and Fisher's exact test.
  - Paired on (task, initial state, seed): McNemar's exact test (`statsmodels.stats.contingency_tables.mcnemar(exact=True)`). Optionally, a stratified version (Cochran-Mantel-Haenszel across tasks).
  - Multiple comparisons: Holm-Bonferroni correction; compact letter display for ranking more than 2 policies.
- **Sequential testing:** use TRI's STEP code if it can be installed and fits Bernoulli outcomes (check [the project page](https://tri-ml.github.io/step/)). Otherwise implement a simpler anytime-valid test (for example a mixture sequential probability ratio test) and say so clearly.
- **Power by resampling:** given the per-episode pool for two policies, repeatedly draw evaluations of size n under a given protocol (unpaired, paired, sequential), run the test, and record the rejection rate. This is the empirical power at sample size n.

**Validation tests in `tests/`** (this is where your UQ background shows):
- **Coverage:** simulate Bernoulli data with known p (including p = 0.95 and 0.98, where Wald intervals fail) and check that each interval's empirical coverage is close to 95%. Report the table.
- **Type-I error:** simulate two identical policies and check that each test rejects at about 5%, or less for exact tests.
- **Power sanity:** for a known gap, analytic power and resampled power roughly agree.

**Deliverables:** `src/evalstats/stats/`, `tests/`, `results/tables/coverage.csv`, `results/tables/type1.csv`.
**Done when:** all tests pass and the coverage and Type-I tables look right. Note in the README which intervals under-cover at high success rates; that table is a result in itself.

---

### Step 6: Analysis 1: how uncertain is one success rate?

**Goal:** answer RQ1 with numbers and figures.

**Tasks**
- For each policy: point estimate, Wilson interval, Beta posterior, stratified interval and cluster-bootstrap interval. Compare the widths of the naive pooled interval and the cluster-aware one, and explain the difference.
- **Variance decomposition** of the success indicator Y with the law of total variance:

  Var(Y) = Var_task( E[Y | task] ) + E_task[ Var_init( E[Y | task, init] ) ] + E[ Var(Y | task, init) ]

  The three terms are **between tasks**, **between initial states within a task**, and **within (seed / policy randomness)**. Estimate them from the grid; the last term needs at least 2 seeds per initial state. If the policy turns out deterministic given an initial state, say so: it means seeds add no information, and only initial states do.
- **Protocol check:** LeRobot recommends 10 episodes per task for LIBERO, and many papers use 50. For both, compute the typical interval half-width around the observed success rate. One sentence answer: "at 10 episodes per task, a reported X% is really X ± Y%".

**Figures**
- Forest plot: each policy's success with the different intervals side by side.
- Stacked bar: variance decomposition per policy.
- Per-task success heatmap with uncertainty (or small-multiple Beta violins).

**Deliverables:** `results/tables/rq1_*.csv`, `figures/rq1_*.png`, notebook entry.
**Done when:** every figure is reproducible from one script, and the one-sentence answer is written.

---

### Step 7: Analysis 2: how many rollouts to tell two policies apart?

**Goal:** answer RQ2. This is the headline result.

**Tasks**
- For each policy pair (A vs A′, A vs B), compute the observed gap and its interval.
- **Power curves:** empirical power as a function of total episodes n for three protocols:
  1. **Unpaired fixed-n** (Barnard).
  2. **Paired fixed-n** (McNemar on matched initial states and seeds).
  3. **Sequential** (STEP or your anytime-valid test): report the average number of episodes used before stopping, not just power.
- Headline numbers: the **n needed for 80% power** under each protocol, and the **percentage saved** by pairing and by sequential testing.
- **Generalize beyond the observed gaps:** use a simple simulation model (task-level success rates from the data, plus a shifted copy for the second policy) to draw a chart of "episodes needed versus true gap" for gaps of 1, 2, 5 and 10 points around a 90% baseline. This is the chart people will screenshot.
- Look at **discordant pairs**: when A succeeds and A′ fails on the same initial state, is it a few hard initial states or spread out? Watch 5 to 10 discordant videos and write down failure modes in plain words (missed grasp, wrong object, timeout, etc.).

**Figures**
- Power-versus-n curves, one line per protocol, with the 80% line marked.
- "Episodes needed versus gap" chart.
- 2×2 discordance table and a short gallery of failure frames.

**Deliverables:** `results/tables/rq2_*.csv`, `figures/rq2_*.png`, failure-mode notes.
**Done when:** you can state, with numbers from the data: "Detecting a 2-point gap at 80% power takes about N rollouts unpaired, M paired, and K on average with sequential testing."

---

### Step 8: Analysis 3: robustness under perturbations (recommended)

**Goal:** answer RQ3 with proper uncertainty. This is the most "robotics" part of the project.

**Tasks**
- Create a **separate** uv environment in `envs/libero_plus/` (its own `pyproject.toml` and `.venv`), because LIBERO-Plus replaces vanilla LIBERO. Follow the LeRobot LIBERO-Plus doc: system packages, pinned `robosuite==1.4.1`, the assets download from Hugging Face, `--env.hard_reset=true` for reproducibility.
- Choose a budget-sized subset: the same suite as the main grid, the 7 perturbation dimensions (object layout, camera viewpoint, robot initial state, language, lighting, background, sensor noise), at one or two levels each.
- For each policy × perturbation: the success rate with its interval, and the **drop from the unperturbed baseline** with a paired test where the design allows it. Apply Holm correction across the 7 dimensions.
- Report which degradations are **statistically significant** and which are within noise. That distinction is exactly what most robustness tables leave out.

**Fallback** if LIBERO-Plus cannot be installed on the workstation (system packages need root, which you do not have): implement two perturbations yourself as env or observation wrappers on vanilla LIBERO: (1) image-space photometric noise and brightness shift, and (2) paraphrased or emptied language instructions. Keep the same statistics.

**Figures**
- Heatmap: policy × perturbation, colored by drop, with markers for significant drops after correction.
- Forest plot of drops with intervals.

**Deliverables:** `configs/grid_robustness.yaml`, `results/raw/robustness/`, `figures/rq3_*.png`.
**Done when:** every cell in the heatmap has an interval behind it, and the corrected significance is shown.

---

### Step 9: Optional: leaderboard audit

**Goal:** a short, attention-grabbing sanity check of published comparisons.

**Tasks**
- Collect reported LIBERO success rates for several well-known VLAs from their papers or model cards, **with the number of evaluation episodes they used**. Record the source of every number in `results/tables/leaderboard_sources.csv`. Skip any number whose episode count you cannot find.
- For each adjacent pair on the leaderboard, compute whether the gap is distinguishable at that number of episodes (an unpaired test, since you lack their per-episode data).
- Report the share of adjacent gaps that are within noise. Word it carefully: "not distinguishable from these reported numbers alone", not "these papers are wrong".

**Deliverables:** sourced table and one figure.
**Done when:** every number links to a source.

---

### Step 10: Package it (README, report, reproducibility)

**Goal:** a repo a screener understands in 30 seconds and an engineer can rerun.

**README structure**
1. **One-line hook** with the headline number from Step 7, for example: "On LIBERO, telling apart two VLA policies 2 points apart takes about N rollouts; pairing initial states cuts that by X%."
2. **Headline figure** (the "episodes needed versus gap" chart) and a GIF of a rollout and a failure.
3. **Key findings:** 3 to 5 bullets, each tied to a figure.
4. **Recommendations for practitioners:** for example, "report Wilson or Beta intervals, pair initial states, use at least N episodes per task for gaps under Y points".
5. **Reproduce:** environment files, one command per step, expected GPU-hours.
6. **Toolkit usage:** a 10-line example of using `evalstats` on any per-episode CSV.
7. **Limitations:** one benchmark, simulation only, a few checkpoints, inference-setting ablation as a proxy for "two policies".
8. **References.**

**Also**
- `docs/report.md`: a 3 to 5 page write-up (question, setup, methods, results, limitations). Optionally turn it into a short blog post or arXiv-style PDF.
- A single `make reproduce-figures` (or script) that rebuilds every figure from `results/`.
- Clean commit history, MIT license, CI that runs the unit tests.
- Optionally, contribute back: the LeRobot issue tracker already discusses putting structured evaluation results into model cards ([issue #4481](https://github.com/huggingface/lerobot/issues/4481)). A small, well-scoped PR (for example per-episode results in the eval output, if missing) would be a strong signal.

**Deliverables:** README, report, figures, tagged release `v1.0`.
**Done when:** a fresh clone plus the documented commands rebuilds the figures from the saved results.

---

## 5. Scope ladder (if time is short)

| Level | Steps | What you can claim |
|---|---|---|
| **Minimum viable** | 0 to 7, one suite, policies A and A′ | Uncertainty, variance decomposition, paired vs unpaired power |
| **Target** | + Policy B, + Step 8 | Adds robustness with proper statistics |
| **Stretch** | + second suite, + Step 9, + upstream PR | Leaderboard audit, open-source contribution |

You can apply to jobs as soon as the minimum viable level is done. Link the repo as "ongoing" before that if needed.

---

## 6. Compute and time estimates (to be replaced by Step 2 measurements)

- The main grid at the example size is a few thousand episodes. The real cost depends on the throughput measured in Step 2; do not launch before filling in the cost table.
- All compute is one RTX A5000 on verdier, so GPU-hours equal wall-clock hours and the policies run one after the other. Size the grid to what fits in a few days of unattended running.
- π0.5 loads in about 10 GB of the 24 GB. If it turns out too slow for the budget, use a second SmolVLA variant as Policy B and say so.
- The statistics (Steps 5 to 7) run on a laptop once the parquet files exist.

---

## 7. Risks and fallbacks

| Risk | Fallback |
|---|---|
| LeRobot eval does not expose per-episode results or initial-state control | Thin wrapper around the eval loop (Step 2) |
| Published success not reproduced | Debug control mode, image preprocessing and versions; if it remains, document the gap and continue (the statistics still hold for what you measured) |
| Policy A is near 100% on the chosen suite (little variance to study) | Switch to `libero_10` (long-horizon, lower success) |
| SSH session drops during a long run | Always launch inside `tmux`; logs in `logs/` |
| Machine rebooted, or GPU taken by another user, mid-run | Resumable harness: rerun the same command; check `nvidia-smi` before launching |
| `/Data` is local to verdier and may be wiped | Code and `results/tables/` in git; copy `results/raw/` elsewhere after each long run |
| Home quota (30 GB) fills up | Keep every cache on `/Data` via `env.sh` |
| LIBERO-Plus install fails | Two home-made perturbations (Step 8 fallback) |
| STEP code unusable | Simple anytime-valid sequential test, clearly labeled |

---

## 8. CV and interview material (fill in at the end)

**CV entry (template; replace the brackets with real results):**

> **Robot Learning Project: Statistically Rigorous Evaluation of VLA Policies** (GitHub)
> - Evaluated open vision-language-action policies (SmolVLA, π0.5) on [N] simulated LIBERO manipulation rollouts on a remote GPU workstation; decomposed success-rate variance into task, initial-state and policy-noise components
> - Showed that pairing initial states [cuts the rollouts needed to detect a 2-point gap by X%]; quantified robustness under 7 perturbation types with multiple-comparison-corrected tests; released a tested evaluation toolkit

**Interview story (2 minutes):** the same "diagnose before you build" instinct as NYGC. Robot learning compares policies on noisy, expensive rollouts. You measured how noisy those comparisons are and how to make them cheaper and more reliable, using the uncertainty quantification you studied at Polytechnique.

**Questions to be ready for:** why Wilson over Wald at high success rates; fixed versus random effects for tasks; why peeking breaks fixed-sample tests; what pairing assumes; how this would change on a real robot (fewer episodes, non-stationary resets, so sequential testing matters even more).

---

## 9. References

- Kress-Gazit et al., *Robot Learning as an Empirical Science: Best Practices for Policy Evaluation* (2024): [arXiv:2409.09491](https://arxiv.org/abs/2409.09491)
- TRI blog, *Statistical Thinking for Robot Policy Evaluation: From Rigorous A/B Testing to Effective Visualization*: [Medium](https://medium.com/toyotaresearch/statistical-thinking-for-robot-policy-evaluation-from-rigorous-a-b-testing-to-effective-0ae886fbd68d)
- Snyder et al., *Is Your Imitation Learning Policy Better than Mine? Policy Comparison with Near-Optimal Stopping* (STEP): [arXiv:2503.10966](https://arxiv.org/abs/2503.10966), [code](https://tri-ml.github.io/step/)
- *Beyond Binary Success: Sample-Efficient and Statistically Rigorous Robot Policy Comparison* (N-SCORE): [arXiv:2603.13616](https://arxiv.org/abs/2603.13616)
- *LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models*: [arXiv:2510.13626](https://arxiv.org/abs/2510.13626), [code](https://github.com/sylvestf/LIBERO-plus)
- LeRobot LIBERO docs: [huggingface.co/docs/lerobot/en/libero](https://huggingface.co/docs/lerobot/en/libero)
- LeRobot LIBERO-Plus docs: [GitHub](https://github.com/huggingface/lerobot/blob/main/docs/source/libero_plus.mdx)
- SmolVLA paper: [arXiv:2506.01844](https://arxiv.org/abs/2506.01844); checkpoint: [lerobot/smolvla_libero](https://huggingface.co/lerobot/smolvla_libero)
