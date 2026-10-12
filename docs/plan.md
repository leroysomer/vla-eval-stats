# Analysis plan

Written on 2026-10-11 and extended with Part 3 on 2026-10-12, before the main grid is run. Anything decided after seeing results is
recorded in `docs/lab_notebook.md` with its date and labelled as post hoc.

The report will be about five pages long and carries **three questions and one idea**: the
variance structure of a LIBERO success rate (RQ1) determines how much a paired comparison gains
over an unpaired one, both when the number of rollouts is fixed in advance (RQ2) and when the
evaluator may stop as soon as the evidence is sufficient (RQ3). Robustness to perturbations and
the leaderboard audit are out of scope (Section 9).

## 1. Setup fixed in advance

| Item | Choice |
|---|---|
| Suite | `libero_spatial`, all 10 tasks |
| Initial states | all 50 shipped by LIBERO per task |
| Seeds | 4 per (task, initial state): 0, 1, 2, 3 |
| Episodes | 10 × 50 × 4 = 2,000 per policy, same keys for every policy |
| Success | LIBERO's built-in task success flag at the end of the episode |
| Step limit | 280 (LeRobot's value for `libero_spatial`) |
| Crashed episodes | missing data, rerun; never counted as failures |
| Level and sidedness | every test two-sided at α = 0.05; every interval 95% |

**Policies**

| Name | Checkpoint | Inference setting |
|---|---|---|
| A | `lerobot/smolvla_libero` | default (`n_action_steps = 50`) |
| A′ | same checkpoint | `n_action_steps = 10` (replans five times as often) |
| B | `lerobot/pi05_libero_finetuned` | default, with `compile_model = false` |

The value 10 for A′ is fixed now and is not tuned on results. B runs without `torch.compile`
because the compiled model is not reproducible for a fixed seed (lab notebook, 2026-10-08).

**Planned comparisons:** A vs A′ (small gap expected, the kind of ablation papers report) and
A vs B (large gap expected). There is no third comparison. Since there are only two, each
answering its own question, no multiple-comparison correction is applied. Both comparisons are
used in RQ2 and in RQ3.

**Randomness of one episode.** An episode is identified by its key $k = (t, i, s)$: task,
initial state, seed. The harness derives all randomness (policy noise, simulator) from the key
alone, not from the policy and not from run order. Two policies run on the same key therefore
share the initial state and the random numbers.

## 2. Estimand

$Y_\pi(t, i, s) \in \{0, 1\}$ is the success of policy $\pi$ on key $(t, i, s)$.

- **Tasks are fixed.** The 10 tasks are the benchmark and get equal weight.
- **Initial states are random.** The 50 states of a task are treated as independent draws from
  that task's distribution of starting configurations.
- **Seeds are random.**

The success rate of a policy is

$$p_\pi = \frac{1}{10} \sum_{t=1}^{10} \mathbb{E}_{i, s}\left[ Y_\pi(t, i, s) \right],$$

and the gap between two policies is $\delta = p_{\pi'} - p_\pi$. Every claim in the report is
about these 10 tasks. Nothing is claimed about other tasks or suites.

## 3. RQ1: how uncertain is one reported success rate, and why?

**Variance decomposition.** For an episode drawn with a uniformly random task, the law of total
variance gives

$$p(1 - p) = \underbrace{\sigma^2_{\text{task}}}_{\text{between tasks}}
           + \underbrace{\sigma^2_{\text{init}}}_{\text{between initial states, within task}}
           + \underbrace{\sigma^2_{\text{seed}}}_{\text{between seeds, within initial state}} .$$

The three components are estimated for each policy with the unbiased moment estimators of the
nested design (10 tasks × 50 states × 4 seeds), with intervals from the bootstrap below.

**What the decomposition predicts.** A protocol that runs $K$ initial states per task with $m$
seeds each (so $n = 10Km$ episodes) estimates $p$ with variance

$$\operatorname{Var}(\hat p) = \frac{\sigma^2_{\text{init}}}{10K} + \frac{\sigma^2_{\text{seed}}}{10Km},
\qquad \text{while the binomial formula assumes} \qquad \frac{p(1-p)}{10Km} .$$

The ratio of the two is

$$\text{design effect} = \frac{m\,\sigma^2_{\text{init}} + \sigma^2_{\text{seed}}}
                               {\sigma^2_{\text{task}} + \sigma^2_{\text{init}} + \sigma^2_{\text{seed}}} .$$

Below 1, the binomial interval is too wide (task differences are counted as noise although
tasks are fixed). Above 1, it is too narrow (repeated seeds on one initial state are counted
as independent episodes). Which case holds on LIBERO is the result.

**Two intervals, compared.**

1. *Naive:* Clopper-Pearson on the pooled count, as if the 2,000 episodes were independent
   draws with one success probability.
2. *Cluster-aware:* percentile bootstrap, 10,000 replicates, resampling the 50 initial states
   with replacement within each task and keeping the 4 seeds of each drawn state together.

**Answer in one sentence**, for the two protocols in common use (10 and 50 episodes per task,
one seed each, so $m = 1$): "a reported X% on `libero_spatial` is X ± Y points", with Y from
the variance formula above and the estimated components.

**Figure 1** (two panels): variance shares per policy as stacked bars; success rate per policy
with both intervals. **Table 1:** the components, the design effect and Y for the two protocols.

## 4. RQ2: how many rollouts to tell two policies apart, paired vs unpaired?

Throughout, $n$ is the number of rollouts **per policy**; both designs cost $2n$ rollouts.

**The two designs.** Keys are drawn independently and uniformly from the grid (task, then
initial state, then seed).

- *Unpaired:* $n$ keys for one policy and $n$ other, independent keys for the other.
  Test: Barnard's exact test on the two success counts.
- *Paired:* the same $n$ keys for both policies.
  Test: McNemar's exact test on the discordant pairs.

Both tests are exact for the null $\delta = 0$ under this sampling, because the pairs (or the
episodes) are independent and identically distributed.

**Why pairing helps.** With $b$ the probability that only the first policy succeeds on a key
and $c$ the probability that only the second does, $\delta = c - b$ and the variance of the
estimated gap is $\sigma^2 / n$ with

$$\sigma^2_{\text{unpaired}} = p_\pi(1 - p_\pi) + p_{\pi'}(1 - p_{\pi'}),
\qquad
\sigma^2_{\text{paired}} = b + c - (c - b)^2
                         = \sigma^2_{\text{unpaired}} - 2\operatorname{Cov}(Y_\pi, Y_{\pi'}) .$$

The rollouts needed for power $1 - \beta$ are approximately
$n \approx (z_{1-\alpha/2} + z_{1-\beta})^2 \, \sigma^2 / \delta^2$ in both designs, so

$$\frac{n_{\text{paired}}}{n_{\text{unpaired}}} \approx
  1 - \frac{2\operatorname{Cov}(Y_\pi, Y_{\pi'})}{\sigma^2_{\text{unpaired}}} .$$

The covariance splits into the same three levels as in RQ1 (shared task, shared initial state,
shared seed). This is the link between the two questions: pairing removes exactly the part of
the noise that the two policies have in common.

**What is computed**, for A vs A′ and for A vs B:

1. The gap $\hat\delta$ on the full grid, with a 95% interval from the bootstrap of Section 3
   applied to the per-key differences. This is the only inference made on the full grid,
   where the 4 seeds of an initial state are not independent.
2. $\hat b$, $\hat c$ and the covariance split by level (Table 2).
3. Power curves: for $n$ from 50 to 5,000, draw 5,000 evaluations of each design from the
   grid, apply the test, record the rejection rate.
4. $n$ for 80% power under each design, from the curves and from the formula, and the
   resulting saving.
5. A null control: A against itself on the same initial states with different seeds. The true
   gap is zero, so both tests must reject in at most 5% of draws.
6. Ten discordant episodes of A vs A′ are re-run with video (keys are reproducible) and the
   failure modes described in one paragraph.

**Beyond the two observed gaps.** A chart of rollouts needed against the true gap, for gaps of
1, 2, 5 and 10 points, from the formula above. The paired curve assumes that the rate of
discordant pairs $b + c$ stays at the value measured for A vs A′ while $c - b$ varies. This
is a modelling assumption and is stated on the chart.

**Answer in one sentence:** "detecting a 2-point gap at 80% power on `libero_spatial` takes
about N rollouts per policy unpaired and M paired."

**Figure 2:** power against $n$, one panel per comparison, one curve per design, formula
overlaid. **Figure 3:** rollouts needed against gap. **Table 2:** gaps, discordance, covariance
split, $n$ at 80% power.

## 5. RQ3: how many rollouts when the evaluator may stop early?

RQ2 answers "how many rollouts for a given gap", which requires knowing the gap. Here the
evaluator fixes only a budget and stops as soon as a decision is reached. This completes the
grid of designs:

| | Fixed $n$ (RQ2) | Sequential (RQ3) |
|---|---|---|
| **Unpaired** | Barnard | STEP |
| **Paired** | McNemar | the test below |

**Common setting.** Budget $N = 500$ rollouts per policy, two-sided level α = 0.05, risk spent
uniformly over the budget: after $n$ rollouts per policy a test may have used at most
$\alpha n / N$ of its false-positive probability. Rollouts arrive one per policy at a time, on
keys drawn as in Section 4.

**Unpaired: STEP** (Snyder et al., 2025), from the authors' package
`sequentialized_barnard_tests`, with $n_{\max} = 500$ and its default uniform risk budget. It is
one-sided, so it is run in both directions at level α/2 each. Each policy gets its own
independent keys, so the independence STEP assumes holds exactly. The package is not tested
beyond $n_{\max} = 500$, which is why the budget is 500.

**Paired: a sequential McNemar test.** Both policies run on the same key at each step. Let
$D_n = Y_{\pi'} - Y_\pi \in \{-1, 0, 1\}$ be the difference on the $n$-th key, $B_n = |D_n|$
indicate a discordant pair, and $W_k$ be the sum of the first $k$ non-zero differences.

1. *Null distribution.* Pairs are independent and identically distributed, so given the whole
   sequence $(B_n)$ the signs of the discordant pairs are independent and each is $+1$ with
   probability $\theta = c / (b + c)$. Under $H_0: \delta = 0$ we have $\theta = 1/2$, so
   $W_k$ is a symmetric simple random walk **whatever the discordance rate** $b + c$. The null
   is a single known distribution; there is no nuisance parameter.
2. *Boundary.* Let $n_k$ be the step at which the $k$-th discordant pair arrives. The test
   rejects at that step if $|W_k| \ge a_k$, where $a_k$ is the smallest integer such that

   $$\Pr\nolimits_{H_0}\left(\exists j \le k : |W_j| \ge a_j\right) \le \alpha \, \frac{n_k}{N} .$$

   The left-hand side is computed exactly by propagating the distribution of the walk over
   the paths that have not yet stopped. Risk that cannot be spent because the walk is discrete
   carries over to later steps.
3. *Validity.* Given $(B_n)$ the thresholds $a_1, a_2, \dots$ are fixed numbers, and by
   construction a symmetric walk crosses them with probability at most $\alpha\, n_K / N \le
   \alpha$, where $K$ is the number of discordant pairs within the budget. This holds for every
   $(B_n)$, hence unconditionally and for every discordance rate. The false-positive rate is at
   most α in finite samples, with no approximation.
4. *Relation to known tests.* Spending all the risk at $n = N$ gives McNemar's exact test.
   Spending it over time according to the rollout count, not the count of informative pairs,
   is the alpha-spending idea of Lan and DeMets (1983), computed exactly here.

If the budget is reached without rejection, the test stops without a decision. This is not
evidence that the two policies are equal.

**What is computed**, for A vs A′ and for A vs B, from 5,000 simulated evaluations of each
design drawn from the grid:

1. The probability of having rejected by step $n$, for $n = 1, \dots, 500$.
2. The power within the budget, next to the power of Barnard and McNemar at $n = 500$.
3. The expected number of rollouts per policy before stopping (500 when no decision is reached).
4. The null control of Section 4, applied to both sequential tests.

**Expected pattern, to be confirmed or refuted.** On A vs B both sequential tests should stop
long before the budget. On A vs A′ the unpaired test may rarely decide within 500 rollouts; the
question is whether pairing makes the comparison decidable.

**Answer in one sentence:** "with a budget of 500 rollouts per policy, A vs B is decided after
about N rollouts unpaired and M paired, and A vs A′ is decided in X% of evaluations unpaired
and Y% paired."

**Figure 4:** probability of having rejected against $n$, one panel per comparison, one curve
per sequential design, with the fixed-$n$ power at 500 marked. **Table 3:** power within budget
and expected rollouts for the four designs.

## 6. Validation before real data

Each method is checked on simulated data with known truth (a nested model with chosen
$\sigma^2_{\text{task}}, \sigma^2_{\text{init}}, \sigma^2_{\text{seed}}$ and a chosen
correlation between two policies):

- the moment estimators recover the three components;
- the bootstrap interval covers $p$ about 95% of the time, and the naive interval departs from
  95% in the direction the design effect predicts;
- Barnard and McNemar reject a true null at most 5% of the time;
- resampled power agrees with the formula;
- the sequential McNemar test rejects a true null at most 5% of the time for discordance rates
  from 2% to 50%, and its boundary computed with all the risk at the last step reproduces
  McNemar's exact test;
- STEP, run in both directions at α/2, rejects a true null at most 5% of the time on
  independent Bernoulli sequences.

## 7. Rules

- No policy comparison is looked at until the grid is complete (every key present once).
- The Step 3 sanity check (100 episodes of A) may be compared with published numbers to debug
  the setup. Its episodes are part of the grid and are not rerun.
- If a setting has to change after the grid starts, all policies are rerun on all keys.
- Results are reported whatever they are: a gap near zero for A vs A′ or a saving near zero
  from pairing is a finding.

## 8. Limits of the claims

- One suite, one simulator, three policy variants, one of which differs only by an inference
  setting.
- Power figures and stopping times are plug-in estimates: they treat the measured grid as the
  population.
- The sequential results hold for one budget (500) and one way of spending the risk (uniform).
  The sequential McNemar test is valid but is not claimed to be optimal.
- Initial states are assumed to be independent draws; LIBERO does not document how its 50
  states were generated.

## 9. Out of scope

- **Perturbations (LIBERO-Plus).** Its difficulty levels are defined by other models' outcomes
  and easy instances were filtered out, so a rigorous treatment needs its own design.
- **Other sequential methods** (anytime-valid tests, the Lai procedure, budgets above 500).
- **Leaderboard audit.**
