# An Analysis of Diffusion Policy under Varying Demonstration Quality and Quantity

Training a diffusion policy to perform square-nut peg insertion, where the demonstrations
come not from a human teleoperator but from **three classical controllers of deliberately
different quality**. By sweeping demonstration quality against quantity, this project asks a
practical question about imitation learning: *when is more cheap, noisy data worth more than
fewer clean demonstrations — and where does noisy data stop paying off?*

The headline finding is that a learned policy is shaped far less by the raw *precision* of its
demonstrations than by **how much of the task's state space those demonstrations cover**. A
noisy demonstrator that wanders through more states can beat a flawless one that always traces
the same narrow path.

---

## Motivation & Background

**Diffusion Policy.** A diffusion policy starts from Gaussian noise and iteratively *denoises*
it into an action sequence, conditioned on the current observation. Rather than classifying the
next action, it learns the gradient of the action distribution and represents that distribution
*implicitly*. This differs from tokenized/autoregressive approaches, which discretize actions
into tokens and predict them one step at a time.

**Why peg insertion, why classical controllers.** Contact-rich tasks like peg insertion have a
long history of being solved *classically* — with closed-loop control and force feedback — so we
can generate demonstrations at any quality level we like, on demand, and know exactly how they
were produced. That makes it a clean testbed for the real question:

> If the same task is demonstrated by controllers of different quality — a precise expert, a
> competent-but-shaky operator, and a near-random flailer — **how does the learned policy
> differ**, and how much does demonstration *quantity* compensate for demonstration *quality*?

**The imitation-learning tension.** Behavior cloning only learns the states it actually sees. A
flawless expert traces a *narrow tube* through state space, so a policy that drifts even
slightly falls off-distribution and has no learned response — the classic **covariate-shift**
failure of BC. Injecting noise into the demonstrations *widens that tube*: the policy sees, and
learns to recover from, states to the side of the nominal path. This project measures that
trade-off directly.

---

## Implementation

The pipeline is built in three stages: **(1)** a custom simulation environment, **(2)** three
classical controllers that generate the demonstrations, and **(3)** training + evaluation of 12
diffusion policies on GPU.

### Step 1 — The Environment

The task is **square-nut peg insertion**, built as a subclass of robosuite's `NutAssemblySquare`
(robosuite 1.5.2, MuJoCo 3.8.1) with a Panda arm. The subclass — `Manipulation_Enviroment` —
adds insertion-aware **staged rewards** on top of the parent's four stages, giving a smooth
six-rung reward staircase:

```
reach → grasp → lift → hover → insert → seat
 0.10    0.35    0.50   0.70     0.80    0.90
```

The first four rungs are inherited unchanged; `insert` and `seat` are new. Both fold continuous
geometric quantities (lateral XY error, descent onto the peg) through `tanh` gains chosen from
the task geometry — each gain anchored so it pays half credit at a physically meaningful offset
(e.g. the 6.75 mm mechanical clearance, or `on_peg`'s 30 mm tolerance) rather than being a magic
number. This keeps the reward monotone down the final descent, which the parent's peg-midpoint
distance term does *not* do.

A few facts about the task that drove the design:

- **Orientation is the dominant variation.** At reset, the nut's XY position varies by only ~5 mm
  in x and ~11 cm in y, but its **yaw is uniform over the full 360°**. Yaw is what both the
  grasp phase and the trained policy must generalize over.
- **Success requires releasing the nut.** robosuite's `_check_success` fires while the gripper
  is *still holding* the seated nut, but letting go is the true end of the task — so demos are
  recorded through release + retreat, not cut off at first success.

The env is validated by `test_env.py` (17/17 checks: reward monotonicity, stage caps over 2000
random poses, `reward()` exactly 1.0 at success).

### Step 2 — The Controllers

Three controllers form a deliberate **quality ladder**, differing in *exactly one thing* each so
the ablation stays interpretable. All share a single phase-indexed state machine
(approach → descend → grasp → lift → transport → insert → release → retreat); they differ only
in what corrupts the action:

| Controller | What it is | Role |
|---|---|---|
| **`pd`** — clean PD | precise position/yaw servo + lateral-force correction during insertion | a repeatable expert |
| **`hybrid`** — PD + noise | the same PD loop, but Gaussian noise added to each output action (the loop *still senses and corrects*) | a competent operator with a shaky hand |
| **`pure_noise`** | a naive heuristic dominated by large Gaussian noise, no force feedback, no fine servo | worst case; succeeds mostly by luck |

> **Naming note.** The eval JSONs and analysis use the corrected names above. Some on-disk
> artifact paths (`data/*.hdf5`, checkpoint dirs) keep older filenames — those are just paths,
> not claims about the arm.

Getting the clean PD controller to work was where most of the physics lived. The wins were
**geometric, not force-related**: the nut hangs 5.4 cm from its grasp point and *droops* up to
16.5° on contact (wedging across the peg); a stiff XY servo makes jamming *worse* by twisting a
touched-down nut tighter; and cross-axis coupling during the descent drags the grasp off the
handle. Correcting nut tilt, softening the insert-phase XY gain, using a constant-rate descent,
and relaxing one transport tolerance from 3.5 mm to 5.5 mm took the clean controller from roughly
**one-in-three to ~85%** collection success.

Each controller is run to collect **only successful** demonstrations, at four dataset sizes
**N ∈ {10, 50, 100, 200}** — 3 × 4 = **12 datasets**. Crucially, the three controllers are run
against a **shared per-episode seed list**, so at any given N they face the same initial states.
The per-controller collection *yield* — the share of attempts that produced a usable demo — is
itself a headline result and sets up the ablation:

![Demonstrator quality ladder](peg_insertion/analysis/quality_ladder.png)

The HDF5 layout (`obs` / `actions` / `rewards` / `dones` / `states` per demo) is written directly
in the robomimic-style format that diffusion_policy's low-dim loader expects, so training needs
no conversion. The pinned observation set is 16-dim: `robot0_eef_pos` (3), `robot0_eef_quat` (4),
`robot0_gripper_qpos` (2), `SquareNut_pos` (3), `SquareNut_quat` (4); actions are 7-dim OSC deltas.

### Step 3 — Training & Evaluation on GCP

Each of the 12 datasets trains one **low-dim diffusion policy** (UNet, `diffusion_policy_lowdim`
config), all with identical hyperparameters and epoch counts so the ablation is fair. The only
thing that changes across runs is the dataset.

Training and evaluation both run on a **GCP L4 GPU** rather than the M2 Mac. The reason is the
diffusion sampler: a 100-step DDPM chain costs ~3.7 s per prediction on MPS, making the sampler
~96% of eval wall-time — the full 12 × 50-rollout sweep takes ~2 days locally versus a few hours
on the L4. The GCP runbook (`gcp/README.md`) provisions the VM, transfers code + data, trains all
12 in `tmux`, and evaluates in place. Evaluation needs no headless-GL setup because rollouts run
pure MuJoCo physics (`has_offscreen_renderer=use_camera_obs=False`).

Evaluation is deliberately **harder than collection**, in two ways that matter for reading the
numbers:

- **Unfiltered.** Each policy is scored over **50 rollouts from `env.reset()` with uniform yaw**
  over ±180°, whereas the *training* set kept only successful demos and is therefore **yaw-biased
  toward easy poses**. Eval success sits *below* collection success by construction.
- **Paired.** Placement is reseeded per rollout from a fixed base, so **all 12 policies are
  scored on the same 50 initial states**.

At 50 rollouts, the binomial standard error is ~6–7% near 50% — that is the ruler for whether a
gap between cells is real.

---

## Analysis: the training distribution shapes the policy

The central result is a 3 (demonstrator) × 4 (N) grid of eval success rates:

```
controller        | N=10   | N=50   | N=100  | N=200
pd (clean)        |   4%   |  34%   |  68%   |  74%
hybrid (PD+noise) |  14%   |  58%   |  68%   |  80%   <- best cell
pure noise        |   6%   |  68%   |  62%   |  72%
```
*(± ~3–7% SE per cell.)*

![Success vs. demo count, by demonstrator](peg_insertion/analysis/ablation.png)

**The organizing idea is training-distribution coverage.** A behavior-cloning policy only learns
the states it saw. Reading each demonstrator through that lens:

- **`pd` (clean, narrow distribution).** *Data-hungry.* Nearly useless at N=10–50 (4%, 34%): the
  narrow demo tube gives poor coverage, and small N compounds it. Clean PD only becomes
  competitive once N is large enough to densely cover that narrow tube (68% → 74%). This is
  covariate shift made visible — a precise expert that never strays teaches the policy nothing
  about how to recover once it does.

- **`hybrid` = PD + noise (widened, *and still controlled*).** The thesis case. Noise broadens
  coverage **without destroying demo quality** — the underlying PD loop still senses and corrects,
  so the extra states are *recoverable* ones. Coverage that stays useful keeps paying off as data
  grows, which is why `hybrid` leads at scale (80% at N=200) and never trails clean PD.

- **`pure_noise` (widest distribution, *but uncontrolled*).** The same coverage mechanism taken
  too far. It is the **best cell at N=50 (68%)** — maximal coverage helps most when data is
  scarce — but then **plateaus** (62–72%) instead of climbing. The demonstrations themselves lack
  controlled movement, so coverage gets you *started* but demo quality *caps the ceiling*. Adding
  data cannot teach precision that was never demonstrated.

**The yaw breakdown is the clincher.** Pooling all rollouts by spawn orientation, `pure_noise`
collapses to **25%** on the hardest poses (|yaw| 120–180°) while `hybrid` holds at **54%** —
exactly the poses where real control authority, not luck, is required:

![Success by spawn orientation](peg_insertion/analysis/yaw_breakdown.png)

### What the data supports — and how firmly

- **Firm finding: wider distribution ⇒ better sample efficiency.** At N=50, both noisy
  demonstrators (58%, 68%) beat clean PD (34%) by margins well outside the error bars. When data
  is scarce, coverage dominates precision.
- **Suggestive, not decisive: wider distribution ⇒ higher ceiling.** At N=200 the three arms are
  72 / 74 / 80% with ~6% SE each, so `hybrid`'s lead is within ~1 SE. Sharpening this would take
  more rollouts per cell (SE shrinks like 1/√n) and a **paired** per-seed test that exploits the
  shared seed list, rather than comparing marginal rates.

### Takeaway

Noise helps a learned policy by **widening the training distribution**. *Controlled* noise
(`hybrid`) widens it with recoverable states and wins on sample efficiency — and tentatively at
scale — while *uncontrolled* noise (`pure_noise`) widens it with junk: great for a cold start,
but it hits a quality ceiling and fails where precision matters. For real-world data collection,
the implication is concrete: a cheaper, shakier demonstrator can be *better* than a pristine one,
provided its imperfections still trace **recoverable** states — but there is a floor on quality
below which more data stops buying you anything.

---

## Repository layout

```
peg_insertion/
  envs/Manipulation_Enviroment.py      custom env: 6 staged rewards
  controllers/                         pd / hybrid / pure_noise, shared FSM
  data_collection/collect_demos.py     env + controller -> HDF5 (12 datasets)
  training/eval.py                     receding-horizon rollout -> success JSON
  analysis/analyze_ablation.py         12 JSONs -> table + curve + yaw breakdown
data/                                  12 HDF5 demo datasets (3 controllers x 4 N)
eval_results/                          12 eval JSONs (success_rate, per_rollout)
gcp/                                   train + eval runbook and scripts for the L4 VM
docs/analyzing_results.md              full walkthrough of the ablation result
```

**Reproduce the analysis** (from `proj/`, in the `diffpol` env):

```bash
python peg_insertion/analysis/analyze_ablation.py   # table + yaw breakdown + ablation.png
```
