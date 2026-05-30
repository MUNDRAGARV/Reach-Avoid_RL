# Safety and Liveness Guarantees through Reach-Avoid Reinforcement Learning

Implementation of **safe reinforcement learning for reach-avoid problems** using **Double Deep Q Networks (DDQN)** combined with an **HJ-inspired adaptive safety shield** for a **Dubins car pursuit-evasion environment**.

This project is based on the RSS 2021 work:

> **"Safety and Liveness Guarantees through Reach-Avoid Reinforcement Learning"**

and extends it with an **adaptive Hamilton–Jacobi inspired safety shield**.

---

## Related Papers

### Primary Paper

- **RSS 2021:**  
  *Safety and Liveness Guarantees through Reach-Avoid Reinforcement Learning*  
  https://arxiv.org/abs/2112.12288

### Safety Shield Inspiration

- **L4DC 2025:**  
  *Safe Learning in the Real World via Adaptive Shielding with Hamilton-Jacobi Reachability*
  https://raw.githubusercontent.com/mlresearch/v283/main/assets/lu25a/lu25a.pdf
---

# I. Project Overview

This project studies **safe reinforcement learning** in a **reach-avoid pursuit-evasion problem** involving **Dubins car dynamics**.

The objective of the **evader agent** is to:

1. Reach the target region (**liveness objective**)
2. Avoid unsafe/capture regions (**safety objective**)

A **DDQN agent** learns the navigation policy, while an **HJ-inspired adaptive safety shield** prevents unsafe actions during training and execution.

The framework enables safe exploration while maintaining safety guarantees.

---

# II. Environment

## Dubins Car Pursuit-Evasion Setup

The environment consists of:

- **Evader vehicle** (controlled using RL)
- **Pursuer/adversary**
- **Goal region**
- **Constraint/unsafe regions**

### State Space (6D)

The complete state vector is:

```math
[x_e,\ y_e,\ \theta_e,\ x_p,\ y_p,\ \theta_p]
```

Where:

- `(x_e, y_e)` → Evader position
- `θ_e` → Evader heading angle
- `(x_p, y_p)` → Pursuer position
- `θ_p` → Pursuer heading angle

### Action Space

#### Evader Actions (3 actions)

| Action | Description |
|--------|-------------|
| 0 | Turn Left |
| 1 | Go Straight |
| 2 | Turn Right |

The pursuer acts adversarially using a **worst-case assumption**.

---

# III. Methodology

## Training Pipeline

The proposed framework combines:

### 1. Double Deep Q Network (DDQN)

The RL policy is represented by a neural network architecture:

```text
6 → 512 → 512 → 512 → 9
```

Where:

- **6 inputs** = environment state
- **3 hidden layers (512 neurons each)** = feature learning
- **9 outputs** = action-value predictions

The network learns through:

- Experience Replay
- Target Networks
- Temporal Difference (TD) Learning
- ε-greedy exploration

At every timestep, the DDQN proposes an action for the evader.

---

### 2. Safety Shield

Before executing the DDQN action, a **safety verification step** is performed.

The shield evaluates a **Hamilton–Jacobi inspired safety value function**:

```math
V(s) = safety\_margin(s)
```

Interpretation:

- **V(s) > 0** → Safe state
- **V(s) < 0** → Unsafe state

If the DDQN action is unsafe, the shield overrides it with a safer action.

---

## Safety Shield Logic

### Step 1 — DDQN Proposes Action

The RL agent first proposes an evader action.

### Step 2 — Safety Evaluation

The shield predicts the next state and evaluates:

```math
V(next\_state)
```

If:

```math
V(next\_state) \ge threshold
```

the action is executed.

Otherwise, intervention occurs.

### Step 3 — Safe Action Search

The shield evaluates all possible evader actions and chooses:

- A safe action if one exists
- Otherwise, the **least unsafe action**

---

# IV. Adaptive HJ-Inspired Safety Shield (Our Extension)

This work extends the RSS 2021 framework using an **adaptive robust HJ-inspired safety shield**.

The implementation includes:

## 1. Adaptive Thresholding

Instead of using a fixed intervention threshold, the shield adapts dynamically:

```math
threshold = \max(\kappa \cdot \bar{\xi}, \epsilon)
```

Where:

- `ε` = minimum safety threshold
- `ξ̄` = maximum observed safety slack
- `κ` = scaling factor

This makes the shield dynamically adjust to uncertainty.

---

## 2. Soft Intervention / Blending Zone

Rather than abrupt switching, a **probabilistic blending region** is used near unsafe boundaries.

Benefits:

- Reduced oscillations
- Smoother control
- Less conservative behavior

---

## 3. Multi-Step Lookahead Rollout

The shield evaluates future safety over a rollout horizon instead of only one step ahead.

This enables **trajectory-aware decisions** rather than myopic corrections.

---

## 4. Robust Worst-Case Pursuer Handling

The pursuer is modeled adversarially.

For every evader action, the environment simulates the **worst-case pursuer response**, making the learned policy robust.

---

## 5. Unavoidable Failure Handling

If no safe action exists:

- The event is logged
- The least unsafe action is executed

This prevents silent failures.

---

# V. Safety Shield Code Structure

The shield is implemented in:

```text
safety_shield.py
```

Three progressively stronger shield versions are implemented.

## Level 0 — `BasicSafetyShield`

Features:

- One-step safety checking
- Action override mechanism
- Adaptive threshold support
- Safety slack tracking

---

## Level 1 — `AdaptiveSafetyShield`

Adds:

- Adaptive thresholding
- Soft blending near unsafe regions
- Slack decay
- Intervention statistics

---

## Level 2 — `RobustHJCBFSafetyShield`

Adds:

- Multi-step rollout reasoning
- Trajectory-aware safety
- Safety statistics
- Unavoidable-event handling

---

# VI. Repository Structure

```text
project/
│── gym_reachability/
│── RARL/safety_shield.py
│── sim_car_pe_old_without_shield.py
│── sim_car_pe_withshield.py
│── requirements.txt
│── README.md
```

---

# VII. Dependencies

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

# VIII. How to Run

Each script automatically creates an experiment folder under:

```text
experiments/
```

Generated outputs include:

- Training curves
- Reward plots
- Success statistics
- Trajectory visualizations
- Model checkpoints
- Safety metrics

A `train.pkl` file is also generated containing:

- Training loss
- Training accuracy
- Rollout outcomes
- Grid-based trajectory analysis
- Action selection maps

## 1. Dubins Car Without Shield

```bash
python3 sim_car_pe_old_without_shield.py -sf
```

## 2. Dubins Car With Shield

```bash
python3 sim_car_pe_withshield.py -sf
```

---

# IX. Results

The shielded agent demonstrates:

-Reduced safety violations  
-Safer exploration during training  
-Improved robustness near unsafe regions  
-Better reach-avoid performance under disturbances

However, excessive interventions may reduce task success due to the **safety-performance tradeoff**.

---

# X. Future Improvements

Potential future directions:

- Continuous action spaces
- SAC/PPO-based controllers
- Hybrid RL + Control frameworks
- Real-world robotic deployment
- Multi-agent adversarial learning

---

# XI. Citation

If you use this repository, please cite:

```bibtex
@article{rss2021,
  title={Safety and Liveness Guarantees through Reach-Avoid Reinforcement Learning},
  year={2021}
}
```

---
