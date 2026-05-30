"""
safety_shield.py  —  Robust HJ-CBF Safety Shield for Dubins Car Pursuit-Evasion

Implements the Robust HJ-CBF Safety Filter from:
  "Safe Learning in the Real World via Adaptive Shielding with
   Hamilton-Jacobi Reachability" (Lu et al., L4DC 2025)

Equation references throughout refer to that paper.

Core idea
---------
At every timestep the DDQN proposes an evader action.  The shield
checks V(s) = safety_margin(s):
  - V(s) > threshold  →  allow the DDQN action (safe enough)
  - V(s) ≤ threshold  →  override with the safest evader action
                          and tighten the threshold for next time

Three levels are provided so you can start simple and layer on
complexity as you validate each stage:

  Level 0  BasicSafetyShield          — fixes all bugs, minimal logic
  Level 1  AdaptiveSafetyShield       — adds adaptive threshold (paper eq. 10)
  Level 2  RobustHJCBFSafetyShield    — full paper method + lookahead rollout
"""

import numpy as np
from collections import deque


# ─────────────────────────────────────────────────────────────────────────────
# Level 0 — Basic shield (all original bugs fixed, nothing extra)
# ─────────────────────────────────────────────────────────────────────────────

class BasicSafetyShield:



    def __init__(self, env, epsilon=0.05, kappa=1.0):
        """
        Args:
            env:      The DubinsCarPEEnv instance.
            epsilon:  Initial safety threshold.  Shield fires when V(s) < epsilon.
                      Typical range: 0.02 – 0.15.
            kappa:    Multiplier on max slack for adaptive threshold (eq. 10).
                      Set kappa=0 to disable adaptation and use fixed epsilon.
        """
        self.env = env
        self.epsilon = epsilon
        self.kappa = kappa

        # Tracks ξ_bar = max slack needed so far  (paper eq. 10)
        self.max_violation = 0.0

        # ── BUG 2 FIX: iterate over EVADER actions only (0, 1, 2) ──
        self.num_evader_actions = env.numActionList[0]   # 3
        self.evader_action_list = list(range(self.num_evader_actions))

        # Logging
        self.n_interventions = 0
        self.n_steps = 0

    # ── BUG 1 FIX: V(s) = safety_margin only ─────────────────────────────────
    def hj_value(self, state):
        """
        Safety value V(s).

        Returns
        -------
        float   positive  →  evader is safe (farther from capture/boundary)
                negative  →  evader is inside failure set
        """
        return self.env.safety_margin(state)

    def _current_threshold(self):
        """Adaptive threshold: max(κ · ξ_bar, ε)  [paper eq. 10]."""
        return max(self.kappa * self.max_violation, self.epsilon)

    def should_intervene(self, state):
        """True if the shield should override the DDQN action."""
        return self.hj_value(state) < self._current_threshold()

    def find_safest_evader_action(self, state):
        """
        One-step lookahead over evader actions using simulate_step.

        simulate_step already internalises worst-case pursuer (it loops
        over all pursuer actions and picks the minimum), so we only need
        to iterate over the 3 evader actions here.

        Returns
        -------
        best_action  int    evader action index (0-2)
        best_value   float  V(next_state) for that action
        """
        best_action = 0
        best_value = -np.inf

        for evader_action in self.evader_action_list:
            next_state = self.env.simulate_step(state, evader_action)
            value = self.hj_value(next_state)

            if value > best_value:
                best_value = value
                best_action = evader_action

        return best_action, best_value

    # ── BUG 3 FIX: track actual CBF slack ξ ──────────────────────────────────
    def update_slack(self, next_state):
        """
        Computes ξ = max(0, -V(s'))  — how far into the unsafe region
        the system moved despite the shield acting.  Accumulates ξ_bar.

        """
        V_next = self.hj_value(next_state)
        xi = max(0.0, -V_next)
        self.max_violation = max(self.max_violation, xi)

    def get_safe_evader_action(self, state, ddqn_evader_action):
        """
        Main shield entry point.

        Parameters
        ----------
        state               np.ndarray, shape (6,) — full env state
        ddqn_evader_action  int — evader action index proposed by DDQN (0-2)

        Returns
        -------
        int  — safe evader action to execute (may equal ddqn_evader_action)
"""

        self.n_steps += 1
        threshold = self._current_threshold()

        # CHECK DDQN ACTION FIRST
        next_state_ddqn = self.env.simulate_step(state, ddqn_evader_action)

        if self.hj_value(next_state_ddqn) >= threshold:
            return ddqn_evader_action

        # OTHERWISE FIND SAFE ACTION
        safe_actions = []
        best_action = 0
        best_value = -np.inf

        for a in self.evader_action_list:
            next_state = self.env.simulate_step(state, a)
            value = self.hj_value(next_state)

            if value >= threshold:
                safe_actions.append(a)

            if value > best_value:
                best_value = value
                best_action = a

        self.n_interventions += 1

        if safe_actions:
            return safe_actions[0]   # or smarter selection
        else:
            return best_action      # fallback

    @property
    def intervention_rate(self):
        """Fraction of steps where the shield overrode the DDQN."""
        if self.n_steps == 0:
            return 0.0
        return self.n_interventions / self.n_steps

    def reset_episode(self):
        """Call at the start of each episode (optional — doesn't reset max_violation)."""
        pass  # max_violation intentionally persists across episodes

    def __repr__(self):
        return (
            f"BasicSafetyShield(ε={self.epsilon}, κ={self.kappa}, "
            f"ξ_bar={self.max_violation:.4f}, "
            f"threshold={self._current_threshold():.4f}, "
            f"interventions={self.n_interventions}/{self.n_steps})"
        )

# Level 1 — Adaptive shield  (adds value-based softening near boundary)


class AdaptiveSafetyShield(BasicSafetyShield):
    """
    Extends BasicSafetyShield with:

    1. Soft blending zone
       Rather than a hard switch at threshold, linearly blend the shield's
       safe action with the DDQN action in a band [threshold, threshold + blend_band].
       Outside the band: full DDQN.  Inside: full shield.  In between: probabilistic.
       This reduces jitter at the boundary (mentioned as a limitation of hard
       switching in Section 3.3 of the paper).

    2. Episode-level slack decay
       Optionally decay max_violation slowly over time so the shield becomes
       less conservative if the environment turns out to be safer than feared.

    3. Intervention logging per episode
       Track how many interventions happen each episode so you can monitor
       convergence in your training script.
    """

    def __init__(
        self, env, epsilon=0.05, kappa=1.0,
        blend_band=0.05, slack_decay=0.9999
    ):
        """
        Args:
            blend_band:   Width of the probabilistic blending zone above threshold.
                          Set to 0 to recover hard switching (same as BasicSafetyShield).
            slack_decay:  Each step, max_violation *= slack_decay.
                          1.0 = no decay (paper default).
                          0.999 = slow forgetting over ~1000 steps.
        """
        super().__init__(env, epsilon=epsilon, kappa=kappa)
        self.blend_band = blend_band
        self.slack_decay = slack_decay

        # Per-episode tracking
        self.episode_interventions = []
        self._ep_interventions = 0

    def get_safe_evader_action(self, state, ddqn_evader_action):
        """
        Same as BasicSafetyShield but with soft blending near the boundary.
        """
        self.n_steps += 1
        threshold = self._current_threshold()

        # Evaluate DDQN action
        next_state_ddqn = self.env.simulate_step(state, ddqn_evader_action)
        V_ddqn = self.hj_value(next_state_ddqn)

        # Fully safe → allow
        if V_ddqn >= threshold + self.blend_band:
            return ddqn_evader_action

        # Compute safest action
        safe_candidates = []
        best_action = None
        best_value = -np.inf

        for a in self.evader_action_list:
            ns = self.env.simulate_step(state, a)
            v = self.hj_value(ns)

            if v >= threshold:
                safe_candidates.append(a)

            if v > best_value:
                best_value = v
                best_action = a

        if safe_candidates:
            safe_action = max(safe_candidates, key=lambda a: self.hj_value(self.env.simulate_step(state, a)))
        else:
            safe_action = best_action

        # Unsafe → always shield
        if V_ddqn < threshold:
            self.n_interventions += 1
            self._ep_interventions += 1
            return safe_action

        # Blending zone
        # V_ddqn ∈ [threshold, threshold + blend_band]
        danger = 1.0 - (V_ddqn - threshold) / self.blend_band

        if np.random.rand() < danger:
            self.n_interventions += 1
            self._ep_interventions += 1
            return safe_action

        return ddqn_evader_action

    def update_slack(self, next_state):
        """Updates slack with optional decay."""
        super().update_slack(next_state)
        self.max_violation *= self.slack_decay

    def reset_episode(self):
        """Call at episode end to log per-episode intervention count."""
        self.episode_interventions.append(self._ep_interventions)
        self._ep_interventions = 0

    def mean_interventions_per_episode(self, last_n=50):
        hist = self.episode_interventions[-last_n:]
        return np.mean(hist) if hist else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Level 2 — Full Robust HJ-CBF shield  (multi-step rollout lookahead)
# ─────────────────────────────────────────────────────────────────────────────

class RobustHJCBFSafetyShield(AdaptiveSafetyShield):
    """

    Adds over AdaptiveSafetyShield:

    1. Multi-step lookahead
       Instead of evaluating V only one step ahead, roll out H steps using
       the shield's own greedy-safe policy and take the minimum V seen.
       This makes the shield reason about trajectories, not just single steps.
       In the paper this is implicit in the BRT computation; here we approximate
       it with a short greedy rollout.

    2. Running safety statistics
       Tracks a sliding window of V values so you can detect if the environment
       is drifting (e.g. pursuer getting faster) and report it.

    3. Hard-stop fallback
       If ALL evader actions lead to V < 0 (imminent capture unavoidable),
       log the event but still return the least-bad action.  This prevents
       silent failures where the shield does nothing because no action is safe.
    """

    def __init__(
        self, env, epsilon=0.01, kappa=1.0,
        blend_band=0.05, slack_decay=0.9999,
        horizon=2, window_size=500
    ):
        """
        Args:
            horizon:      Number of lookahead steps for rollout evaluation.
                          1 = same as AdaptiveSafetyShield.
            window_size:  Sliding window length for safety statistics.
        """
        super().__init__(
            env, epsilon=epsilon, kappa=kappa,
            blend_band=blend_band, slack_decay=slack_decay
        )
        self.horizon = horizon
        self._v_window = deque(maxlen=window_size)

        # Counters
        self.n_unavoidable = 0   # times no safe action existed

    def _rollout_value(self, state, first_evader_action):
        """
        Simulate H steps: take first_evader_action at step 0, then greedily
        choose the safest evader action at each subsequent step.

        Returns min V seen across the rollout (worst-case safety measure).
        """
        threshold = self._current_threshold()

        current_state = self.env.simulate_step(state, first_evader_action)
        min_V = self.hj_value(current_state)

        for _ in range(self.horizon - 1):

            safe_candidates = []
            best_a = None
            best_v = -np.inf

            for a in self.evader_action_list:
                ns = self.env.simulate_step(current_state, a)
                v = self.hj_value(ns)

                if v >= threshold:
                    safe_candidates.append(a)

                if v > best_v:
                    best_v = v
                    best_a = a

            if safe_candidates:
                chosen_a = safe_candidates[0]
            else:
                chosen_a = best_a

            current_state = self.env.simulate_step(current_state, chosen_a)
            min_V = min(min_V, self.hj_value(current_state))

        return min_V

    def find_safest_evader_action(self, state):
        """
        Override: use multi-step rollout instead of one-step lookahead.
        """
        best_action = 0
        best_value = -np.inf
        all_unsafe = True

        for evader_action in self.evader_action_list:
            value = self._rollout_value(state, evader_action)

            threshold = self._current_threshold()

            if value >= threshold:
                all_unsafe = False
            if value > best_value:
                best_value = value
                best_action = evader_action

        if all_unsafe:
            self.n_unavoidable += 1

        return best_action, best_value

    def update_slack(self, next_state):
        """Updates slack and records V for statistics."""
        super().update_slack(next_state)
        self._v_window.append(self.hj_value(next_state))

    @property
    def mean_safety_value(self):
        """Mean V over recent steps. Should stay positive during safe learning."""
        return float(np.mean(self._v_window)) if self._v_window else 0.0

    @property
    def safety_violation_rate(self):
        """Fraction of recent steps where V(next_state) < 0."""
        if not self._v_window:
            return 0.0
        return sum(v < 0 for v in self._v_window) / len(self._v_window)

    def __repr__(self):
        return (
            f"RobustHJCBFSafetyShield("
            f"ε={self.epsilon}, κ={self.kappa}, H={self.horizon}, "
            f"ξ_bar={self.max_violation:.4f}, "
            f"threshold={self._current_threshold():.4f}, "
            f"mean_V={self.mean_safety_value:.4f}, "
            f"viol_rate={self.safety_violation_rate:.3f}, "
            f"unavoidable={self.n_unavoidable}, "
            f"interventions={self.n_interventions}/{self.n_steps})"
        )