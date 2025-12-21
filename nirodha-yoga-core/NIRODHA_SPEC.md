# Nirodha Specification: Formalized Breakthrough

## Part I — Axioms

### Axiom 1 — Cognitive State
There exists a cognitive state $C_t \in \mathbb{R}^n$ evolving in discrete time under internal dynamics.

### Axiom 2 — Anchor (Observer Reference)
There exists a fixed reference state $C_0 \in \mathbb{R}^n$ such that all regulation is defined relative to $C_0$. No claim is made that $C_0$ is correct, optimal, or true.

### Axiom 3 — Update Dynamics
Unregulated dynamics evolve as $\tilde{C}_{t+1} = C_t + \Delta_t$, where $\Delta_t$ may be stochastic, adversarial, or unbounded.

### Axiom 4 — Nirodha Operator (Non-Destructive Suppression)
There exists a suppression operator $\mathcal{N}_\beta : \mathbb{R}^n \to \mathbb{R}^n$ applied elementwise, satisfying:
1. **Identity near zero**: $\mathcal{N}_\beta(x) = x + O(x^2)$
2. **Global boundedness**: $\|\mathcal{N}_\beta(x)\| \le \frac{1}{\beta}$
3. **Contractivity**: $\|\mathcal{N}_\beta(x) - \mathcal{N}_\beta(y)\| \le \|x - y\|$

Instantiation: $\mathcal{N}_\beta(x) = \frac{x}{1 + \beta |x|}$

### Axiom 5 — Regulated Dynamics
The regulated system evolves as: $C_{t+1} = C_0 + \mathcal{N}_\beta(C_t + \Delta_t - C_0)$

### Axiom 6 — Energy Function (Lyapunov Candidate)
Define empirical energy: $E_t = \|C_t - C_0\|^2$

---

## Part II — Theorems

### Theorem 1 — Lyapunov Stability (Expectation)
For any update distribution $\Delta_t$ with finite second moment:
$\mathbb{E}[E_{t+1} - E_t \mid C_t] \le 0 \quad \text{for all } \beta > 0$
*Interpretation: The anchor $C_0$ is a Lyapunov-stable attractor.*

### Theorem 2 — Variance Boundedness Under Scale
Let $S$ denote internal scale. For fixed $\beta > 0$:
$\text{Var}(\Delta E_t) = O(1) \quad \text{as } S \to \infty$
while for $\beta = 0$:
$\text{Var}(\Delta E_t) \to \infty \quad \text{superlinearly in } S$
*Interpretation: Nirodha removes catastrophic tail risk from scaling.*

### Theorem 3 — Architecture Independence
Theorems 1 and 2 hold empirically for residual additive systems (Transformers), iterative stochastic systems (Diffusion), and message-passing systems (GNNs).

### Theorem 4 — Stability–Correctness Separation
Stability holds for arbitrary anchors $C_0$, including corrupted ones.

### Corollary — Sensitivity Becomes a Free Variable
Once instability is removed as a failure mode, sensitivity $\perp$ instability. Scale can increase sensitivity without inducing divergence.

---

## Part III — Safe Operating Envelope (SOE) Specification

The system must operate within the following boundaries:

1. **SOE-1: Mean Energy Dissipation**: $\mathbb{E}[\Delta E_t] \le 0$
2. **SOE-2: Variance Bound**: $\text{Var}(\Delta E_t) \le V_{\max}$
3. **SOE-3: No Limit Cycles**: $\lim_{k \to \infty} \|C_{t+k} - C_t\| = 0$
4. **SOE-4: Observer Invariance**: $\|C_0^{(t)} - C_0^{(t+k)}\| = 0$

### Operating Regions
- **Region I (Under-Damped)**: β too small. Early sensitivity but high risk of divergence. [UNSAFE]
- **Region II (Critical Window)**: β ≈ β*. Delayed but clean onset. Lyapunov stability holds. [OPTIMAL]
- **Region III (Over-Damped)**: β too large. Stability guaranteed but no breakthrough capability. [INERT]
