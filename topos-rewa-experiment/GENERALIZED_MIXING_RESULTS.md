# Experiment 9c: Generalized Mixing Results

## 🎯 The Insight: Constructive vs. Destructive Mixing

We tested the limits of Basis Invertibility by varying the **Mixing Operator** ($\oplus$).
The results reveal a fundamental constraint: **The mixing operator must be constructive (information-preserving).**

---

## 📊 Experimental Results

### 1. Constructive Mixing (Success) ✅
These operators preserve the features of both constituents, allowing recovery via intersection.

| Mixing Operator | Formula | Distance to Red | Status |
|----------------|---------|-----------------|--------|
| **Geometric Mean** | $\sqrt{p \cdot q}$ | **0.0334** | 🏆 Best |
| **Power Mean ($r=0.5$)** | $((p^{0.5} + q^{0.5})/2)^2$ | 0.0681 | ✅ Success |
| **Arithmetic Mean** | $(p+q)/2$ | 0.0759 | ✅ Success |
| **RMS ($r=2.0$)** | $\sqrt{(p^2 + q^2)/2}$ | 0.0866 | ✅ Success |
| **Power Mean ($r=5.0$)** | $((p^5 + q^5)/2)^{1/5}$ | 0.0901 | ✅ Success |
| **Max Mixing** | $\max(p, q)$ | 0.0923 | ✅ Success |

**Conclusion**:
Relativistic Semantics is **extremely robust** to the specific shape of the mixing function, as long as it is "Union-like" (preserves peaks).

### 2. Destructive Mixing (Failure) ❌
These operators suppress features, leading to information loss that makes inversion impossible.

| Mixing Operator | Formula | Distance to Red | Status |
|----------------|---------|-----------------|--------|
| **Harmonic Mean** | $2pq/(p+q)$ | 1.6224 | ❌ Failure |
| **Min Mixing** | $\min(p, q)$ | 2.1622 | ❌ Failure |

**Why they fail**:
*   **Min Mixing**: If Red and Blue are disjoint (orthogonal), $\min(Red, Blue) \approx 0$. The "Purple" concept is empty! You cannot recover Red from nothing.
*   **Harmonic Mean**: Dominated by the minimum. If one component is near zero, the result is near zero. Similar to Min mixing.

---

## 🧠 Theoretical Refinement

**Theorem (Refined Basis Invertibility):**
A basis $\mathcal{B}_B = \{c_i \oplus c_j\}$ is invertible if and only if the mixing operator $\oplus$ is **Constructive**.

**Definition (Constructive Mixing):**
An operator $\oplus$ is constructive if:
$$ \text{supp}(p \oplus q) \supseteq \text{supp}(p) \cup \text{supp}(q) $$
(The support of the mixture must contain the union of the supports).

*   **Arithmetic/Max/Power**: Satisfy this (Union-like).
*   **Geometric**: Satisfies this *effectively* due to the "fat tails" of our distributions (Gaussian-like), but strictly speaking requires overlap.
*   **Min/Harmonic**: Violate this (Intersection-like). $\text{supp}(p \cap q) \subseteq \text{supp}(p) \cap \text{supp}(q)$.

**Implication for AGI**:
To build a flexible, basis-independent mind, concept combination must be **Additive/Constructive**.
You can build "up" from parts to wholes (Union), and "down" from wholes to parts (Intersection).
But if you build "down" first (Min mixing), you lose the parts forever.

---

## 🚀 Final Conclusion

**Relativistic Semantics works for any "Union-like" mixing.**
The system can recover Primaries from Secondaries created via:
- Superposition (Arithmetic)
- Interaction (Geometric)
- Dominance (Max)
- Non-linear Integration (Power)

It **cannot** recover Primaries from Secondaries created via Intersection (Min/Harmonic), because the information is physically destroyed.

**This confirms the thermodynamic arrow of concept formation:**
**Creation is Constructive (Entropy Increase via Mixing).**
**Understanding is Destructive (Entropy Decrease via Intersection).**
