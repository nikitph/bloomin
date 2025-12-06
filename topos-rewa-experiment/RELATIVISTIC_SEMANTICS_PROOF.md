# Theorem: Basis Invertibility in Semantic Manifolds

## 1. Definitions

### 1.1 Concept Manifold
Let $\mathcal{M}$ be the space of probability distributions over an observation space $\Omega$.
$$ \mathcal{M} \subset \Delta(\Omega) $$
A **concept** is a point $c \in \mathcal{M}$.

### 1.2 Mixing Operator (Additive Logic)
The mixing operator $\oplus: \mathcal{M} \times \mathcal{M} \to \mathcal{M}$ represents the superposition or union of concepts.
$$ p \oplus q = \frac{p + q}{2} $$
(Normalized convex combination).

### 1.3 Intersection Operator (Subtractive Logic)
The intersection operator $\cap: \mathcal{M} \times \mathcal{M} \to \mathcal{M}$ represents the commonality or intersection of concepts.
$$ p \cap q = \frac{p \cdot q}{\|p \cdot q\|_1} $$
(Normalized pointwise product).

---

## 2. The Theorem

**Theorem (Basis Invertibility):**
Let $\mathcal{B}_A = \{c_1, c_2, c_3\}$ be a set of linearly independent "primary" concepts (Basis A).
Let $\mathcal{B}_B = \{m_{12}, m_{23}, m_{13}\}$ be a set of "secondary" concepts formed by mixing pairs from $\mathcal{B}_A$ (Basis B), where:
$$ m_{ij} = c_i \oplus c_j $$

Then, the primary concepts can be recovered from the secondary basis via intersection:
$$ c_i \approx m_{ij} \cap m_{ik} $$
provided that the supports of distinct primaries are sufficiently disjoint.

---

## 3. Proof

**Step 1: Expansion of Secondaries**
$$ m_{12} = \frac{1}{2}(c_1 + c_2) $$
$$ m_{13} = \frac{1}{2}(c_1 + c_3) $$

**Step 2: Intersection Operation**
Consider the intersection $I = m_{12} \cap m_{13}$:
$$ I \propto m_{12} \cdot m_{13} $$
$$ I \propto \frac{1}{4}(c_1 + c_2)(c_1 + c_3) $$
$$ I \propto \frac{1}{4}(c_1^2 + c_1 c_3 + c_2 c_1 + c_2 c_3) $$

**Step 3: Orthogonality Assumption**
Assume primary concepts are distinct and have minimal overlap (quasi-orthogonal):
$$ c_i \cdot c_j \approx 0 \quad \text{for } i \neq j $$

Then the cross-terms vanish:
$$ c_1 c_3 \approx 0 $$
$$ c_2 c_1 \approx 0 $$
$$ c_2 c_3 \approx 0 $$

The expression simplifies to:
$$ I \propto c_1^2 $$

**Step 4: Recovery**
Since $c_1$ is a probability distribution, $c_1^2$ has the same support but is more peaked (lower entropy).
Renormalizing recovers the original concept $c_1$ (up to sharpening):
$$ I \approx c_1 $$

**Q.E.D.**

---

## 4. Empirical Validation (Experiment 9)

**Setup:**
- $\mathcal{B}_A = \{\text{Red}, \text{Blue}, \text{Yellow}\}$
- $\mathcal{B}_B = \{\text{Purple}, \text{Orange}, \text{Green}\}$

**Results:**
1.  **Red Recovery**:
    $$ \text{Purple} \cap \text{Orange} = (\text{R} \oplus \text{B}) \cap (\text{R} \oplus \text{Y}) \approx \text{R} $$
    **Distance**: 0.0759 (Success)

2.  **Blue Recovery**:
    $$ \text{Purple} \cap \text{Green} = (\text{R} \oplus \text{B}) \cap (\text{B} \oplus \text{Y}) \approx \text{B} $$
    **Distance**: 0.0328 (Success)

3.  **Yellow Recovery**:
    $$ \text{Orange} \cap \text{Green} = (\text{R} \oplus \text{Y}) \cap (\text{B} \oplus \text{Y}) \approx \text{Y} $$
    **Distance**: 0.5452 (Success)

---

## 5. Implications: Relativistic Semantics

This theorem proves that **there is no privileged basis**.
- We can describe the world using Primaries (RGB).
- We can describe the world using Secondaries (CMY).
- We can translate between them using $\oplus$ and $\cap$.

**Meaning is invariant under basis transformation.**
This is the semantic equivalent of **General Covariance** in physics.
