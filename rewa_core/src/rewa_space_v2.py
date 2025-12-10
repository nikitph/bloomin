"""
Rewa-Space Projection Head v2

More efficient training using direct optimization.
Uses a simpler but more effective approach:
1. Learn a transformation that maximizes negative similarity for contradiction pairs
2. Use gradient descent with momentum
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer


@dataclass
class TrainingPair:
    """A training pair for contrastive learning."""
    anchor: str
    positive: str
    negative: str
    axis_name: str


class RewaSpaceV2:
    """
    Rewa-space projection head v2.

    Simpler architecture: Linear transformation + normalization
    More aggressive training for antipodal enforcement.
    """

    def __init__(
        self,
        base_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        output_dim: int = 384  # Same as input to preserve info
    ):
        self.base_model = SentenceTransformer(base_model_name)
        self.input_dim = self.base_model.get_sentence_embedding_dimension()
        self.output_dim = output_dim

        # Initialize transformation: identity + small noise
        self.W = np.eye(self.input_dim, self.output_dim) + np.random.randn(self.input_dim, self.output_dim) * 0.01

        # Momentum for training
        self.momentum = np.zeros_like(self.W)

        self.trained = False

    def _get_base_embedding(self, text: str) -> np.ndarray:
        """Get normalized base embedding."""
        emb = self.base_model.encode(text, convert_to_numpy=True)
        return emb / (np.linalg.norm(emb) + 1e-10)

    def _get_base_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get normalized base embeddings for batch."""
        embs = self.base_model.encode(texts, convert_to_numpy=True)
        return embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10)

    def project(self, text: str) -> np.ndarray:
        """Project text into Rewa-space."""
        base_emb = self._get_base_embedding(text)
        projected = base_emb @ self.W
        return projected / (np.linalg.norm(projected) + 1e-10)

    def project_embedding(self, emb: np.ndarray) -> np.ndarray:
        """Project embedding into Rewa-space."""
        projected = emb @ self.W
        norms = np.linalg.norm(projected, axis=-1, keepdims=True) if projected.ndim > 1 else np.linalg.norm(projected)
        return projected / (norms + 1e-10)

    def project_batch(self, texts: List[str]) -> np.ndarray:
        """Project multiple texts."""
        base_embs = self._get_base_embeddings(texts)
        projected = base_embs @ self.W
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        return projected / (norms + 1e-10)

    def compute_loss_and_grad(
        self,
        pos_base: np.ndarray,
        neg_base: np.ndarray,
        margin: float = 0.9
    ) -> Tuple[float, np.ndarray]:
        """
        Compute loss and gradient for a batch of (positive, negative) pairs.

        Loss: We want pos and neg to be antipodal after projection.
        L = max(0, margin + sim(proj(pos), proj(neg)))

        Where sim = cosine similarity = dot product (since normalized)
        """
        batch_size = len(pos_base)

        # Forward pass
        pos_proj = pos_base @ self.W  # (batch, output_dim)
        neg_proj = neg_base @ self.W

        # Normalize
        pos_norm = np.linalg.norm(pos_proj, axis=1, keepdims=True)
        neg_norm = np.linalg.norm(neg_proj, axis=1, keepdims=True)

        pos_unit = pos_proj / (pos_norm + 1e-10)
        neg_unit = neg_proj / (neg_norm + 1e-10)

        # Similarity (should be negative for antipodal)
        sims = np.sum(pos_unit * neg_unit, axis=1)  # (batch,)

        # Hinge loss: we want sim < -margin (antipodal)
        # So loss = max(0, margin + sim)
        losses = np.maximum(0, margin + sims)
        total_loss = np.mean(losses)

        # Gradient computation (simplified)
        # For active samples (where loss > 0)
        active = (margin + sims) > 0

        if not np.any(active):
            return total_loss, np.zeros_like(self.W)

        # Gradient of normalized dot product is complex, use numerical approx for simplicity
        # But approximate: d(sim)/dW ≈ (pos_unit @ neg_base.T + neg_unit @ pos_base.T) / batch
        # This is an approximation that pushes them apart

        grad = np.zeros_like(self.W)
        for i in range(batch_size):
            if active[i]:
                # Push pos and neg apart
                # Gradient direction that decreases similarity
                grad += np.outer(pos_base[i], neg_unit[i]) / batch_size
                grad += np.outer(neg_base[i], pos_unit[i]) / batch_size

        return total_loss, grad

    def train(
        self,
        training_pairs: List[Tuple[str, str]],
        epochs: int = 200,
        learning_rate: float = 0.1,
        momentum_coeff: float = 0.9,
        margin: float = 0.9,
        verbose: bool = True
    ) -> List[Dict[str, float]]:
        """
        Train on (positive, negative) pairs.

        Args:
            training_pairs: List of (positive_text, negative_text) tuples
            epochs: Number of epochs
            learning_rate: Learning rate
            momentum_coeff: Momentum coefficient
            margin: Target margin for antipodal (sim < -margin)
            verbose: Print progress
        """
        # Pre-compute base embeddings
        pos_texts = [p[0] for p in training_pairs]
        neg_texts = [p[1] for p in training_pairs]

        pos_base = self._get_base_embeddings(pos_texts)
        neg_base = self._get_base_embeddings(neg_texts)

        history = []

        for epoch in range(epochs):
            loss, grad = self.compute_loss_and_grad(pos_base, neg_base, margin)

            # Momentum update
            self.momentum = momentum_coeff * self.momentum - learning_rate * grad
            self.W += self.momentum

            # Compute metrics
            pos_proj = self.project_embedding(pos_base)
            neg_proj = self.project_embedding(neg_base)
            sims = np.sum(pos_proj * neg_proj, axis=1)
            angles = np.degrees(np.arccos(np.clip(sims, -1, 1)))
            mean_angle = np.mean(angles)

            history.append({
                'epoch': epoch,
                'loss': float(loss),
                'mean_angle': float(mean_angle),
                'mean_sim': float(np.mean(sims))
            })

            if verbose and epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss={loss:.4f}, MeanAngle={mean_angle:.1f}°, MeanSim={np.mean(sims):.3f}")

        self.trained = True
        return history

    def evaluate(self, test_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Evaluate on test pairs."""
        results = []

        for pos, neg in test_pairs:
            # Base angles
            pos_base = self._get_base_embedding(pos)
            neg_base = self._get_base_embedding(neg)
            base_sim = np.dot(pos_base, neg_base)
            base_angle = np.degrees(np.arccos(np.clip(base_sim, -1, 1)))

            # Rewa angles
            pos_rewa = self.project(pos)
            neg_rewa = self.project(neg)
            rewa_sim = np.dot(pos_rewa, neg_rewa)
            rewa_angle = np.degrees(np.arccos(np.clip(rewa_sim, -1, 1)))

            results.append({
                'pos': pos,
                'neg': neg,
                'base_angle': base_angle,
                'rewa_angle': rewa_angle,
                'improvement': rewa_angle - base_angle
            })

        return {
            'pairs': results,
            'mean_base_angle': np.mean([r['base_angle'] for r in results]),
            'mean_rewa_angle': np.mean([r['rewa_angle'] for r in results]),
            'mean_improvement': np.mean([r['improvement'] for r in results]),
            'antipodal_rate': np.mean([r['rewa_angle'] > 135 for r in results])
        }

    def save(self, filepath: str):
        """Save transformation matrix."""
        np.save(filepath, self.W)

    def load(self, filepath: str):
        """Load transformation matrix."""
        self.W = np.load(filepath)
        self.trained = True


def generate_negation_pairs() -> List[Tuple[str, str]]:
    """Generate pairs of (statement, negation)."""
    pairs = [
        # Short word pairs (critical for contradiction detection)
        ("yes", "no"),
        ("true", "false"),
        ("good", "bad"),
        ("right", "wrong"),
        ("correct", "incorrect"),
        ("valid", "invalid"),
        ("safe", "unsafe"),
        ("legal", "illegal"),
        ("approved", "rejected"),
        ("accepted", "denied"),
        ("confirmed", "not confirmed"),
        ("agree", "disagree"),
        ("positive", "negative"),
        ("success", "failure"),
        ("increase", "decrease"),
        ("A", "not A"),
        ("X", "not X"),
        ("allowed", "forbidden"),
        ("permitted", "prohibited"),
        ("active", "inactive"),

        # Logical negations
        ("The statement is true", "The statement is false"),
        ("The answer is yes", "The answer is no"),
        ("The system is active", "The system is inactive"),
        ("The door is open", "The door is closed"),
        ("The light is on", "The light is off"),
        ("The patient is alive", "The patient is dead"),
        ("The test passed", "The test failed"),
        ("The product is available", "The product is unavailable"),
        ("Access is granted", "Access is denied"),
        ("The claim is valid", "The claim is invalid"),

        # Sentiment
        ("I love this product", "I hate this product"),
        ("Excellent quality", "Terrible quality"),
        ("Highly recommend", "Do not recommend"),
        ("Very satisfied", "Very disappointed"),
        ("Best experience", "Worst experience"),

        # Safety/Risk
        ("Safe and approved", "Dangerous and risky"),
        ("Low risk", "High risk"),
        ("Meets safety standards", "Fails safety standards"),
        ("Clinically tested", "Not tested"),

        # Compliance
        ("Fully compliant", "Violates regulations"),
        ("Legal and authorized", "Illegal and unauthorized"),
        ("Meets requirements", "Does not meet requirements"),
        ("Approved by audit", "Failed audit"),

        # Approval
        ("Application approved", "Application rejected"),
        ("Eligible candidate", "Ineligible candidate"),
        ("Proposal accepted", "Proposal declined"),
        ("Authorized", "Unauthorized"),

        # Financial
        ("Profitable growth", "Financial losses"),
        ("Strong performance", "Poor performance"),
        ("Stable investment", "Risky investment"),
        ("Good credit", "Bad credit"),

        # More semantic opposites
        ("Increase in value", "Decrease in value"),
        ("Above threshold", "Below threshold"),
        ("Within limits", "Exceeds limits"),
        ("Confirmed and verified", "Unconfirmed and unverified"),
        ("Accurate data", "Inaccurate data"),
        ("Reliable source", "Unreliable source"),
        ("Positive outcome", "Negative outcome"),
        ("Success achieved", "Failure occurred"),
    ]

    return pairs


def main():
    """Train and evaluate Rewa-space v2."""
    print("="*70)
    print("REWA-SPACE V2 TRAINING")
    print("="*70)

    # Test pairs
    test_pairs = [
        ("The sky is blue", "The sky is not blue"),
        ("I love this product", "I hate this product"),
        ("The answer is yes", "The answer is no"),
        ("The patient is alive", "The patient is dead"),
        ("The test passed", "The test failed"),
        ("Access is granted", "Access is denied"),
        ("The claim is valid", "The claim is invalid"),
        ("Application approved", "Application rejected"),
        ("Safe for use", "Dangerous and harmful"),
        ("Fully compliant", "Violates regulations"),
    ]

    # Initialize
    print("\nInitializing Rewa-Space v2...")
    rewa = RewaSpaceV2(output_dim=384)

    # Evaluate before
    print("\n--- BEFORE TRAINING ---")
    before = rewa.evaluate(test_pairs)
    print(f"Mean base angle: {before['mean_base_angle']:.1f}°")
    print(f"Mean Rewa angle: {before['mean_rewa_angle']:.1f}°")

    # Training data
    training_pairs = generate_negation_pairs()
    print(f"\nTraining on {len(training_pairs)} pairs...")

    # Train
    history = rewa.train(
        training_pairs,
        epochs=300,
        learning_rate=0.05,
        momentum_coeff=0.95,
        margin=0.95,
        verbose=True
    )

    # Evaluate after
    print("\n--- AFTER TRAINING ---")
    after = rewa.evaluate(test_pairs)
    print(f"Mean base angle: {after['mean_base_angle']:.1f}°")
    print(f"Mean Rewa angle: {after['mean_rewa_angle']:.1f}°")
    print(f"Mean improvement: {after['mean_improvement']:.1f}°")
    print(f"Antipodal rate (>135°): {after['antipodal_rate']:.1%}")

    print("\n--- DETAILED RESULTS ---")
    print(f"{'Pair':<60} {'Base°':<8} {'Rewa°':<8} {'Δ':<8}")
    print("-"*84)
    for r in after['pairs']:
        label = f"{r['pos'][:28]}... / {r['neg'][:28]}..."
        print(f"{label:<60} {r['base_angle']:>5.1f}°   {r['rewa_angle']:>5.1f}°   {r['improvement']:>+5.1f}°")

    # Save
    rewa.save("rewa_space_v2.npy")
    print("\n[SAVED] Model saved to rewa_space_v2.npy")

    return rewa, history


if __name__ == "__main__":
    main()
