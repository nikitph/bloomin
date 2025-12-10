"""
Rewa-Space Projection Head

Transforms base embeddings into a space where:
1. Contradictions are antipodal (angle ≈ π)
2. Policy axes are clean and geometric
3. Compliant/violating pairs are well-separated

Architecture:
    Base Encoder → Normalize → Rewa Projection Head → Rewa-Space (S^{k-1})

Training:
    Contrastive loss to enforce antipodal behavior along policy axes:
    L = max(0, m - u_p · u_c) + max(0, m + u_p · u_v)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import json


@dataclass
class TrainingPair:
    """A training pair for contrastive learning."""
    anchor: str           # Policy/concept
    positive: str         # Compliant/aligned example
    negative: str         # Violating/contradicting example
    axis_name: str        # Name of the semantic axis (e.g., "safety", "compliance")


class RewaSpaceHead:
    """
    Rewa-space projection head.

    Maps base embeddings z ∈ S^{d-1} to Rewa-space u ∈ S^{k-1}
    where antipodal structure is enforced along policy axes.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 128,
        hidden_dims: List[int] = [256, 128],
        activation: str = "relu"
    ):
        """
        Args:
            input_dim: Dimension of base embeddings
            output_dim: Dimension of Rewa-space (k)
            hidden_dims: Hidden layer dimensions
            activation: Activation function
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        # Initialize weights
        self._init_weights()

        # Training state
        self.trained = False
        self.training_history = []

    def _init_weights(self):
        """Initialize network weights with Xavier initialization."""
        self.weights = {}
        self.biases = {}

        prev_dim = self.input_dim
        for i, hidden_dim in enumerate(self.hidden_dims):
            # Xavier initialization
            scale = np.sqrt(2.0 / (prev_dim + hidden_dim))
            self.weights[f'W{i}'] = np.random.randn(prev_dim, hidden_dim) * scale
            self.biases[f'b{i}'] = np.zeros(hidden_dim)
            prev_dim = hidden_dim

        # Output layer
        scale = np.sqrt(2.0 / (prev_dim + self.output_dim))
        self.weights['W_out'] = np.random.randn(prev_dim, self.output_dim) * scale
        self.biases['b_out'] = np.zeros(self.output_dim)

    def forward(self, z: np.ndarray) -> np.ndarray:
        """
        Forward pass through the projection head.

        Args:
            z: Base embedding(s) on unit sphere, shape (d,) or (batch, d)

        Returns:
            u: Rewa-space embedding(s) on unit sphere, shape (k,) or (batch, k)
        """
        single_input = z.ndim == 1
        if single_input:
            z = z.reshape(1, -1)

        x = z

        # Hidden layers with ReLU
        for i in range(len(self.hidden_dims)):
            x = x @ self.weights[f'W{i}'] + self.biases[f'b{i}']
            x = np.maximum(0, x)  # ReLU

        # Output layer
        x = x @ self.weights['W_out'] + self.biases['b_out']

        # Normalize to unit sphere
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        u = x / (norms + 1e-10)

        if single_input:
            return u[0]
        return u

    def __call__(self, z: np.ndarray) -> np.ndarray:
        """Alias for forward."""
        return self.forward(z)


class AntipodalContrastiveLoss:
    """
    Contrastive loss for enforcing antipodal behavior.

    For each (anchor, positive, negative) triplet:
    L = max(0, margin - u_a · u_p) + max(0, margin + u_a · u_n)

    This pushes:
    - Anchor and positive to have high similarity (same hemisphere)
    - Anchor and negative to have low similarity (opposite hemispheres)
    """

    def __init__(self, margin: float = 0.8):
        """
        Args:
            margin: Target margin for dot products
        """
        self.margin = margin

    def compute(
        self,
        anchor: np.ndarray,
        positive: np.ndarray,
        negative: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute contrastive loss for a triplet.

        Returns:
            (loss, details)
        """
        # Dot products
        pos_sim = np.dot(anchor, positive)
        neg_sim = np.dot(anchor, negative)

        # Hinge losses
        pos_loss = max(0, self.margin - pos_sim)
        neg_loss = max(0, self.margin + neg_sim)

        total_loss = pos_loss + neg_loss

        return total_loss, {
            'pos_sim': float(pos_sim),
            'neg_sim': float(neg_sim),
            'pos_loss': float(pos_loss),
            'neg_loss': float(neg_loss),
            'pos_angle_deg': float(np.degrees(np.arccos(np.clip(pos_sim, -1, 1)))),
            'neg_angle_deg': float(np.degrees(np.arccos(np.clip(neg_sim, -1, 1))))
        }

    def compute_batch(
        self,
        anchors: np.ndarray,
        positives: np.ndarray,
        negatives: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """Compute mean loss over batch."""
        batch_size = len(anchors)

        # Vectorized dot products
        pos_sims = np.sum(anchors * positives, axis=1)
        neg_sims = np.sum(anchors * negatives, axis=1)

        # Hinge losses
        pos_losses = np.maximum(0, self.margin - pos_sims)
        neg_losses = np.maximum(0, self.margin + neg_sims)

        total_loss = float(np.mean(pos_losses + neg_losses))

        return total_loss, {
            'mean_pos_sim': float(np.mean(pos_sims)),
            'mean_neg_sim': float(np.mean(neg_sims)),
            'mean_pos_loss': float(np.mean(pos_losses)),
            'mean_neg_loss': float(np.mean(neg_losses)),
            'mean_pos_angle_deg': float(np.mean(np.degrees(np.arccos(np.clip(pos_sims, -1, 1))))),
            'mean_neg_angle_deg': float(np.mean(np.degrees(np.arccos(np.clip(neg_sims, -1, 1)))))
        }


class RewaSpaceTrainer:
    """
    Trainer for the Rewa-space projection head.

    Uses gradient descent with numerical gradients (simple implementation).
    Production would use PyTorch for autograd.
    """

    def __init__(
        self,
        head: RewaSpaceHead,
        base_model: SentenceTransformer,
        loss_fn: Optional[AntipodalContrastiveLoss] = None,
        learning_rate: float = 0.01,
        batch_size: int = 32
    ):
        self.head = head
        self.base_model = base_model
        self.loss_fn = loss_fn or AntipodalContrastiveLoss(margin=0.8)
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def _get_base_embedding(self, text: str) -> np.ndarray:
        """Get normalized base embedding."""
        emb = self.base_model.encode(text, convert_to_numpy=True)
        return emb / (np.linalg.norm(emb) + 1e-10)

    def _numerical_gradient(
        self,
        param_name: str,
        anchors_base: np.ndarray,
        positives_base: np.ndarray,
        negatives_base: np.ndarray,
        eps: float = 1e-5
    ) -> np.ndarray:
        """Compute numerical gradient for a parameter."""
        if param_name in self.head.weights:
            param = self.head.weights[param_name]
        else:
            param = self.head.biases[param_name]
        grad = np.zeros_like(param)

        # For efficiency, sample a subset of parameters
        flat_param = param.flatten()
        n_samples = min(100, len(flat_param))  # Sample 100 params max
        indices = np.random.choice(len(flat_param), n_samples, replace=False)

        for idx in indices:
            # Perturb +eps
            flat_param[idx] += eps
            if param_name in self.head.weights:
                self.head.weights[param_name] = flat_param.reshape(param.shape)
            else:
                self.head.biases[param_name] = flat_param.reshape(param.shape)

            u_a = self.head.forward(anchors_base)
            u_p = self.head.forward(positives_base)
            u_n = self.head.forward(negatives_base)
            loss_plus, _ = self.loss_fn.compute_batch(u_a, u_p, u_n)

            # Perturb -eps
            flat_param[idx] -= 2 * eps
            if param_name in self.head.weights:
                self.head.weights[param_name] = flat_param.reshape(param.shape)
            else:
                self.head.biases[param_name] = flat_param.reshape(param.shape)

            u_a = self.head.forward(anchors_base)
            u_p = self.head.forward(positives_base)
            u_n = self.head.forward(negatives_base)
            loss_minus, _ = self.loss_fn.compute_batch(u_a, u_p, u_n)

            # Restore
            flat_param[idx] += eps
            if param_name in self.head.weights:
                self.head.weights[param_name] = flat_param.reshape(param.shape)
            else:
                self.head.biases[param_name] = flat_param.reshape(param.shape)

            # Gradient
            grad.flat[idx] = (loss_plus - loss_minus) / (2 * eps)

        return grad

    def train_step(
        self,
        triplets: List[TrainingPair]
    ) -> Dict[str, float]:
        """
        Perform one training step on a batch of triplets.
        """
        # Get base embeddings
        anchors_base = np.array([self._get_base_embedding(t.anchor) for t in triplets])
        positives_base = np.array([self._get_base_embedding(t.positive) for t in triplets])
        negatives_base = np.array([self._get_base_embedding(t.negative) for t in triplets])

        # Forward pass
        u_a = self.head.forward(anchors_base)
        u_p = self.head.forward(positives_base)
        u_n = self.head.forward(negatives_base)

        # Compute loss
        loss, details = self.loss_fn.compute_batch(u_a, u_p, u_n)

        # Compute gradients and update (simplified SGD)
        for param_name in list(self.head.weights.keys()) + list(self.head.biases.keys()):
            grad = self._numerical_gradient(param_name, anchors_base, positives_base, negatives_base)

            if param_name in self.head.weights:
                self.head.weights[param_name] -= self.learning_rate * grad
            else:
                self.head.biases[param_name] -= self.learning_rate * grad

        details['loss'] = loss
        return details

    def train(
        self,
        training_pairs: List[TrainingPair],
        epochs: int = 100,
        verbose: bool = True
    ) -> List[Dict[str, float]]:
        """
        Train the projection head on training pairs.
        """
        history = []

        for epoch in range(epochs):
            # Shuffle
            np.random.shuffle(training_pairs)

            # Process in batches
            epoch_losses = []
            for i in range(0, len(training_pairs), self.batch_size):
                batch = training_pairs[i:i + self.batch_size]
                step_details = self.train_step(batch)
                epoch_losses.append(step_details['loss'])

            mean_loss = np.mean(epoch_losses)
            history.append({'epoch': epoch, 'loss': mean_loss})

            if verbose and epoch % 10 == 0:
                # Evaluate on full dataset
                all_anchors = np.array([self._get_base_embedding(t.anchor) for t in training_pairs])
                all_positives = np.array([self._get_base_embedding(t.positive) for t in training_pairs])
                all_negatives = np.array([self._get_base_embedding(t.negative) for t in training_pairs])

                u_a = self.head.forward(all_anchors)
                u_p = self.head.forward(all_positives)
                u_n = self.head.forward(all_negatives)

                _, details = self.loss_fn.compute_batch(u_a, u_p, u_n)

                print(f"Epoch {epoch}: Loss={mean_loss:.4f}, "
                      f"PosAngle={details['mean_pos_angle_deg']:.1f}°, "
                      f"NegAngle={details['mean_neg_angle_deg']:.1f}°")

        self.head.trained = True
        self.head.training_history = history

        return history

    def evaluate(
        self,
        test_pairs: List[TrainingPair]
    ) -> Dict[str, Any]:
        """Evaluate the trained head on test pairs."""
        anchors_base = np.array([self._get_base_embedding(t.anchor) for t in test_pairs])
        positives_base = np.array([self._get_base_embedding(t.positive) for t in test_pairs])
        negatives_base = np.array([self._get_base_embedding(t.negative) for t in test_pairs])

        u_a = self.head.forward(anchors_base)
        u_p = self.head.forward(positives_base)
        u_n = self.head.forward(negatives_base)

        loss, details = self.loss_fn.compute_batch(u_a, u_p, u_n)

        # Check antipodal success
        pos_sims = np.sum(u_a * u_p, axis=1)
        neg_sims = np.sum(u_a * u_n, axis=1)

        antipodal_success = np.mean(neg_sims < -0.5)  # Negative pairs should have negative similarity
        alignment_success = np.mean(pos_sims > 0.5)   # Positive pairs should have positive similarity

        return {
            'loss': loss,
            'mean_pos_angle_deg': details['mean_pos_angle_deg'],
            'mean_neg_angle_deg': details['mean_neg_angle_deg'],
            'antipodal_success_rate': float(antipodal_success),
            'alignment_success_rate': float(alignment_success),
            'details': details
        }


def generate_training_data() -> List[TrainingPair]:
    """
    Generate training data for Rewa-space.

    Covers multiple semantic axes:
    - Negation (X vs not X)
    - Sentiment (positive vs negative)
    - Safety (safe vs unsafe)
    - Compliance (compliant vs violating)
    - Truth (true vs false)
    - Risk (low risk vs high risk)
    """
    pairs = []

    # Axis 1: Logical Negation
    negation_pairs = [
        ("The statement is true", "The statement is true and verified", "The statement is false"),
        ("The answer is yes", "Yes, that is correct", "No, that is incorrect"),
        ("The system is active", "The system is running and active", "The system is inactive"),
        ("The door is open", "The door is open and unlocked", "The door is closed"),
        ("The light is on", "The light is on and bright", "The light is off"),
        ("The patient is alive", "The patient is alive and stable", "The patient is dead"),
        ("The test passed", "The test passed successfully", "The test failed"),
        ("The product is available", "The product is in stock", "The product is unavailable"),
        ("Access is granted", "Access is permitted", "Access is denied"),
        ("The claim is valid", "The claim is legitimate", "The claim is invalid"),
    ]

    for anchor, pos, neg in negation_pairs:
        pairs.append(TrainingPair(anchor, pos, neg, "negation"))

    # Axis 2: Sentiment
    sentiment_pairs = [
        ("positive sentiment", "I love this product, it's amazing", "I hate this product, it's terrible"),
        ("positive review", "Excellent quality and fast shipping", "Poor quality and slow delivery"),
        ("customer satisfaction", "Very satisfied with the service", "Extremely disappointed with the service"),
        ("recommendation", "I highly recommend this", "I would never recommend this"),
        ("experience quality", "Best experience ever", "Worst experience ever"),
    ]

    for anchor, pos, neg in sentiment_pairs:
        pairs.append(TrainingPair(anchor, pos, neg, "sentiment"))

    # Axis 3: Safety
    safety_pairs = [
        ("safety assessment", "The procedure is safe and approved", "The procedure is dangerous and risky"),
        ("risk level", "Low risk with minimal side effects", "High risk with severe consequences"),
        ("medical safety", "Clinically tested and safe for use", "Not tested and potentially harmful"),
        ("product safety", "Meets all safety standards", "Fails safety requirements"),
        ("operational safety", "Safe operating conditions", "Hazardous operating conditions"),
    ]

    for anchor, pos, neg in safety_pairs:
        pairs.append(TrainingPair(anchor, pos, neg, "safety"))

    # Axis 4: Compliance
    compliance_pairs = [
        ("regulatory compliance", "Fully compliant with regulations", "Violates multiple regulations"),
        ("policy adherence", "Follows all company policies", "Breaches company policies"),
        ("legal status", "Legally permitted and authorized", "Illegal and unauthorized"),
        ("standard compliance", "Meets industry standards", "Below industry standards"),
        ("audit result", "Passed compliance audit", "Failed compliance audit"),
    ]

    for anchor, pos, neg in compliance_pairs:
        pairs.append(TrainingPair(anchor, pos, neg, "compliance"))

    # Axis 5: Truth/Accuracy
    truth_pairs = [
        ("factual accuracy", "Verified and accurate information", "Misinformation and false claims"),
        ("data quality", "Accurate and reliable data", "Inaccurate and unreliable data"),
        ("statement validity", "Factually correct statement", "Factually incorrect statement"),
        ("evidence support", "Supported by strong evidence", "Contradicted by evidence"),
        ("source reliability", "From a credible source", "From an unreliable source"),
    ]

    for anchor, pos, neg in truth_pairs:
        pairs.append(TrainingPair(anchor, pos, neg, "truth"))

    # Axis 6: Financial Risk
    risk_pairs = [
        ("investment risk", "Safe investment with stable returns", "Risky investment with volatile returns"),
        ("credit assessment", "Excellent credit score and history", "Poor credit score and history"),
        ("financial health", "Financially stable and profitable", "Financially unstable with losses"),
        ("market position", "Strong market position", "Weak market position"),
        ("growth outlook", "Positive growth trajectory", "Declining performance"),
    ]

    for anchor, pos, neg in risk_pairs:
        pairs.append(TrainingPair(anchor, pos, neg, "financial_risk"))

    # Axis 7: Approval/Rejection
    approval_pairs = [
        ("approval status", "Application approved", "Application rejected"),
        ("eligibility", "Meets all eligibility criteria", "Does not meet eligibility criteria"),
        ("qualification", "Fully qualified candidate", "Unqualified candidate"),
        ("acceptance", "Proposal accepted", "Proposal declined"),
        ("authorization", "Authorized and permitted", "Unauthorized and forbidden"),
    ]

    for anchor, pos, neg in approval_pairs:
        pairs.append(TrainingPair(anchor, pos, neg, "approval"))

    return pairs


class RewaProjectionHead:
    """
    Complete Rewa-space projection system.

    Combines:
    1. Base encoder (SentenceTransformer)
    2. Trained projection head
    3. Evaluation utilities
    """

    def __init__(
        self,
        base_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        output_dim: int = 128,
        pretrained_weights: Optional[Dict[str, np.ndarray]] = None
    ):
        self.base_model = SentenceTransformer(base_model_name)
        self.input_dim = self.base_model.get_sentence_embedding_dimension()
        self.output_dim = output_dim

        self.head = RewaSpaceHead(
            input_dim=self.input_dim,
            output_dim=output_dim
        )

        if pretrained_weights:
            self.head.weights = pretrained_weights['weights']
            self.head.biases = pretrained_weights['biases']
            self.head.trained = True

    def project(self, text: str) -> np.ndarray:
        """Project text into Rewa-space."""
        # Base embedding
        base_emb = self.base_model.encode(text, convert_to_numpy=True)
        base_emb = base_emb / (np.linalg.norm(base_emb) + 1e-10)

        # Project to Rewa-space
        return self.head.forward(base_emb)

    def project_batch(self, texts: List[str]) -> np.ndarray:
        """Project multiple texts."""
        base_embs = self.base_model.encode(texts, convert_to_numpy=True)
        base_embs = base_embs / (np.linalg.norm(base_embs, axis=1, keepdims=True) + 1e-10)
        return self.head.forward(base_embs)

    def train(
        self,
        training_pairs: Optional[List[TrainingPair]] = None,
        epochs: int = 100,
        learning_rate: float = 0.01,
        verbose: bool = True
    ) -> List[Dict[str, float]]:
        """Train the projection head."""
        if training_pairs is None:
            training_pairs = generate_training_data()

        trainer = RewaSpaceTrainer(
            head=self.head,
            base_model=self.base_model,
            learning_rate=learning_rate
        )

        return trainer.train(training_pairs, epochs=epochs, verbose=verbose)

    def evaluate_antipodal(self, pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Evaluate antipodal behavior on negation pairs.

        Args:
            pairs: List of (positive, negative) text pairs

        Returns:
            Evaluation metrics
        """
        results = []

        for pos_text, neg_text in pairs:
            pos_emb = self.project(pos_text)
            neg_emb = self.project(neg_text)

            dot = np.dot(pos_emb, neg_emb)
            angle = np.arccos(np.clip(dot, -1, 1))

            results.append({
                'positive': pos_text,
                'negative': neg_text,
                'dot_product': float(dot),
                'angle_rad': float(angle),
                'angle_deg': float(np.degrees(angle)),
                'is_antipodal': angle > np.pi * 0.75  # > 135°
            })

        # Summary statistics
        angles = [r['angle_deg'] for r in results]
        antipodal_rate = sum(1 for r in results if r['is_antipodal']) / len(results)

        return {
            'pairs': results,
            'mean_angle_deg': float(np.mean(angles)),
            'min_angle_deg': float(np.min(angles)),
            'max_angle_deg': float(np.max(angles)),
            'antipodal_success_rate': float(antipodal_rate)
        }

    def save(self, filepath: str):
        """Save the trained head."""
        data = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'weights': {k: v.tolist() for k, v in self.head.weights.items()},
            'biases': {k: v.tolist() for k, v in self.head.biases.items()},
            'trained': self.head.trained
        }

        with open(filepath, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filepath: str, base_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """Load a trained head."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        instance = cls(
            base_model_name=base_model_name,
            output_dim=data['output_dim']
        )

        instance.head.weights = {k: np.array(v) for k, v in data['weights'].items()}
        instance.head.biases = {k: np.array(v) for k, v in data['biases'].items()}
        instance.head.trained = data['trained']

        return instance
