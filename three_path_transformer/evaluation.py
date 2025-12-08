import torch
import numpy as np
import time
from collections import defaultdict

class Evaluator:
    """
    Evaluates the Three-Path Transformer System.
    """
    def __init__(self, training_protocol):
        self.tp = training_protocol
        self.model = training_protocol.model
        self.memory = training_protocol.memory
        self.device = training_protocol.device
        
    def evaluate_all(self):
        print("Running full evaluation...")
        results = {}
        
        # 1. Hierarchy Maintenance
        results['hierarchy'] = self.test_hierarchy_depth()
        
        # 2. Constituent Recovery
        results['recovery'] = self.test_constituent_recovery()
        
        # 3. Cost Analysis
        results['costs'] = self.measure_costs()
        
        return results
        
    def attempt_recovery(self, concept_name):
        """
        Attempt to recover parent from child using dynamic relationship map.
        We know Child = Mix(P1, P2). We want to find P1.
        We need P2 (sibling) to do Intersection(Child, P2) -> P1.
        """
        relationships = self.tp.relationships
        if concept_name not in relationships:
            return None, None
            
        p1, p2 = relationships[concept_name]
        
        # Let's say we want to recover p1. We need p2.
        # sibling = p2. Target = p1.
        sibling_name = p2
        target_name = p1
        
        # Get embeddings
        id1 = self.tp.get_token_id(concept_name)
        id2 = self.tp.get_token_id(sibling_name)
        
        with torch.no_grad():
            emb1 = self.model.encode(id1, path='slow')
            emb2 = self.model.encode(id2, path='slow')
            
            # Intersection logic
            intersection = emb1 * emb2
            if intersection.sum() > 1e-6:
                intersection = intersection / intersection.sum()
            else:
                intersection = torch.ones_like(intersection) / intersection.shape[-1]
                
        return intersection.squeeze(0), target_name

    def test_hierarchy_depth(self):
        """Test recovery accuracy at all depths present in hierarchy"""
        hierarchy = self.tp.hierarchy
        accuracies = {}
        
        # Skip depth 0 (Primaries have no parents)
        depths = sorted([d for d in hierarchy.keys() if d > 0])
        
        for depth in depths:
            concepts = hierarchy[depth]
            correct = 0
            total = 0
            
            for name in concepts:
                recovered_emb, target_name = self.attempt_recovery(name)
                if recovered_emb is None: continue
                
                target_id = self.tp.get_token_id(target_name)
                with torch.no_grad():
                    target_emb = self.model.encode(target_id, path='slow').squeeze(0)
                
                # Check distance
                dist = torch.norm(recovered_emb - target_emb, p=2).item()
                
                if dist < 0.25: 
                    correct += 1
                total += 1
                
            accuracies[depth] = correct / total if total > 0 else 0.0
            
        return accuracies

    def test_constituent_recovery(self):
        """Specific test cases"""
        cases = [
            ('Purple', 'Orange', 'Red'), 
            ('Mauve', 'Chartreuse', 'Orange')
        ]
        results = []
        
        for c1, c2, target in cases:
            id1 = self.tp.get_token_id(c1)
            id2 = self.tp.get_token_id(c2)
            tgt_id = self.tp.get_token_id(target)
            
            with torch.no_grad():
                e1 = self.model.encode(id1, path='slow')
                e2 = self.model.encode(id2, path='slow')
                et = self.model.encode(tgt_id, path='slow')
                
                rec = e1 * e2
                rec = rec / (rec.sum() + 1e-10)
                
                dist = torch.norm(rec - et, p=2).item()
                results.append({'case': f"{c1}+{c2}->{target}", 'distance': dist, 'success': dist < 0.2})
                
        return results

    def measure_costs(self):
        """Measure inference time"""
        dummy_id = torch.tensor([[0]], device=self.device)
        
        # Warmup
        self.model.encode(dummy_id, path='fast')
        
        # Fast
        start = time.time()
        for _ in range(100):
            self.model.encode(dummy_id, path='fast')
        fast_time = (time.time() - start) / 100
        
        # Slow
        start = time.time()
        for _ in range(100):
            self.model.encode(dummy_id, path='slow')
        slow_time = (time.time() - start) / 100
        
        return {
            'fast_ms': fast_time * 1000, 
            'slow_ms': slow_time * 1000, 
            'ratio': slow_time/fast_time
        }
