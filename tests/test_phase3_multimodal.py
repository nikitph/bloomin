import torch
import numpy as np
import pytest
from rads.rads_field import RADSField
from rads.discretization import PatternExtractor, HyperedgeExtractor
from rads.reasoning import SymbolicHypergraph

class TestPhase3MultiModal:

    def load_mock_image(self):
        """
        Create a mock 3-channel image field (224x224).
        Shape: [Person inside Car]
        Person = Red blob
        Car = Blue box surrounding it
        """
        size = 64 # Scale down for test speed vs 224
        img = torch.zeros(1, 3, size, size)
        
        # Car: Blue box (Channel 2)
        # Center box
        car_r, car_c = 32, 32
        car_h, car_w = 20, 30
        img[0, 2, car_r-car_h:car_r+car_h, car_c-car_w:car_c+car_w] = 1.0
        
        # Person: Red blob (Channel 0) inside car
        p_r, p_c = 32, 32
        p_r, p_c = 35, 30 # Slightly off center but inside
        p_size = 5
        img[0, 0, p_r-p_size:p_r+p_size, p_c-p_size:p_c+p_size] = 1.0
        
        return img

    def test_exp_3_3_multi_modal_reasoning(self):
        """
        Experiment 3.3: Multi-Modal Reasoning (Image -> Field -> Graph -> NLP).
        """
        print("\nRunning Exp 3.3: Multi-Modal Reasoning")
        
        # STEP 1: Image -> Field
        image = self.load_mock_image()
        
        # Map 3 channels to d=16 field via random projection or padding
        # Simple padding
        d = 16
        field = RADSField(size=64, d=d)
        padded = torch.zeros(1, d, 64, 64)
        padded[0, :3, :, :] = image[0]
        field.phi = padded
        
        # STEP 2: Evolve
        # Diffusion should smooth boundaries and link contained objects?
        # A person "inside" a car shares spatial location.
        # Diffusion will make their signals overlap spatially.
        for t in range(20):
            field.step(dt=0.1)
            
        # STEP 3: Extract Scene Graph
        p_extractor = PatternExtractor()
        
        # We need to distinguish features.
        # PatternExtractor usually flattens to 1 activation map.
        # But we want to detect "Red Blob" vs "Blue Box".
        # We should extract patterns PRE-flattening or per-channel?
        # For a general scene graph, we extract features from field.
        # Let's assume we extract vertices based on peak activations in specific semantic channels
        # (simulated by our Red/Blue channels).
        
        # Custom extraction for multi-modal test:
        # Extract Red vertices (Person) and Blue vertices (Car)
        
        # Channel 0: Person
        person_act = field.phi[0, 0].detach().cpu().numpy()
        car_act = field.phi[0, 2].detach().cpu().numpy()
        
        vocab = []
        
        import scipy.ndimage as ndimage
        
        # Person Node
        mask_p = person_act > 0.5
        lbl_p, n_p = ndimage.label(mask_p)
        if n_p > 0:
            vocab.append({'name': 'person', 'mask': mask_p, 'pos': ndimage.center_of_mass(person_act, lbl_p, 1)})
            
        # Car Node
        mask_c = car_act > 0.5
        lbl_c, n_c = ndimage.label(mask_c)
        if n_c > 0:
            vocab.append({'name': 'car', 'mask': mask_c, 'pos': ndimage.center_of_mass(car_act, lbl_c, 1)})
            
        print(f"Extracted objects: {[v['name'] for v in vocab]}")
        assert len(vocab) >= 2
        
        # Build Hypergraph
        scene_graph = SymbolicHypergraph()
        
        # Add edges based on spatial containment
        # Containment: P center is inside C mask?
        # Or overlap IoU?
        
        for i in range(len(vocab)):
            for j in range(len(vocab)):
                if i == j: continue
                
                obj1 = vocab[i]
                obj2 = vocab[j]
                
                # Check if obj1 inside obj2
                pos1 = obj1['pos'] # (r, c) float
                r, c = int(pos1[0]), int(pos1[1])
                
                # Check if this pixel is true in obj2 mask
                # Check bounds
                if 0 <= r < 64 and 0 <= c < 64:
                    if obj2['mask'][r, c]:
                        # 1 is inside 2
                        print(f"Detected relation: {obj1['name']} IN {obj2['name']}")
                        # Edge: Person -> Car
                        scene_graph.add_hyperedge(obj1['name'], {obj2['name']})
                        # Add 'inside' node? user prompt implies "has_hyperedge('person', {'inside', 'car'})"
                        # This implies 'inside' is a node or attribute.
                        # Detailed: person -> {inside, car} (hyperedge)
                        # Let's add 'inside' as a node/attribute if needed.
                        # For query_nlp logic, I implemented "Is X in Y", which checks X -> Y path.
                        # So connecting Person -> Car is sufficient.
                        
        # STEP 4: Query
        query = "Is the person in the car?"
        result = scene_graph.query_nlp(query)
        
        print("NLP Result:", result)
        
        assert result['answer'] == 'YES'
        assert result['confidence'] > 0.9
