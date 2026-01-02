import numpy as np
import scipy.ndimage as ndimage
from scipy.spatial.distance import pdist, squareform
import torch

class PatternExtractor:
    """
    Extract vertices from field patterns.
    """
    def __init__(self):
        pass

    def extract_vertices_from_field(self, field, threshold=0.1, min_size=5):
        """
        Find stable patterns -> vertices
        
        Algorithm:
        1. Identify local maxima in field (using 'u' component or raw phi)
        2. Apply connected components / labeling
        3. Label each region as vertex
        """
        # Get activation map (using 'v' for Gray-Scott usually, or just norm of phi)
        if field.reaction == 'gray_scott':
             # Patterns are high 'v' (actually 'v' is the spot in some formulations, 'u' is background)
             # Let's use the property field.u if defined (which extracts channel 0)
             # Actually GS spots are usually low U, high V. 
             # Let's try finding peaks in channel 1 (V).
             if field.phi.shape[1] > 1:
                 activation = field.phi[0, 1].detach().cpu().numpy()
             else:
                 activation = field.phi[0, 0].detach().cpu().numpy()
        else:
             # General case: Norm or mean
             # If d > 1, take mean across features. If d=1, just take it.
             if field.phi.shape[1] > 1:
                 # activation = torch.mean(field.phi[0], dim=0).detach().cpu().numpy()
                 # Use norm for general signal detection (signed values cancel in mean)
                 activation = torch.norm(field.phi[0], dim=0).detach().cpu().numpy()
             else:
                 activation = field.phi[0, 0].detach().cpu().numpy()
             
        # Peaks
        # Simple thresholding + connected components
        mask = activation > threshold
        
        labeled, num_features = ndimage.label(mask)
        
        vertices = []
        for i in range(1, num_features + 1):
            region_mask = (labeled == i)
            # center = ndimage.center_of_mass(activation, labeled, i)
            # center_of_mass can return tuple of floats
            center = ndimage.center_of_mass(activation, labeled, i)
            
            strength = activation[region_mask].mean()
            size = region_mask.sum()
            
            # Filter tiny noise
            if size >= min_size:
                vertices.append({
                    'id': i,
                    'position': center,
                    'strength': strength,
                    'mask': region_mask,
                    'size': size
                })
        
        return vertices

class HyperedgeExtractor:
    """
    Extract hyperedges from co-occurring patterns.
    """
    def __init__(self):
        pass
        
    def extract_hyperedges_from_history(self, history, vertices, correlation_threshold=0.7):
        """
        Find co-occurring patterns -> hyperedges.
        
        Args:
            history: List of field states (numpy arrays) over time. 
                     Wait, history comes from where? The experiment loop must provide it.
                     Let's assume history is list of activation maps (H, W).
            vertices: List of vertex dicts extracted from the "reference" state (usually last).
            
        Returns:
            List of hyperedges: {'source': id, 'targets': [id, ...], 'weight': float}
        """
        if len(history) < 2:
            return []
            
        # Build activation matrix: (Time, Vertices)
        num_t = len(history)
        num_v = len(vertices)
        activations = np.zeros((num_t, num_v))
        
        for t, state in enumerate(history):
            # state is (H, W) or complete phi
            # Assume state is activation map matching vertex masks
            if isinstance(state, torch.Tensor):
                 if state.ndim == 4: # B,C,H,W
                     # assuming channel 1 for GS or mean
                     if state.shape[1] > 1:
                         act_map = state[0, 1].detach().cpu().numpy()
                     else:
                         act_map = state[0, 0].detach().cpu().numpy()
                 else:
                     act_map = state.detach().cpu().numpy()
            else:
                 act_map = state
                 
            for idx, v in enumerate(vertices):
                mask = v['mask']
                # Mean activation in the vertex region
                if mask.shape == act_map.shape:
                    activations[t, idx] = act_map[mask].mean()
                else:
                    activations[t, idx] = 0.0 # mismatch shape
                    
        # Compute correlation
        # We need variance to compute correlation. Constant signals give NaN.
        # Add tiny noise to avoid div by zero if perfectly const
        activations += np.random.normal(0, 1e-6, activations.shape)
        
        corr_matrix = np.corrcoef(activations.T) # (V, V)
        np.fill_diagonal(corr_matrix, 0) # ignore self
        
        hyperedges = []
        
        # Simple extraction strategy:
        # For each vertex, find others highly correlated with it.
        # If A is correlated with {B, C}, form A -> {B, C}
        
        for i in range(num_v):
            # Find strongly correlated
            correlated_indices = np.where(corr_matrix[i] > correlation_threshold)[0]
            
            if len(correlated_indices) > 0:
                # Group them
                # Note: this simple logic creates star-like edges.
                targets = [vertices[j]['id'] for j in correlated_indices]
                source = vertices[i]['id']
                
                weight = np.mean(corr_matrix[i, correlated_indices])
                
                hyperedges.append({
                    'source': source,
                    'targets': targets,
                    'weight': weight
                })
                
        return hyperedges

class HyperedgeReasoner:
    """
    Simple container for extracted hypergraph to compute stats.
    Compatible with exp 3 interface later.
    """
    def __init__(self, n=0):
        self.n = n
        self.hyperedges = [] # List of dicts
        
    def add_hyperedge(self, source, targets, weight=1.0):
        self.hyperedges.append({
            'source': source,
            'targets': targets,
            'weight': weight
        })
        
    @property
    def num_hyperedges(self):
        return len(self.hyperedges)
        
    @property
    def avg_hyperedge_size(self):
        if not self.hyperedges:
            return 0
        sizes = [len(h['targets'])+1 for h in self.hyperedges] # count source + targets
        return np.mean(sizes)

    def to_adjacency_matrix(self):
        # Flatten to standard graph adj for simple correlation checks
        # If A->{B,C}, add A-B, A-C
        # We need mapping from ID to index 0..N-1
        # Assumes IDs are 1-based from pattern extractor? Or just use self.n
        # Extractor output IDs are 1..K. self.n should be K+1 or max ID+1
        
        # Let's map IDs safely
        all_ids = set()
        for h in self.hyperedges:
            all_ids.add(h['source'])
            for t in h['targets']:
                all_ids.add(t)
        
        if not all_ids:
            return np.zeros((self.n, self.n))
            
        max_id = max(all_ids)
        size = max(self.n, max_id + 1)
        adj = np.zeros((size, size))
        
        for h in self.hyperedges:
            s = h['source']
            for t in h['targets']:
                adj[s, t] = h['weight']
                adj[t, s] = h['weight'] # Directed or undirected? Correlation is symmetric
                
        return adj
