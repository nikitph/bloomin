import torch
import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class GraphConfig:
    num_clusters: int = 5
    nodes_per_cluster: int = 20
    intra_cluster_prob: float = 0.3
    inter_cluster_prob: float = 0.01
    seed: int = 42

class HierarchicalGraphGenerator:
    def __init__(self, config: GraphConfig):
        self.config = config
        self.graph = None
        self.shortest_paths = None
        self.node_to_cluster = {}
        
    def generate(self):
        np.random.seed(self.config.seed)
        G = nx.Graph()
        
        total_nodes = self.config.num_clusters * self.config.nodes_per_cluster
        nodes = range(total_nodes)
        G.add_nodes_from(nodes)
        
        # Assign clusters
        for i in nodes:
            cluster_id = i // self.config.nodes_per_cluster
            self.node_to_cluster[i] = cluster_id
            
        # Add edges
        for i in nodes:
            for j in range(i + 1, total_nodes):
                cluster_i = self.node_to_cluster[i]
                cluster_j = self.node_to_cluster[j]
                
                if cluster_i == cluster_j:
                    prob = self.config.intra_cluster_prob
                else:
                    prob = self.config.inter_cluster_prob
                    
                if np.random.random() < prob:
                    G.add_edge(i, j)
                    
        # Ensure connectivity (add minimum spanning tree edges if needed, 
        # but for now let's just keep the giant component or retry)
        # For simplicity, we'll just use the largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        
        # Relabel nodes to be 0..N-1
        G = nx.convert_node_labels_to_integers(G, label_attribute='old_label')
        
        # Update node_to_cluster mapping
        new_node_to_cluster = {}
        for n, data in G.nodes(data=True):
            old_label = data['old_label']
            new_node_to_cluster[n] = self.node_to_cluster[old_label]
        self.node_to_cluster = new_node_to_cluster
        
        self.graph = G
        print(f"Generated graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        
        # Precompute all pairs shortest paths
        print("Computing all-pairs shortest paths...")
        self.shortest_paths = dict(nx.all_pairs_shortest_path_length(G))
        print("Done.")
        
        return G

    def get_dataset(self, num_samples=10000):
        """
        Returns a dataset of (start_node, end_node, distance)
        """
        if self.graph is None:
            self.generate()
            
        nodes = list(self.graph.nodes())
        samples = []
        
        for _ in range(num_samples):
            u = np.random.choice(nodes)
            v = np.random.choice(nodes)
            
            if v in self.shortest_paths[u]:
                dist = self.shortest_paths[u][v]
                samples.append((u, v, dist))
            else:
                # Unreachable (shouldn't happen with connected component, but safe to handle)
                samples.append((u, v, -1)) 
                
        return samples

    def get_length_split_dataset(self, train_range=(1, 5), test_range=(6, 10), num_samples=20000):
        if self.graph is None:
            self.generate()
            
        nodes = list(self.graph.nodes())
        train_samples = []
        test_samples = []
        
        # Naive sampling might be slow for specific ranges, so let's iterate all pairs if graph is small
        # With 100 nodes, 100*99/2 = 4950 pairs. We can just classify all of them.
        all_pairs = []
        for u in nodes:
            for v in nodes:
                if u >= v: continue
                if v in self.shortest_paths[u]:
                    dist = self.shortest_paths[u][v]
                    all_pairs.append((u, v, dist))
        
        np.random.shuffle(all_pairs)
        
        for u, v, dist in all_pairs:
            if train_range[0] <= dist <= train_range[1]:
                train_samples.append((u, v, dist))
            elif test_range[0] <= dist <= test_range[1]:
                test_samples.append((u, v, dist))
                
        # Resample if we need more, but for this experiment let's just return what we found
        # or cap it.
        return train_samples, test_samples

    def get_cluster_split_dataset(self, train_clusters=[0, 1, 2], test_pair=(0, 2)):
        """
        Train on pairs within train_clusters and adjacent pairs (e.g. 0-1, 1-2).
        Test on the 'jump' pair (e.g. 0-2) which is not directly seen but implied.
        Actually, let's follow the user's spec:
        Train: Within A, B, C; and A<->B, B<->C.
        Test: A<->C.
        """
        if self.graph is None:
            self.generate()
            
        nodes = list(self.graph.nodes())
        train_samples = []
        test_samples = []
        
        c0, c1, c2 = train_clusters
        
        all_pairs = []
        for u in nodes:
            for v in nodes:
                if u >= v: continue
                if v in self.shortest_paths[u]:
                    dist = self.shortest_paths[u][v]
                    all_pairs.append((u, v, dist))
                    
        np.random.shuffle(all_pairs)
        
        for u, v, dist in all_pairs:
            cu = self.node_to_cluster[u]
            cv = self.node_to_cluster[v]
            
            # Check if pair involves only relevant clusters
            if cu not in train_clusters or cv not in train_clusters:
                continue
                
            # Train conditions:
            # 1. Same cluster
            same_cluster = (cu == cv)
            # 2. Adjacent clusters (0-1 or 1-2)
            adj_01 = (cu == c0 and cv == c1) or (cu == c1 and cv == c0)
            adj_12 = (cu == c1 and cv == c2) or (cu == c2 and cv == c1)
            
            # Test condition:
            # 3. Jump clusters (0-2)
            jump_02 = (cu == c0 and cv == c2) or (cu == c2 and cv == c0)
            
            if same_cluster or adj_01 or adj_12:
                train_samples.append((u, v, dist))
            elif jump_02:
                test_samples.append((u, v, dist))
                
        return train_samples, test_samples


if __name__ == "__main__":
    config = GraphConfig()
    gen = HierarchicalGraphGenerator(config)
    gen.generate()
    data = gen.get_dataset(10)
    print("Sample data:", data[:5])
