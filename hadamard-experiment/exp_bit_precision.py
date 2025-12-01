import numpy as np
from scipy.linalg import hadamard
from sklearn.metrics.pairwise import euclidean_distances

def run_bit_precision(N=2000, d=1024, m=128):
    print(f"--- Experiment 4: Bit Precision (Integer Quantization) ---")
    print(f"{'Precision':<10} | {'WP Recall':<10}")
    print("-" * 30)
    
    # Setup Planted Cluster (Same as Experiment 1, sigma=0.01)
    np.random.seed(42)
    X = np.random.randn(N, d)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    
    center = np.random.randn(d)
    center /= np.linalg.norm(center)
    sigma = 0.01
    
    for i in range(10):
        point = center + np.random.randn(d) * sigma
        point /= np.linalg.norm(point)
        X[i] = point
        
    Q = (center + np.random.randn(d) * sigma).reshape(1, -1)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    
    # Ground Truth
    dists_gt = euclidean_distances(Q, X)[0]
    gt_indices = np.argsort(dists_gt)[:10]
    
    # Test Precisions
    precisions = ['float32', 'int16', 'int8']
    
    for prec in precisions:
        # Quantize X and Q
        if prec == 'float32':
            X_q = X
            Q_q = Q
        elif prec == 'int16':
            # Scale to range [-32767, 32767]
            scale = 32767
            X_q = np.round(X * scale).astype(np.int16)
            Q_q = np.round(Q * scale).astype(np.int16)
        elif prec == 'int8':
            # Scale to range [-127, 127]
            scale = 127
            X_q = np.round(X * scale).astype(np.int8)
            Q_q = np.round(Q * scale).astype(np.int8)
            
        # Witness Polar (Hadamard)
        # Note: Hadamard matrix is +1/-1, so it's just addition/subtraction.
        # We simulate this with matrix mult, but ensuring we use the quantized types if possible.
        # Numpy might upcast during matmul, but the input information is quantized.
        
        D = np.random.choice([-1, 1], size=d)
        H = hadamard(d)
        
        # Apply D
        X_d = X_q * D
        Q_d = Q_q * D
        
        # Apply H
        # For int8/int16, we need to be careful about overflow during accumulation.
        # Hadamard transform sums d elements. 
        # Max value for int8 input: 127 * 1024 = 130048 (overflows int16, fits in int32)
        # Max value for int16 input: 32767 * 1024 = 33M (fits in int32)
        
        X_polar = X_d @ H
        Q_polar = Q_d @ H
        
        # Keep top m
        X_enc = X_polar[:, :m]
        Q_enc = Q_polar[:, :m]
        
        dists_wp = euclidean_distances(Q_enc, X_enc)[0]
        wp_indices = np.argsort(dists_wp)[:10]
        wp_recall = len(set(gt_indices).intersection(wp_indices)) / 10
        
        print(f"{prec:<10} | {wp_recall:.2f}")

if __name__ == "__main__":
    run_bit_precision()
