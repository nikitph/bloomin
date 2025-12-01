import torch
import time
import math

def fwht_layer(x):
    """
    Fast Walsh-Hadamard Transform in PyTorch.
    Input: (N, d) where d must be a power of 2.
    """
    N, d = x.shape
    k = int(math.log2(d))
    assert 2**k == d, "Dimension d must be a power of 2"
    
    # Reshape to (N, 2, 2, ..., 2)
    # We want to operate on each of the k dimensions.
    x = x.view(N, *([2] * k))
    
    for i in range(k):
        # We want to operate on dimension i+1 (0 is batch)
        # But for efficiency, let's just use the fact that we can access slices.
        # Actually, iterating over the dimensions of the hypercube is exactly what we need.
        
        # Let's define slices dynamically or use torch.unbind
        # x has shape (N, 2, 2, ..., 2)
        # We want to perform the butterfly on dimension i+1
        
        # Slice for 0 and 1 along dimension i+1
        # We can use torch.split or unbind
        # x0, x1 = x.unbind(dim=i+1)
        # But unbind creates copies. In-place is better?
        # x[..., 0, ...] = x0 + x1
        # x[..., 1, ...] = x0 - x1
        
        # To do this efficiently without complex indexing:
        # Move the target dimension to the end, operate, then move back?
        # Or just use slice indexing.
        
        # Construct slices
        idx0 = [slice(None)] * (k + 1)
        idx0[i+1] = 0
        
        idx1 = [slice(None)] * (k + 1)
        idx1[i+1] = 1
        
        # We need to clone to avoid overwriting while reading
        # x0 = x[tuple(idx0)].clone()
        # x1 = x[tuple(idx1)].clone()
        
        # x[tuple(idx0)] = x0 + x1
        # x[tuple(idx1)] = x0 - x1
        
        # Optimization:
        # x0 + x1
        # x0 - x1
        # This is just a linear transform on that dimension.
        # We can do:
        # y = x.transpose(i+1, -1)
        # y0 = y[..., 0]
        # y1 = y[..., 1]
        # y[..., 0] = y0 + y1
        # y[..., 1] = y0 - y1
        # x = y.transpose(i+1, -1)
        
        # Even better:
        # Just reshape to (Pre, 2, Post)
        # But the dimensions are entangled.
        # Actually, the order of dimensions in FWHT doesn't matter! H_2n = H_2 x H_n = H_n x H_2.
        # So we can just iterate k times, always operating on the LAST dimension, 
        # but we need to permute/reshape to bring "fresh" interactions to the last dimension.
        # Wait, if we just operate on the last dimension k times, we are just doing H_2 on the same bits?
        # No.
        # The standard algorithm:
        # for h in [1, 2, 4, ...]:
        #   for i in 0..N step 2h:
        #     ...
        # This pairs indices (j, j+h).
        
        # In tensor view (N, 2, 2, ..., 2):
        # Dimension 1 corresponds to stride 2^(k-1)
        # Dimension k corresponds to stride 1
        # We need to apply the butterfly on EACH dimension 1..k.
        # The order DOES NOT MATTER.
        # So we can just loop i from 1 to k:
        #   x0 = x.select(i, 0)
        #   x1 = x.select(i, 1)
        #   sum = x0 + x1
        #   diff = x0 - x1
        #   x.select(i, 0).copy_(sum)
        #   x.select(i, 1).copy_(diff)
        pass
        
    # Implementation
    for i in range(k):
        # dim 0 is batch. dim 1..k are the hypercube dims.
        dim = i + 1
        x0 = x.select(dim, 0).clone()
        x1 = x.select(dim, 1).clone()
        x.select(dim, 0).copy_(x0 + x1)
        x.select(dim, 1).copy_(x0 - x1)
        
    return x.view(N, d)

def fwht_kronecker(x):
    """
    FWHT using the Kronecker Product trick.
    H_1024 = H_32 (x) H_32.
    We can compute (H (x) H) vec(X) as vec(H X H').
    This replaces O(d log d) scalar ops with O(d^1.5) matrix ops,
    but uses highly optimized GEMM kernels.
    """
    N, d = x.shape
    assert d == 1024, "This optimization is hardcoded for d=1024 (32x32)"
    
    # Precompute H_32
    # We can construct it using the recursive definition or just use the slow fwht once
    # H_1 = [1]
    # H_2 = [[1, 1], [1, -1]]
    # ...
    # Let's just build it.
    
    # Construct H_32 on the fly (cached in practice)
    # We can use the same fwht_layer on identity matrix
    # But fwht_layer is slow.
    # Let's just hardcode construction or use a small helper.
    
    # Efficient H_32 construction:
    h = torch.tensor([[1., 1.], [1., -1.]], device=x.device)
    for _ in range(4): # 2 -> 4 -> 8 -> 16 -> 32
        h = torch.kron(h, torch.tensor([[1., 1.], [1., -1.]], device=x.device))
    
    # Reshape X to (N, 32, 32)
    X_mat = x.view(N, 32, 32)
    
    # Compute H X H
    # H is (32, 32). X_mat is (N, 32, 32).
    # We want H @ X_i @ H for each i.
    # torch.matmul broadcasts.
    # Y = H @ X_mat  -> (32, 32) @ (N, 32, 32) -> (N, 32, 32) ?
    # Wait, matmul rules:
    # If both arguments are at least 1-dimensional and at least one argument is N-dimensional (where N > 2), then a batched matrix multiplication is returned.
    # If the first argument is 2-D and the second argument is > 2-D, the matrix is prepended with 1s to its dimensions...
    # Actually, let's just be explicit.
    # We want to multiply H on the left of every 32x32 matrix in the batch.
    # X_mat is (N, 32, 32).
    # We can treat it as (N*32, 32) ? No.
    # We can use einsum: 'ij,njk->nik' -> H @ X
    
    # Y = torch.einsum('ij,njk->nik', h, X_mat)
    # Z = torch.einsum('nik,lk->nil', Y, h) # Y @ H^T (H is symmetric)
    
    # Or using matmul with broadcasting:
    # h_broadcast = h.unsqueeze(0) # (1, 32, 32)
    # Y = h_broadcast @ X_mat # (1, 32, 32) @ (N, 32, 32) -> (N, 32, 32)
    # Z = Y @ h_broadcast # (N, 32, 32) @ (1, 32, 32) -> (N, 32, 32)
    
    # This should be very fast.
    h_broadcast = h.unsqueeze(0)
    Y = h_broadcast @ X_mat
    Z = Y @ h_broadcast
    
    return Z.view(N, 1024)

def run_gpu_benchmark():
    print(f"--- Experiment 7: GPU Benchmark (PyTorch) ---")
    
    # Check device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Device: MPS (Mac GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using Device: CUDA")
    else:
        device = torch.device("cpu")
        print("Using Device: CPU (Warning: Slow)")
        
    N = 10000
    d = 1024
    m = 128
    
    print(f"Encoding {N} vectors of dimension {d} -> {m}")
    
    # Data
    X = torch.randn(N, d, device=device)
    G = torch.randn(d, m, device=device)
    D = torch.randint(0, 2, (d,), device=device).float() * 2 - 1 # {-1, 1}
    
def fwht_strided(x):
    """
    FWHT using explicit strided stages (Cooley-Tukey).
    This maps to the user's pseudocode:
    for stage in range(log2(d)):
        stride = 1 << stage
        butterfly_stage(x, stride)
    """
    N, d = x.shape
    steps = int(math.log2(d))
    
    # We can do this in-place if we are careful, or out-of-place.
    # In-place is harder with views.
    # Let's try to be efficient with views.
    
    for i in range(steps):
        stride = 1 << i
        # The butterfly pairs are (j, j+stride) for j where (j & stride) == 0.
        # This structure corresponds to reshaping to (..., 2, stride, ...)
        # Specifically: (N, d // (2*stride), 2, stride)
        
        batch_dim = d // (2 * stride)
        
        # View as (N, batch_dim, 2, stride)
        y = x.view(N, batch_dim, 2, stride)
        
        # Split
        a = y[:, :, 0, :]
        b = y[:, :, 1, :]
        
        # Compute
        sum_ = a + b
        diff = a - b
        
        # We need to write back.
        # Constructing a new tensor is cleaner but allocates.
        # x = torch.cat([sum_.unsqueeze(2), diff.unsqueeze(2)], dim=2).reshape(N, d)
        
        # Can we do it in-place?
        # y[:, :, 0, :] = sum_  <-- This modifies 'a' if 'a' is a view of 'y'!
        # But 'sum_' depends on 'a'.
        # So we need temporary storage.
        
        # Optimization: use one temp buffer.
        # temp = a + b
        # b = a - b  (writes to y[:,:,1,:])
        # a = temp   (writes to y[:,:,0,:])
        
        # But we need to be careful about overwriting.
        # a and b are views into x.
        # sum_ = a + b (allocates new memory for result)
        # diff = a - b (allocates new memory for result)
        
        # y[:, :, 0, :] = sum_
        # y[:, :, 1, :] = diff
        
        # This works and uses 2 allocations per stage.
        # Can we do better?
        # a.add_(b) -> a becomes sum
        # b.sub_(a) -> b - (a+b) = -a. Wrong.
        
        # Standard in-place butterfly:
        # u = a
        # v = b
        # a = u + v
        # b = u - v
        
        # We need to store one of them.
        # temp = b.clone()
        # b.sub_(a).neg_() -> -(b-a) = a-b. Correct.
        # a.add_(temp) -> a+b. Correct.
        
        # Let's try this in-place version to save memory bandwidth.
        
        temp = b.clone()
        b.sub_(a).neg_() # b = a - b
        a.add_(temp)     # a = a + b_old
        
    return x

def run_gpu_benchmark():
    print(f"--- Experiment 7: GPU Benchmark (PyTorch) ---")
    
    # Check device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Device: MPS (Mac GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using Device: CUDA")
    else:
        device = torch.device("cpu")
        print("Using Device: CPU (Warning: Slow)")
        
    N = 10000
    d = 1024
    m = 128
    
    print(f"Encoding {N} vectors of dimension {d} -> {m}")
    
    # Data
    X = torch.randn(N, d, device=device)
    G = torch.randn(d, m, device=device)
    D = torch.randint(0, 2, (d,), device=device).float() * 2 - 1 # {-1, 1}
    
    # Warmup
    _ = X @ G
    _ = fwht_layer(X * D)
    _ = fwht_kronecker(X * D)
    _ = fwht_strided(X * D)
    
    iterations = 50
    print(f"Running {iterations} iterations...")
    
    # Random Projection
    start = time.perf_counter()
    for _ in range(iterations):
        _ = X @ G
        if device.type == 'cuda': torch.cuda.synchronize()
        if device.type == 'mps': torch.mps.synchronize()
    time_rp = (time.perf_counter() - start) / iterations * 1000
    
    # Witness Polar (Loop)
    start = time.perf_counter()
    for _ in range(iterations):
        X_flipped = X * D
        X_trans = fwht_layer(X_flipped)
        _ = X_trans[:, :m]
        if device.type == 'cuda': torch.cuda.synchronize()
        if device.type == 'mps': torch.mps.synchronize()
    time_wp_loop = (time.perf_counter() - start) / iterations * 1000
    
    # Witness Polar (Kronecker)
    start = time.perf_counter()
    for _ in range(iterations):
        X_flipped = X * D
        X_trans = fwht_kronecker(X_flipped)
        _ = X_trans[:, :m]
        if device.type == 'cuda': torch.cuda.synchronize()
        if device.type == 'mps': torch.mps.synchronize()
    time_wp_kron = (time.perf_counter() - start) / iterations * 1000
    
    # Witness Polar (Strided)
    start = time.perf_counter()
    for _ in range(iterations):
        X_flipped = X * D
        X_trans = fwht_strided(X_flipped)
        _ = X_trans[:, :m]
        if device.type == 'cuda': torch.cuda.synchronize()
        if device.type == 'mps': torch.mps.synchronize()
    time_wp_strided = (time.perf_counter() - start) / iterations * 1000
    
    print(f"Random Projection:      {time_rp:.2f} ms")
    print(f"Witness-Polar (Loop):   {time_wp_loop:.2f} ms")
    print(f"Witness-Polar (Kron):   {time_wp_kron:.2f} ms")
    print(f"Witness-Polar (Stride): {time_wp_strided:.2f} ms")
    print(f"Speedup (Stride vs RP): {time_rp / time_wp_strided:.2f}x")

if __name__ == "__main__":
    run_gpu_benchmark()
