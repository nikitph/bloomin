import torch

def lorenz_dynamics(x, sigma=10.0, rho=28.0, beta=8/3.0):
    """
    Lorenz system equations.
    x: Tensor of shape (3,)
    """
    dxdt = torch.zeros_like(x)
    dxdt[0] = sigma * (x[1] - x[0])
    dxdt[1] = x[0] * (rho - x[2]) - x[1]
    dxdt[2] = x[0] * x[1] - beta * x[2]
    return dxdt

def augmented_dynamics(state, sigma=10.0, rho=28.0, beta=8/3.0):
    """
    Computes both dx/dt and dJ/dt = Df(x) * J.
    state: Tensor of shape (6,) -> [x, y, z, jx, jy, jz]
    """
    x = state[:3].detach().requires_grad_(True)
    v = state[3:]
    
    # Compute flow
    dx = lorenz_dynamics(x, sigma, rho, beta)
    
    # Compute Jacobian-Vector Product (JVP)
    # Using torch.func.jvp for modern PyTorch, or autograd.grad for older versions
    # We want Jv = (df/dx) * v
    
    # We'll use a functional approach for JVP
    def flow_fn(x_in):
        return lorenz_dynamics(x_in, sigma, rho, beta)
    
    _, jv = torch.autograd.functional.jvp(flow_fn, (x,), (v,))
    
    return torch.cat([dx, jv])

def rk4_step(fn, state, dt):
    """Simple RK4 integrator."""
    k1 = fn(state)
    k2 = fn(state + 0.5 * dt * k1)
    k3 = fn(state + 0.5 * dt * k2)
    k4 = fn(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

if __name__ == "__main__":
    # Test initialization
    x0 = torch.tensor([1.0, 1.0, 1.0])
    v0 = torch.tensor([0.1, 0.0, 0.0])
    state0 = torch.cat([x0, v0])
    
    print("State 0:", state0)
    d_state = augmented_dynamics(state0)
    print("dState/dt:", d_state)
