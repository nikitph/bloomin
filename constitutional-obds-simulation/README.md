# Constitutional OBDS Programming Demo

This repository demonstrates the difference between systems that learn rules and systems where rules are part of execution semantics.

In this system, violations are not unlikely — they are **impossible**.

## The Claim

| System | Can violation occur? | Why |
| :--- | :--- | :--- |
| **Constitutional OBDS** | ❌ **No** | Forbidden states are unreachable by construction. |
| **GNN** | ✅ **Yes** | Penalties superpose and can be outweighed by incentives. |
| **Transformer** | ✅ **Yes** | Compliance is a probabilistic attention weight. |

## How to Run

```bash
python3 demo.py
```

## Core Logic

### Constitutional OBDS (Governance as Control Flow)
```python
def step_constitutional(agent, norms, institutions):
    # ... proposal ...
    for norm in norms:
        if norm.violates(agent.state, proposed, agent.identity):
            proposed = agent.state  # Hard projection
    # ... commit ...
```

### Baselines (Governance as Weights)
- **GNN**: Adds a `-penalty` vector to the gradient.
- **Transformer**: Applies compliance with some `probability`.

Both baselines eventually fail as the drive towards the target (incentive) or random noise overwhelms the learned governance. OBDS remains invariant.
