from data_generation import generate_color_dataset

examples, concepts, hierarchy, relationships = generate_color_dataset(n_samples=100, dim=32, max_depth=5)

print(f"Hierarchy keys: {list(hierarchy.keys())}")
for d in sorted(hierarchy.keys()):
    print(f"Depth {d}: {hierarchy[d]}")

if 5 in hierarchy:
    print("SUCCESS: Depth 5 generated.")
else:
    print("FAILURE: Depth 5 missing.")
