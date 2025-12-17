#!/usr/bin/env python3
"""
Killer Visualization for Thermal Bloom Paper
=============================================
The single figure that explains everything:
- Panel A: Discrete Bloom (no gradient)
- Panel B: Thermal Bloom (continuous field)
- Panel C: Query following heat gradient
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.ndimage import gaussian_filter
from sklearn.datasets import make_blobs
import matplotlib.colors as mcolors

# Set style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14


def generate_clustered_data(n_samples=500, n_clusters=8, seed=42):
    """Generate clustered 2D data"""
    np.random.seed(seed)
    X, _ = make_blobs(n_samples=n_samples, n_features=2, centers=n_clusters,
                      cluster_std=1.2, random_state=seed)
    # Normalize to [-8, 8]
    X = (X - X.mean(axis=0)) / X.std(axis=0) * 3
    return X


def build_discrete_grid(points, grid_size=64, range_val=8):
    """Build discrete bloom grid"""
    grid = np.zeros((grid_size, grid_size))
    scale = (grid_size - 1) / (2 * range_val)

    for pt in points:
        x = int((pt[0] + range_val) * scale)
        y = int((pt[1] + range_val) * scale)
        x = np.clip(x, 0, grid_size - 1)
        y = np.clip(y, 0, grid_size - 1)
        grid[x, y] = 1

    return grid


def build_thermal_grid(points, grid_size=64, range_val=8, sigma=2.0):
    """Build thermal bloom grid with diffusion"""
    grid = build_discrete_grid(points, grid_size, range_val)
    grid = gaussian_filter(grid, sigma=sigma)
    return grid


def trace_gradient_path(grid, start_point, grid_size=64, range_val=8, max_steps=50):
    """Trace gradient ascent path from query to nearest peak"""
    scale = (grid_size - 1) / (2 * range_val)

    # Convert start point to grid coords
    x = (start_point[0] + range_val) * scale
    y = (start_point[1] + range_val) * scale

    path = [(start_point[0], start_point[1])]

    for _ in range(max_steps):
        xi, yi = int(x), int(y)

        if xi <= 0 or xi >= grid_size - 1 or yi <= 0 or yi >= grid_size - 1:
            break

        # Compute gradient
        dx = (grid[xi + 1, yi] - grid[xi - 1, yi]) / 2
        dy = (grid[xi, yi + 1] - grid[xi, yi - 1]) / 2

        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            break

        # Move uphill (in grid coords)
        step_size = 0.8
        x += dx * step_size / (abs(dx) + abs(dy) + 1e-8) * 2
        y += dy * step_size / (abs(dx) + abs(dy) + 1e-8) * 2

        # Convert back to world coords
        world_x = x / scale - range_val
        world_y = y / scale - range_val
        path.append((world_x, world_y))

    return np.array(path)


def create_killer_figure():
    """Create the killer 3-panel figure for the paper"""
    # Generate data
    points = generate_clustered_data(n_samples=400, n_clusters=8)

    # Create query point (in sparse region)
    query = np.array([-4.5, 2.5])

    # Find nearest actual point for ground truth
    distances = np.sqrt(np.sum((points - query) ** 2, axis=1))
    nearest_idx = np.argmin(distances)
    nearest = points[nearest_idx]

    # Build grids
    grid_size = 64
    range_val = 8
    sigma = 2.5

    discrete_grid = build_discrete_grid(points, grid_size, range_val)
    thermal_grid = build_thermal_grid(points, grid_size, range_val, sigma)

    # Trace gradient path
    path = trace_gradient_path(thermal_grid, query, grid_size, range_val)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    extent = [-range_val, range_val, -range_val, range_val]

    # Custom colormap: black to orange to white
    colors = ['#000000', '#1a0a00', '#3d1a00', '#662b00', '#994400',
              '#cc5500', '#ff6600', '#ff8533', '#ffa366', '#ffc299', '#ffe5cc', '#ffffff']
    thermal_cmap = mcolors.LinearSegmentedColormap.from_list('thermal', colors)

    # ========== Panel A: Discrete Bloom ==========
    ax1 = axes[0]

    # Show discrete grid (transposed for correct orientation)
    ax1.imshow(discrete_grid.T, origin='lower', extent=extent, cmap='Greys',
               aspect='equal', alpha=0.3, interpolation='nearest')

    # Plot data points
    ax1.scatter(points[:, 0], points[:, 1], s=30, c='#3366cc', alpha=0.6,
                edgecolors='white', linewidths=0.5, label='Stored items')

    # Plot query
    ax1.scatter(query[0], query[1], s=300, c='#cc0000', marker='*',
                edgecolors='black', linewidths=1.5, zorder=10, label='Query')

    # Add "?" to indicate no guidance
    ax1.text(query[0] + 0.5, query[1] + 0.5, '?', fontsize=24, color='#cc0000',
             fontweight='bold', ha='left', va='bottom')

    ax1.set_xlim(-range_val, range_val)
    ax1.set_ylim(-range_val, range_val)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('(A) Discrete Bloom Filter\nNo gradient information', fontsize=13)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.2)

    # ========== Panel B: Thermal Bloom (Heat Field) ==========
    ax2 = axes[1]

    # Show thermal field
    im = ax2.imshow(thermal_grid.T, origin='lower', extent=extent, cmap=thermal_cmap,
                    aspect='equal', interpolation='bilinear')

    # Plot data points
    ax2.scatter(points[:, 0], points[:, 1], s=30, c='white', alpha=0.8,
                edgecolors='black', linewidths=0.5)

    # Plot query
    ax2.scatter(query[0], query[1], s=300, c='#00ff00', marker='*',
                edgecolors='black', linewidths=1.5, zorder=10)

    # Add gradient vectors
    skip = 4
    X, Y = np.meshgrid(np.linspace(-range_val, range_val, grid_size // skip),
                       np.linspace(-range_val, range_val, grid_size // skip))

    # Compute gradient field
    grad_y, grad_x = np.gradient(thermal_grid)
    grad_x_sampled = grad_x[::skip, ::skip].T
    grad_y_sampled = grad_y[::skip, ::skip].T

    # Normalize for visualization
    mag = np.sqrt(grad_x_sampled**2 + grad_y_sampled**2) + 1e-8
    grad_x_norm = grad_x_sampled / mag * 0.4
    grad_y_norm = grad_y_sampled / mag * 0.4

    # Only show significant gradients
    mask = mag > 0.002
    ax2.quiver(X[mask], Y[mask], grad_x_norm[mask], grad_y_norm[mask],
               color='white', alpha=0.5, scale=15, width=0.003)

    ax2.set_xlim(-range_val, range_val)
    ax2.set_ylim(-range_val, range_val)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('(B) Thermal Bloom Filter\nContinuous potential field', fontsize=13)
    ax2.set_aspect('equal')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Heat intensity', fontsize=10)

    # ========== Panel C: Gradient Descent Path ==========
    ax3 = axes[2]

    # Show thermal field (lighter)
    ax3.imshow(thermal_grid.T, origin='lower', extent=extent, cmap=thermal_cmap,
               aspect='equal', alpha=0.5, interpolation='bilinear')

    # Plot data points
    ax3.scatter(points[:, 0], points[:, 1], s=30, c='#3366cc', alpha=0.4,
                edgecolors='white', linewidths=0.3)

    # Plot gradient descent path
    ax3.plot(path[:, 0], path[:, 1], 'g-', linewidth=3, label='Gradient path',
             solid_capstyle='round')

    # Add arrows along path
    for i in range(0, len(path) - 1, 3):
        dx = path[i + 1, 0] - path[i, 0]
        dy = path[i + 1, 1] - path[i, 1]
        ax3.annotate('', xy=(path[i + 1, 0], path[i + 1, 1]),
                     xytext=(path[i, 0], path[i, 1]),
                     arrowprops=dict(arrowstyle='->', color='#00cc00', lw=2))

    # Plot query (start)
    ax3.scatter(query[0], query[1], s=300, c='#cc0000', marker='*',
                edgecolors='black', linewidths=1.5, zorder=10, label='Query')

    # Plot endpoint (result)
    ax3.scatter(path[-1, 0], path[-1, 1], s=200, c='#00ff00', marker='o',
                edgecolors='black', linewidths=2, zorder=10, label='Result')

    # Plot nearest ground truth
    ax3.scatter(nearest[0], nearest[1], s=150, c='white', marker='D',
                edgecolors='#00cc00', linewidths=2, zorder=9, label='Nearest item')

    # Add step counter
    ax3.text(query[0] + 0.3, query[1] - 0.5, f'{len(path)} steps',
             fontsize=10, color='white', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#333333', alpha=0.8))

    ax3.set_xlim(-range_val, range_val)
    ax3.set_ylim(-range_val, range_val)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('(C) Query Follows Heat Gradient\nGradient ascent to nearest neighbor', fontsize=13)
    ax3.set_aspect('equal')
    ax3.legend(loc='lower right', fontsize=9)

    plt.tight_layout()

    # Save
    plt.savefig('figure1_thermal_bloom.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('figure1_thermal_bloom.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: figure1_thermal_bloom.png")
    print("Saved: figure1_thermal_bloom.pdf")

    plt.show()


def create_abstract_figure():
    """Create a simplified single-panel figure for talks/abstracts"""
    # Generate data
    points = generate_clustered_data(n_samples=300, n_clusters=6, seed=42)

    # Query in sparse region
    query = np.array([-5.0, 3.0])

    # Build thermal grid
    grid_size = 64
    range_val = 8
    sigma = 2.5
    thermal_grid = build_thermal_grid(points, grid_size, range_val, sigma)

    # Trace path
    path = trace_gradient_path(thermal_grid, query, grid_size, range_val)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    extent = [-range_val, range_val, -range_val, range_val]

    # Custom colormap
    colors = ['#000000', '#1a0a00', '#3d1a00', '#662b00', '#994400',
              '#cc5500', '#ff6600', '#ff8533', '#ffa366', '#ffc299', '#ffe5cc', '#ffffff']
    thermal_cmap = mcolors.LinearSegmentedColormap.from_list('thermal', colors)

    # Show thermal field
    ax.imshow(thermal_grid.T, origin='lower', extent=extent, cmap=thermal_cmap,
              aspect='equal', interpolation='bilinear')

    # Plot data points
    ax.scatter(points[:, 0], points[:, 1], s=60, c='white', alpha=0.9,
               edgecolors='black', linewidths=0.5, label='Stored items')

    # Plot path
    ax.plot(path[:, 0], path[:, 1], 'g-', linewidth=4, label='Gradient ascent')

    # Plot query and result
    ax.scatter(query[0], query[1], s=400, c='#ff0000', marker='*',
               edgecolors='white', linewidths=2, zorder=10, label='Query')
    ax.scatter(path[-1, 0], path[-1, 1], s=250, c='#00ff00', marker='o',
               edgecolors='white', linewidths=2, zorder=10, label='Result')

    ax.set_xlim(-range_val, range_val)
    ax.set_ylim(-range_val, range_val)
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    ax.set_title('Thermal Bloom Filter\nGradient-guided nearest neighbor search', fontsize=16)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('figure_abstract.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: figure_abstract.png")
    plt.show()


if __name__ == '__main__':
    create_killer_figure()
    create_abstract_figure()
