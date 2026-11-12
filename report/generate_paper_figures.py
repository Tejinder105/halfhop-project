import os
import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.5,
    'figure.titlesize': 13,
    'figure.dpi': 300,
    'lines.linewidth': 1.8,
    'grid.alpha': 0.35,
    'grid.linestyle': '--'
})

out_dir = "report/figures"
os.makedirs(out_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# Figure 1: 2D Heterophilic Graph Reproduction (GCN & GraphSAGE)
# -----------------------------------------------------------------------------
def plot_reproduction_bars():
    datasets = ['Texas', 'Wisconsin', 'Actor', 'Squirrel', 'Chameleon', 'Cornell']
    gcn_base = [58.65, 52.35, 28.68, 26.83, 37.83, 41.35]
    gcn_hh   = [73.51, 55.49, 30.99, 29.63, 44.41, 62.97]
    sage_base = [76.76, 75.10, 33.88, 36.27, 50.48, 72.43]
    sage_hh   = [81.35, 79.22, 34.89, 37.10, 50.88, 67.84]

    x = np.arange(len(datasets))
    width = 0.20

    fig, ax = plt.subplots(figsize=(7.2, 3.6))

    rects1 = ax.bar(x - 1.5*width, gcn_base, width, label='GCN Base', color='#93c5fd', edgecolor='#1e40af')
    rects2 = ax.bar(x - 0.5*width, gcn_hh,   width, label='GCN + Half-Hop', color='#1d4ed8', edgecolor='#1e3a8a')
    rects3 = ax.bar(x + 0.5*width, sage_base, width, label='GraphSAGE Base', color='#a7f3d0', edgecolor='#065f46')
    rects4 = ax.bar(x + 1.5*width, sage_hh,   width, label='GraphSAGE + Half-Hop', color='#047857', edgecolor='#064e3b')

    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Reproduction: 10-Split Mean Accuracy on Heterophilic Benchmarks')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(20, 92)
    ax.legend(ncol=2, loc='upper left', frameon=True, framealpha=0.95)
    ax.grid(axis='y')

    # Add delta annotations for prominent gains
    ax.annotate('+14.86%', xy=(x[0] - 0.5*width, 73.51), xytext=(x[0] - 0.8*width, 79),
                arrowprops=dict(arrowstyle="->", color='#1d4ed8', lw=1.2),
                fontsize=8.5, fontweight='bold', color='#1d4ed8')
    ax.annotate('+21.62%', xy=(x[5] - 0.5*width, 62.97), xytext=(x[5] - 0.8*width, 68),
                arrowprops=dict(arrowstyle="->", color='#1d4ed8', lw=1.2),
                fontsize=8.5, fontweight='bold', color='#1d4ed8')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig_reproduction_bars.pdf'))
    plt.savefig(os.path.join(out_dir, 'fig_reproduction_bars.png'), dpi=300)
    plt.close()
    print("Saved fig_reproduction_bars")

# -----------------------------------------------------------------------------
# Figure 2: Conceptual Diagram of Geometry-Aware Boundary Half-Hop
# -----------------------------------------------------------------------------
def plot_concept_diagram():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))

    # Subplot 1: Baseline EdgeConv (Feature Bleeding)
    ax1.set_title('(a) Baseline Point Cloud Graph\n(Indiscriminate Message Mixing)', fontsize=10.5, pad=8)
    # Part A (Seat/Top) points
    top_x = np.array([0.2, 0.4, 0.6, 0.8, 0.5])
    top_y = np.array([0.7, 0.72, 0.69, 0.71, 0.75])
    # Part B (Leg) points
    leg_x = np.array([0.5, 0.5, 0.52, 0.48])
    leg_y = np.array([0.5, 0.35, 0.2, 0.1])

    ax1.scatter(top_x, top_y, color='#2563eb', s=90, label='Part A (e.g. Tabletop)', zorder=4)
    ax1.scatter(leg_x, leg_y, color='#d97706', s=90, label='Part B (e.g. Leg)', zorder=4)

    # Internal edges
    for i in range(len(top_x)-1):
        ax1.plot([top_x[i], top_x[i+1]], [top_y[i], top_y[i+1]], color='#93c5fd', lw=1.5, zorder=2)
    for i in range(len(leg_x)-1):
        ax1.plot([leg_x[i], leg_x[i+1]], [leg_y[i], leg_y[i+1]], color='#fde68a', lw=1.5, zorder=2)

    # Boundary cross-edge (bleeds features)
    ax1.annotate('', xy=(top_x[4], top_y[4]), xytext=(leg_x[0], leg_y[0]),
                 arrowprops=dict(arrowstyle="<->", color='#dc2626', lw=2.5, ls='--'), zorder=3)
    ax1.text(0.53, 0.61, 'Rapid Feature\nBleeding (1-hop)', color='#dc2626', fontsize=8.5, fontweight='bold')

    ax1.set_xlim(0.05, 0.95)
    ax1.set_ylim(0.0, 0.9)
    ax1.axis('off')
    ax1.legend(loc='lower left', fontsize=8, frameon=True)

    # Subplot 2: Geometry-Aware Boundary Half-Hop
    ax2.set_title('(b) Geometry-Aware Boundary Half-Hop\n(Dual-Gated Crease Throttling)', fontsize=10.5, pad=8)
    ax2.scatter(top_x, top_y, color='#2563eb', s=90, zorder=4)
    ax2.scatter(leg_x, leg_y, color='#d97706', s=90, zorder=4)

    # Internal edges (unimpeded)
    for i in range(len(top_x)-1):
        ax2.plot([top_x[i], top_x[i+1]], [top_y[i], top_y[i+1]], color='#93c5fd', lw=1.5, zorder=2)
    for i in range(len(leg_x)-1):
        ax2.plot([leg_x[i], leg_x[i+1]], [leg_y[i], leg_y[i+1]], color='#fde68a', lw=1.5, zorder=2)

    # Virtual Slow Node at boundary
    slow_x = (top_x[4] + leg_x[0]) / 2.0
    slow_y = (top_y[4] + leg_y[0]) / 2.0
    ax2.scatter(slow_x, slow_y, color='#059669', s=130, marker='D', label='Virtual Slow Node (ν_k)', zorder=5)

    # Asymmetric connections: leg -> slow <-> top
    ax2.annotate('', xy=(slow_x, slow_y), xytext=(leg_x[0], leg_y[0]),
                 arrowprops=dict(arrowstyle="->", color='#059669', lw=2.0), zorder=3)
    ax2.annotate('', xy=(top_x[4], top_y[4]), xytext=(slow_x, slow_y),
                 arrowprops=dict(arrowstyle="<->", color='#059669', lw=2.0), zorder=3)
    ax2.text(0.56, 0.60, 'Boundary Delay\nPreserves Seam', color='#059669', fontsize=8.5, fontweight='bold')

    ax2.text(0.12, 0.78, 'Smooth Plane: Direct 1-hop', color='#1e40af', fontsize=8, style='italic')

    ax2.set_xlim(0.05, 0.95)
    ax2.set_ylim(0.0, 0.9)
    ax2.axis('off')
    ax2.legend(loc='lower left', fontsize=8, frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig_boundary_halfhop_concept.pdf'))
    plt.savefig(os.path.join(out_dir, 'fig_boundary_halfhop_concept.png'), dpi=300)
    plt.close()
    print("Saved fig_boundary_halfhop_concept")

# -----------------------------------------------------------------------------
# Figure 3: Anti-Starvation Threshold Tuning (Chair Ablation)
# -----------------------------------------------------------------------------
def plot_chair_threshold_sweep():
    quantiles = [0.85, 0.90, 0.95, 0.98]
    miou = [84.76, 84.78, 85.13, 85.28]
    accuracy = [93.77, 93.64, 94.13, 94.18]

    fig, ax1 = plt.subplots(figsize=(6.0, 3.2))

    color_miou = '#2563eb'
    ax1.set_xlabel('Geometric Crease Quantile Threshold (Q)')
    ax1.set_ylabel('Test mIoU (%)', color=color_miou, fontweight='bold')
    line1 = ax1.plot(quantiles, miou, color=color_miou, marker='o', markersize=7, lw=2.2, label='Test mIoU')
    ax1.tick_params(axis='y', labelcolor=color_miou)
    ax1.set_ylim(84.5, 85.5)
    ax1.set_xticks(quantiles)
    ax1.set_xticklabels(['Q=0.85\n(Top 15%)', 'Q=0.90\n(Top 10%)', 'Q=0.95\n(Top 5%)', 'Q=0.98\n(Top 2%)'])

    # Annotate gain
    ax1.annotate('Monotonic Gain (+0.52%)', xy=(0.98, 85.28), xytext=(0.91, 85.35),
                 arrowprops=dict(arrowstyle="->", color=color_miou, lw=1.5),
                 fontweight='bold', color=color_miou, fontsize=9)

    ax2 = ax1.twinx()
    color_acc = '#059669'
    ax2.set_ylabel('Test Accuracy (%)', color=color_acc, fontweight='bold')
    line2 = ax2.plot(quantiles, accuracy, color=color_acc, marker='s', markersize=7, lw=2.2, linestyle='--', label='Test Accuracy')
    ax2.tick_params(axis='y', labelcolor=color_acc)
    ax2.set_ylim(93.4, 94.4)

    plt.title('Threshold Optimization: Eliminating Receptive-Field Starvation')
    ax1.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig_chair_threshold_sweep.pdf'))
    plt.savefig(os.path.join(out_dir, 'fig_chair_threshold_sweep.png'), dpi=300)
    plt.close()
    print("Saved fig_chair_threshold_sweep")

# -----------------------------------------------------------------------------
# Figure 4: Rigid Category Positive Gains (Table & Airplane)
# -----------------------------------------------------------------------------
def plot_rigid_gains():
    categories = ['Table\n(Overall mIoU)', 'Table\n(Boundary mIoU)',
                  'Airplane\n(Overall mIoU)', 'Airplane\n(Boundary mIoU)']
    baseline = [74.32, 54.63, 77.95, 57.41]
    boundary_hh = [76.19, 56.73, 78.23, 58.44]
    deltas = ['+1.87%', '+2.10%', '+0.28%', '+1.03%']

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.8, 3.4))

    rects1 = ax.bar(x - width/2, baseline, width, label='Standard EdgeConv Baseline', color='#cbd5e1', edgecolor='#475569')
    rects2 = ax.bar(x + width/2, boundary_hh, width, label='Geometry-Aware Boundary Half-Hop', color='#0284c7', edgecolor='#0369a1')

    ax.set_ylabel('Score (%)')
    ax.set_title('Performance Gains on Rigid Categories (ShapeNet-Part)')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(45, 86)
    ax.legend(loc='upper left', frameon=True, framealpha=0.95)
    ax.grid(axis='y')

    # Add delta badges on top of bars
    for idx, (b, val, delta) in enumerate(zip(baseline, boundary_hh, deltas)):
        ax.text(idx + width/2, val + 1.2, delta, ha='center', va='bottom',
                fontweight='bold', fontsize=9.5, color='#0369a1')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig_rigid_gains.pdf'))
    plt.savefig(os.path.join(out_dir, 'fig_rigid_gains.png'), dpi=300)
    plt.close()
    print("Saved fig_rigid_gains")

if __name__ == "__main__":
    plot_reproduction_bars()
    plot_concept_diagram()
    plot_chair_threshold_sweep()
    plot_rigid_gains()
    print("All publication figures successfully generated!")
