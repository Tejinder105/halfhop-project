import os
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# UNIFIED PUBLICATION COLOR PALETTE
# =============================================================================
# Primary Method (Half-Hop / Ours / +HH / Tabletop / mIoU): Deep Royal Blue
CLR_PRIMARY        = '#1D4ED8' # Deep Royal Blue
CLR_PRIMARY_LIGHT  = '#93C5FD' # Soft Light Blue (for Baseline paired with GCN)
CLR_PRIMARY_DARK   = '#1E3A8A' # Dark Navy outline

# Secondary / Complementary (Part B Legs / Accuracy / SAGE): Warm Amber
CLR_SECONDARY      = '#D97706' # Warm Amber / Gold
CLR_SECONDARY_LIGHT= '#FDE68A' # Soft Light Amber (for Baseline paired with SAGE)
CLR_SECONDARY_DARK = '#B45309' # Dark Amber outline

# Baseline Neutral: Cool Slate Gray
CLR_BASE_SLATE     = '#94A3B8' # Slate-400
CLR_BASE_DARK      = '#475569' # Slate-600 outline
CLR_BASE_LIGHT     = '#E2E8F0' # Slate-200 background fill

# Virtual Slow Nodes & Sharp Seam Highlights: Emerald Green
CLR_SLOW_NODE      = '#059669' # Emerald Green
CLR_GREEN_TINT     = '#D1FAE5' # Soft Green Tint
CLR_GREEN_DARK     = '#065F46' # Dark Emerald outline

# Crease Edges & Feature Bleeding Alerts: Crimson Red
CLR_CREASE         = '#DC2626' # Crimson Red
CLR_RED_TINT       = '#FEE2E2' # Soft Red Tint
CLR_RED_DARK       = '#991B1B' # Dark Red outline

# Publication Style Configuration
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 10.5,
    'axes.titlesize': 11.5,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 9.0,
    'figure.titlesize': 12.5,
    'figure.dpi': 300,
    'lines.linewidth': 1.8,
    'grid.alpha': 0.35,
    'grid.linestyle': '--'
})

out_dir = "report/figures"
os.makedirs(out_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# Figure 3: 2D Heterophilic Graph Reproduction (GCN & GraphSAGE)
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

    rects1 = ax.bar(x - 1.5*width, gcn_base, width, label='GCN Base',
                    color=CLR_PRIMARY_LIGHT, edgecolor=CLR_PRIMARY_DARK, lw=1.0)
    rects2 = ax.bar(x - 0.5*width, gcn_hh,   width, label='GCN + Half-Hop',
                    color=CLR_PRIMARY, edgecolor=CLR_PRIMARY_DARK, lw=1.2)
    rects3 = ax.bar(x + 0.5*width, sage_base, width, label='GraphSAGE Base',
                    color=CLR_SECONDARY_LIGHT, edgecolor=CLR_SECONDARY_DARK, lw=1.0)
    rects4 = ax.bar(x + 1.5*width, sage_hh,   width, label='GraphSAGE + Half-Hop',
                    color=CLR_SECONDARY, edgecolor=CLR_SECONDARY_DARK, lw=1.2)

    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Reproduction: 10-Split Mean Accuracy on Heterophilic Benchmarks')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(20, 92)
    ax.legend(ncol=2, loc='upper left', frameon=True, framealpha=0.95)
    ax.grid(axis='y')

    # Add delta annotations for prominent gains
    ax.annotate('+14.86%', xy=(x[0] - 0.5*width, 73.51), xytext=(x[0] - 0.85*width, 79),
                arrowprops=dict(arrowstyle="->", color=CLR_PRIMARY, lw=1.3),
                fontsize=8.5, fontweight='bold', color=CLR_PRIMARY)
    ax.annotate('+21.62%', xy=(x[5] - 0.5*width, 62.97), xytext=(x[5] - 0.85*width, 68),
                arrowprops=dict(arrowstyle="->", color=CLR_PRIMARY, lw=1.3),
                fontsize=8.5, fontweight='bold', color=CLR_PRIMARY)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig_reproduction_bars.pdf'))
    plt.savefig(os.path.join(out_dir, 'fig_reproduction_bars.png'), dpi=300)
    plt.close()
    print("Saved fig_reproduction_bars (Uniform Palette)")

# -----------------------------------------------------------------------------
# Figure 1: Conceptual Diagram of Geometry-Aware Boundary Half-Hop
# -----------------------------------------------------------------------------
def plot_concept_diagram():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))

    # Subplot 1: Baseline EdgeConv (Feature Bleeding)
    ax1.set_title('(a) Baseline Point Cloud Graph\n(Indiscriminate Message Mixing)', fontsize=10.5, pad=8)
    top_x = np.array([0.2, 0.4, 0.6, 0.8, 0.5])
    top_y = np.array([0.7, 0.72, 0.69, 0.71, 0.75])
    leg_x = np.array([0.5, 0.5, 0.52, 0.48])
    leg_y = np.array([0.5, 0.35, 0.2, 0.1])

    ax1.scatter(top_x, top_y, color=CLR_PRIMARY, s=90, label='Part A (Tabletop)', zorder=4, edgecolors=CLR_PRIMARY_DARK)
    ax1.scatter(leg_x, leg_y, color=CLR_SECONDARY, s=90, label='Part B (Leg)', zorder=4, edgecolors=CLR_SECONDARY_DARK)

    # Internal edges
    for i in range(len(top_x)-1):
        ax1.plot([top_x[i], top_x[i+1]], [top_y[i], top_y[i+1]], color=CLR_PRIMARY_LIGHT, lw=1.5, zorder=2)
    for i in range(len(leg_x)-1):
        ax1.plot([leg_x[i], leg_x[i+1]], [leg_y[i], leg_y[i+1]], color=CLR_SECONDARY_LIGHT, lw=1.5, zorder=2)

    # Boundary cross-edge (bleeds features)
    ax1.annotate('', xy=(top_x[4], top_y[4]), xytext=(leg_x[0], leg_y[0]),
                 arrowprops=dict(arrowstyle="<->", color=CLR_CREASE, lw=2.5, ls='--'), zorder=3)
    ax1.text(0.53, 0.61, 'Rapid Feature\nBleeding (1-hop)', color=CLR_CREASE, fontsize=8.5, fontweight='bold')

    ax1.set_xlim(0.05, 0.95)
    ax1.set_ylim(0.0, 0.9)
    ax1.axis('off')
    ax1.legend(loc='lower left', fontsize=8, frameon=True)

    # Subplot 2: Geometry-Aware Boundary Half-Hop
    ax2.set_title('(b) Geometry-Aware Boundary Half-Hop\n(Dual-Gated Crease Throttling)', fontsize=10.5, pad=8)
    ax2.scatter(top_x, top_y, color=CLR_PRIMARY, s=90, zorder=4, edgecolors=CLR_PRIMARY_DARK)
    ax2.scatter(leg_x, leg_y, color=CLR_SECONDARY, s=90, zorder=4, edgecolors=CLR_SECONDARY_DARK)

    # Internal edges (unimpeded)
    for i in range(len(top_x)-1):
        ax2.plot([top_x[i], top_x[i+1]], [top_y[i], top_y[i+1]], color=CLR_PRIMARY_LIGHT, lw=1.5, zorder=2)
    for i in range(len(leg_x)-1):
        ax2.plot([leg_x[i], leg_x[i+1]], [leg_y[i], leg_y[i+1]], color=CLR_SECONDARY_LIGHT, lw=1.5, zorder=2)

    # Virtual Slow Node at boundary
    slow_x = (top_x[4] + leg_x[0]) / 2.0
    slow_y = (top_y[4] + leg_y[0]) / 2.0
    ax2.scatter(slow_x, slow_y, color=CLR_SLOW_NODE, s=130, marker='D',
                edgecolors=CLR_GREEN_DARK, lw=1.0, label='Virtual Slow Node (ν_k)', zorder=5)

    # Asymmetric connections: leg -> slow <-> top
    ax2.annotate('', xy=(slow_x, slow_y), xytext=(leg_x[0], leg_y[0]),
                 arrowprops=dict(arrowstyle="->", color=CLR_SLOW_NODE, lw=2.0), zorder=3)
    ax2.annotate('', xy=(top_x[4], top_y[4]), xytext=(slow_x, slow_y),
                 arrowprops=dict(arrowstyle="<->", color=CLR_SLOW_NODE, lw=2.0), zorder=3)
    ax2.text(0.56, 0.60, 'Boundary Delay\nPreserves Seam', color=CLR_SLOW_NODE, fontsize=8.5, fontweight='bold')

    ax2.text(0.12, 0.78, 'Smooth Plane: Direct 1-hop', color=CLR_PRIMARY_DARK, fontsize=8, style='italic')

    ax2.set_xlim(0.05, 0.95)
    ax2.set_ylim(0.0, 0.9)
    ax2.axis('off')
    ax2.legend(loc='lower left', fontsize=8, frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig_boundary_halfhop_concept.pdf'))
    plt.savefig(os.path.join(out_dir, 'fig_boundary_halfhop_concept.png'), dpi=300)
    plt.close()
    print("Saved fig_boundary_halfhop_concept (Uniform Palette)")

# -----------------------------------------------------------------------------
# Figure 4: Anti-Starvation Threshold Tuning (Chair Ablation)
# -----------------------------------------------------------------------------
def plot_chair_threshold_sweep():
    quantiles = [0.85, 0.90, 0.95, 0.98]
    miou = [84.76, 84.78, 85.13, 85.28]
    accuracy = [93.77, 93.64, 94.13, 94.18]

    fig, ax1 = plt.subplots(figsize=(6.0, 3.2))

    ax1.set_xlabel('Geometric Crease Quantile Threshold (Q)')
    ax1.set_ylabel('Test mIoU (%)', color=CLR_PRIMARY, fontweight='bold')
    line1 = ax1.plot(quantiles, miou, color=CLR_PRIMARY, marker='o', markersize=7, lw=2.2, label='Test mIoU')
    ax1.tick_params(axis='y', labelcolor=CLR_PRIMARY)
    ax1.set_ylim(84.5, 85.5)
    ax1.set_xticks(quantiles)
    ax1.set_xticklabels(['Q=0.85\n(Top 15%)', 'Q=0.90\n(Top 10%)', 'Q=0.95\n(Top 5%)', 'Q=0.98\n(Top 2%)'])

    # Annotate gain
    ax1.annotate('Monotonic Gain (+0.52%)', xy=(0.98, 85.28), xytext=(0.91, 85.35),
                 arrowprops=dict(arrowstyle="->", color=CLR_PRIMARY, lw=1.5),
                 fontweight='bold', color=CLR_PRIMARY, fontsize=9)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Test Accuracy (%)', color=CLR_SECONDARY, fontweight='bold')
    line2 = ax2.plot(quantiles, accuracy, color=CLR_SECONDARY, marker='s', markersize=7, lw=2.2, linestyle='--', label='Test Accuracy')
    ax2.tick_params(axis='y', labelcolor=CLR_SECONDARY)
    ax2.set_ylim(93.4, 94.4)

    plt.title('Threshold Optimization: Eliminating Receptive-Field Starvation')
    ax1.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig_chair_threshold_sweep.pdf'))
    plt.savefig(os.path.join(out_dir, 'fig_chair_threshold_sweep.png'), dpi=300)
    plt.close()
    print("Saved fig_chair_threshold_sweep (Uniform Palette)")

# -----------------------------------------------------------------------------
# Figure 6: Rigid Category Positive Gains (Table & Airplane)
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

    rects1 = ax.bar(x - width/2, baseline, width, label='Standard EdgeConv Baseline',
                    color=CLR_BASE_SLATE, edgecolor=CLR_BASE_DARK, lw=1.0)
    rects2 = ax.bar(x + width/2, boundary_hh, width, label='Geometry-Aware Boundary Half-Hop',
                    color=CLR_PRIMARY, edgecolor=CLR_PRIMARY_DARK, lw=1.2)

    ax.set_ylabel('Score (%)')
    ax.set_title('Performance Gains on Rigid Categories (ShapeNet-Part)')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(45, 86)
    ax.legend(loc='upper left', frameon=True, framealpha=0.95)
    ax.grid(axis='y')

    # Add delta badges on top of bars in uniform Primary Blue
    for idx, (b, val, delta) in enumerate(zip(baseline, boundary_hh, deltas)):
        ax.text(idx + width/2, val + 1.2, delta, ha='center', va='bottom',
                fontweight='bold', fontsize=9.5, color=CLR_PRIMARY)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig_rigid_gains.pdf'))
    plt.savefig(os.path.join(out_dir, 'fig_rigid_gains.png'), dpi=300)
    plt.close()
    print("Saved fig_rigid_gains (Uniform Palette)")

if __name__ == "__main__":
    plot_reproduction_bars()
    plot_concept_diagram()
    plot_chair_threshold_sweep()
    plot_rigid_gains()
    print("All standard figures re-generated with unified palette!")
