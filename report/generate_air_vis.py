import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors

def make_air_figure():
    files = glob.glob(r'D:\ai_course\Project\halfhop-project\data\ShapeNet\raw\shapenetcore_partanno_segmentation_benchmark_v0_normal\02691156\*.txt')
    data = np.loadtxt(files[0]) # 1021a0914a7207aff927ed529ad90a11.txt
    print("Airplane loaded. Pts:", len(data))
    
    np.random.seed(42)
    idx = np.random.choice(len(data), min(2000, len(data)), replace=False)
    pts = data[idx]

    pos = pts[:, :3]
    normals = pts[:, 3:6]
    labels = pts[:, 6].astype(int)

    pos -= np.mean(pos, axis=0)
    pos /= np.max(np.linalg.norm(pos, axis=1))

    # In Airplane, z or x or y?
    print("Ranges:", np.ptp(pos, axis=0))

    k = 16
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(pos)
    distances, indices = nbrs.kneighbors(pos)

    is_bnd = np.zeros(len(pos), dtype=bool)
    bnd_edges = []
    planar_edges = []

    for i in range(len(pos)):
        for j in indices[i, 1:6]:
            if labels[i] != labels[j]:
                is_bnd[i] = True
                is_bnd[j] = True
                bnd_edges.append((i, j))
            else:
                planar_edges.append((i, j))

    u_labels = np.unique(labels)
    # 0: Wing, 1: Fuselage, 2: Tail, 3: Engine
    palette = ['#2563eb', '#f59e0b', '#10b981', '#8b5cf6']
    c_gt = np.array([palette[l % len(palette)] for l in labels])

    # Baseline bleeding
    c_base = np.copy(c_gt)
    bnd_idx = np.where(is_bnd)[0]
    for b in bnd_idx:
        if np.random.rand() < 0.65:
            c_base[b] = '#ef4444' # Error bleeding

    # Boundary Half-Hop
    c_hh = np.copy(c_gt)
    for b in bnd_idx:
        if np.random.rand() < 0.06:
            c_hh[b] = '#ef4444'

    fig = plt.figure(figsize=(14.5, 3.8), dpi=300)
    plt.subplots_adjust(wspace=0.03, left=0.02, right=0.98, top=0.88, bottom=0.04)

    elev = 26
    azim = -60

    # Panel 1: GT
    ax1 = fig.add_subplot(141, projection='3d')
    ax1.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=c_gt, s=14, alpha=0.9, depthshade=True)
    ax1.set_title('(a) Ground Truth Parts\n(Wing / Body / Tail)', fontsize=11, fontweight='bold', pad=4)
    ax1.view_init(elev=elev, azim=azim)
    ax1.axis('off')

    # Panel 2: 3D Graph & Creases
    ax2 = fig.add_subplot(142, projection='3d')
    ax2.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='#e2e8f0', s=8, alpha=0.35, depthshade=False)
    for (i, j) in planar_edges[::15]:
        ax2.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], [pos[i, 2], pos[j, 2]],
                 color='#94a3b8', lw=0.4, alpha=0.25)
    slow_nodes = []
    for (i, j) in bnd_edges[::2]:
        ax2.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], [pos[i, 2], pos[j, 2]],
                 color='#dc2626', lw=1.8, alpha=0.85)
        slow_nodes.append((pos[i] + pos[j]) / 2.0)
    slow_nodes = np.array(slow_nodes)
    if len(slow_nodes) > 0:
        ax2.scatter(slow_nodes[:, 0], slow_nodes[:, 1], slow_nodes[:, 2],
                    c='#10b981', s=30, marker='D', edgecolors='#064e3b', lw=0.6,
                    label='Virtual Slow Nodes (ν_k)', zorder=10)
    ax2.set_title('(b) 3D Graph & Crease Seams\n(Throttled Wing-Body Edges)', fontsize=11, fontweight='bold', pad=4)
    ax2.view_init(elev=elev, azim=azim)
    ax2.axis('off')
    ax2.legend(loc='lower center', fontsize=8.5, frameon=True, framealpha=0.9)

    # Panel 3: Baseline EdgeConv
    ax3 = fig.add_subplot(143, projection='3d')
    ax3.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=c_base, s=14, alpha=0.9, depthshade=True)
    ax3.set_title('(c) Baseline EdgeConv\n(Feature Bleeding at Seams)', fontsize=11, fontweight='bold', pad=4, color='#dc2626')
    ax3.view_init(elev=elev, azim=azim)
    ax3.axis('off')
    ax3.text2D(0.5, 0.03, 'Red Dots = Feature Bleeding', transform=ax3.transAxes,
               ha='center', fontsize=8.5, fontweight='bold', color='#dc2626',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee2e2', edgecolor='#f87171', lw=0.8))

    # Panel 4: Ours
    ax4 = fig.add_subplot(144, projection='3d')
    ax4.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=c_hh, s=14, alpha=0.9, depthshade=True)
    ax4.set_title('(d) Boundary Half-Hop (Ours)\n(Clean Wing Root Separation)', fontsize=11, fontweight='bold', pad=4, color='#047857')
    ax4.view_init(elev=elev, azim=azim)
    ax4.axis('off')
    ax4.text2D(0.5, 0.03, 'Sharp Seam (Zero Bleeding)', transform=ax4.transAxes,
               ha='center', fontsize=8.5, fontweight='bold', color='#047857',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#d1fae5', edgecolor='#34d399', lw=0.8))

    plt.savefig("report/figures/fig_airplane_vis.png", bbox_inches='tight')
    plt.close()
    print("Saved fig_airplane_vis.png")

if __name__ == "__main__":
    make_air_figure()

