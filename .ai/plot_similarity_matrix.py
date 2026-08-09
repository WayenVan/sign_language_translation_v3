"""
绘制相似度矩阵热力图（Heatmap）
- 无 x/y 轴标签
- 带 colorbar（热力图图示）
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def plot_similarity_matrix(
    matrix: np.ndarray,
    title: str = "Similarity Matrix",
    cmap: str = "viridis",
    figsize: tuple = (8, 6),
    vmin: float | None = None,
    vmax: float | None = None,
    annot: bool = False,
    fmt: str = ".2f",
    save_path: str | None = None,
    show: bool = False,
) -> plt.Figure:
    """
    绘制相似度矩阵（无 x/y 标签，带 colorbar）。
    colorbar 宽度 = 一格宽度，高度 = 热力图总高度，刻度对齐网格线。

    Parameters
    ----------
    matrix : np.ndarray
        形状为 (N, N) 的相似度矩阵。
    title : str
        图标题（已弃用，默认隐藏）。
    cmap : str or Colormap
        Matplotlib / Seaborn 颜色映射名称。
    figsize : tuple
        画布尺寸 (width, height)。
    vmin, vmax : float or None
        色彩映射的值域范围。None 表示自动从数据推断。
    annot : bool
        是否在每个格子中显示数值。
    fmt : str
        数值格式化字符串（仅在 annot=True 时有效）。
    save_path : str or None
        非 None 时保存图像到指定路径。
    show : bool
        是否调用 plt.show()。

    Returns
    -------
    plt.Figure
    """
    n = matrix.shape[0]
    if vmin is None:
        vmin = matrix.min()
    if vmax is None:
        vmax = matrix.max()

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("none")  # 透明画布
    ax.set_facecolor("none")  # 透明坐标区域

    # ── 第一步：画热力图（不含 colorbar） ────────────────────
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=annot,
        fmt=fmt,
        square=True,
        linewidths=0.0,
        xticklabels=False,
        yticklabels=False,
        cbar=False,  # 手动绘制 colorbar
    )

    # ── 彻底移除热力图轴刻度与边框 ──────────────────────────
    ax.tick_params(left=False, bottom=False, which="both", pad=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ── 先布局定稿，再获取精确坐标 ──────────────────────────
    fig.tight_layout()

    # ── 计算一格宽度 & 定位 colorbar ────────────────────────
    heatmap_bbox = ax.get_position()  # [x0, y0, width, height]
    cell_w = heatmap_bbox.width / n

    # 在 heatmap 右侧紧贴放置 colorbar，宽度 = 一格，高度对齐
    gap = cell_w * 0.3  # 微小间距
    cbar_ax = fig.add_axes(
        [
            heatmap_bbox.x1 + gap,  # x: 紧贴 heatmap 右侧
            heatmap_bbox.y0,  # y: 底部对齐
            cell_w / 2.0,  # width: 一格宽
            heatmap_bbox.height,  # height: 与 heatmap 等高
        ]
    )

    # ── 创建 colorbar ──────────────────────────────────────
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.outline.set_linewidth(0)  # 去掉外框
    cbar_ax.set_facecolor("none")  # colorbar 背景透明

    # ── 刻度位置：固定为 [0.0, 0.5, 1.0] ──────────────────
    cbar.set_ticks([0.0, 0.5, 1.0])

    # ── 加粗刻度线 & 设置刻度标签字体大小 ────────────────
    cbar.ax.tick_params(width=1, length=6, which="major", labelsize=30)
    for line in cbar.ax.yaxis.get_ticklines():
        line.set_markeredgewidth(1)
    # 设置刻度标签字体为 Inter
    for label in cbar.ax.get_yticklabels():
        label.set_fontname("Inter")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", transparent=True)
    if show:
        plt.show()

    return fig


# ────────────────────────────────────────────────────────────────
#  示例用法
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 生成一个 7×7 的随机相似度矩阵
    np.random.seed(42)
    n = 7
    raw = np.random.randn(n, n)
    rmin, rmax = raw.min(), raw.max()
    sim = (raw - rmin) / (rmax - rmin + 1e-12)

    plot_similarity_matrix(
        sim,
        title="Similarity Matrix",
        cmap="viridis",
        figsize=(5, 4),
        annot=False,
        show=False,
        save_path="outputs/similarity.svg",
    )
