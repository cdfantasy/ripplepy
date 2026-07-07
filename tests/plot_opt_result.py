#!/usr/bin/env python3
"""Plot optimisation results with publication-quality styling.

Usage
-----
    python tests/plot_opt_result.py
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import numpy as np
import csv


# ═══════════════════════════════════════════════════════════════════════════
# 0. 出版级全局样式
# ═══════════════════════════════════════════════════════════════════════════

def set_publication_style():
    """Configure matplotlib rcParams for publication-quality output."""
    plt.rcParams.update({
        # ── 字体 ──
        'font.family': 'serif',
        'font.size': 13,
        'axes.labelsize': 15,
        'axes.titlesize': 15,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        # ── 线条 / 标记 ──
        'lines.linewidth': 1.4,
        'lines.markersize': 7,
        'lines.markeredgewidth': 0.6,
        # ── 刻度 ──
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4.5,
        'ytick.major.size': 4.5,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.minor.size': 2.5,
        'ytick.minor.size': 2.5,
        'xtick.minor.width': 0.7,
        'xtick.top': True,
        'ytick.right': True,
        # ── 坐标轴 ──
        'axes.linewidth': 1.1,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linewidth': 0.5,
        'grid.linestyle': '--',
        # ── 图例 ──
        'legend.frameon': True,
        'legend.framealpha': 0.85,
        'legend.edgecolor': '#cccccc',
        'legend.fancybox': False,
        # ── 输出 ──
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.08,
        # ── LaTeX ──
        'text.usetex': False,
        'mathtext.fontset': 'stix',
    })


set_publication_style()


# ═══════════════════════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════════════════════

QUANTITY_LABELS = {
    'epsilon_eff': r'$\varepsilon_{\mathrm{eff}}^{3/2}$',
    'iota':        r'$\iota$',
    'volume':      r'$V$  [m$^3$]',
    'Aspect ratio': r'$A$  ($R_0 / a$)',
    'average B':   r'$\langle B \rangle$  [T]',
}

PHYSICAL_QUANTITIES = list(QUANTITY_LABELS.keys())


# ═══════════════════════════════════════════════════════════════════════════
# 1. 辅助函数：数据验证
# ═══════════════════════════════════════════════════════════════════════════

def is_valid_entry(entry):
    """
    检查个体是否有效。
    
    根据 failure_flag 判断：
    - failure_flag 为 False 或 'none' 表示有效
    - 其他值（如 'tracing_failed'）表示无效
    """
    if entry is None:
        return False
    # 从CSV读取的 failure_flag 可能是 bool 或 string
    failure_flag = entry.get('failure_flag', True)
    # 如果是字符串，检查是否为 'none' 或 'False'
    if isinstance(failure_flag, str):
        return failure_flag.lower() in ('none', 'false', '')
    # 如果是布尔值
    return not bool(failure_flag)


# ═══════════════════════════════════════════════════════════════════════════
# 2. 数据加载
# ═══════════════════════════════════════════════════════════════════════════

def load_optimization_log(log_file):
    """读取 CSV 日志，返回结构化数据.

    Returns
    -------
    data : dict[generation][individual] → {物理量: float, 'extcur': [...], 'failure_flag': ...}
    generations : list[int]
    max_individual : int
    start_data : dict or None
    """
    data = {}
    start_data = None
    
    def safe_float(val):
        """安全转换，空值返回 NaN."""
        if val is None or val == '' or val.strip() == '':
            return float('nan')
        try:
            return float(val)
        except (ValueError, TypeError):
            return float('nan')
    
    def parse_failure_flag(val):
        """解析 failure_flag."""
        if val is None or val == '':
            return None
        if val.lower() in ('none', 'false', '0'):
            return False
        if val.lower() in ('true', '1'):
            return True
        return val  # 返回原始字符串（如 'tracing_failed'）
    
    with open(log_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gen_raw = row.get('Generation', '')
            ind = int(row.get('Individual', 0))
            
            if gen_raw == 'start':
                gen = 'start'
            else:
                try:
                    gen = int(gen_raw)
                except ValueError:
                    continue
            
            if gen not in data:
                data[gen] = {}
            
            extcur_raw = row.get('extcur', '[]')
            extcur = [float(x) for x in extcur_raw.strip('[]').split(',') if x.strip()]
            
            entry = {
                'epsilon_eff':   safe_float(row.get('epsilon_eff', '')),
                'iota':          safe_float(row.get('iota', '')),
                'volume':        safe_float(row.get('volume', '')),
                'Aspect ratio':  safe_float(row.get('Aspect ratio', '')),
                'average B':     safe_float(row.get('average B', '')),
                'extcur':        extcur,
                'failure_flag':  parse_failure_flag(row.get('failure_flag', '')),
                'failure_reason': row.get('failure_reason', ''),
            }
            data[gen][ind] = entry
            
            if gen == 'start':
                start_data = entry

    generations = sorted([g for g in data.keys() if isinstance(g, int)])
    max_individual = 0
    for g, inds in data.items():
        if isinstance(g, int) and inds:
            max_individual = max(max_individual, max(inds.keys()))
    
    return data, generations, max_individual, start_data


# ═══════════════════════════════════════════════════════════════════════════
# 3. Plot 1 — Generation (横轴) vs 物理量 (纵轴)
# ═══════════════════════════════════════════════════════════════════════════

def plot_generation_vs_quantity(
    data,
    generations,
    y_key='epsilon_eff',
    log_scale=None,
    figsize=(14, 10),
    title=None,
    output=None,
    cmap='viridis',
):
    """Plot 1: 横轴 = Generation，纵轴 = 物理量，颜色表示代数"""
    if log_scale is None:
        log_scale = (y_key == 'epsilon_eff')

    # 收集有效数据（只根据 failure_flag 过滤）
    gen_list, ind_list, y_list = [], [], []
    for gen in generations:
        for ind, entry in data[gen].items():
            # 只检查 failure_flag 是否有效
            if not is_valid_entry(entry):
                continue
            y_val = entry.get(y_key, float('nan'))
            if np.isfinite(y_val):
                gen_list.append(gen)
                ind_list.append(ind)
                y_list.append(y_val)

    if not y_list:
        print(f"Warning: No valid data for {y_key} (after filtering)")
        return

    max_ind = max(ind_list) if ind_list else 0

    fig, ax = plt.subplots(figsize=figsize)
    
    norm = plt.Normalize(vmin=min(generations), vmax=max(generations))
    cmap_obj = plt.get_cmap(cmap)
    
    for ind in range(max_ind + 1):
        mask = [i == ind for i in ind_list]
        if not any(mask):
            continue
        g = np.array([gen_list[i] for i, m in enumerate(mask) if m])
        y = np.array([y_list[i] for i, m in enumerate(mask) if m])
        
        for gi, gen in enumerate(g):
            color = cmap_obj(norm(gen))
            ax.scatter(gen, y[gi], s=25, color=color, 
                      marker='o', alpha=0.6, edgecolors='white', 
                      linewidths=0.3, zorder=3)

    # Start marker
    start_data = data.get('start', {}).get(0)
    if start_data is not None and is_valid_entry(start_data):
        start_x = generations[0] - 1 if generations else -1
        ax.scatter([start_x], [start_data[y_key]], s=100, marker='D',
                   facecolor='#333333', edgecolors='#111111',
                   linewidths=1.0, zorder=7, label='Start')
        ax.axvline(x=-0.5, color='#aaaaaa', linewidth=0.6, linestyle=':')

    if log_scale:
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=6))
        ax.yaxis.set_minor_locator(
            ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12)
        )

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xlabel('Generation')
    ax.set_ylabel(QUANTITY_LABELS.get(y_key, y_key))
    if title:
        ax.set_title(title, pad=8)

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Generation', fontsize=12)
    cbar.ax.tick_params(labelsize=11)
    cbar.outline.set_linewidth(0.8)

    if output:
        fig.savefig(output, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Plot 2 — 物理量 vs 物理量
# ═══════════════════════════════════════════════════════════════════════════

def plot_quantity_vs_quantity(
    data,
    generations,
    x_key='iota',
    y_key='epsilon_eff',
    best_key='epsilon_eff',
    best_mode='min',
    log_x=False,
    log_y=None,
    figsize=(14, 10),
    title=None,
    output=None,
    cmap='viridis',
):
    """Plot 2: 物理量 vs 物理量，颜色表示代数"""
    if log_y is None:
        log_y = (y_key == 'epsilon_eff')

    norm = plt.Normalize(vmin=min(generations), vmax=max(generations))
    cmap_obj = plt.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=figsize)
    
    for gen in generations:
        gen_color = cmap_obj(norm(gen))
        
        # 获取有效个体（只根据 failure_flag 过滤）
        valid_entries = {
            ind: entry for ind, entry in data[gen].items()
            if is_valid_entry(entry)
        }
        
        if not valid_entries:
            continue
        
        # 找最优个体
        best_idx = min(valid_entries.keys(), 
                      key=lambda i: valid_entries[i].get(best_key, float('inf')))
        
        for ind, entry in valid_entries.items():
            x = entry.get(x_key, float('nan'))
            y = entry.get(y_key, float('nan'))
            
            if not (np.isfinite(x) and np.isfinite(y)):
                continue

            if ind == best_idx:
                ax.scatter([x], [y], s=180, marker='*',
                          facecolor=gen_color, edgecolors='#222222',
                          linewidths=0.5, zorder=6)
            else:
                ax.scatter([x], [y], s=42, marker='o',
                          facecolor=gen_color, edgecolors='white',
                          linewidths=0.5, alpha=0.7, zorder=4)

    # Start marker
    start_entry = data.get('start', {}).get(0)
    if start_entry is not None and is_valid_entry(start_entry):
        sx = start_entry.get(x_key, float('nan'))
        sy = start_entry.get(y_key, float('nan'))
        if np.isfinite(sx) and np.isfinite(sy):
            ax.scatter([sx], [sy], s=100, marker='D',
                      facecolor='#333333', edgecolors='#111111',
                      linewidths=0.5, zorder=8, label='Start')

    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=6))
        ax.yaxis.set_minor_locator(
            ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12)
        )

    ax.set_xlabel(QUANTITY_LABELS.get(x_key, x_key))
    ax.set_ylabel(QUANTITY_LABELS.get(y_key, y_key))
    if title:
        ax.set_title(title, pad=8)

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Generation', fontsize=12)
    cbar.ax.tick_params(labelsize=11)
    cbar.outline.set_linewidth(0.8)

    if output:
        fig.savefig(output, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Plot 3 — 线圈电流 vs ε_eff
# ═══════════════════════════════════════════════════════════════════════════

def plot_coils_vs_epsilon(
    data,
    generations,
    best_key='epsilon_eff',
    best_mode='min',
    figsize=None,
    output=None,
    cmap='viridis',
    max_coils_per_row=5,
):
    """分面散点图：每个线圈 vs ε_eff，颜色表示代数"""
    # 动态检测线圈数量
    n_coils = None
    for gen in generations:
        for ind in data[gen].keys():
            extcur = data[gen][ind].get('extcur', [])
            if extcur:
                n_coils = len(extcur)
                break
        if n_coils is not None:
            break
    
    if n_coils is None:
        print("Error: No valid extcur data found")
        return
    
    print(f"Detected {n_coils} coils")
    
    # 动态计算图形尺寸
    if figsize is None:
        n_rows = (n_coils + max_coils_per_row - 1) // max_coils_per_row
        n_cols = min(n_coils, max_coils_per_row)
        figsize = (n_cols * 4.5, n_rows * 4.0)
    
    norm = plt.Normalize(vmin=min(generations), vmax=max(generations))
    cmap_obj = plt.get_cmap(cmap)
    
    n_rows = (n_coils + max_coils_per_row - 1) // max_coils_per_row
    n_cols = min(n_coils, max_coils_per_row)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, 
                            sharey=True, squeeze=False)
    axes = axes.flatten()
    
    # 隐藏多余的子图
    for idx in range(n_coils, len(axes)):
        axes[idx].set_visible(False)
    
    for ci in range(n_coils):
        ax = axes[ci]
        
        # 收集该线圈的数据范围（只包含有效个体）
        all_vals = []
        for gen in generations:
            for ind, entry in data[gen].items():
                if not is_valid_entry(entry):
                    continue
                extcur = entry.get('extcur', [])
                if ci < len(extcur):
                    all_vals.append(extcur[ci])
        
        if all_vals:
            margin = (max(all_vals) - min(all_vals)) * 0.1 or abs(max(all_vals)) * 0.05
            ax.set_xlim(min(all_vals) - margin, max(all_vals) + margin)
        
        # 绘图
        for gen in generations:
            gen_color = cmap_obj(norm(gen))
            
            # 获取有效个体（只根据 failure_flag 过滤）
            valid_entries = {
                ind: entry for ind, entry in data[gen].items()
                if is_valid_entry(entry)
            }
            
            if not valid_entries:
                continue
            
            best_idx = min(valid_entries.keys(),
                          key=lambda i: valid_entries[i].get(best_key, float('inf')))
            
            xs, ys = [], []
            for ind, entry in valid_entries.items():
                extcur = entry.get('extcur', [])
                if ci >= len(extcur):
                    continue
                    
                coil_val = extcur[ci]
                eps = entry.get(best_key, float('nan'))
                
                if not np.isfinite(eps):
                    continue
                    
                xs.append(coil_val)
                ys.append(eps)
                
                if ind == best_idx:
                    ax.scatter([coil_val], [eps], s=140, marker='*',
                              facecolor=gen_color, edgecolors='#222222',
                              linewidths=0.8, zorder=6)
            
            if xs:
                ax.scatter(xs, ys, s=28, marker='o',
                          facecolor=gen_color, edgecolors='white',
                          linewidths=0.4, alpha=0.7, zorder=4)
        
        coil_labels = [f'Coil {i}' for i in range(n_coils)]
        ax.set_xlabel(coil_labels[ci] + '  [A]')
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(-2, 4))
        if ci % n_cols == 0:
            ax.set_ylabel(QUANTITY_LABELS.get(best_key, best_key))
        ax.set_yscale('log')
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
        
        # Start marker
        start_entry = data.get('start', {}).get(0)
        if start_entry is not None and is_valid_entry(start_entry):
            extcur = start_entry.get('extcur', [])
            if ci < len(extcur):
                sc = extcur[ci]
                se = start_entry.get(best_key, float('nan'))
                if np.isfinite(se):
                    ax.scatter([sc], [se], s=100, marker='D',
                              facecolor='#333333', edgecolors='#111111',
                              linewidths=0.5, zorder=8, label='Start' if ci == 0 else "")
    
    fig.subplots_adjust(right=0.88, hspace=0.3, wspace=0.25)
    
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[:n_coils].tolist(), 
                       fraction=0.02, pad=0.02)
    cbar.set_label('Generation', fontsize=12)
    cbar.ax.tick_params(labelsize=11)
    cbar.outline.set_linewidth(0.8)
    
    # 如果有start，添加图例
    if start_entry is not None and is_valid_entry(start_entry):
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, 
                      loc='upper right', bbox_to_anchor=(0.98, 0.98),
                      frameon=True, framealpha=0.85)

    if output:
        fig.savefig(output, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 6. 入口
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    log_file = 'tests/h1_optimisation/h1_optimisation_log.csv'
    data, generations, max_ind, start_data = load_optimization_log(log_file)
    print(f'Loaded {len(generations)} generation(s), max individual = {max_ind}')
    
    cmap_choice = 'viridis'
    
    # Plot 1
    plot_generation_vs_quantity(
        data, generations, y_key='epsilon_eff',
        output='tests/h1_optimisation/ripple_generation.png',
        cmap=cmap_choice,
    )
    plot_generation_vs_quantity(
        data, generations, y_key='iota',
        output='tests/h1_optimisation/iota_generation.png',
        cmap=cmap_choice,
    )

    # Plot 2
    plot_quantity_vs_quantity(
        data, generations,
        x_key='iota', y_key='epsilon_eff',
        output='tests/h1_optimisation/iota_ripple.png',
        cmap=cmap_choice,
    )
    plot_quantity_vs_quantity(
        data, generations,
        x_key='Aspect ratio', y_key='epsilon_eff',
        output='tests/h1_optimisation/asp_ripple.png',
        cmap=cmap_choice,
    )

    # Plot 3
    plot_coils_vs_epsilon(
        data, generations,
        output='tests/h1_optimisation/coils_vs_eps.png',
        cmap=cmap_choice,
    )