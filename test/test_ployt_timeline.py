import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator
import numpy as np

# 示例数据（替换为实际从simulator获取的数据）
timeline_data = [
    # 格式: [rank, event_type, microbatch, start_time, end_time]
    [0, "COMPUTE_FWD", 1, 0, 0.00814718],
    [0, "PP_COMM_FWD", 1, 0.00814718, -1],
    [1, "PP_COMM_FWD", 1, 0.00814718, 0.00893361],
    [1, "COMPUTE_FWD", 1, 0.00893361, 0.0170808],
    [1, "COMPUTE_BWD", -1, 0.0170808, 0.0316296],
    [1, "PP_COMM_BWD", -1, 0.0316296, -1],
    [0, "PP_COMM_BWD", -1, 0.0316296, 0.0324161],
    [0, "COMPUTE_BWD", -1, 0.0324161, 0.0469649]
]

def plot_timeline_with_layers(timeline_data):
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    
    # 颜色映射
    color_map = {
        "COMPUTE_FWD": "#4E79A7",
        "COMPUTE_BWD": "#F28E2B",
        "TP_COMM_FWD": "#59A14F",
        "TP_COMM_BWD": "#8CD17D",
        "PP_COMM_FWD": "#E15759", 
        "PP_COMM_BWD": "#FF9D9A",
        "DP_COMM_EVENT": "#B07AA1"
    }
    
    # 获取所有rank并建立y轴位置映射
    ranks = sorted(set([e[0] for e in timeline_data]))
    y_pos = {rank: i for i, rank in enumerate(ranks)}
    
    # 为每个rank维护活跃事件计数
    active_events = {rank: [] for rank in ranks}
    
    # 先按开始时间排序
    timeline_data_sorted = sorted(timeline_data, key=lambda x: x[3])
    
    # 预处理：为每个事件确定它所在的层
    event_layers = {}
    for event in timeline_data_sorted:
        rank, etype, mb, start, end = event
        duration = end - start
        
        if duration <= 0:
            continue
        
        # 查找可用的层
        layer = 0
        while layer < len(active_events[rank]):
            if start >= active_events[rank][layer]:
                break
            layer += 1
        
        # 更新活跃事件
        if layer < len(active_events[rank]):
            active_events[rank][layer] = end
        else:
            active_events[rank].append(end)
        
        event_layers[tuple(event)] = layer
    
    # 绘制每个事件
    for event in timeline_data_sorted:
        rank, etype, mb, start, end = event
        duration = end - start

        if duration > 0:  # 只打印有效事件
            print(f"Rank {rank}: {etype} MB{mb} ({start:.6f}-{end:.6f})")
        
        if duration <= 0:
            continue
        
        layer = event_layers.get(tuple(event), 0)
        total_layers = max(1, len(active_events.get(rank, [1])))
        
        # 计算垂直位置和高度
        y_bottom = y_pos[rank] - 0.4  # rank区域的底部
        y_height = 0.8  # rank区域的总高度
        
        # 从下往上分配空间
        layer_height = y_height / total_layers
        y_center = y_bottom + layer_height * (layer + 0.5)
        height = layer_height * 0.9  # 留一点间隙
        
        # 创建矩形块
        rect = patches.Rectangle(
            (start, y_center - height/2),
            duration,
            height,
            facecolor=color_map.get(etype, "#DDDDDD"),
            edgecolor='black',
            linewidth=0.5
        )
        ax.add_patch(rect)
        
        # 添加文本标注
        label = f"{etype}\nMB{mb}" if mb != 0 else etype
        ax.text(
            start + duration/2,
            y_center,
            label,
            ha='center',
            va='center',
            fontsize=8,
            color='white' if etype in ["TP_FWD", "PP_FWD", "DP_COMM"] else 'black'
        )

        print(f"Rank {rank} event at layer {layer}/{total_layers}:")
        print(f"  y_bottom: {y_bottom:.2f}, y_center: {y_center:.2f}, height: {height:.4f}")
        print(f"  vertical range: [{y_center-height/2:.4f}, {y_center+height/2:.4f}]")
    
    # 设置坐标轴
    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels([f"Rank {r}" for r in y_pos.keys()])
    ax.set_xlabel("Time (ms)")
    ax.set_title("GPU Cluster Timeline with Event Layering")
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # 自动调整时间轴范围
    max_time = max([e[4] for e in timeline_data])
    ax.set_xlim(0, max_time*1.1)
    
    # 设置y轴范围确保所有rank都可见
    ax.set_ylim(-0.5, len(ranks)-0.5)
    
    # 创建图例
    legend_handles = [
        patches.Patch(color=color, label=label)
        for label, color in color_map.items()
    ]
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('gpu_timeline.png', dpi=300, bbox_inches='tight')
    plt.show()

# 运行可视化
plot_timeline_with_layers(timeline_data)