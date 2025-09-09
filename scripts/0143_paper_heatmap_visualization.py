import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_attention_data(file_path):
    """加载注意力权重数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_avg_attention(attention_data):
    """计算平均注意力权重"""
    if not attention_data:
        return None
        
    # 获取层数
    num_layers = len(attention_data[0]['attentions']['cause'])
    
    # 初始化累加器
    avg_attention = {
        'cause': np.zeros(num_layers),
        'effect1': np.zeros(num_layers),
        'effect2': np.zeros(num_layers)
    }
    
    # 计算每个组件的平均注意力权重
    for component in ['cause', 'effect1', 'effect2']:
        accumulator = np.zeros(num_layers)
        sample_count = 0
        
        for sample in attention_data:
            # 对每一层的16个head求和
            sample_attention = np.array(sample['attentions'][component])
            layer_sums = np.sum(sample_attention, axis=1)
            accumulator += layer_sums
            sample_count += 1
        
        avg_attention[component] = accumulator / sample_count
    
    return avg_attention

def plot_paper_heatmaps(save_path=None):
    """绘制2x2的热力图对比图"""
    # 设置中文字体

    
    # 设置数据目录
    base_dir = Path("outputs/try/causal_attention")
    
    # 加载数据
    versions = ['en', 'en_para', 'zh', 'zh_para']
    attention_data = {}
    
    for version in versions:
        data_dir = base_dir / f"attention_weights_{version}"
        all_samples = []
        
        # 加载正确样本
        correct_dir = data_dir / "correct"
        if correct_dir.exists():
            for json_file in correct_dir.glob("attention_*.json"):
                all_samples.extend(load_attention_data(json_file))
        
        # 加载错误样本
        incorrect_dir = data_dir / "incorrect"
        if incorrect_dir.exists():
            for json_file in incorrect_dir.glob("attention_*.json"):
                all_samples.extend(load_attention_data(json_file))
        
        attention_data[version] = all_samples
    
    # 创建图形和网格
    fig = plt.figure(figsize=(18, 9))
    gs = plt.GridSpec(2, 2, width_ratios=[1, 1.22])  # 右边的子图更宽一些
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(2)])
    
    # 设置标题映射
    title_map = {
        'en': 'English (Forward Causal Chain)',
        'en_para': 'English (Reversed Causal Chain)',
        'zh': 'Chinese (Forward Causal Chain)',
        'zh_para': 'Chinese (Reversed Causal Chain)'
    }
    
    # 绘制每个版本的热力图
    for idx, version in enumerate(versions):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # 计算平均注意力权重
        avg_attention = calculate_avg_attention(attention_data[version])
        if avg_attention is None:
            continue
        
        # 准备数据
        data = np.array([
            avg_attention['cause'],
            avg_attention['effect1'],
            avg_attention['effect2']
        ])
        
        # 创建热力图
        sns.heatmap(
            data,
            ax=ax,
            cmap='YlOrRd',
            vmin=0,
            vmax=1,
            xticklabels=[f'L{i+1}' if (i+1) % 4 == 0 or i == 0 else '' for i in range(data.shape[1])],
            yticklabels=['Cause', 'Effect1', 'Effect2'],
            cbar=col == 1,  # 只在最右边的子图显示颜色条
            cbar_kws={'label': 'Attention Weight'} if col == 1 else None
        )
        
        # 设置标题和标签
        ax.set_title(title_map[version], fontsize=18, pad=10)
        ax.set_xlabel('Layer', fontsize=16)
        ax.set_ylabel('Component', fontsize=16)
        ax.tick_params(axis='both', labelsize=14)
        
        # 调整颜色条标签字体大小（只在最右边的子图）
        if col == 1:
            cbar = ax.collections[0].colorbar
            cbar.ax.set_ylabel('RCAR', fontsize=15)
            cbar.ax.tick_params(labelsize=14)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()

def main():
    # 设置输出目录
    output_dir = Path("outputs/try/causal_attention/visualizations/paper_figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成热力图
    save_path = output_dir / "paper_heatmaps.png"
    plot_paper_heatmaps(save_path)
    print(f"✅ Heatmap visualization saved to {save_path}")

if __name__ == "__main__":
    main() 