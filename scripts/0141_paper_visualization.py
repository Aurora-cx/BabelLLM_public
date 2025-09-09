import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 全局配置
BASE_DIR = Path('outputs/try/attention_weights')

def load_attention_data(file_path):
    """加载attention权重数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_avg_attentions(samples):
    """计算平均attention权重"""
    if not samples:
        return None
        
    avg_attentions = {}
    # 获取所有样本中出现的所有组件
    all_components = set()
    for sample in samples:
        all_components.update(sample['attentions'].keys())
    
    print(f"\nAll components found: {sorted(all_components)}")
    
    # 对每个组件计算平均值
    for component in all_components:
        # 收集所有样本的attention数据，跳过缺失的样本
        component_data = []
        skipped_samples = []
        for sample in samples:
            if component in sample['attentions']:
                component_data.append(sample['attentions'][component])
            else:
                skipped_samples.append(sample['sample_id'])
        
        if skipped_samples:
            print(f"Warning: Component '{component}' missing in {len(skipped_samples)} samples, e.g. {skipped_samples[:3]}...")
        
        if component_data:  # 只要有数据就计算平均值
            avg_attentions[component] = np.mean(component_data, axis=0)
            print(f"Processed {len(component_data)} samples for {component}")
    
    print(f"\nProcessed components: {sorted(avg_attentions.keys())}")
    return avg_attentions

def plot_paper_comparison(en_avg_attentions, zh_avg_attentions, save_path):
    """绘制论文中使用的固定布局组件对比图"""
    # 获取中英文样本共有的组件
    components = sorted(set(en_avg_attentions.keys()) & set(zh_avg_attentions.keys()))
    print(f"\nAvailable components: {components}")
    
    # 固定布局：2行4列，增加图片宽度
    n_rows = 2
    n_cols = 4
    fig_width = 20  # 增加图片宽度
    
    # 创建大图
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, 8))
    axes = axes.flatten()
    
    # 定义组件位置映射（使用实际的组件名称）
    component_positions = {
        'cause_subject': [0],        # 第一行第一个位置
        'cause_verb': [4],           # 第一行第二个位置
        'intermediate_subject': [1],  # 第一行第三个位置
        'intermediate_verb': [5],     # 第一行第四个位置
        'final_subject': [2],         # 第二行第一个位置
        'final_verb': [6],           # 第二行第二个位置
        'connective_once': [3],       # 第二行第三个位置
        'connective_then': [7]        # 第二行第四个位置
    }
    
    # 设置颜色
    en_color = '#1f77b4'  # 蓝色
    zh_color = '#ff7f0e'  # 橙色
    
    # 为每个组件绘制子图
    for component_type, positions in component_positions.items():
        print(f"\nProcessing component: {component_type}")
        for pos in positions:
            if component_type in components:
                print(f"Drawing {component_type} at position {pos}")
                # 获取该组件在所有层的attention总和
                en_values = np.sum(en_avg_attentions[component_type], axis=1)
                zh_values = np.sum(zh_avg_attentions[component_type], axis=1)
                print(f"en_values shape: {en_values.shape}, zh_values shape: {zh_values.shape}")
                print(f"en_values: {en_values}")
                print(f"zh_values: {zh_values}")
                
                # 绘制柱状图
                x = np.arange(len(en_values))
                width = 0.35
                
                # 绘制柱状图，使用自定义颜色
                axes[pos].bar(x - width/2, en_values, width, label='English', color=en_color, alpha=0.7)
                axes[pos].bar(x + width/2, zh_values, width, label='Chinese', color=zh_color, alpha=0.7)
                
                # 设置子图属性
                axes[pos].set_title(component_type, fontsize=17, pad=10)
                axes[pos].set_xticks(x[::4])  # 每4个层显示一个刻度
                axes[pos].set_xticklabels(range(0, len(en_values), 4), fontsize=10)  # 显示对应的层号
                axes[pos].set_ylim(0, 1)
                axes[pos].grid(True, axis='y', linestyle='--', alpha=0.3)
                
                # 设置刻度标签字体大小
                axes[pos].tick_params(axis='both', which='major', labelsize=15)
                
                # 只在第一列的子图添加y轴标签
                if pos % n_cols == 0:
                    axes[pos].set_ylabel('Relative Component Attention Ratio', fontsize=14)
                
                # 只在最后一行的子图添加x轴标签
                if pos >= (n_rows-1)*n_cols:
                    axes[pos].set_xlabel('Layer', fontsize=15)
            else:
                print(f"Component {component_type} not found in data")
                # 如果组件不存在，清空该子图
                axes[pos].axis('off')
    
    # 添加总图例，放在底部
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.52, -0.03),
                       ncol=2, fontsize=16, frameon=True, fancybox=True)
    # 设置图例边框样式
    legend.get_frame().set_alpha(0.2)
    legend.get_frame().set_edgecolor('black')
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # 为图例留出空间
    
    # 保存图片
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight', dpi=300)
    plt.close()

def main():
    """主函数"""
    # 设置输出目录
    vis_dir = BASE_DIR / 'visualizations' / 'paper_figures'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集英文和中文样本
    en_samples = []
    zh_samples = []
    
    # 加载英文数据（包括correct和incorrect）
    en_dir = BASE_DIR / 'attention_weights_en'
    for subdir in ['correct', 'incorrect']:
        current_dir = en_dir / subdir
        print(f"\nLooking for English files in: {current_dir}")
        for file in current_dir.glob('attention_*.json'):
            print(f"Found English file: {file}")
            data = load_attention_data(file)
            print(f"Loaded {len(data)} samples from {file}")
            print(f"Sample keys: {data[0].keys() if data else 'No data'}")
            en_samples.extend(data)
    
    # 加载中文数据（包括correct和incorrect）
    zh_dir = BASE_DIR / 'attention_weights_zh'
    for subdir in ['correct', 'incorrect']:
        current_dir = zh_dir / subdir
        print(f"\nLooking for Chinese files in: {current_dir}")
        for file in current_dir.glob('attention_*.json'):
            print(f"Found Chinese file: {file}")
            data = load_attention_data(file)
            print(f"Loaded {len(data)} samples from {file}")
            print(f"Sample keys: {data[0].keys() if data else 'No data'}")
            zh_samples.extend(data)
    
    print(f"\nTotal: Loaded {len(en_samples)} English samples and {len(zh_samples)} Chinese samples")
    
    if not en_samples or not zh_samples:
        print("Error: No samples loaded!")
        return
    
    # 计算平均attention权重
    print("\nProcessing English samples:")
    en_avg_attentions = calculate_avg_attentions(en_samples)
    print("\nProcessing Chinese samples:")
    zh_avg_attentions = calculate_avg_attentions(zh_samples)
    
    if not en_avg_attentions or not zh_avg_attentions:
        print("Error: Failed to calculate average attentions!")
        return
    
    print("\nEnglish components:", sorted(en_avg_attentions.keys()))
    print("Chinese components:", sorted(zh_avg_attentions.keys()))
    
    # 绘制论文中使用的对比图
    save_path = vis_dir / 'paper_component_comparison.png'
    plot_paper_comparison(en_avg_attentions, zh_avg_attentions, save_path)
    print(f"Saved paper comparison plot")
    
    print("\n✅ Visualization complete!")

if __name__ == "__main__":
    main() 