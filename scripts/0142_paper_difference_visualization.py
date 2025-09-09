import json
import numpy as np
import matplotlib.pyplot as plt
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

def calculate_differences(en_avg_attentions, zh_avg_attentions):
    """计算所有组件的中英文差值"""
    differences = {}
    for component in en_avg_attentions.keys():
        if component in zh_avg_attentions:
            # 获取该组件在所有层的attention总和
            en_values = np.sum(en_avg_attentions[component], axis=1)
            zh_values = np.sum(zh_avg_attentions[component], axis=1)

            # 计算绝对差值
            difference = (zh_values - en_values).sum()
            differences[component] = difference
    return differences

def plot_differences(differences, save_path):
    """绘制中英文差值对比图"""
    # 设置中文字体
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 准备数据并按值排序
    sorted_items = sorted(differences.items(), key=lambda x: x[1], reverse=True)
    components = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    
    # 设置颜色：正值用红色，负值用蓝色
    colors = ['#ff7f0e' if v > 0 else '#1f77b4' for v in values]
    
    # 绘制竖直条形图
    bars = ax.bar(components, values, color=colors, alpha=0.7)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        label_y_pos = height + 0.01 if height > 0 else height - 0.01
        ax.text(bar.get_x() + bar.get_width()/2, label_y_pos,
                f'{height:.3f}', ha='center', va='bottom' if height > 0 else 'top',
                fontsize=14)
    
    # 设置标题和标签
    #ax.set_title('RCAR Sum Differences (Chinese - English)', fontsize=14, pad=20)
    ax.set_ylabel('RCAR Difference', fontsize=16)
    
    # 添加零线
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 设置网格
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # 旋转x轴标签以防重叠
    plt.xticks(rotation=45, ha='right',fontsize=16)
    plt.yticks(fontsize=16)
    
    # 调整布局
    plt.tight_layout()
    
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
    
    # 计算差值
    differences = calculate_differences(en_avg_attentions, zh_avg_attentions)
    
    # 绘制差值对比图
    save_path = vis_dir / 'paper_differences.png'
    plot_differences(differences, save_path)
    print(f"Saved differences plot")
    
    print("\n✅ Visualization complete!")

if __name__ == "__main__":
    main() 