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

def plot_component_comparison(en_avg_attentions, zh_avg_attentions, component, save_path):
    """为单个组件绘制中英文对比图"""
    # 获取该组件在所有层的attention总和
    en_values = np.sum(en_avg_attentions[component], axis=1)  # 对每个层的所有head求和
    zh_values = np.sum(zh_avg_attentions[component], axis=1)
    
    # 设置图形大小和样式
    plt.figure(figsize=(15, 8))
    x = np.arange(len(en_values))  # 层数
    width = 0.35  # 柱状图宽度
    
    # 绘制柱状图
    plt.bar(x - width/2, en_values, width, label='English', alpha=0.7)
    plt.bar(x + width/2, zh_values, width, label='Chinese', alpha=0.7)
    
    # 设置图形属性
    plt.xlabel('Layer')
    plt.ylabel('Sum of Attention Weights')
    plt.title(f'Attention Weights Comparison for {component}')
    plt.xticks(x, range(len(en_values)))
    plt.ylim(0, 1)
    plt.legend()
    
    # 添加网格线
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # 保存图片
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

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

def plot_all_components_comparison(en_avg_attentions, zh_avg_attentions, save_path):
    """绘制所有组件的组合对比图"""
    # 获取中英文样本共有的组件
    components = sorted(set(en_avg_attentions.keys()) & set(zh_avg_attentions.keys()))
    n_components = len(components)
    
    # 计算子图的行列数
    n_cols = 3
    n_rows = (n_components + n_cols - 1) // n_cols
    
    # 创建大图
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    # 为每个组件绘制子图
    for idx, component in enumerate(components):
        # 获取该组件在所有层的attention总和
        en_values = np.sum(en_avg_attentions[component], axis=1)
        zh_values = np.sum(zh_avg_attentions[component], axis=1)
        
        # 绘制柱状图
        x = np.arange(len(en_values))
        width = 0.35
        
        axes[idx].bar(x - width/2, en_values, width, label='English', alpha=0.7)
        axes[idx].bar(x + width/2, zh_values, width, label='Chinese', alpha=0.7)
        
        # 设置子图属性
        axes[idx].set_title(component)
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(range(len(en_values)))
        axes[idx].set_ylim(0, 1)
        axes[idx].grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # 只在第一列的子图添加y轴标签
        if idx % n_cols == 0:
            axes[idx].set_ylabel('Sum of Attention Weights')
        
        # 只在最后一行的子图添加x轴标签
        if idx >= (n_rows-1)*n_cols:
            axes[idx].set_xlabel('Layer')
    
    # 添加总图例
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2)
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # 为图例留出空间
    
    # 保存图片
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_fixed_layout_comparison(en_avg_attentions, zh_avg_attentions, save_path):
    """绘制固定布局的组件对比图"""
    # 获取中英文样本共有的组件
    components = sorted(set(en_avg_attentions.keys()) & set(zh_avg_attentions.keys()))
    
    # 固定布局：2行4列
    n_rows = 2
    n_cols = 4
    
    # 创建大图
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 8))
    axes = axes.flatten()
    
    # 定义组件位置映射
    component_positions = {
        'subject': [0, 1, 2],  # 第一行的前三个位置
        'verb': [4, 5, 6],     # 第二行的前三个位置
        'once': [3],           # 第一行最后一个位置
        'then': [7]            # 第二行最后一个位置
    }
    
    # 为每个组件绘制子图
    for component_type, positions in component_positions.items():
        for pos in positions:
            if component_type in components:
                # 获取该组件在所有层的attention总和
                en_values = np.sum(en_avg_attentions[component_type], axis=1)
                zh_values = np.sum(zh_avg_attentions[component_type], axis=1)
                
                # 绘制柱状图
                x = np.arange(len(en_values))
                width = 0.35
                
                axes[pos].bar(x - width/2, en_values, width, label='English', alpha=0.7)
                axes[pos].bar(x + width/2, zh_values, width, label='Chinese', alpha=0.7)
                
                # 设置子图属性
                axes[pos].set_title(component_type)
                axes[pos].set_xticks(x)
                axes[pos].set_xticklabels(range(len(en_values)))
                axes[pos].set_ylim(0, 1)
                axes[pos].grid(True, axis='y', linestyle='--', alpha=0.7)
                
                # 只在第一列的子图添加y轴标签
                if pos % n_cols == 0:
                    axes[pos].set_ylabel('Sum of Attention Weights')
                
                # 只在最后一行的子图添加x轴标签
                if pos >= (n_rows-1)*n_cols:
                    axes[pos].set_xlabel('Layer')
            else:
                # 如果组件不存在，清空该子图
                axes[pos].axis('off')
    
    # 添加总图例
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2)
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # 为图例留出空间
    
    # 保存图片
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_attention_heatmap(avg_attentions, title, save_path, vmin=None, vmax=None):
    """绘制注意力热力图"""
    # 获取所有组件和层数
    components = sorted(avg_attentions.keys())
    n_layers = len(avg_attentions[components[0]])
    
    # 创建数据矩阵
    data = np.zeros((n_layers, len(components)))
    for i, component in enumerate(components):
        # 对每个层的所有head求和
        data[:, i] = np.sum(avg_attentions[component], axis=1)
    
    # 创建图形
    plt.figure(figsize=(15, 8))
    
    # 绘制热力图
    sns.heatmap(data, 
                xticklabels=components,
                yticklabels=range(n_layers),
                cmap='YlOrRd',
                vmin=vmin,
                vmax=vmax,
                cbar_kws={'label': 'Sum of Attention Weights'})
    
    # 设置图形属性
    plt.title(title)
    plt.xlabel('Components')
    plt.ylabel('Layer')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def calculate_cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

def calculate_similarities(en_avg_attentions, zh_avg_attentions):
    """计算所有组件的中英文相似度"""
    similarities = {}
    for component in en_avg_attentions.keys():
        if component in zh_avg_attentions:
            # 获取该组件在所有层的attention总和
            en_values = np.sum(en_avg_attentions[component], axis=1)
            zh_values = np.sum(zh_avg_attentions[component], axis=1)
            print(f"en_values: {en_values.shape}, zh_values: {zh_values.shape}")
            
            # 计算余弦相似度
            similarity = calculate_cosine_similarity(en_values, zh_values)
            similarities[component] = similarity
    return similarities

def absolute_difference(en_avg_attentions, zh_avg_attentions):
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

def plot_similarity_heatmap(similarities, save_path):
    """绘制相似度bar图"""
    # 按相似度排序
    sorted_components = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    components = [x[0] for x in sorted_components]
    values = [x[1] for x in sorted_components]
    
    # 创建图形
    plt.figure(figsize=(12, 6))
    
    # 绘制热力图
    plt.bar(range(len(components)), values)
    plt.xticks(range(len(components)), components, rotation=45, ha='right')
    plt.ylim(0, 1)
    
    # 设置图形属性
    plt.title('Cosine Similarity between Chinese and English Attention Distributions')
    plt.xlabel('Components')
    plt.ylabel('Cosine Similarity')
    
    # 添加数值标签
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_difference_heatmap(differences, save_path):
    """绘制差值bar图"""
    # 按差值排序
    sorted_components = sorted(differences.items(), key=lambda x: x[1], reverse=True)
    components = [x[0] for x in sorted_components]
    values = [x[1] for x in sorted_components]
    
    # 创建图形
    plt.figure(figsize=(12, 6))
    
    # 绘制热力图
    plt.bar(range(len(components)), values)
    plt.xticks(range(len(components)), components, rotation=45, ha='right')
    plt.ylim(-0.5, 2)
    
    # 设置图形属性
    plt.title('Absolute Difference between Chinese and English Attention Distributions')
    plt.xlabel('Components')
    plt.ylabel('Absolute Difference')
    
    # 添加数值标签
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def calculate_component_similarities(avg_attentions):
    """计算组件之间的两两相似度"""
    components = sorted(avg_attentions.keys())
    n_components = len(components)
    similarity_matrix = np.zeros((n_components, n_components))
    
    for i, comp1 in enumerate(components):
        for j, comp2 in enumerate(components):
            # 获取两个组件在所有层的attention总和
            values1 = np.sum(avg_attentions[comp1], axis=1)
            values2 = np.sum(avg_attentions[comp2], axis=1)
            
            # 计算余弦相似度
            similarity = calculate_cosine_similarity(values1, values2)
            similarity_matrix[i, j] = similarity
    
    return components, similarity_matrix

def plot_component_similarity_heatmap(components, similarity_matrix, save_path):
    """绘制组件相似度条形图"""
    # 获取所有非对角线的相似度值（不包括自身与自身的相似度）
    pairs = []
    similarities = []
    
    for i in range(len(components)):
        for j in range(i+1, len(components)):  # 只取上三角矩阵，避免重复
            pair = f"{components[i]}-{components[j]}"
            pairs.append(pair)
            similarities.append(similarity_matrix[i, j])
    
    # 按相似度排序
    sorted_indices = np.argsort(similarities)[::-1]  # 降序排序
    pairs = [pairs[i] for i in sorted_indices]
    similarities = [similarities[i] for i in sorted_indices]
    
    # 创建图形
    plt.figure(figsize=(15, 8))
    
    # 绘制条形图
    bars = plt.bar(range(len(pairs)), similarities)
    
    # 设置x轴标签
    plt.xticks(range(len(pairs)), pairs, rotation=45, ha='right')
    
    # 设置图形属性
    plt.title('Pairwise Component Similarities in English Data')
    plt.xlabel('Component Pairs')
    plt.ylabel('Cosine Similarity')
    plt.ylim(0, 1)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    # 添加网格线
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    """主函数"""
    # 设置输出目录
    vis_dir = BASE_DIR / 'visualizations' / 'en_zh_comparison'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义要处理的样本类型
    sample_types = ['correct', 'incorrect', 'all']
    
    for sample_type in sample_types:
        print(f"\n{'='*50}")
        print(f"Processing {sample_type} samples")
        print(f"{'='*50}")
        
        # 设置当前样本类型的输出目录
        current_vis_dir = vis_dir / sample_type
        current_vis_dir.mkdir(parents=True, exist_ok=True)
        
        # 收集英文和中文样本
        en_samples = []
        zh_samples = []
        
        # 加载英文数据
        en_dir = BASE_DIR / 'attention_weights_en'
        if sample_type == 'all':
            # 对于all类型，加载所有样本
            for subdir in ['correct', 'incorrect']:
                current_dir = en_dir / subdir
                print(f"\nLooking for English files in: {current_dir}")
                for file in current_dir.glob('attention_*.json'):
                    print(f"Found English file: {file}")
                    data = load_attention_data(file)
                    print(f"Loaded {len(data)} samples from {file}")
                    en_samples.extend(data)
        else:
            # 对于correct和incorrect类型，只加载对应目录
            current_dir = en_dir / sample_type
            print(f"\nLooking for English files in: {current_dir}")
            for file in current_dir.glob('attention_*.json'):
                print(f"Found English file: {file}")
                data = load_attention_data(file)
                print(f"Loaded {len(data)} samples from {file}")
                en_samples.extend(data)
        
        # 加载中文数据
        zh_dir = BASE_DIR / 'attention_weights_zh'
        if sample_type == 'all':
            # 对于all类型，加载所有样本
            for subdir in ['correct', 'incorrect']:
                current_dir = zh_dir / subdir
                print(f"\nLooking for Chinese files in: {current_dir}")
                for file in current_dir.glob('attention_*.json'):
                    print(f"Found Chinese file: {file}")
                    data = load_attention_data(file)
                    print(f"Loaded {len(data)} samples from {file}")
                    zh_samples.extend(data)
        else:
            # 对于correct和incorrect类型，只加载对应目录
            current_dir = zh_dir / sample_type
            print(f"\nLooking for Chinese files in: {current_dir}")
            for file in current_dir.glob('attention_*.json'):
                print(f"Found Chinese file: {file}")
                data = load_attention_data(file)
                print(f"Loaded {len(data)} samples from {file}")
                zh_samples.extend(data)
        
        print(f"\nTotal: Loaded {len(en_samples)} English samples and {len(zh_samples)} Chinese samples")
        
        if not en_samples or not zh_samples:
            print(f"Error: No {sample_type} samples loaded!")
            continue
        
        # 计算平均attention权重
        print("\nProcessing English samples:")
        en_avg_attentions = calculate_avg_attentions(en_samples)
        print("\nProcessing Chinese samples:")
        zh_avg_attentions = calculate_avg_attentions(zh_samples)
        
        if not en_avg_attentions or not zh_avg_attentions:
            print(f"Error: Failed to calculate average attentions for {sample_type} samples!")
            continue
        
        # 获取中英文样本共有的组件
        common_components = set(en_avg_attentions.keys()) & set(zh_avg_attentions.keys())
        print(f"\nCommon components between Chinese and English: {sorted(common_components)}")
        
        # 为每个共有组件绘制对比图
        for component in common_components:
            save_path = current_vis_dir / f'{component}_comparison.png'
            plot_component_comparison(en_avg_attentions, zh_avg_attentions, component, save_path)
            print(f"Saved comparison plot for {component}")
        
        # 绘制所有组件的组合图
        save_path = current_vis_dir / 'all_components_comparison.png'
        plot_all_components_comparison(en_avg_attentions, zh_avg_attentions, save_path)
        print(f"Saved combined comparison plot for all components")
        
        # 绘制固定布局的组件对比图
        save_path = current_vis_dir / 'fixed_layout_comparison.png'
        plot_fixed_layout_comparison(en_avg_attentions, zh_avg_attentions, save_path)
        print(f"Saved fixed layout comparison plot")
        
        # 计算所有样本的最大最小值，用于统一颜色范围
        all_values = []
        for avg_attentions in [en_avg_attentions, zh_avg_attentions]:
            for component in avg_attentions:
                all_values.extend(np.sum(avg_attentions[component], axis=1))
        
        vmin = 0
        vmax = np.max(all_values)
        print(f"\nHeatmap value range: [{vmin:.2f}, {vmax:.2f}]")
        
        # 绘制热力图
        save_path = current_vis_dir / 'english_attention_heatmap.png'
        plot_attention_heatmap(en_avg_attentions, f'English Attention Weights Heatmap ({sample_type})', save_path, vmin, vmax)
        print(f"Saved English attention heatmap")
        
        save_path = current_vis_dir / 'chinese_attention_heatmap.png'
        plot_attention_heatmap(zh_avg_attentions, f'Chinese Attention Weights Heatmap ({sample_type})', save_path, vmin, vmax)
        print(f"Saved Chinese attention heatmap")
        
        # 计算并绘制相似度
        similarities = calculate_similarities(en_avg_attentions, zh_avg_attentions)
        save_path = current_vis_dir / 'attention_similarity.png'
        plot_similarity_heatmap(similarities, save_path)
        print(f"Saved attention similarity plot")
        
        # 计算并绘制差值
        differences = absolute_difference(en_avg_attentions, zh_avg_attentions)
        save_path = current_vis_dir / 'attention_difference.png'
        plot_difference_heatmap(differences, save_path)
        print(f"Saved attention difference plot")
        
        # 计算并绘制英文组件之间的两两相似度
        components, similarity_matrix = calculate_component_similarities(en_avg_attentions)
        save_path = current_vis_dir / 'english_component_similarities.png'
        plot_component_similarity_heatmap(components, similarity_matrix, save_path)
        print(f"Saved English component similarities heatmap")
        
        # 打印相似度排序
        print(f"\nComponent similarities for {sample_type} samples (sorted by similarity):")
        sorted_components = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        for component, similarity in sorted_components:
            print(f"{component}: {similarity:.3f}")
        
        print(f"\n✅ Visualization complete for {sample_type} samples!")

if __name__ == "__main__":
    main() 