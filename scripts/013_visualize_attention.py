import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 全局配置
BASE_DIR = Path('outputs/qwen15_18b_v2/attention_weights')

def load_attention_data(file_path):
    """加载attention权重数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_attention_heatmap(attention_matrix, title, save_path, vmin=None, vmax=None):
    """绘制attention热力图"""
    plt.figure(figsize=(15, 10))
    sns.heatmap(attention_matrix,
                cmap='YlOrRd',
                xticklabels=range(attention_matrix.shape[1]),  # head indices
                yticklabels=range(attention_matrix.shape[0]),  # layer indices
                cbar_kws={'label': 'Attention Weight'},
                vmin=vmin,
                vmax=vmax)
    
    plt.title(title)
    plt.xlabel('Attention Head')
    plt.ylabel('Layer')
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def plot_component_sum_attention(avg_attentions, title, save_path, vmin=None, vmax=None):
    """绘制所有组件的attention总和热力图"""
    # 初始化矩阵
    num_layers = len(avg_attentions[list(avg_attentions.keys())[0]])
    components = list(avg_attentions.keys())
    sum_matrix = np.zeros((num_layers, len(components)))
    
    # 计算每个组件在每个层的attention总和
    for i, component in enumerate(components):
        for layer in range(num_layers):
            sum_matrix[layer, i] = np.sum(avg_attentions[component][layer])
    
    # 创建热力图
    plt.figure(figsize=(15, 10))
    sns.heatmap(sum_matrix,
                cmap='YlOrRd',
                xticklabels=components,
                yticklabels=range(num_layers),
                cbar_kws={'label': 'Sum of Attention Weights'},
                vmin=vmin,
                vmax=vmax)
    
    plt.title(title)
    plt.xlabel('Component')
    plt.ylabel('Layer')
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    """主函数"""
    # 设置输出目录
    vis_dir = BASE_DIR / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集所有样本
    samples = {
        'en_correct': [],
        'en_incorrect': [],
        'zh_correct': [],
        'zh_incorrect': []
    }
    
    # 遍历英文和中文数据
    for lang in ['en', 'zh']:
        lang_dir = BASE_DIR / f'attention_weights_{lang}'
        # 遍历correct和incorrect文件夹
        for data_type in ['correct', 'incorrect']:
            data_dir = lang_dir / data_type
            # 遍历所有主题文件
            for file in data_dir.glob('attention_*.json'):
                attention_data = load_attention_data(file)
                # 直接将文件夹中的所有样本添加到对应的类别中
                if lang == 'en':
                    if data_type == 'correct':
                        samples['en_correct'].extend(attention_data)
                    else:
                        samples['en_incorrect'].extend(attention_data)
                else:  # zh
                    if data_type == 'correct':
                        samples['zh_correct'].extend(attention_data)
                    else:
                        samples['zh_incorrect'].extend(attention_data)
    
    # 首先计算所有样本的最大最小值，用于统一颜色范围
    all_attention_values = []
    all_sum_values = []
    
    for sample_list in samples.values():
        if not sample_list:
            continue
            
        components = list(sample_list[0]['attentions'].keys())
        num_layers = len(sample_list[0]['attentions'][components[0]])
        
        # 收集所有attention值
        for sample in sample_list:
            for component in components:
                all_attention_values.extend(sample['attentions'][component])
        
        # 计算每个样本的attention总和
        for sample in sample_list:
            sum_matrix = np.zeros((num_layers, len(components)))
            for i, component in enumerate(components):
                for layer in range(num_layers):
                    sum_matrix[layer, i] = np.sum(sample['attentions'][component][layer])
            all_sum_values.extend(sum_matrix.flatten())
    
    # 计算全局最大最小值
    vmin_attention = np.min(all_attention_values)
    vmax_attention = np.max(all_attention_values)
    vmin_sum = 0
    vmax_sum = 32
    
    print(f"Attention value range: [{vmin_attention:.2f}, {vmax_attention:.2f}]")
    print(f"Sum value range: [{vmin_sum:.2f}, {vmax_sum:.2f}]")
    
    # 计算每种情况的平均attention权重
    for sample_type, sample_list in samples.items():
        if not sample_list:
            continue
            
        # 计算平均attention权重
        avg_attentions = {}
        components = list(sample_list[0]['attentions'].keys())
        
        for component in components:
            # 收集所有样本的attention数据
            component_data = [sample['attentions'][component] for sample in sample_list]
            # 计算平均值
            avg_attentions[component] = np.mean(component_data, axis=0)
        
        # 为每个组件绘制热力图
        for component in components:
            title = f'Attention Weights for {component} ({sample_type}, n={len(sample_list)})'
            save_path = vis_dir / f'{component}_attention_{sample_type}.png'
            plot_attention_heatmap(avg_attentions[component], title, save_path, vmin_attention, vmax_attention)
        
        # 绘制所有组件的总和热力图
        title = f'Sum of Attention Weights Across All Components ({sample_type}, n={len(sample_list)})'
        save_path = vis_dir / f'component_sum_attention_{sample_type}.png'
        plot_component_sum_attention(avg_attentions, title, save_path, vmin_sum, vmax_sum)
    
    print("\n✅ Visualization complete!")

if __name__ == "__main__":
    main()