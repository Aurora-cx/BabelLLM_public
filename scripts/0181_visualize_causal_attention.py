import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from itertools import combinations
import sys
from svcca import cca_core

def load_attention_data(file_path):
    """加载注意力权重数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def plot_attention_heatmap(attention_data, save_path=None, lang="en"):
    """绘制注意力热力图
    
    Args:
        attention_data: 注意力权重数据
        save_path: 保存图片的路径
        lang: 语言版本 ("en" 或 "zh")
    """
    # 检查是否有数据
    if not attention_data:
        print(f"Warning: No attention data available for {lang}")
        return
        
    # 获取层数
    num_layers = len(attention_data[0]['attentions']['cause'])
    
    print(f"  Creating heatmap with {num_layers} layers...")
    
    # 对每个样本的注意力权重进行平均
    print("  Computing average attention weights...")
    avg_attention = {
        'cause': np.zeros(num_layers),
        'effect1': np.zeros(num_layers),
        'effect2': np.zeros(num_layers)
    }
    
    # 先计算每个样本每层的head和，再计算所有样本的平均值
    for component in ['cause', 'effect1', 'effect2']:
        print(f"    Processing {component}...")
        # 初始化累加器
        accumulator = np.zeros(num_layers)
        sample_count = 0
        
        # 逐个处理样本
        for sample in attention_data:
            # 对每一层的16个head求和
            sample_attention = np.array(sample['attentions'][component])
            layer_sums = np.sum(sample_attention, axis=1)  # 对每层的16个head求和
            accumulator += layer_sums
            sample_count += 1
            
            # 每处理100个样本打印一次进度
            if sample_count % 100 == 0:
                print(f"      Processed {sample_count} samples...")
        
        # 计算最终平均值
        avg_attention[component] = accumulator / sample_count
    
    # 创建图形
    print("  Creating figure...")
    plt.figure(figsize=(15, 8))
    
    # 绘制热力图
    print("  Drawing heatmap...")
    # 准备数据
    data = np.array([
        avg_attention['cause'],
        avg_attention['effect1'],
        avg_attention['effect2']
    ])
    
    # 创建热力图
    sns.heatmap(
        data,
        cmap='YlOrRd',
        vmin=0,
        vmax=1,
        xticklabels=[f'Layer {i+1}' for i in range(num_layers)],
        yticklabels=['Cause', 'Effect1', 'Effect2']
    )
    
    plt.title(f'Attention Weight Heatmap (Correct Samples Only - {lang.upper()})')
    plt.xlabel('Layer')
    plt.ylabel('Component')
    
    print("  Adjusting layout...")
    plt.tight_layout()
    
    # 保存图片
    print(f"  Saving figure to {save_path}...")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("  Heatmap generation completed.")

def plot_component_attention(attention_data, save_path=None, lang="en"):
    """绘制不同组件的注意力分布
    
    Args:
        attention_data: 注意力权重数据
        save_path: 保存图片的路径
        lang: 语言版本 ("en" 或 "zh")
    """
    # 检查是否有数据
    if not attention_data:
        print(f"Warning: No attention data available for {lang}")
        return
        
    print("  Computing component attention distributions...")
    
    # 计算每个组件的平均注意力权重
    components = ['cause', 'effect1', 'effect2']
    avg_attention = {comp: [] for comp in components}
    
    # 逐个处理样本
    sample_count = 0
    for sample in attention_data:
        for component in components:
            # 计算当前样本当前组件的平均attention（所有层所有head的平均）
            component_attention = np.array(sample['attentions'][component])
            component_mean = np.mean(component_attention)  # 对所有层所有head求平均
            avg_attention[component].append(component_mean)
        
        sample_count += 1
        # 每处理100个样本打印一次进度
        if sample_count % 100 == 0:
            print(f"    Processed {sample_count} samples...")
    
    # 创建箱线图
    print("  Creating box plot...")
    plt.figure(figsize=(10, 6))
    data = [avg_attention[comp] for comp in components]
    plt.boxplot(data, labels=['Cause', 'Effect1', 'Effect2'])
    plt.title(f'Attention Distribution by Component (Correct Samples Only - {lang.upper()})')
    plt.ylabel('Average Attention Weight')
    
    print(f"  Saving figure to {save_path}...")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("  Component distribution plot completed.")

def plot_component_comparison(attention_data_dict, save_dir, component):
    """绘制不同版本下某个组件的注意力对比柱状图
    
    Args:
        attention_data_dict: 包含不同版本数据的字典
        save_dir: 保存图片的目录
        component: 要绘制的组件 ('cause', 'effect1', 或 'effect2')
    """
    print(f"\nDrawing comparison bar plot for {component}...")
    
    # 获取层数
    num_layers = len(next(iter(attention_data_dict.values()))[0]['attentions'][component])
    
    # 计算每个版本的平均注意力权重
    avg_attention = {}
    for version, data in attention_data_dict.items():
        print(f"  Processing {version}...")
        accumulator = np.zeros(num_layers)
        sample_count = 0
        
        for sample in data:
            # 对每一层的16个head求和
            sample_attention = np.array(sample['attentions'][component])
            layer_sums = np.sum(sample_attention, axis=1)
            accumulator += layer_sums
            sample_count += 1
            
            if sample_count % 100 == 0:
                print(f"    Processed {sample_count} samples...")
        
        avg_attention[version] = accumulator / sample_count
    
    # 创建柱状图
    plt.figure(figsize=(15, 8))
    
    # 设置柱状图的位置
    x = np.arange(num_layers)
    width = 0.2  # 柱子的宽度
    
    # 绘制每个版本的柱状图
    versions = ['en', 'zh', 'en_para', 'zh_para']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f1c40f']
    
    for i, version in enumerate(versions):
        if version in avg_attention:
            plt.bar(x + i*width, avg_attention[version], width, 
                   label=version.upper(), color=colors[i])
    
    # 设置图表属性
    plt.title(f'Attention Weight Comparison for {component.upper()}')
    plt.xlabel('Layer')
    plt.ylabel('Attention Weight')
    plt.xticks(x + width*1.5, [f'Layer {i+1}' for i in range(num_layers)], rotation=45)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    save_path = save_dir / f"{component}_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison plot to {save_path}")

def calculate_cosine_similarity(attention_data_dict, save_dir):
    """计算不同版本之间注意力权重的余弦相似度
    
    Args:
        attention_data_dict: 包含不同版本数据的字典
        save_dir: 保存结果的目录
    """
    print("\nCalculating cosine similarities...")
    
    # 计算每个版本的平均注意力权重向量
    avg_attention = {}
    versions = ['en', 'zh', 'en_para', 'zh_para']
    components = ['cause', 'effect1', 'effect2']
    
    for version in versions:
        if version not in attention_data_dict:
            continue
            
        print(f"  Processing {version}...")
        avg_attention[version] = {}
        
        for component in components:
            # 初始化累加器
            accumulator = np.zeros(24)  # 24层
            sample_count = 0
            
            for sample in attention_data_dict[version]:
                # 对每一层的16个head求和
                sample_attention = np.array(sample['attentions'][component])
                layer_sums = np.sum(sample_attention, axis=1)
                accumulator += layer_sums
                sample_count += 1
            
            # 计算平均值
            avg_attention[version][component] = accumulator / sample_count
    
    # 计算余弦相似度
    results = {}
    for component in components:
        print(f"\nCosine similarities for {component}:")
        results[component] = {}
        
        # 计算所有版本对之间的相似度
        for v1 in versions:
            if v1 not in avg_attention:
                continue
            results[component][v1] = {}
            
            for v2 in versions:
                if v2 not in avg_attention:
                    continue
                    
                # 计算余弦相似度
                vec1 = avg_attention[v1][component]
                vec2 = avg_attention[v2][component]
                
                # 归一化向量
                vec1_norm = vec1 / np.linalg.norm(vec1)
                vec2_norm = vec2 / np.linalg.norm(vec2)
                
                # 计算余弦相似度
                similarity = np.dot(vec1_norm, vec2_norm)
                results[component][v2] = similarity
                
                print(f"  {v1} vs {v2}: {similarity:.4f}")
    
    # 保存结果到文件
    save_path = save_dir / "cosine_similarities.txt"
    with open(save_path, 'w') as f:
        f.write("Cosine Similarities between Different Versions\n")
        f.write("=============================================\n\n")
        
        for component in components:
            f.write(f"\n{component.upper()}:\n")
            f.write("-" * 50 + "\n")
            
            # 写入表头
            f.write("Version".ljust(10))
            for v2 in versions:
                if v2 in avg_attention:
                    f.write(f"{v2.upper()}".ljust(15))
            f.write("\n")
            
            # 写入数据
            for v1 in versions:
                if v1 not in avg_attention:
                    continue
                f.write(f"{v1.upper()}".ljust(10))
                for v2 in versions:
                    if v2 not in avg_attention:
                        continue
                    f.write(f"{results[component][v2]:.4f}".ljust(15))
                f.write("\n")
    
    print(f"\nResults saved to {save_path}")

def calculate_total_attention_and_differences(attention_data_dict, save_dir):
    """计算每个组件下四个版本24维向量的总和，并输出所有两两之间的绝对值差异"""
    print("\nCalculating total attention sums and pairwise absolute differences...")
    
    versions = ['en', 'zh', 'en_para', 'zh_para']
    components = ['cause', 'effect1', 'effect2']
    
    # 计算每个版本每个组件的总和
    total_sums = {comp: {} for comp in components}
    for version in versions:
        if version not in attention_data_dict:
            continue
        for component in components:
            accumulator = np.zeros(24)
            sample_count = 0
            for sample in attention_data_dict[version]:
                sample_attention = np.array(sample['attentions'][component])
                layer_sums = np.sum(sample_attention, axis=1)
                accumulator += layer_sums
                sample_count += 1
            avg_vector = accumulator / sample_count
            total_sum = np.sum(avg_vector)
            total_sums[component][version] = total_sum
    
    # 计算两两之间的绝对值差异
    results = {}
    for component in components:
        results[component] = {}
        print(f"\nComponent: {component}")
        for v1, v2 in combinations(versions, 2):
            if v1 in total_sums[component] and v2 in total_sums[component]:
                diff = abs(total_sums[component][v1] - total_sums[component][v2])
                results[component][f"{v1} vs {v2}"] = diff
                print(f"  {v1} sum: {total_sums[component][v1]:.6f}, {v2} sum: {total_sums[component][v2]:.6f}, |diff|: {diff:.6f}")
    
    # 保存结果到文件
    save_path = save_dir / "total_attention_sums_and_differences.txt"
    with open(save_path, 'w') as f:
        for component in components:
            f.write(f"Component: {component}\n")
            for version in versions:
                if version in total_sums[component]:
                    f.write(f"  {version}: {total_sums[component][version]:.6f}\n")
            for k, v in results[component].items():
                f.write(f"  {k}: {v:.6f}\n")
            f.write("\n")
    print(f"\nResults saved to {save_path}")

def calculate_en_component_layerwise_similarity(attention_data_dict, save_dir):
    """计算EN版本下cause、effect1、effect2三个组件的24维向量两两之间的余弦相似度"""
    print("\nCalculating EN component-wise layer similarity (cosine)...")
    version = 'en'
    components = ['cause', 'effect1', 'effect2']
    if version not in attention_data_dict:
        print("  EN version not found in data.")
        return
    # 计算每个组件的24维向量
    avg_vectors = {}
    for component in components:
        accumulator = np.zeros(24)
        sample_count = 0
        for sample in attention_data_dict[version]:
            sample_attention = np.array(sample['attentions'][component])
            layer_sums = np.sum(sample_attention, axis=1)
            accumulator += layer_sums
            sample_count += 1
        avg_vector = accumulator / sample_count
        avg_vectors[component] = avg_vector
    # 计算两两余弦相似度
    from numpy import dot
    from numpy.linalg import norm
    results = {}
    for i in range(3):
        for j in range(i+1, 3):
            c1, c2 = components[i], components[j]
            v1, v2 = avg_vectors[c1], avg_vectors[c2]
            sim = dot(v1, v2) / (norm(v1) * norm(v2))
            results[f"{c1} vs {c2}"] = sim
            print(f"  {c1} vs {c2}: {sim:.4f}")
    # 保存结果
    save_path = save_dir / "en_component_layerwise_cosine_similarity.txt"
    with open(save_path, 'w') as f:
        for k, v in results.items():
            f.write(f"{k}: {v:.6f}\n")
    print(f"Results saved to {save_path}")

def calculate_svcca_between_versions(attention_data_dict, save_dir):
    """
    计算 en, zh, en_para, zh_para 四个版本 (3, 24) 表征的两两 SVCCA 相似度
    """
    print("\nCalculating SVCCA similarities between all versions...")
    try:
        from svcca import cca_core
    except ImportError:
        print("  Please install svcca: pip install svcca")
        return
    import numpy as np
    from itertools import combinations

    versions = ['en', 'zh', 'en_para', 'zh_para']
    components = ['cause', 'effect1', 'effect2']
    matrices = {}

    # 构建每个版本的 (3, 24) 矩阵
    for version in versions:
        if version not in attention_data_dict:
            continue
        mat = []
        for component in components:
            accumulator = np.zeros(24)
            sample_count = 0
            for sample in attention_data_dict[version]:
                sample_attention = np.array(sample['attentions'][component])
                layer_sums = np.sum(sample_attention, axis=1)
                accumulator += layer_sums
                sample_count += 1
            avg_vector = accumulator / sample_count
            mat.append(avg_vector)
        matrices[version] = np.stack(mat, axis=0)  # shape (3, 24)

    # 计算所有两两组合的 SVCCA
    results = {}
    for v1, v2 in combinations(versions, 2):
        if v1 in matrices and v2 in matrices:
            mat1 = matrices[v1]
            mat2 = matrices[v2]
            svcca_result = cca_core.get_cca_similarity(mat1, mat2, epsilon=1e-10, verbose=False)
            # 兼容不同返回结构
            if 'cca_coef' in svcca_result:
                mean_svcca = svcca_result['cca_coef'].mean()
            elif 'cca_coef1' in svcca_result:
                mean_svcca = svcca_result['cca_coef1'].mean()
            elif 'cca_coefs' in svcca_result:
                mean_svcca = svcca_result['cca_coefs'].mean()
            else:
                raise ValueError(f"Unknown SVCCA result keys: {svcca_result.keys()}")
            results[f"{v1} vs {v2}"] = mean_svcca
            print(f"  SVCCA({v1} vs {v2}): {mean_svcca:.4f}")

    # 保存结果
    save_path = save_dir / "svcca_similarities.txt"
    with open(save_path, 'w') as f:
        for k, v in results.items():
            f.write(f"{k}: {v:.6f}\n")
    print(f"Results saved to {save_path}")

def main():
    # 设置数据目录
    base_dir = Path("outputs/try/causal_attention")
    output_dir = Path("outputs/try/causal_attention/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集所有版本的数据
    attention_data_dict = {
        'correct': {},    # 正确样本
        'incorrect': {},  # 错误样本
        'all': {}        # 全量样本
    }
    
    # 处理所有数据文件
    for data_dir in base_dir.glob("attention_weights_*"):
        if not data_dir.is_dir():
            continue
            
        print(f"\nProcessing {data_dir.name}...")
        
        # 确定版本
        version = data_dir.name.replace("attention_weights_", "")
        
        # 收集正确样本的注意力数据
        correct_samples = []
        incorrect_samples = []
        
        # 处理correct目录中的每个主题文件
        correct_dir = data_dir / "correct"
        incorrect_dir = data_dir / "incorrect"
        
        # 处理正确样本
        if correct_dir.exists():
            json_files = list(correct_dir.glob("attention_*.json"))
            total_files = len(json_files)
            
            for i, json_file in enumerate(json_files, 1):
                print(f"  Processing correct file {i}/{total_files}: {json_file.name}")
                attention_data = load_attention_data(json_file)
                correct_samples.extend(attention_data)
                del attention_data
        
        # 处理错误样本
        if incorrect_dir.exists():
            json_files = list(incorrect_dir.glob("attention_*.json"))
            total_files = len(json_files)
            
            for i, json_file in enumerate(json_files, 1):
                print(f"  Processing incorrect file {i}/{total_files}: {json_file.name}")
                attention_data = load_attention_data(json_file)
                incorrect_samples.extend(attention_data)
                del attention_data
        
        print(f"  Found {len(correct_samples)} correct samples and {len(incorrect_samples)} incorrect samples")
        
        # 保存数据
        if correct_samples:
            attention_data_dict['correct'][version] = correct_samples
        if incorrect_samples:
            attention_data_dict['incorrect'][version] = incorrect_samples
        if correct_samples or incorrect_samples:
            attention_data_dict['all'][version] = correct_samples + incorrect_samples
    
    # 为每种样本类型生成分析结果
    for sample_type in ['correct', 'incorrect', 'all']:
        print(f"\n{'='*50}")
        print(f"Processing {sample_type} samples...")
        print(f"{'='*50}")
        
        # 创建对应的输出目录
        type_output_dir = output_dir / sample_type
        type_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取当前类型的数据
        current_data = attention_data_dict[sample_type]
        
        if not current_data:
            print(f"No {sample_type} samples found for any version")
            continue
        
        # 为每个版本生成热力图和组件分布图
        for version, samples in current_data.items():
            print(f"\nGenerating visualizations for {version}...")
            
            # 生成热力图
            print("  Generating heatmap...")
            heatmap_path = type_output_dir / f"{version}_heatmap.png"
            plot_attention_heatmap(samples, heatmap_path, lang=version.split('_')[0])
            
            # 生成组件分布图
            print("  Generating component distribution plot...")
            component_path = type_output_dir / f"{version}_components.png"
            plot_component_attention(samples, component_path, lang=version.split('_')[0])
        
        # 为每个组件生成对比图
        for component in ['cause', 'effect1', 'effect2']:
            plot_component_comparison(current_data, type_output_dir, component)
        
        # 计算各种统计指标
        calculate_cosine_similarity(current_data, type_output_dir)
        calculate_total_attention_and_differences(current_data, type_output_dir)
        calculate_en_component_layerwise_similarity(current_data, type_output_dir)
        calculate_svcca_between_versions(current_data, type_output_dir)
        
        print(f"\n✅ {sample_type} samples analysis completed!")

if __name__ == "__main__":
    main() 