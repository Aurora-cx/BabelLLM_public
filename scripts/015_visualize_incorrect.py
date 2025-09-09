import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import random
import jsonlines
import matplotlib as mpl

# 设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 全局配置
BASE_DIR = Path('outputs/try/attention_weights')
RESULTS_DIR = Path('outputs/qwen15_18b_00_v2')

def load_attention_data(file_path):
    """加载attention权重数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_answers_from_results(sample_id, theme, lang='en'):
    """从results.jsonl文件中获取答案"""
    results_file = RESULTS_DIR / f'{theme}_results.jsonl'
    with jsonlines.open(results_file) as reader:
        for item in reader:
            if item['id'] == sample_id:
                return {
                    'correct_answer': item[f'ground_truth_{lang}'],
                    'model_answer': item[f'predicted_answer_{lang}']
                }
    return None

def plot_sample_heatmap(sample, title, save_path, answers, vmin=None, vmax=None):
    """绘制单个样本的注意力热力图"""
    # 获取所有组件和层数
    components = sorted(sample['attentions'].keys())
    n_layers = len(sample['attentions'][components[0]])
    
    # 创建数据矩阵
    data = np.zeros((n_layers, len(components)))
    for i, component in enumerate(components):
        # 对每个层的所有head求和
        data[:, i] = np.sum(sample['attentions'][component], axis=1)
    
    # 绘制热力图
    sns.heatmap(data, 
                xticklabels=components,
                yticklabels=range(n_layers),
                cmap='YlOrRd',
                vmin=vmin,
                vmax=vmax,
                cbar_kws={'label': '注意力权重总和'})
    
    # 设置图形属性
    title_text = f"{title}\n样本ID: {sample['sample_id']}\n"
    if answers:
        title_text += f"正确答案: {answers['correct_answer']}\n"
        title_text += f"模型答案: {answers['model_answer']}"
    plt.title(title_text)
    plt.xlabel('组件')
    plt.ylabel('层数')

def get_available_themes():
    """获取所有可用的主题"""
    themes = set()
    # 从英文目录获取主题
    en_dir = BASE_DIR / 'attention_weights_en' / 'incorrect'
    for file in en_dir.glob('attention_*.json'):
        name_list = file.stem.split('_')
        if len(name_list) < 3:
            themes.add(name_list[1])
        else:
            themes.add(name_list[1] + '_' + name_list[2])
    return sorted(list(themes))

def get_random_incorrect_sample(lang='en'):
    """随机获取一个错误样本"""
    dir_path = BASE_DIR / f'attention_weights_{lang}' / 'incorrect'
    files = list(dir_path.glob('attention_*.json'))
    if not files:
        return None, None
    
    # 随机选择一个文件
    file = random.choice(files)
    samples = load_attention_data(file)
    if len(samples) == 0:
        return None, None
    
    # 随机选择一个样本
    sample = random.choice(samples)
    
    # 获取主题
    name_list = file.stem.split('_')
    if len(name_list) < 3:
        theme = name_list[1]
    else:
        theme = name_list[1] + '_' + name_list[2]
    
    return sample, theme

def find_matching_samples():
    """找到同一个样本的中英文版本"""
    # 加载所有英文正确样本
    en_correct_samples = []
    en_correct_dir = BASE_DIR / 'attention_weights_en' / 'correct'
    for file in en_correct_dir.glob('attention_*.json'):
        samples = load_attention_data(file)
        en_correct_samples.extend(samples)
    
    # 加载所有中文错误样本
    zh_incorrect_samples = []
    zh_incorrect_dir = BASE_DIR / 'attention_weights_zh' / 'incorrect'
    for file in zh_incorrect_dir.glob('attention_*.json'):
        samples = load_attention_data(file)
        zh_incorrect_samples.extend(samples)
    
    # 找到匹配的样本
    for en_sample in en_correct_samples:
        for zh_sample in zh_incorrect_samples:
            if en_sample['sample_id'] == zh_sample['sample_id']:
                return en_sample, zh_sample
    
    return None, None

def find_sample_by_id(sample_id):
    """根据样本ID找到中英文样本"""
    # 加载英文正确样本
    en_correct_dir = BASE_DIR / 'attention_weights_en' / 'incorrect'
    for file in en_correct_dir.glob('attention_*.json'):
        samples = load_attention_data(file)
        for sample in samples:
            if sample['sample_id'] == sample_id:
                en_sample = sample
                en_file = file
                break
        if 'en_sample' in locals():
            break
    
    # 加载中文错误样本
    zh_incorrect_dir = BASE_DIR / 'attention_weights_zh' / 'correct'
    for file in zh_incorrect_dir.glob('attention_*.json'):
        samples = load_attention_data(file)
        for sample in samples:
            if sample['sample_id'] == sample_id:
                zh_sample = sample
                zh_file = file
                break
        if 'zh_sample' in locals():
            break
    
    if 'en_sample' not in locals() or 'zh_sample' not in locals():
        return None, None, None, None
    
    return en_sample, zh_sample, en_file, zh_file

def main():
    """主函数"""
    # 设置输出目录
    vis_dir = BASE_DIR / 'visualizations' / 'incorrect_samples'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # 指定样本ID
    sample_id = input("请输入要分析的样本ID: ")
    
    # 找到对应的样本
    en_sample, zh_sample, en_file, zh_file = find_sample_by_id(sample_id)
    if en_sample is None or zh_sample is None:
        print("错误: 未找到对应的样本!")
        return
    
    print(f"找到样本: {sample_id}")
    print(f"英文文件: {en_file}")
    print(f"中文文件: {zh_file}")
    
    # 从文件名获取主题
    name_list = en_file.stem.split('_')
    if len(name_list) < 3:
        theme = name_list[1]
    else:
        theme = name_list[1] + '_' + name_list[2]
    print(f"主题: {theme}")
    
    # 获取答案
    en_answers = get_answers_from_results(sample_id, theme, 'en')
    zh_answers = get_answers_from_results(sample_id, theme, 'zh')
    
    if en_answers:
        print(f"英文正确答案: {en_answers['correct_answer']}")
        print(f"英文模型答案: {en_answers['model_answer']}")
    
    if zh_answers:
        print(f"中文正确答案: {zh_answers['correct_answer']}")
        print(f"中文模型答案: {zh_answers['model_answer']}")
    
    # 计算两个样本的注意力权重范围
    en_data = np.zeros((len(en_sample['attentions'][list(en_sample['attentions'].keys())[0]]), len(en_sample['attentions'])))
    zh_data = np.zeros((len(zh_sample['attentions'][list(zh_sample['attentions'].keys())[0]]), len(zh_sample['attentions'])))
    
    for i, component in enumerate(sorted(en_sample['attentions'].keys())):
        en_data[:, i] = np.sum(en_sample['attentions'][component], axis=1)
    
    for i, component in enumerate(sorted(zh_sample['attentions'].keys())):
        zh_data[:, i] = np.sum(zh_sample['attentions'][component], axis=1)
    
    vmin = min(np.min(en_data), np.min(zh_data))
    vmax = max(np.max(en_data), np.max(zh_data))
    
    # 创建一个大图
    plt.figure(figsize=(20, 8))
    
    # 绘制英文热力图
    plt.subplot(1, 2, 1)
    plot_sample_heatmap(en_sample, '英文正确样本注意力权重', None, en_answers, vmin, vmax)
    
    # 绘制中文热力图
    plt.subplot(1, 2, 2)
    plot_sample_heatmap(zh_sample, '中文错误样本注意力权重', None, zh_answers, vmin, vmax)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    save_path = vis_dir / 'attention_heatmaps.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"\n已保存热力图到: {save_path}")
    print("\n✅ 可视化完成!")

if __name__ == "__main__":
    main() 