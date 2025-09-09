import json
from pathlib import Path
from tqdm import tqdm

# 配置
DATA_DIR = Path("data/paraphrased_causal_data_v2")
BACKUP_DIR = Path("data/paraphrased_causal_data_backup_v2")  # 备份原始数据

def add_ids_to_file(file_path: Path, backup: bool = True):
    """给单个文件的所有样本添加ID"""
    print(f"\n处理文件: {file_path.name}")
    
    # 读取数据
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    
    # 添加ID
    for idx, sample in enumerate(samples):
        sample["id"] = f"{file_path.stem}_{idx:04d}"  # 例如：healthcare_0001
    
    # 如果需要备份
    if backup:
        backup_path = BACKUP_DIR / file_path.name
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        print(f"备份原始文件到: {backup_path}")
        with open(backup_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    # 保存添加了ID的文件
    print(f"保存添加了ID的文件到: {file_path}")
    with open(file_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"完成! 共处理 {len(samples)} 个样本")

def main():
    """主函数"""
    # 获取所有jsonl文件
    jsonl_files = list(DATA_DIR.glob("*.jsonl"))
    print(f"找到 {len(jsonl_files)} 个文件需要处理")
    
    # 处理所有文件
    for file_path in tqdm(jsonl_files, desc="处理文件", unit="file"):
        add_ids_to_file(file_path)
    
    print("\n✅ 所有文件处理完成！")

if __name__ == "__main__":
    main() 