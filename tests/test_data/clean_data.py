import json
import numpy as np
from tqdm import tqdm

# ==== 配置 ====
input_path = "/home/gulihui/workspace/dev_github/SpecForge/cache/dataset/allava4v_qwen2_5_vl_train.jsonl"      # 原始文件路径
output_path = "/home/gulihui/workspace/dev_github/SpecForge/cache/dataset/allava4v_qwen2_5_vl_clean_train.jsonl"  # 输出文件路径
cut_ratio = 0.01  # 删除前 1%

# ==== 读取数据 ====
data = []
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

print(f"共读取 {len(data)} 条样本")

# ==== 统计字数 ====
lengths = []
for item in tqdm(data, desc="统计字数"):
    total_text = ""
    for sentence in item.get("conversations", []):
        content = sentence.get("content", "")
        if isinstance(content, list):
            for c in content:
                if c.get("type") == "text":
                    total_text += c.get("text", "")
        elif isinstance(content, str):
            total_text += content
    lengths.append(len(total_text))

lengths = np.array(lengths)
num_samples = len(lengths)
cut_num = int(num_samples * cut_ratio)

# ==== 找出前 1% 长的样本 ====
sorted_indices = np.argsort(lengths)[::-1]  # 从长到短排序
drop_indices = set(sorted_indices[:cut_num])

print(f"将删除前 {cut_ratio*100:.1f}% （{cut_num} 条）样本，最长字数 = {lengths[sorted_indices[0]]}")

# ==== 重新保存 ====
with open(output_path, "w", encoding="utf-8") as f:
    kept = 0
    for i, item in enumerate(data):
        if i not in drop_indices:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1

print(f"✅ 已保存至: {output_path}")
print(f"保留样本数: {kept}/{num_samples} ({kept/num_samples*100:.2f}%)")