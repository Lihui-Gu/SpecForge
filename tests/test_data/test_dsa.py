import torch
import time
import torch
import time
import numpy as np
from specforge.modeling.draft.llama3_eagle import optimized_topk_attention

def generate_realistic_topk_indices(query, keys, K):
    """基于真实attention分数生成topk索引"""
    # 计算attention分数
    scores = torch.matmul(query, keys.transpose(-2, -1))  # (bsz, heads, q_len, kv_len)
    
    # 取topk
    topk_scores, topk_indices = torch.topk(scores, k=K, dim=-1)
    return topk_indices

def benchmark_attention_methods(bsz=2, head_num=8, kv_len=1000, head_dim=64, K=100, q_len=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 生成测试数据 - 统一格式为 (bsz, heads, seq_len, dim)
    all_key_states = torch.randn(bsz, head_num, kv_len, head_dim, device=device)
    all_value_states = torch.randn(bsz, head_num, kv_len, head_dim, device=device)
    query_states = torch.randn(bsz, head_num, q_len, head_dim, device=device)
    
    # 生成真实的topk索引
    topk_indices = generate_realistic_topk_indices(query_states, all_key_states, K)
    
    print(f"测试配置: bsz={bsz}, heads={head_num}, kv_len={kv_len}, K={K}, q_len={q_len}")
    print("=" * 60)
    
    # 预热GPU（减少次数）
    for _ in range(100):
        _ = torch.matmul(torch.randn(1000, 1000, device=device), 
                        torch.randn(1000, 1000, device=device))
    
    # 测试次数减少，增加稳定性
    num_iterations = 1000

    # 转置为F.scaled_dot_product_attention期望的格式
    q_transposed = query_states.transpose(1, 2)  # (bsz, q_len, heads, dim)
    k_transposed = all_key_states.transpose(1, 2)  # (bsz, kv_len, heads, dim)
    v_transposed = all_value_states.transpose(1, 2)

    torch.cuda.synchronize()

    sdpa_cost_time = []
    x = kv_len
    for _ in range(num_iterations):
        all_key_states = torch.randn(bsz, head_num, x, head_dim, device=device)
        all_value_states = torch.randn(bsz, head_num, x, head_dim, device=device)
        query_states = torch.randn(bsz, head_num, q_len, head_dim, device=device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        full_out = torch.nn.functional.scaled_dot_product_attention(
            q_transposed, k_transposed, v_transposed, dropout_p=0.0
        )
        end_event.record()
        torch.cuda.synchronize()
        sdpa_cost_time.append(start_event.elapsed_time(end_event))
        x = x + 1

    torch.cuda.synchronize()
    full_time = np.mean(sdpa_cost_time)
    
    print(f"Full Attention 平均耗时: {full_time:.3f} ms")
    
    # 测试topk attention
    torch.cuda.synchronize()

    dsa_cost_time = []
    x = kv_len
    for _ in range(num_iterations):
        all_key_states = torch.randn(bsz, head_num, x, head_dim, device=device)
        all_value_states = torch.randn(bsz, head_num, x, head_dim, device=device)
        query_states = torch.randn(bsz, head_num, q_len, head_dim, device=device)
        
        # 生成真实的topk索引
        topk_indices = generate_realistic_topk_indices(query_states, all_key_states, K)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        topk_out = optimized_topk_attention(
            query_states, all_key_states, all_value_states, topk_indices, head_dim
        )
        end_event.record()
        torch.cuda.synchronize()
        dsa_cost_time.append(start_event.elapsed_time(end_event))
        x = x + 1

    torch.cuda.synchronize()
    topk_time = np.mean(dsa_cost_time)
    
    print(f"TopK Attention 平均耗时: {topk_time:.3f} ms")
    
    speedup = full_time / topk_time
    print(f"加速比: {speedup:.2f}x")
    print(f"稀疏度: {K/kv_len*100:.1f}%")

# 更合理的测试配置
if __name__ == "__main__":
    configs = [
        # (bsz, heads, kv_len, head_dim, K, q_len)
        (1, 28, 1122, 128, 128, 1),   # 10% 稀疏度
        (1, 28, 1122, 128, 128, 1),  # 20% 稀疏度
        (1, 28, 15000, 128, 1500, 1), # 10% 稀疏度
        (1, 28, 15000, 128, 3000, 1), # 20% 稀疏度
    ]
    
    for config in configs:
        benchmark_attention_methods(*config)