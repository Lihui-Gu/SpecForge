import torch
import time
from specforge.modeling.draft.llama3_eagle import optimized_topk_attention

def full_attention(query_states, all_key_states, all_value_states):
    """
    全量attention计算，用于性能对比
    """
    # 标准full attention计算
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,  # (bsz, q_len, head_num, dim)
        all_key_states,  # (bsz, kv_len, head_num, dim)
        all_value_states,
        dropout_p=0.0,
    )
    return attn_output


def benchmark_attention_methods(bsz=2, head_num=8, kv_len=1000, head_dim=64, K=100, q_len=1):
    """
    对比topk attention和full attention的性能
    """
    # 生成测试数据
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 模拟cache数据
    all_key_states = torch.randn(bsz, head_num, kv_len, head_dim, device=device)
    all_value_states = torch.randn(bsz, head_num, kv_len, head_dim, device=device)
    query_states = torch.randn(bsz, head_num, q_len, head_dim, device=device)
    
    # 生成topk索引（模拟topk选择）
    topk_indices = torch.randint(0, kv_len, (bsz, head_num, q_len, K), device=device)
    
    print(f"测试配置: bsz={bsz}, heads={head_num}, kv_len={kv_len}, K={K}, q_len={q_len}")
    print("=" * 60)
    
    # 预热GPU
    for _ in range(1000):
        _ = torch.randn(1000, 1000, device=device).matmul(torch.randn(1000, 1000, device=device))
    
    # 测试full attention
    full_times = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(10000):  # 多次测试取平均
        start_event.record()
        full_out = full_attention(query_states, all_key_states, all_value_states)
        end_event.record()
        torch.cuda.synchronize()
        full_times.append(start_event.elapsed_time(end_event))
    
    avg_full_time = sum(full_times) / len(full_times)
    print(f"Full Attention 平均耗时: {avg_full_time:.3f} ms")
    
    # 测试topk attention
    topk_times = []
    for _ in range(10000):
        start_event.record()
        topk_out = optimized_topk_attention(query_states, all_key_states, all_value_states, topk_indices, head_dim)
        end_event.record()
        torch.cuda.synchronize()
        topk_times.append(start_event.elapsed_time(end_event))
    
    avg_topk_time = sum(topk_times) / len(topk_times)
    print(f"TopK Attention 平均耗时: {avg_topk_time:.3f} ms")
    
    # 性能对比
    speedup = avg_full_time / avg_topk_time
    print(f"加速比: {speedup:.2f}x")
    
    # 验证结果一致性（可选）
    if K == kv_len:  # 当K=kv_len时，结果应该相近
        print(f"结果差异: {torch.abs(full_out - topk_out).max().item():.6f}")

# 运行测试
if __name__ == "__main__":
    # 测试不同配置
    configs = [
        (1, 28, 506, 128, 128, 1),   # 小模型
        (1, 28, 1538, 128, 128, 1),  # 中等模型
        (1, 28, 2043, 128, 128, 1),  # 大模型
    ]
    for config in configs:
        benchmark_attention_methods(*config)
        print()