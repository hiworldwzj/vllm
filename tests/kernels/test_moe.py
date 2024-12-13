"""Tests for the MOE layers.

Run `pytest tests/kernels/test_moe.py`.
"""
import sys
sys.path.append("/nvme/wzj/dev1/latest_test_lightllm/vllm")

# import pytest
import torch
from transformers import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock


import vllm.model_executor.layers.fused_moe  # noqa
from utils import (compute_max_diff, opcheck, stack_and_dev,
                                 torch_moe, torch_moe_single)
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe import fused_moe
from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_topk, moe_align_block_size)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    marlin_quantize)
from vllm.model_executor.models.mixtral import MixtralMoE
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

NUM_EXPERTS = [8, 64]
TOP_KS = [2, 6]

g_test_count = 20
# @pytest.mark.parametrize("m", [1, 33, 64, 222, 1024 * 128])
# @pytest.mark.parametrize("n", [128, 1024, 2048])
# @pytest.mark.parametrize("k", [128, 511, 1024])
# @pytest.mark.parametrize("e", NUM_EXPERTS)
# @pytest.mark.parametrize("topk", TOP_KS)
# @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.no_grad()
def test_fused_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
):
    input_tuples = []
    for _ in range(g_test_count):
        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
        w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
        w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        input_tuples.append((a, w1, w2, score))
    
    triton_output = fused_moe(a, w1, w2, score, topk, renormalize=False)

    import time
    torch.cuda.synchronize()
    start = time.time()
    for index in range(g_test_count):
        a, w1, w2, score = input_tuples[index]
        triton_output = fused_moe(a, w1, w2, score, topk, renormalize=False)
    torch.cuda.synchronize()
    
    cost_time = (time.time() - start) * 1000
    
    print(f"bf16 {m} cost time: {cost_time} ms")
    return cost_time
    
    # torch_output = torch_moe(a, w1, w2, score, topk)
    # torch.testing.assert_close(triton_output, torch_output, atol=2e-2, rtol=0)
    



def quantize_moe(weight):
    num_experts = weight.shape[0]
    qweights = []
    weight_scales = []
    qweights = torch.empty_like(weight, dtype=torch.float8_e4m3fn).cuda()
    for i in range(num_experts):
        qweight, weight_scale = ops.scaled_fp8_quant(weight[i].cuda(), scale=None, use_per_token_if_dynamic=False)
        qweights[i] = qweight
        weight_scales.append(weight_scale)
    weight_scale = torch.cat(weight_scales, dim=0).reshape(-1)
    return qweights, weight_scale

@torch.no_grad()
def test_fused_moe_fp8(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
):   
    input_tuples = []
    for _ in range(g_test_count):
        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
        w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
        w1, w1_scale = quantize_moe(w1)
        w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
        w2, w2_scale = quantize_moe(w2)

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        input_tuples.append((a, w1, w1_scale, w2, w2_scale))
    
    triton_output = fused_moe(a, w1, w2, score, topk, renormalize=False, use_fp8_w8a8=True, w1_scale=w1_scale, w2_scale=w2_scale)

    import time
    torch.cuda.synchronize()
    start = time.time()
    for index in range(g_test_count):
        a, w1, w1_scale, w2, w2_scale = input_tuples[index]
        triton_output = fused_moe(a, w1, w2, score, topk, renormalize=False, use_fp8_w8a8=True, w1_scale=w1_scale, w2_scale=w2_scale)
    torch.cuda.synchronize()
    cost_time = (time.time() - start) * 1000
    
    print(f"dynamic fp8 {m} cost time: {cost_time} ms")
    return cost_time
    
    # torch_output = torch_moe(a, w1, w2, score, topk)
    # torch.testing.assert_close(triton_output, torch_output, atol=2e-2, rtol=0)
    



@torch.no_grad()
def test_fused_moe_fp8_static(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
):
    input_tuples = []
    for _ in range(g_test_count):
        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
        a_t, a1_scale = ops.scaled_fp8_quant(a, scale=None, use_per_token_if_dynamic=False)
        w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
        w1, w1_scale = quantize_moe(w1)
        w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
        w2, w2_scale = quantize_moe(w2)

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        input_tuples.append((a, w1, w2, score, w1_scale, w2_scale, a1_scale))
    
    triton_output = fused_moe(a, w1, w2, score, topk, renormalize=False, use_fp8_w8a8=True, w1_scale=w1_scale, w2_scale=w2_scale, a1_scale=a1_scale, a2_scale=a1_scale)

    import time
    torch.cuda.synchronize()
    start = time.time()
    for index in range(g_test_count):
        a, w1, w2, score, w1_scale, w2_scale, a1_scale = input_tuples[index]
        triton_output = fused_moe(a, w1, w2, score, topk, renormalize=False, use_fp8_w8a8=True, w1_scale=w1_scale, w2_scale=w2_scale, a1_scale=a1_scale, a2_scale=a1_scale)
    torch.cuda.synchronize()
    cost_time = (time.time() - start) * 1000
    print(f"static fp8 {m} cost time: {cost_time} ms")
    return cost_time
    # torch_output = torch_moe(a, w1, w2, score, topk)
    # torch.testing.assert_close(triton_output, torch_output, atol=2e-2, rtol=0)


# test_fused_moe(128, 192, 5120, 160, 6, torch.bfloat16)
# test_fused_moe(256, 192, 5120, 160, 6, torch.bfloat16)
# test_fused_moe(512, 192, 5120, 160, 6, torch.bfloat16)
# test_fused_moe(1024, 192, 5120, 160, 6, torch.bfloat16)

# test_fused_moe_fp8(128, 192, 5120, 160, 6, torch.bfloat16)
# test_fused_moe_fp8(256, 192, 5120, 160, 6, torch.bfloat16)
# test_fused_moe_fp8(512, 192, 5120, 160, 6, torch.bfloat16)
# test_fused_moe_fp8(1024, 192, 5120, 160, 6, torch.bfloat16)

# test_fused_moe_fp8_static(128, 192, 5120, 160, 6, torch.bfloat16)
# test_fused_moe_fp8_static(256, 192, 5120, 160, 6, torch.bfloat16)
# test_fused_moe_fp8_static(512, 192, 5120, 160, 6, torch.bfloat16)
# test_fused_moe_fp8_static(1024,192, 5120, 160, 6, torch.bfloat16)

import subprocess
import json
from multiprocessing import Process, Queue

def worker(m, n, k, e, topk, dtype, test_configs, queue):
    from vllm.model_executor.layers.fused_moe.fused_moe import get_default_config

    def fix_get_default_config(
            M: int,
            E: int,
            N: int,
            K: int,
            topk: int,
            dtype: str,
            is_marlin: bool,
        ):
        import os
        return os.config
    
    get_default_config.__code__ = fix_get_default_config.__code__

    try:
        for index in range(len(test_configs)):
            import os
            os.config = test_configs[index]
            cost_time = test_fused_moe_fp8(m, n, k, e, topk, dtype)
            queue.put(cost_time)  # Put result in queue
    except Exception as ex:
        import sys
        sys.exit(-1) 
        pass


def tuning_configs(m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype):
    
    best_config, best_cost_time = None, 10000000
    queue = Queue()
    test_configs = []
    for block_m in [16, 32, 64, 128, 256]:
        for block_n in [16, 32, 64, 128, 256]:
            for block_k in [32, 64, 128, 256]:
                for group_m in [1, 2, 4, 8]:
                    for num_warps in [4, 8]:
                        for num_stages in [1, 2, 3]:
                            t_config = {        
                                'BLOCK_SIZE_M': block_m,
                                'BLOCK_SIZE_N': block_n,
                                'BLOCK_SIZE_K': block_k,
                                'GROUP_SIZE_M': group_m,
                                'num_warps': num_warps,
                                'num_stages': num_stages,
                            }
                            test_configs.append(t_config)
                            if len(test_configs) < 64:
                                continue
                            
                            p = Process(target=worker, args=(m, n, k, e, topk, dtype, test_configs, queue))
                            p.start()
                            p.join()
                            get_count = 0
                            while get_count < len(test_configs):
                                try:
                                    print(test_configs[get_count])
                                    cost_time = queue.get_nowait()
                                    get_count += 1
                                    print(cost_time)
                                    if cost_time < best_cost_time:
                                        best_config = t_config
                                        best_cost_time = cost_time
                                except:
                                    break
                            test_configs = test_configs[get_count + 1:]
                            
    
    p = Process(target=worker, args=(m, n, k, e, topk, dtype, test_configs, queue))
    p.start()
    p.join()
    get_count = 0
    while get_count < len(test_configs):
        try:
            print(test_configs[get_count])
            cost_time = queue.get_nowait()
            get_count += 1
            print(cost_time)
            if cost_time < best_cost_time:
                best_config = t_config
                best_cost_time = cost_time
        except:
            break
    test_configs = test_configs[get_count + 1:]                        
                                
    print(best_config, best_cost_time)

# tuning_configs(256, 192, 5120, 160, 6, torch.bfloat16)
tuning_configs(1024, 192, 5120, 160, 6, torch.bfloat16)
    
# {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256, 'GROUP_SIZE_M': 8, 'num_warps': 8, 'num_stages': 2}
# {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4, 'num_warps': 8, 'num_stages': 1} for batch size 258
# {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 2, 'num_warps': 4, 'num_stages': 2} for batch size 199
import os
os.config =  {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 2, 'num_warps': 4, 'num_stages': 2}
g_test_count = 60
# test_fused_moe_fp8(128, 192, 5120, 160, 6, torch.bfloat16)
# test_fused_moe_fp8(180, 192, 5120, 160, 6, torch.bfloat16)
# test_fused_moe_fp8(199, 192, 5120, 160, 6, torch.bfloat16)
# test_fused_moe_fp8(200, 192, 5120, 160, 6, torch.bfloat16)
# test_fused_moe_fp8(256, 192, 5120, 160, 6, torch.bfloat16)
# test_fused_moe_fp8(512, 192, 5120, 160, 6, torch.bfloat16)
# test_fused_moe_fp8(1024, 192, 5120, 160, 6, torch.bfloat16)

# for i in range(201):
#     test_fused_moe_fp8(i + 1, 192, 5120, 160, 6, torch.bfloat16)
