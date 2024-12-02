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
    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10

    score = torch.randn((m, e), device="cuda", dtype=dtype)
    
    triton_output = fused_moe(a, w1, w2, score, topk, renormalize=False)

    import time
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        triton_output = fused_moe(a, w1, w2, score, topk, renormalize=False)
    torch.cuda.synchronize()
    
    print(f"bf16 {m} cost time: {(time.time() - start) * 1000}")
    
    # torch_output = torch_moe(a, w1, w2, score, topk)
    # torch.testing.assert_close(triton_output, torch_output, atol=2e-2, rtol=0)

test_fused_moe(128, 5120, 192, 160, 6, torch.bfloat16)
test_fused_moe(256, 5120, 192, 160, 6, torch.bfloat16)
test_fused_moe(512, 5120, 192, 160, 6, torch.bfloat16)
test_fused_moe(1024, 5120, 192, 160, 6, torch.bfloat16)

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
    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    w1, w1_scale = quantize_moe(w1)
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
    w2, w2_scale = quantize_moe(w2)

    score = torch.randn((m, e), device="cuda", dtype=dtype)
    
    triton_output = fused_moe(a, w1, w2, score, topk, renormalize=False, use_fp8_w8a8=True, w1_scale=w1_scale, w2_scale=w2_scale)

    import time
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        triton_output = fused_moe(a, w1, w2, score, topk, renormalize=False, use_fp8_w8a8=True, w1_scale=w1_scale, w2_scale=w2_scale)
    torch.cuda.synchronize()
    
    print(f"fp8 {m} cost time: {(time.time() - start) * 1000}")
    
    # torch_output = torch_moe(a, w1, w2, score, topk)
    # torch.testing.assert_close(triton_output, torch_output, atol=2e-2, rtol=0)
    
test_fused_moe_fp8(128, 5120, 192, 160, 6, torch.bfloat16)
test_fused_moe_fp8(256, 5120, 192, 160, 6, torch.bfloat16)
test_fused_moe_fp8(512, 5120, 192, 160, 6, torch.bfloat16)
test_fused_moe_fp8(1024, 5120, 192, 160, 6, torch.bfloat16)