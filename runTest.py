import os
import torch
import subprocess

import triton
import triton.language as tl

import passList
from utils import common, passdiff


@triton.jit
def matmul_kernel_fp32(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr = 32,
    BLOCK_N: tl.constexpr = 32,
    BLOCK_K: tl.constexpr = 32,
):
    # ── program ID (block coordinates) ────────────
    pid_m = tl.program_id(0)         # row tile index
    pid_n = tl.program_id(1)         # column tile index

    # ── offsets for each tile ────────────
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # ── A, B matrix tile pointers ────────────
    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    # ── K-dimension loop: accumulate BLOCK_K ────────────
    for k in range(0, K, BLOCK_K):
        a = tl.load(
            A_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K),
            other=0.0,
        )
        b = tl.load(
            B_ptrs,
            mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b)          # FP32 dot product accumulation

        # move pointer to next K-tile
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    # ── save result ─────────────────────────────────
    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        C_ptrs, acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def triton_matmul_fp32(a: torch.Tensor, b: torch.Tensor):
    """
    a: [M, K] FP32 CUDA tensor
    b: [K, N] FP32 CUDA tensor
    return: c = a @ b (FP32, [M, N])
    """
    assert a.is_cuda and b.is_cuda, "Input must be a GPU Tensor."
    assert a.dtype == torch.float32 and b.dtype == torch.float32, "FP32 only kernel."
    assert a.shape[1] == b.shape[0], "K dimension must match."

    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    grid = (triton.cdiv(M, 32), triton.cdiv(N, 32))
    compiled_kernel = matmul_kernel_fp32[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )

    return compiled_kernel


# ── simple test ──────────────────────────────────
def run_inference_test():
    torch.manual_seed(0)
    M, K, N = 128, 256, 64
    a = torch.randn((M, K), device="cuda", dtype=torch.float32)
    b = torch.randn((K, N), device="cuda", dtype=torch.float32)

    compiled_kernel = triton_matmul_fp32(a, b)

    return compiled_kernel


if __name__ == "__main__":
    cfg = common.config

    compiled_kernel = run_inference_test()

    # Print IRs in _compiled directory
    common.print_ir(compiled_kernel)

    # Run triton-opt
    print("Running TritonIR Transform Pass")
    passList.run_triton_transform_pass_test(cfg)

    print("Running TritonToTritonGPU Pass")
    passList.run_triton_to_triton_gpu_pass_test(cfg)

    print("Running TritonGPUToLLVM Pass")
    passList.run_triton_gpu_to_llvm_pass_test(cfg)

    print("Compile done.")



