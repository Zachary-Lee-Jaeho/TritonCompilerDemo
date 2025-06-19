# Triton Compilation Pipeline Test Suite

## Overview

This project provides a comprehensive test suite for analyzing and understanding the Triton compilation pipeline. It demonstrates the complete transformation process from high-level Triton kernels to LLVM Dialect code through multiple intermediate representations (IRs). The test suite is specifically designed to work with Triton version `v3.3.1` and provides detailed insights into each compilation stage.

## Project Structure

```
TritonTest/
├── runTest.py              # Main test runner with matrix multiplication kernel
├── README.md               # This documentation file
├── .gitignore              # Git ignore rules for temporary files
├── utils/                  # Utility modules
│   ├── common.py           # Configuration and IR printing utilities
│   └── passdiff.py         # MLIR pass difference analysis tool
├── passList/               # Compilation pass definitions
│   ├── __init__.py         # Module exports
│   ├── tritonTransform.py  # Triton IR transformation passes
│   ├── tritonToTritonGPU.py # Triton to TritonGPU conversion passes
│   └── tritonGPUToLLVM.py  # TritonGPU to LLVM IR conversion passes
├── _compiled/              # Generated IR files (TTIR, TTGIR, LLVM, PTX)
├── _output_mlir/           # Intermediate MLIR files after each pass
├── _mlir_dump/             # Detailed pass-by-pass transformation dumps
└── _pass_diff/             # Diff files showing changes between passes
```

## Prerequisites

### Triton Installation
This tutorial was tested on Triton version `v3.3.1`. Ensure you have Triton properly installed in your environment.

### Required Dependencies
- Python 3.8+
- PyTorch with CUDA support
- Triton v3.3.1
- triton-opt binary (compiled from Triton source)

### Setting up triton-opt Path

The `triton-opt` binary is essential for running the MLIR transformation passes. You need to set the binary path in your environment:

```bash
export PATH="$PATH_TO_TRITON/python/build/cmake.linux-x86_64-cpython-3.12/bin":$PATH
```

Replace `$PATH_TO_TRITON` with the actual path to your Triton installation directory.

## Core Components

### 1. Matrix Multiplication Kernel (`runTest.py`)

The main test file contains a complete FP32 matrix multiplication kernel implemented in Triton:

- **Kernel Function**: `matmul_kernel_fp32` - A tiled matrix multiplication kernel with configurable block sizes
- **Wrapper Function**: `triton_matmul_fp32` - PyTorch-compatible wrapper for the kernel
- **Test Function**: `run_inference_test` - Simple test with 128×256×64 matrices

#### Key Features:
- **Block-based computation**: Uses 32×32×32 tile sizes for optimal GPU utilization
- **Memory coalescing**: Optimized memory access patterns for GPU efficiency
- **Masking**: Proper boundary handling for non-divisible matrix dimensions
- **FP32 precision**: Full single-precision floating-point arithmetic

### 2. Compilation Pipeline

The project implements a three-stage compilation pipeline:

#### Stage 1: Triton IR Transformations (`tritonTransform.py`)
Applies high-level optimizations to the initial Triton IR:
- `-inline`: Function inlining
- `-triton-rewrite-tensor-pointer`: Tensor pointer optimizations
- `-canonicalize`: Canonical form conversion
- `-triton-combine`: Operation combination
- `-triton-reorder-broadcast`: Broadcast operation reordering
- `-cse`: Common subexpression elimination
- `-symbol-dce`: Dead code elimination
- `-triton-loop-unroll`: Loop unrolling optimizations

#### Stage 2: Triton to TritonGPU Conversion (`tritonToTritonGPU.py`)
Converts Triton IR to GPU-specific TritonGPU IR:
- `--convert-triton-to-tritongpu`: Main conversion pass with GPU parameters
- `-tritongpu-coalesce`: Memory coalescing optimizations
- `-tritongpu-F32DotTC`: FP32 dot product tensor core utilization
- `-triton-nvidia-gpu-plan-cta`: Cooperative thread array planning
- `-tritongpu-accelerate-matmul`: Matrix multiplication acceleration
- `-tritongpu-pipeline`: Instruction pipelining
- `-tritongpu-prefetch`: Memory prefetching optimizations
- And many more GPU-specific optimizations...

#### Stage 3: TritonGPU to LLVM IR Conversion (`tritonGPUToLLVM.py`)
Final conversion to LLVM IR for GPU code generation:
- `-convert-scf-to-cf`: Structured control flow conversion
- `-convert-index-to-llvm`: Index type conversion
- `-allocate-shared-memory`: Shared memory allocation
- `-convert-triton-gpu-to-llvm`: Main conversion with compute capability 8.6
- `-convert-nv-gpu-to-llvm`: NVIDIA GPU dialect conversion
- `-convert-arith-to-llvm`: Arithmetic operation conversion

### 3. Utility Modules

#### Configuration (`utils/common.py`)
- **Config class**: Centralized configuration management
- **Directory management**: Automatic creation of output directories
- **IR printing**: Extracts and saves all intermediate representations (TTIR, TTGIR, LLVM, PTX)

#### Pass Difference Analysis (`utils/passdiff.py`)
- **MLIR dump parsing**: Analyzes triton-opt output with `-mlir-print-ir-after-all`
- **Diff generation**: Creates unified diff format files for VSCode compatibility
- **Colorized output**: Terminal-friendly diff display with ANSI colors
- **Patch file generation**: Saves differences in standard patch format

## Usage

### Running the Complete Test

```bash
cd TritonTest
python runTest.py
```

This will:
1. Compile the matrix multiplication kernel
2. Generate all intermediate IR files in `_compiled/`
3. Run the three-stage compilation pipeline
4. Create detailed transformation dumps in `_mlir_dump/`
5. Generate diff files in `_pass_diff/`

### Output Files

After running the test, you'll find:

#### Compiled IRs (`_compiled/`)
- `{kernel_name}.ttir.mlir`: Initial Triton IR
- `{kernel_name}.ttgir.mlir`: TritonGPU IR
- `{kernel_name}.ll`: LLVM IR
- `{kernel_name}.ptx`: Final PTX assembly

#### Intermediate MLIR Files (`_output_mlir/`)
- `{kernel_name}.TritonTransform.mlir`: After Stage 1 transformations
- `{kernel_name}.TritonToTritonGPU.mlir`: After Stage 2 conversion
- `{kernel_name}.TritonGPUToLLVM.mlir`: After Stage 3 conversion

#### Detailed Dumps (`_mlir_dump/`)
- `{kernel_name}.TritonTransform.dump`: Step-by-step Stage 1 transformations
- `{kernel_name}.TritonToTritonGPU.dump`: Step-by-step Stage 2 transformations
- `{kernel_name}.TritonGPUToLLVM.dump`: Step-by-step Stage 3 transformations

#### Diff Files (`_pass_diff/`)
- `{kernel_name}.TritonTransform.patch`: Changes in Stage 1
- `{kernel_name}.TritonToTritonGPU.patch`: Changes in Stage 2
- `{kernel_name}.TritonGPUToLLVM.patch`: Changes in Stage 3

### Analyzing Transformations

To analyze the differences between compilation stages:

```bash
# View diff for Triton Transform stage
python utils/passdiff.py _mlir_dump/{kernel_name}.TritonTransform.dump

# Generate patch file for VSCode
python utils/passdiff.py _mlir_dump/{kernel_name}.TritonTransform.dump --patch _pass_diff/analysis.patch --patch-only
```

## GPU Configuration

The current configuration targets:
- **Compute Capability**: 8.6 (RTX 40 series)
- **PTX Version**: 8.4
- **Thread Configuration**: 4 warps, 32 threads per warp
- **Cooperative Thread Arrays**: 1 CTA

To modify for different GPU architectures, update the parameters in:
- `tritonToTritonGPU.py`: Line 6 (convert-triton-to-tritongpu parameters)
- `tritonGPUToLLVM.py`: Line 7 (convert-triton-gpu-to-llvm parameters)

## Troubleshooting

### Common Issues

1. **triton-opt not found**: Ensure the PATH is correctly set as described in the prerequisites
2. **CUDA errors**: Verify PyTorch CUDA installation and GPU availability
3. **Permission errors**: Check write permissions for output directories

### Debugging

- Enable verbose output by modifying the subprocess calls in pass files
- Check the dump files in `_mlir_dump/` for detailed transformation logs
- Use the diff files to understand what each pass does

## Contributing

To extend this test suite:
1. Add new kernels in `runTest.py`
2. Modify pass lists in the respective `passList/` modules
3. Update the configuration in `utils/common.py` if needed
4. Test with different GPU configurations


## References

- [Triton Documentation](https://triton-lang.org/)
- [MLIR Documentation](https://mlir.llvm.org/)

