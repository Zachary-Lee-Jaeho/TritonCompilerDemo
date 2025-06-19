import os
import subprocess

from utils import common


def tritonGPUToLLVMPasses():
    passlist = [
        "-tritongpu-combine-tensor-select-and-if",
        "-convert-scf-to-cf",
        "-convert-index-to-llvm",
        "-allocate-shared-memory",
        "-tritongpu-global-scratch-memory-allocation",
        "-convert-triton-gpu-to-llvm=compute-capability=86 ptx-version=84",
        "-convert-nv-gpu-to-llvm",
        "-convert-arith-to-llvm",
        "-canonicalize",
        "-cse",
        "-symbol-dce",
    ]
    return passlist


def run_triton_gpu_to_llvm_pass_test(config: common.Config):
    pass_list = tritonGPUToLLVMPasses()
    compiledTTIRPath = os.path.join(config.OUTPUT_MLIR_DIR, f"{config.compiled_kernel_name}.TritonToTritonGPU.mlir")

    # Run triton-opt
    output_mlir_dir = config.OUTPUT_MLIR_DIR
    output_mlir_dump_dir = config.OUTPUT_MLIR_DUMP_DIR
    pass_diff_dir = config.PASS_DIFF_DIR

    # Output MLIR path
    output_mlir_path = os.path.join(output_mlir_dir, f"{config.compiled_kernel_name}.TritonGPUToLLVM.mlir")
    # MLIR pass step-by-step dump
    dump_file_path = os.path.join(output_mlir_dump_dir, f"{config.compiled_kernel_name}.TritonGPUToLLVM.dump")

    # Pass list
    cmd_pass_args = []
    for ttpass in pass_list:
        cmd_pass_args.append(ttpass)
    # Run triton-opt and capture stdout to file
    with open(dump_file_path, 'w') as dump_file:
        subprocess.run(["triton-opt", compiledTTIRPath, *cmd_pass_args, "-mlir-print-ir-after-all", "-o", output_mlir_path], 
                      stdout=subprocess.PIPE, stderr=dump_file, check=True)
    
    # Run passdiff
    patch_file_path = os.path.join(pass_diff_dir, f"{config.compiled_kernel_name}.TritonGPUToLLVM.patch")
    subprocess.run([
        "python",
        "utils/passdiff.py",
        dump_file_path,
        "--patch",
        patch_file_path,
        "--patch-only"
        ], check=True)

    return 