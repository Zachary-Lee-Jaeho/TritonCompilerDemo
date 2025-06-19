import os
import subprocess

from utils import common


def tritonToTritonGPUPasses():
    passlist = [
        "--convert-triton-to-tritongpu=num-ctas=1 num-warps=4 threads-per-warp=32 target=cuda:86",

        "-tritongpu-coalesce",
        "-tritongpu-F32DotTC",
        "-triton-nvidia-gpu-plan-cta",
        "-tritongpu-remove-layout-conversions",
        "-tritongpu-optimize-thread-locality",
        "-tritongpu-accelerate-matmul",
        "-tritongpu-remove-layout-conversions",
        "-tritongpu-optimize-dot-operands",
        "-cse",
        "-tritongpu-optimize-accumulator-init",
        "-tritongpu-combine-tensor-select-and-if",
        "-tritongpu-pipeline",
        "-tritongpu-prefetch",
        "-tritongpu-optimize-dot-operands",
        "-tritongpu-coalesce-async-copy",
        "-tritongpu-remove-layout-conversions",
        "-tritongpu-reduce-data-duplication",
        "-tritongpu-reorder-instructions",
        "-cse",
        "-symbol-dce",
        "-triton-nvidia-gpu-fence-insertion",
        "-triton-nvidia-tma-lowering",
        "-canonicalize",
    ]

    return passlist


def run_triton_to_triton_gpu_pass_test(config: common.Config):
    pass_list = tritonToTritonGPUPasses()
    compiledTTIRPath = os.path.join(config.OUTPUT_MLIR_DIR, f"{config.compiled_kernel_name}.TritonTransform.mlir")

    # Run triton-opt
    output_mlir_dir = config.OUTPUT_MLIR_DIR
    output_mlir_dump_dir = config.OUTPUT_MLIR_DUMP_DIR
    pass_diff_dir = config.PASS_DIFF_DIR

    # Output MLIR path
    output_mlir_path = os.path.join(output_mlir_dir, f"{config.compiled_kernel_name}.TritonToTritonGPU.mlir")
    # MLIR pass step-by-step dump
    dump_file_path = os.path.join(output_mlir_dump_dir, f"{config.compiled_kernel_name}.TritonToTritonGPU.dump")

    # Pass list
    cmd_pass_args = []
    for ttpass in pass_list:
        cmd_pass_args.append(ttpass)
    # Run triton-opt and capture stdout to file
    with open(dump_file_path, 'w') as dump_file:
        subprocess.run(["triton-opt", compiledTTIRPath, *cmd_pass_args, "-mlir-print-ir-after-all", "-o", output_mlir_path], 
                      stdout=subprocess.PIPE, stderr=dump_file, check=True)
    
    # Run passdiff
    patch_file_path = os.path.join(pass_diff_dir, f"{config.compiled_kernel_name}.TritonToTritonGPU.patch")
    subprocess.run([
        "python",
        "utils/passdiff.py",
        dump_file_path,
        "--patch",
        patch_file_path,
        "--patch-only"
        ], check=True)

    return 