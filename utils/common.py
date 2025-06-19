import os

class Config:
    def __init__(self):
        self.ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

        self.COMPILED_DIR = os.path.join(self.ROOT_DIR, "_compiled")
        os.makedirs(self.COMPILED_DIR, exist_ok=True)

        self.OUTPUT_MLIR_DIR = os.path.join(self.ROOT_DIR, "_output_mlir")
        os.makedirs(self.OUTPUT_MLIR_DIR, exist_ok=True)

        self.OUTPUT_MLIR_DUMP_DIR = os.path.join(self.ROOT_DIR, "_mlir_dump")
        os.makedirs(self.OUTPUT_MLIR_DUMP_DIR, exist_ok=True)

        self.PASS_DIFF_DIR = os.path.join(self.ROOT_DIR, "_pass_diff")
        os.makedirs(self.PASS_DIFF_DIR, exist_ok=True)

        self.compiled_kernel_name = None
    
config = Config()


def print_ir(compiled_kernel):
    config.compiled_kernel_name = compiled_kernel.name

    with open(f"{config.COMPILED_DIR}/{compiled_kernel.name}.ttir.mlir", "w", encoding="utf-8") as f:
        f.write(compiled_kernel.asm.get("ttir", "No TTIR available"))

    with open(f"{config.COMPILED_DIR}/{compiled_kernel.name}.ttgir.mlir", "w", encoding="utf-8") as f:
        f.write(compiled_kernel.asm.get("ttgir", "No TTGIR available"))

    with open(f"{config.COMPILED_DIR}/{compiled_kernel.name}.ll", "w", encoding="utf-8") as f:
        f.write(compiled_kernel.asm.get("llir", "No LLIR available"))

    with open(f"{config.COMPILED_DIR}/{compiled_kernel.name}.ptx", "w", encoding="utf-8") as f:
        f.write(compiled_kernel.asm.get("ptx", "No PTX available"))