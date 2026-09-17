"""H100 roofline with LLM kernels placed on it."""

from pathlib import Path

from mlbook.systems.roofline import (H100_SXM, decode_step_kernel, flash_attention_kernel, layernorm_kernel,
                                     matmul_kernel, naive_attention_kernel, plot_roofline)

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures" / "part14_roofline.png"


def main() -> None:
    kernels = [
        matmul_kernel(8192, 4096, 4096, name="prefill matmul (8192 tokens)"),
        matmul_kernel(64, 4096, 4096, name="decode matmul, B=64"),
        decode_step_kernel(7e9, 1),
        decode_step_kernel(7e9, 256),
        layernorm_kernel(8192, 4096),
        naive_attention_kernel(4096, 128),
        flash_attention_kernel(4096, 128),
    ]
    plot_roofline(H100_SXM, kernels, str(OUT))


if __name__ == "__main__":
    main()
