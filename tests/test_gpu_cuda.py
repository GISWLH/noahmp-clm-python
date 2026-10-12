"""阶段 F：CUDA loader 语法/导入测试；本机无 nvcc，kernel 标为未实测。"""

from __future__ import annotations

import shutil

import pytest

cp = pytest.importorskip("cupy")
from noahmp_clm.gpu.cuda import loader


def test_cuda_loader_imports_source() -> None:
    assert "step_columns" in loader.KERNEL_SOURCE
    assert "gridDim.x" in loader.KERNEL_SOURCE


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc 不可用；CUDA kernel [未实测]")
def test_cuda_kernel_compiles_and_runs() -> None:  # pragma: no cover - CUDA-only
    pytest.fail("需要在带 nvcc 的 xuanze 主机执行 N=1024 parity")
