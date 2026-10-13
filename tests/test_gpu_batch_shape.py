"""Regression for a recorded same-input CUDA JAX batch-shape discrepancy."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from noahmp_clm.gpu.jax_core import step_column_jax


def test_recorded_column_step_is_independent_of_batch_shape():
    """The real J4 col2/step17 input must retain all diagnostics at 1e-6."""
    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "gpu":
        if os.environ.get("NOAHMP_REQUIRE_GPU") == "1":
            pytest.fail("Actual CUDA GPU required for the batch-shape regression")
        pytest.skip("CUDA compiler batch-shape regression requires a GPU")
    assert all("cuda" in str(d.client.platform_version).lower() for d in jax.devices())

    fixture = json.loads((Path(__file__).parent / "fixtures" /
                          "gpu_batch_shape_column2.json").read_text())
    values = tuple(np.asarray(fixture[key], dtype=dtype) for key, dtype in
                   (("f", "<f4"), ("i", "<i4"), ("u", "<f4")))
    digest = hashlib.sha256(b"".join(value.tobytes() for value in values)).hexdigest()
    assert digest == fixture["provenance"]["array_sha256"]
    assert tuple(value.shape for value in values) == ((70,), (33,), (8,))

    step = jax.jit(jax.vmap(step_column_jax))
    results = []
    for columns in (1, 64800):
        inputs = tuple(jnp.asarray(np.broadcast_to(value, (columns, value.size)).copy())
                       for value in values)
        results.append(tuple(np.asarray(value) for value in
                             jax.block_until_ready(step(*inputs))))
    single, batch = results
    for label, index in (("diagnostics", 2), ("state", 0)):
        expected = np.broadcast_to(single[index], batch[index].shape)
        assert batch[index].dtype == expected.dtype == np.float32
        difference = np.abs(batch[index] - expected)
        print(json.dumps({"field": label, "max_abs": float(difference.max()),
                          "failed": int(np.count_nonzero(
                              difference > 1e-6 + 1e-6 * np.abs(expected)))}), flush=True)
        np.testing.assert_allclose(batch[index], expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(batch[1], np.broadcast_to(single[1], batch[1].shape))
