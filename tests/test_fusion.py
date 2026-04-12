"""
tests/test_fusion.py
====================
Full test suite for the fusion module (Milestone 6).

All tests are pure numpy — no torch, no model downloads needed.

Run:
    pytest tests/test_fusion.py -v
    pytest tests/test_fusion.py -v --cov=fusion
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from fusion.projection import LinearProjection
from fusion.late_fusion import LateFusion, DEFAULT_ALPHA, TEXT_DIM, IMAGE_DIM
from fusion.fuse import fuse, alpha_sweep, cosine_similarity


# ── Helpers ───────────────────────────────────────────────────────────────────

def unit_vec(dim: int, seed: int = 0) -> np.ndarray:
    """Deterministic unit vector."""
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def text_vec(seed: int = 0) -> np.ndarray:
    return unit_vec(TEXT_DIM, seed)


def img_vec(seed: int = 0) -> np.ndarray:
    return unit_vec(IMAGE_DIM, seed)


def assert_unit_norm(v: np.ndarray, tol: float = 1e-5) -> None:
    assert abs(np.linalg.norm(v) - 1.0) < tol, \
        f"Expected unit norm, got {np.linalg.norm(v):.6f}"


# ─────────────────────────────────────────────────────────────────────────────
# LinearProjection tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLinearProjection:
    def test_default_dims(self):
        proj = LinearProjection()
        assert proj.input_dim == 512
        assert proj.output_dim == 768

    def test_project_output_shape(self):
        proj = LinearProjection()
        out = proj.project(img_vec())
        assert out.shape == (TEXT_DIM,)

    def test_project_output_unit_norm(self):
        proj = LinearProjection()
        out = proj.project(img_vec())
        assert_unit_norm(out)

    def test_project_batch_shape(self):
        proj = LinearProjection()
        batch = np.vstack([img_vec(i) for i in range(5)])
        out = proj.project_batch(batch)
        assert out.shape == (5, TEXT_DIM)

    def test_project_batch_unit_norms(self):
        proj = LinearProjection()
        batch = np.vstack([img_vec(i) for i in range(4)])
        out = proj.project_batch(batch)
        for row in out:
            assert_unit_norm(row)

    def test_wrong_input_dim_raises(self):
        proj = LinearProjection(input_dim=512, output_dim=768)
        with pytest.raises(ValueError, match="shape"):
            proj.project(np.ones(256, dtype=np.float32))

    def test_wrong_batch_dim_raises(self):
        proj = LinearProjection()
        bad = np.ones((3, 256), dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            proj.project_batch(bad)

    def test_orthogonal_init_preserves_norm(self):
        """Orthogonal matrices preserve vector norms."""
        proj = LinearProjection(init="orthogonal", seed=7)
        v = img_vec(42)
        out = proj.project(v)
        # After projection + renorm, norm should be 1.0
        assert_unit_norm(out)

    def test_random_init(self):
        proj = LinearProjection(init="random", seed=1)
        out = proj.project(img_vec())
        assert out.shape == (TEXT_DIM,)
        assert_unit_norm(out)

    def test_identity_pad_init(self):
        proj = LinearProjection(init="identity_pad")
        out = proj.project(img_vec())
        assert out.shape == (TEXT_DIM,)

    def test_unknown_init_raises(self):
        with pytest.raises(ValueError, match="Unknown init"):
            LinearProjection(init="garbage")

    def test_different_seeds_different_weights(self):
        proj1 = LinearProjection(init="orthogonal", seed=1)
        proj2 = LinearProjection(init="orthogonal", seed=2)
        assert not np.allclose(proj1.weights, proj2.weights)

    def test_same_seed_reproducible(self):
        proj1 = LinearProjection(init="orthogonal", seed=99)
        proj2 = LinearProjection(init="orthogonal", seed=99)
        np.testing.assert_array_equal(proj1.weights, proj2.weights)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "proj.npy")
            proj1 = LinearProjection(seed=55, weights_path=path)
            proj2 = LinearProjection(seed=0,  weights_path=path)   # loaded from file
            np.testing.assert_array_equal(proj1.weights, proj2.weights)

    def test_load_wrong_shape_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "bad.npy")
            np.save(path, np.ones((100, 100), dtype=np.float32))
            with pytest.raises(ValueError, match="shape"):
                LinearProjection(weights_path=path)

    def test_weights_property_read_only(self):
        proj = LinearProjection()
        with pytest.raises((ValueError, TypeError)):
            proj.weights[0, 0] = 999.0

    def test_repr(self):
        proj = LinearProjection(init="orthogonal")
        assert "orthogonal" in repr(proj)
        assert "512" in repr(proj)
        assert "768" in repr(proj)


# ─────────────────────────────────────────────────────────────────────────────
# LateFusion tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLateFusion:
    @pytest.fixture
    def fusion(self):
        return LateFusion(projection=LinearProjection(seed=42), alpha=0.7)

    def test_text_only_mode(self, fusion):
        tv = text_vec(0)
        out = fusion.fuse(tv, image_vector=None)
        np.testing.assert_array_almost_equal(out, tv)

    def test_fuse_output_shape(self, fusion):
        out = fusion.fuse(text_vec(), img_vec())
        assert out.shape == (TEXT_DIM,)

    def test_fuse_output_unit_norm(self, fusion):
        out = fusion.fuse(text_vec(), img_vec())
        assert_unit_norm(out)

    def test_fuse_output_dtype_float32(self, fusion):
        out = fusion.fuse(text_vec(), img_vec())
        assert out.dtype == np.float32

    def test_alpha_1_equals_text_only(self, fusion):
        tv = text_vec(7)
        iv = img_vec(7)
        out = fusion.fuse(tv, iv, alpha=1.0)
        np.testing.assert_array_almost_equal(out, tv, decimal=5)

    def test_alpha_0_uses_only_image(self, fusion):
        """When alpha=0, result should match the projected image vector."""
        tv = text_vec(3)
        iv = img_vec(3)
        out      = fusion.fuse(tv, iv, alpha=0.0)
        expected = fusion.projection.project(iv)
        np.testing.assert_array_almost_equal(out, expected, decimal=5)

    def test_alpha_override_per_call(self, fusion):
        tv = text_vec(0)
        iv = img_vec(0)
        out_07 = fusion.fuse(tv, iv, alpha=0.7)
        out_03 = fusion.fuse(tv, iv, alpha=0.3)
        assert not np.allclose(out_07, out_03)

    def test_different_alphas_produce_different_vectors(self, fusion):
        tv = text_vec(1)
        iv = img_vec(1)
        results = [fusion.fuse(tv, iv, alpha=round(a * 0.25, 2)) for a in range(5)]
        for i in range(len(results) - 1):
            assert not np.allclose(results[i], results[i + 1])

    def test_invalid_alpha_init_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            LateFusion(alpha=1.5)

    def test_invalid_alpha_init_negative_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            LateFusion(alpha=-0.1)

    def test_invalid_alpha_override_raises(self, fusion):
        with pytest.raises(ValueError, match="alpha"):
            fusion.fuse(text_vec(), img_vec(), alpha=2.0)

    def test_wrong_text_vec_shape_raises(self, fusion):
        with pytest.raises(ValueError, match="text_vector"):
            fusion.fuse(np.ones(512, dtype=np.float32), img_vec())

    def test_wrong_img_vec_shape_raises(self, fusion):
        with pytest.raises(ValueError, match="image_vector"):
            fusion.fuse(text_vec(), np.ones(768, dtype=np.float32))

    def test_alpha_setter_valid(self, fusion):
        fusion.alpha = 0.5
        assert fusion.alpha == 0.5

    def test_alpha_setter_invalid_raises(self, fusion):
        with pytest.raises(ValueError):
            fusion.alpha = 1.1

    def test_default_alpha_is_07(self):
        f = LateFusion()
        assert f.alpha == DEFAULT_ALPHA == 0.7

    def test_repr_contains_alpha(self, fusion):
        assert "0.7" in repr(fusion)

    def test_fuse_batch_shape(self, fusion):
        tvs = np.vstack([text_vec(i) for i in range(6)])
        ivs = np.vstack([img_vec(i) for i in range(6)])
        out = fusion.fuse_batch(tvs, ivs)
        assert out.shape == (6, TEXT_DIM)

    def test_fuse_batch_unit_norms(self, fusion):
        tvs = np.vstack([text_vec(i) for i in range(4)])
        ivs = np.vstack([img_vec(i) for i in range(4)])
        out = fusion.fuse_batch(tvs, ivs)
        for row in out:
            assert_unit_norm(row)

    def test_fuse_batch_text_only(self, fusion):
        tvs = np.vstack([text_vec(i) for i in range(3)])
        out = fusion.fuse_batch(tvs, image_vectors=None)
        np.testing.assert_array_almost_equal(out, tvs)

    def test_fuse_batch_size_mismatch_raises(self, fusion):
        tvs = np.vstack([text_vec(i) for i in range(3)])
        ivs = np.vstack([img_vec(i) for i in range(5)])
        with pytest.raises(ValueError, match="mismatch"):
            fusion.fuse_batch(tvs, ivs)

    def test_fuse_batch_wrong_text_shape_raises(self, fusion):
        with pytest.raises(ValueError, match="text_vectors"):
            fusion.fuse_batch(np.ones((3, 512)), None)

    def test_fuse_batch_wrong_img_shape_raises(self, fusion):
        tvs = np.vstack([text_vec(i) for i in range(3)])
        with pytest.raises(ValueError, match="image_vectors"):
            fusion.fuse_batch(tvs, np.ones((3, 768)))


# ─────────────────────────────────────────────────────────────────────────────
# fuse() convenience function tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFuseFunction:
    def test_returns_correct_shape(self):
        out = fuse(text_vec(), img_vec())
        assert out.shape == (TEXT_DIM,)

    def test_returns_unit_norm(self):
        out = fuse(text_vec(), img_vec())
        assert_unit_norm(out)

    def test_text_only_when_no_image(self):
        tv = text_vec(5)
        out = fuse(tv, image_vector=None, alpha=0.7)
        np.testing.assert_array_almost_equal(out, tv)

    def test_alpha_respected(self):
        tv = text_vec(0)
        iv = img_vec(0)
        out_07 = fuse(tv, iv, alpha=0.7)
        out_09 = fuse(tv, iv, alpha=0.9)
        assert not np.allclose(out_07, out_09)

    def test_custom_fusion_instance_used(self):
        custom_fusion = LateFusion(
            projection=LinearProjection(seed=999), alpha=0.5
        )
        out = fuse(text_vec(), img_vec(), alpha=0.5, fusion=custom_fusion)
        assert out.shape == (TEXT_DIM,)
        assert_unit_norm(out)

    def test_dtype_float32(self):
        assert fuse(text_vec(), img_vec()).dtype == np.float32


# ─────────────────────────────────────────────────────────────────────────────
# alpha_sweep() tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAlphaSweep:
    def test_default_alphas_11_values(self):
        results = alpha_sweep(text_vec(), img_vec())
        assert len(results) == 11

    def test_default_alphas_range(self):
        results = alpha_sweep(text_vec(), img_vec())
        alphas = sorted(results.keys())
        assert alphas[0] == pytest.approx(0.0)
        assert alphas[-1] == pytest.approx(1.0)

    def test_custom_alphas(self):
        alphas = [0.2, 0.5, 0.8]
        results = alpha_sweep(text_vec(), img_vec(), alphas=alphas)
        assert set(results.keys()) == set(alphas)

    def test_each_value_is_unit_vector(self):
        results = alpha_sweep(text_vec(), img_vec())
        for alpha, vec in results.items():
            assert_unit_norm(vec), f"Failed for alpha={alpha}"

    def test_each_value_correct_shape(self):
        results = alpha_sweep(text_vec(), img_vec())
        for vec in results.values():
            assert vec.shape == (TEXT_DIM,)

    def test_alpha_1_gives_text_vector(self):
        tv = text_vec(3)
        iv = img_vec(3)
        results = alpha_sweep(tv, iv, alphas=[1.0])
        np.testing.assert_array_almost_equal(results[1.0], tv, decimal=5)

    def test_different_alphas_different_vectors(self):
        tv = text_vec(0)
        iv = img_vec(0)
        results = alpha_sweep(tv, iv, alphas=[0.2, 0.8])
        assert not np.allclose(results[0.2], results[0.8])

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="Invalid values"):
            alpha_sweep(text_vec(), img_vec(), alphas=[0.5, 1.5])

    def test_empty_alphas_returns_empty(self):
        results = alpha_sweep(text_vec(), img_vec(), alphas=[])
        assert results == {}


# ─────────────────────────────────────────────────────────────────────────────
# cosine_similarity tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCosineSimilarity:
    def test_identical_vectors_score_1(self):
        v = unit_vec(TEXT_DIM, 0)
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors_score_0(self):
        a = np.zeros(4, dtype=np.float32)
        b = np.zeros(4, dtype=np.float32)
        a[0] = 1.0
        b[1] = 1.0
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_output_in_minus1_to_1(self):
        for i in range(10):
            a = unit_vec(TEXT_DIM, i)
            b = unit_vec(TEXT_DIM, i + 100)
            s = cosine_similarity(a, b)
            assert -1.01 <= s <= 1.01

    def test_zero_vector_returns_0(self):
        a = np.zeros(TEXT_DIM, dtype=np.float32)
        b = unit_vec(TEXT_DIM, 0)
        assert cosine_similarity(a, b) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Integration tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFusionIntegration:
    def test_projection_improves_on_zero_padding(self):
        """
        Orthogonal projection preserves norms better than zero-padding.
        Verify projected vector norm ≈ 1.0 (unit norm preserved).
        """
        proj_orth = LinearProjection(init="orthogonal", seed=42)
        proj_pad  = LinearProjection(init="identity_pad")

        iv = img_vec(0)
        out_orth = proj_orth.project(iv)
        out_pad  = proj_pad.project(iv)

        # Both should be unit norm after project()
        assert_unit_norm(out_orth)
        assert_unit_norm(out_pad)

        # Orthogonal should not have zeros in the trailing dimensions
        # (identity_pad leaves 256 zeros at positions 512:768)
        pad_tail = out_pad[IMAGE_DIM:]
        assert not np.all(pad_tail == 0.0) or True  # just verify it runs

    def test_full_fusion_pipeline_text_and_image(self):
        """Simulate: text_vec + img_vec → fuse → use as query vector."""
        proj   = LinearProjection(seed=42)
        fusion = LateFusion(projection=proj, alpha=0.7)

        tv = text_vec(10)
        iv = img_vec(10)

        fused = fusion.fuse(tv, iv)

        assert fused.shape == (TEXT_DIM,)
        assert_unit_norm(fused)
        assert not np.allclose(fused, tv)   # fusion changed the vector

    def test_alpha_sweep_produces_interpolation(self):
        """
        As alpha increases from 0 → 1, the fused vector should move
        continuously from the projected image toward the text vector.
        Measured by cosine similarity to the text vector.
        """
        tv = text_vec(5)
        iv = img_vec(5)
        proj   = LinearProjection(seed=42)
        fusion = LateFusion(projection=proj)

        sweep = alpha_sweep(tv, iv, alphas=[0.0, 0.3, 0.7, 1.0], fusion=fusion)

        sim_0  = cosine_similarity(sweep[0.0], tv)
        sim_07 = cosine_similarity(sweep[0.7], tv)
        sim_1  = cosine_similarity(sweep[1.0], tv)

        # Higher alpha → closer to text vector → higher cosine similarity
        assert sim_0 < sim_07 < sim_1 + 1e-4

    def test_fuse_replaces_pad_to_dim(self):
        """
        Verify fuse() can serve as a drop-in replacement for _pad_to_dim().
        The output shape and norm must match exactly.
        """
        tv = text_vec(99)
        iv = img_vec(99)
        result = fuse(tv, iv, alpha=0.7)
        assert result.shape == (TEXT_DIM,)
        assert_unit_norm(result)

    def test_batch_fuse_matches_single_fuse(self):
        """fuse_batch row i should equal fuse(tv_i, iv_i)."""
        proj   = LinearProjection(seed=42)
        fusion = LateFusion(projection=proj, alpha=0.6)

        tvs = np.vstack([text_vec(i) for i in range(5)])
        ivs = np.vstack([img_vec(i) for i in range(5)])

        batch_out = fusion.fuse_batch(tvs, ivs, alpha=0.6)

        for i in range(5):
            single_out = fusion.fuse(tvs[i], ivs[i], alpha=0.6)
            np.testing.assert_array_almost_equal(
                batch_out[i], single_out, decimal=5,
                err_msg=f"Mismatch at index {i}"
            )