"""
tests/test_geometry.py
======================
Unit tests for molbuilder.core.geometry.
"""

import numpy as np
import pytest

from molbuilder.core.geometry import (
    get_geometry_vectors,
    infer_geometry,
    resolve_geometry,
    list_geometries,
)


class TestGeometryVectors:
    """All geometry vectors should be unit vectors with correct count."""

    def _assert_unit_vectors(self, vecs):
        for v in vecs:
            assert abs(np.linalg.norm(v) - 1.0) < 1e-6, f"Not unit vector: {v}"

    def test_octahedral_count(self):
        assert len(get_geometry_vectors("oct")) == 6

    def test_octahedral_unit_vectors(self):
        self._assert_unit_vectors(get_geometry_vectors("oct"))

    def test_octahedral_mutual_angles(self):
        """All oct vectors should be mutually 90° or 180°."""
        vecs = get_geometry_vectors("oct")
        for i in range(6):
            for j in range(i + 1, 6):
                d = abs(np.dot(vecs[i], vecs[j]))
                assert d < 0.01 or abs(d - 1.0) < 0.01, \
                    f"oct: unexpected angle between vec {i} and {j}: dot={d:.4f}"

    def test_tetrahedral_count(self):
        assert len(get_geometry_vectors("tet")) == 4

    def test_tetrahedral_angles(self):
        """All tet vectors should have cos(angle) ≈ -1/3 (109.47°)."""
        vecs = get_geometry_vectors("tet")
        cos_tet = -1.0 / 3.0
        for i in range(4):
            for j in range(i + 1, 4):
                d = np.dot(vecs[i], vecs[j])
                assert abs(d - cos_tet) < 0.01, \
                    f"tet: expected cos={cos_tet:.4f}, got {d:.4f}"

    def test_square_planar_count(self):
        assert len(get_geometry_vectors("sqp")) == 4

    def test_square_planar_coplanar(self):
        """All sqp vectors should lie in the same plane (z=0)."""
        vecs = get_geometry_vectors("sqp")
        for v in vecs:
            assert abs(v[2]) < 1e-6, f"sqp vector not in xy plane: {v}"

    def test_tbp_count(self):
        assert len(get_geometry_vectors("tbp")) == 5

    def test_linear_count(self):
        assert len(get_geometry_vectors("lin")) == 2

    def test_linear_antiparallel(self):
        vecs = get_geometry_vectors("lin")
        assert abs(np.dot(vecs[0], vecs[1]) + 1.0) < 1e-6

    def test_sqpy_count(self):
        assert len(get_geometry_vectors("sqpy")) == 5

    def test_pbp_count(self):
        assert len(get_geometry_vectors("pbp")) == 7

    def test_alias_oh_equals_oct(self):
        assert get_geometry_vectors("oh") == get_geometry_vectors("oct")

    def test_alias_td_equals_tet(self):
        v1 = get_geometry_vectors("td")
        v2 = get_geometry_vectors("tet")
        for a, b in zip(v1, v2):
            assert np.allclose(a, b)

    def test_unknown_geometry_raises(self):
        with pytest.raises(KeyError):
            get_geometry_vectors("xyzzy")


class TestInferGeometry:
    def test_cn2(self):  assert infer_geometry(2) == "lin"
    def test_cn3(self):  assert infer_geometry(3) == "tp"
    def test_cn4(self):  assert infer_geometry(4) == "tet"
    def test_cn5(self):  assert infer_geometry(5) == "sqpy"
    def test_cn6(self):  assert infer_geometry(6) == "oct"
    def test_cn7(self):  assert infer_geometry(7) == "pbp"


class TestResolveGeometry:
    def test_canonical_passthrough(self):  assert resolve_geometry("oct")  == "oct"
    def test_alias_oh(self):               assert resolve_geometry("oh")   == "oct"
    def test_alias_sp(self):               assert resolve_geometry("sp")   == "sqp"
    def test_alias_td(self):               assert resolve_geometry("td")   == "tet"
    def test_none_returns_none(self):      assert resolve_geometry(None)   is None


class TestListGeometries:
    def test_returns_list(self):
        geoms = list_geometries()
        assert isinstance(geoms, list)
        assert len(geoms) > 5

    def test_contains_oct(self):
        keys = [g[0] for g in list_geometries()]
        assert "oct" in keys
