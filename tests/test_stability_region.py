"""Independent AB3 boundary and derivative checks."""
import unittest
import numpy as np
import stability_region as stability


class StabilityTests(unittest.TestCase):
    def test_exact_real_axis_boundary(self):
        self.assertAlmostEqual(stability.dt_limit([-1])[0], 6/11, places=8)
        self.assertTrue(stability.is_stable(0))
        self.assertTrue(stability.is_stable(-6/11))
        self.assertTrue(stability.is_stable(-.5))
        self.assertFalse(stability.is_stable(-.6))
        self.assertAlmostEqual(stability.dt_limit([-2000])[0], (6/11)/2000, places=10)

    def test_neutral_and_growing_modes(self):
        limit, skipped = stability.dt_limit([0, .1, -2])
        self.assertEqual(skipped, 2)
        self.assertAlmostEqual(limit, 3/11, places=8)
        neutral_limit, neutral_count = stability.dt_limit([0, 0])
        self.assertTrue(np.isinf(neutral_limit))
        self.assertEqual(neutral_count, 2)
        self.assertEqual(stability.mode_kind(.1), "growing")
        self.assertFalse(stability.is_stable(.1))
        with self.assertRaises(ValueError):
            stability.dt_limit([np.nan])

    def test_complex_ray_exit(self):
        lam = -.25+.97j
        limit, _ = stability.dt_limit([lam, lam.conjugate()])
        self.assertTrue(stability.is_stable(.99*limit*lam))
        self.assertFalse(stability.is_stable(1.01*limit*lam))

    def test_vectorized_region_matches_scalar_roots(self):
        x, y, stable = stability.stability_region(n=13)
        for k in np.ndindex(x.shape):
            w = x[k] + 1j*y[k]
            roots = np.roots([1, -1 - 23*w/12, 4*w/3, -5*w/12])
            self.assertEqual(stable[k], np.max(np.abs(roots)) < 1)

    def test_checked_jacobian_with_known_linear_system(self):
        exact = np.diag([-.2, -.03, 0, .02, -1, -.1])
        exact[0, 2] = .5
        state = np.array([6.4e6, 300., -.1, .2, .3, -2.])
        calculated, _, discrepancy = stability.checked_jacobian(
            lambda x,t: np.tensordot(exact,x,axes=(1,0)), state, 0)
        np.testing.assert_allclose(calculated, exact, rtol=1e-5, atol=1e-9)
        self.assertLessEqual(discrepancy, 1)


if __name__ == "__main__":
    unittest.main()
