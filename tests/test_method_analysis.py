"""Analytical identities and convergence checks for AB3 with Heun startup."""
import unittest
from unittest.mock import patch

import numpy as np

import method_analysis as analysis


class AnalyticalSolutionTests(unittest.TestCase):
    def test_initial_conditions_and_constant_gravity(self):
        for case in analysis.benchmarks():
            self.assertEqual(case.exact([0, 1]).shape, (2, 2))
        np.testing.assert_array_equal(analysis.constant_gravity_exact(0), [10000, 50])
        np.testing.assert_array_equal(analysis.inverse_square_exact(0), [500000, 0])
        np.testing.assert_array_equal(analysis.linear_drag_exact(0), [10000, 50])
        # A simple independently calculable point: g=10, h0=100, v0=5, t=2.
        np.testing.assert_allclose(
            analysis.constant_gravity_exact(2, height=100, speed=5, gravity=10), [70, 25])

    def test_inverse_square_energy_and_differential_equations(self):
        times = np.array([10., 50., 120., 200.])
        states = analysis.inverse_square_exact(times)
        radius = analysis.r_earth + states[:, 0]
        energy = .5 * states[:, 1]**2 - analysis.MU / radius
        np.testing.assert_allclose(energy, -analysis.MU / (analysis.r_earth + 500000),
                                   rtol=2e-15)
        epsilon = .01
        derivative = (analysis.inverse_square_exact(times + epsilon) -
                      analysis.inverse_square_exact(times - epsilon)) / (2 * epsilon)
        np.testing.assert_allclose(derivative[:, 0], -states[:, 1], rtol=1e-8)
        np.testing.assert_allclose(derivative[:, 1], analysis.MU / radius**2, rtol=1e-8)

    def test_linear_drag_terminal_speed_and_differential_equations(self):
        times = np.array([1., 5., 15.])
        states = analysis.linear_drag_exact(times)
        epsilon = .001
        derivative = (analysis.linear_drag_exact(times + epsilon) -
                      analysis.linear_drag_exact(times - epsilon)) / (2 * epsilon)
        np.testing.assert_allclose(derivative[:, 0], -states[:, 1], rtol=1e-8)
        np.testing.assert_allclose(derivative[:, 1], analysis.GRAVITY - .1 * states[:, 1], rtol=1e-8)
        self.assertAlmostEqual(analysis.linear_drag_exact(1000)[1], analysis.GRAVITY / .1)

    def test_vacuum_radial_model_matches_reentry_equations(self):
        import equations_of_motion as eom

        radial_fall = analysis.benchmarks()[1]
        with patch.multiple(eom, cd=0., cl=0., omega_earth=0.), \
                patch.object(eom, "get_rho", return_value=0.):
            for height, speed in ((500000., 50.), (100000., 1000.), (10000., 3000.)):
                state = np.array([analysis.r_earth + height, speed, -np.pi / 2, 0., 0., 0.])
                derivative = eom.f(state, 0.)
                np.testing.assert_allclose(derivative[:2], radial_fall.rhs([height, speed], 0.))
                np.testing.assert_allclose(derivative[2:], 0, atol=1e-15)


class MethodAccuracyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = analysis.benchmarks()
        cls.results = [analysis.analyze_case(case) for case in cls.cases]

    def test_constant_gravity_is_exact_to_roundoff(self):
        for result in self.results[0]:
            with self.subTest(method=result.method, dt=result.dt):
                self.assertLess(result.max_errors[0], 1e-8)
                self.assertLess(result.max_errors[1], 1e-10)
                self.assertEqual(result.times[-1], self.cases[0].duration)

    def test_heun_supplies_the_starting_steps(self):
        from solver import heun

        for case, results in zip(self.cases, self.results):
            for result in results:
                with self.subTest(case=case.name, dt=result.dt):
                    self.assertEqual(result.method, "AB3")
                    startup, _ = heun(case.rhs, case.exact(0.), 2 * result.dt, result.dt)
                    np.testing.assert_array_equal(result.states[:3], startup)

    def test_third_order_convergence(self):
        for case, results in zip(self.cases[1:], self.results[1:]):
            for component in (0, 1):
                with self.subTest(case=case.name, component=component):
                    errors = [result.max_errors[component] for result in results]
                    self.assertTrue(np.all(np.diff(errors) < 0), errors)
                    order = analysis.observed_order(results[-2], results[-1], component)
                    self.assertAlmostEqual(order, 3, delta=.25)

    def test_no_ground_crossings_and_matching_reference_times(self):
        for case, results in zip(self.cases, self.results):
            for result in results:
                with self.subTest(case=case.name, method=result.method, dt=result.dt):
                    self.assertTrue(np.all(result.states[:, 0] > 0))
                    self.assertEqual(result.times[-1], case.duration)
                    self.assertEqual(result.states.shape, result.errors.shape)
                    np.testing.assert_allclose(
                        result.errors, np.abs(result.states - case.exact(result.times)))


if __name__ == "__main__":
    unittest.main()
