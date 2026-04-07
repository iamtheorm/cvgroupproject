import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.physics_model import PhysicsModeling

class TestPhysicsModel(unittest.TestCase):
    def setUp(self):
        self.physics = PhysicsModeling(youngs_modulus=250e9, poisson_ratio=0.3)

    def test_compute_strain_tensors(self):
        # Create a mock displacement field of 10x10 with constant gradient
        # u(x,y) = 2*x + 3*y, v(x,y) = x + 4*y
        # Note: numpy meshgrid indexing is 'xy' by default
        x, y = np.meshgrid(np.arange(10), np.arange(10))
        u = 2*x + 3*y
        v = x + 4*y
        
        displacement = np.stack([u, v], axis=-1).astype(float)
        
        eps_xx, eps_yy, eps_xy = self.physics.compute_strain_tensors(displacement)
        
        # Expected:
        # du/dx = 2
        # dv/dy = 4
        # du/dy = 3, dv/dx = 1 -> eps_xy = 0.5 * (3 + 1) = 2
        
        # We test the interior because np.gradient uses forward/backward differences at edges
        np.testing.assert_allclose(eps_xx[2:-2, 2:-2], 2.0)
        np.testing.assert_allclose(eps_yy[2:-2, 2:-2], 4.0)
        np.testing.assert_allclose(eps_xy[2:-2, 2:-2], 2.0)

    def test_von_mises(self):
        # Known simple case
        sigma_xx = np.array([[100.0]])
        sigma_yy = np.array([[0.0]])
        tau_xy = np.array([[0.0]])
        
        # VM stress for uniaxial tension is equal to the tension
        vm = self.physics.compute_von_mises_stress(sigma_xx, sigma_yy, tau_xy)
        self.assertAlmostEqual(vm[0,0], 100.0)

if __name__ == '__main__':
    unittest.main()
