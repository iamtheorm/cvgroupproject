import numpy as np

class PhysicsModeling:
    def __init__(self, youngs_modulus=250e9, poisson_ratio=0.3):
        '''
        youngs_modulus: Pa (Default is an approximation for structural steel, 200-250 GPa)
        poisson_ratio: unitless (Default typically ~0.3 for steel)
        '''
        self.E = youngs_modulus
        self.nu = poisson_ratio

    def compute_strain_tensors(self, displacement_field):
        '''
        Calculates the 2D Green-Lagrange strain tensor or infinitesimal strain tensor.
        Here we use Cauchy (infinitesimal) strain for micro-deformations.
        
        displacement_field: (H, W, 2) array where [..., 0] is u (x-displacement) and [..., 1] is v (y-displacement)
        Returns:
            eps_xx, eps_yy, eps_xy (each of shape (H, W))
        '''
        u = displacement_field[..., 0]
        v = displacement_field[..., 1]
        
        # Compute gradients (du/dx, du/dy, dv/dx, dv/dy)
        # np.gradient returns (gradient_along_axis_0, gradient_along_axis_1) i.e. (dy, dx)
        du_dy, du_dx = np.gradient(u)
        dv_dy, dv_dx = np.gradient(v)
        
        eps_xx = du_dx
        eps_yy = dv_dy
        eps_xy = 0.5 * (du_dy + dv_dx) # Engineering shear strain is 2*eps_xy, we use tensor shear here
        
        return eps_xx, eps_yy, eps_xy

    def compute_stress_maps(self, eps_xx, eps_yy, eps_xy):
        '''
        Converts 2D strain components to stress components using 2D Hooke's law (plane stress).
        Returns:
            sigma_xx, sigma_yy, tau_xy
        '''
        factor = self.E / (1 - self.nu**2)
        
        sigma_xx = factor * (eps_xx + self.nu * eps_yy)
        sigma_yy = factor * (eps_yy + self.nu * eps_xx)
        # Shear modulus G = E / (2*(1+nu))
        # tau_xy = G * gamma_xy where gamma_xy = 2*eps_xy
        G = self.E / (2 * (1 + self.nu))
        tau_xy = G * (2 * eps_xy)
        
        return sigma_xx, sigma_yy, tau_xy

    def compute_von_mises_stress(self, sigma_xx, sigma_yy, tau_xy):
        '''
        Computes the Von Mises yield stress equivalent.
        '''
        vm_stress = np.sqrt(sigma_xx**2 - sigma_xx*sigma_yy + sigma_yy**2 + 3 * tau_xy**2)
        return vm_stress

    def process_displacement_to_stress(self, displacement_field):
        '''
        Convenience function wrapping the entire pipeline.
        Returns the Von Mises stress map.
        '''
        eps_xx, eps_yy, eps_xy = self.compute_strain_tensors(displacement_field)
        sigma_xx, sigma_yy, tau_xy = self.compute_stress_maps(eps_xx, eps_yy, eps_xy)
        vm_stress = self.compute_von_mises_stress(sigma_xx, sigma_yy, tau_xy)
        return vm_stress
