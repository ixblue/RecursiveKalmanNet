import torch


class DynamicalSystems:
    
    @staticmethod
    def constant_spd_linear_model(dt=1.0, propag_std=1e-2, obs_std_values=None):
        """
        Define the dynamical system parameters with varying observation noise
        """
        def F_t(t): return torch.tensor([[1, dt], [0, 1]], dtype=torch.float32)
        def Q_t(i, t): return torch.tensor([[1e-16, 0], [0, propag_std**2]], dtype=torch.float32)
        def H_t(t): return torch.tensor([[1, 0]], dtype=torch.float32)
        def R_t(i, t): return torch.tensor([[obs_std_values[i, t]**2]], dtype=torch.float32)

        return [F_t, Q_t, H_t, R_t]
