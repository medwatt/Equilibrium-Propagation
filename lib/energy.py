# imports <<<
import abc
import torch
from lib import config
# >>>

class EnergyBasedModel(abc.ABC, torch.nn.Module):
    """
    Base class for energy-based models.
    """
    def __init__(self, dimensions, c_energy, batch_size, phi):
        super(EnergyBasedModel, self).__init__()

        self.batch_size = batch_size
        self.c_energy = c_energy
        self.clamp_du = torch.zeros(len(dimensions), dtype=torch.bool)
        self.dimensions = dimensions
        self.E = None
        self.n_layers = len(dimensions)
        self.phi = phi
        self.u = None

        self.W = torch.nn.ModuleList(
            torch.nn.Linear(dim1, dim2)
            for dim1, dim2 in zip(self.dimensions[:-1], self.dimensions[1:])
        ).to(config.device)

        # Input (u_0) is clamped by default
        self.clamp_du[0] = True

        # Optional input reconstruction term
        self.x0_target = None
        self.x0_mask = None
        self.x0_weight = 0.0

        self.reset_state()

    @abc.abstractmethod
    def update_energy(self):
        return

    # input reconstruction <<<
    def clear_input_target(self):
        self.x0_target = None
        self.x0_mask = None
        self.x0_weight = 0.0
        self.update_energy()

    def set_input_target(self, x0_target, x0_mask=None, weight=1.0):
        self.x0_target = x0_target.detach().clone()
        self.x0_mask = x0_mask
        self.x0_weight = float(weight)
        self.update_energy()
    # >>>

    # state clamping <<<
    def clamp_layer(self, i, u_i):
        self.u[i] = u_i
        self.clamp_du[i] = True
        self.update_energy()

    def release_layer(self, i):
        self.u[i].requires_grad = True
        self.clamp_du[i] = False
        self.update_energy()

    def reset_state(self):
        self.u = []
        for i in range(self.n_layers):
            self.u.append(torch.randn((self.batch_size, self.dimensions[i]),
                                      requires_grad=not(self.clamp_du[i]),
                                      device=config.device))
        self.update_energy()
    # >>>

    # model energies <<<
    def set_C_target(self, target):
        self.c_energy.set_target(target)
        self.update_energy()

    def input_recon_energy(self):
        if self.x0_target is None or self.x0_weight == 0.0:
            return 0
        diff = self.u[0] - self.x0_target
        return 0.5 * self.x0_weight * torch.sum(diff * diff, dim=1)
    # >>>

    # state dynamics <<<
    def u_relax(self, dt, n_relax, tol, tau):
        E_init = self.E.clone().detach()
        E_prev = self.E.clone().detach()

        for _ in range(n_relax):
            du_norm = self.u_step(dt, tau)
            if du_norm < tol:
                break
            E_prev = self.E.clone().detach()

        return torch.sum(E_prev - E_init)

    def u_step(self, dt, tau, u0_free_mask=None):
        """
        Single relaxation step.

        If u0_free_mask is provided and input is released:
          - only pixels where mask==True update
          - fixed pixels never change
        """
        self.zero_grad()
        torch.sum(self.E).backward()

        with torch.no_grad():
            du_norm = 0.0
            for i in range(self.n_layers):
                if self.clamp_du[i]:
                    continue

                du = self.u[i].grad

                if i == 0 and u0_free_mask is not None:
                    m = u0_free_mask
                    if m.dtype != du.dtype:
                        m = m.to(du.dtype)
                    du = du * m

                self.u[i] -= (dt / tau) * du
                du_norm += float(torch.mean(torch.norm(du, dim=1)))

        self.update_energy()
        return du_norm
    # >>>

    # weight update <<<
    def w_get_gradients(self, loss=None):
        self.zero_grad()
        if loss is None:
            loss = torch.mean(self.E)
        return torch.autograd.grad(loss, self.parameters())

    def w_optimize(self, free_grad, nudged_grad, w_optimizer):
        self.zero_grad()
        w_optimizer.zero_grad()

        for p, f_g, n_g in zip(self.parameters(), free_grad, nudged_grad):
            p.grad = (1 / self.c_energy.beta) * (n_g - f_g)

        w_optimizer.step()
        self.update_energy()

    def zero_grad(self):
        self.W.zero_grad()
        for u_i in self.u:
            if u_i.grad is not None:
                u_i.grad.detach_()
                u_i.grad.zero_()

    # >>>


class HopfieldEBM(EnergyBasedModel):
    """
    The Hopfield energy model.
    """
    def __init__(self, dimensions, c_energy, batch_size, phi):
        super(HopfieldEBM, self).__init__(dimensions, c_energy, batch_size, phi)


    def update_energy(self):
        self.E = 0

        for i, layer in enumerate(self.W):
            r_pre = self.phi[i](self.u[i])
            r_post = self.phi[i + 1](self.u[i + 1])

            if i == 0:
                self.E += 0.5 * torch.einsum('bi,bi->b', self.u[i], self.u[i])

            self.E += 0.5 * torch.einsum('bi,bi->b', self.u[i + 1], self.u[i + 1])
            self.E -= 0.5 * torch.einsum('bi,ji,bj->b', r_pre, layer.weight, r_post)
            self.E -= 0.5 * torch.einsum('bi,ij,bj->b', r_post, layer.weight, r_pre)

            self.E -= torch.einsum('i,bi->b', layer.bias, r_post)

        if self.c_energy.target is not None:
            self.E += self.c_energy.compute_energy(self.u[-1])

        self.E += self.input_recon_energy()

