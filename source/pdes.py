import numpy as np
# import tensorflow as tf
import torch


def Burgers1D(x, t, u, nu):
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    e = (u_t + u * u_x - nu / np.pi * u_xx)
    return e


def diffusion_reaction_1d(x, t, u, nu, rho):
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    e = u_t - nu * u_xx - rho * (u - u * u)
    return e
    

def Diffusion_sorption(x, t, u, original_tf_compat=False):
    D = 5e-4
    por = 0.29
    rho_s = 2880
    k_f = 3.5e-4
    n_f = 0.874

    u_safe = torch.clamp(u, min=1e-6)
    retardation_factor = 1 + ((1 - por) / por) * rho_s * k_f * n_f * u_safe ** (n_f - 1)
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    if original_tf_compat:
        # Replicates a bug in the author's TF code (d(u_x)/dt instead of d(u_x)/dx).
        u_xx = torch.autograd.grad(u_x, t, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    else:
        # Correct second spatial derivative as in paper Eq. (12): u_t = D/R(u) * u_xx.
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    e = u_t - D / retardation_factor * u_xx
    return e



def Boundary(x, u):
    D = 5e-4
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    return D * u_x


def SWE_2D(u, v, h, x, y, t, g):
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    h_t = torch.autograd.grad(h, t, grad_outputs=torch.ones_like(h), create_graph=True)[0]

    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    h_x = torch.autograd.grad(h, x, grad_outputs=torch.ones_like(h), create_graph=True)[0]

    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    h_y = torch.autograd.grad(h, y, grad_outputs=torch.ones_like(h), create_graph=True)[0]

    e1 = h_t + h_x * u + h * u_x + h_y * v + h * v_y
    e2 = u_t + u * u_x + v * u_y + g * h_x
    e3 = v_t + u * v_x + v * v_y + g * h_y
    return e1, e2, e3


def Boundary_condition(x, y, u, v):
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]

    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]

    return u_x, v_x, u_y, v_y

def CFD_2D(x, y, t, d, u, v, p, gamma, keci, yifu):
    E = p / (gamma - 1.0) + 0.5 * d * (u**2 + v**2)
    Fu = u * (E + p)
    Fv = v * (E + p)

    du = d * u
    dv = d * v

    d_t = torch.autograd.grad(d, t, grad_outputs=torch.ones_like(d), create_graph=True)[0]
    du_x = torch.autograd.grad(du, x, grad_outputs=torch.ones_like(du), create_graph=True)[0]
    dv_y = torch.autograd.grad(dv, y, grad_outputs=torch.ones_like(dv), create_graph=True)[0]

    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]

    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

    E_t = torch.autograd.grad(E, t, grad_outputs=torch.ones_like(E), create_graph=True)[0]

    Fu_x = torch.autograd.grad(Fu, x, grad_outputs=torch.ones_like(Fu), create_graph=True)[0]
    Fv_y = torch.autograd.grad(Fv, y, grad_outputs=torch.ones_like(Fv), create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]

    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]

    v_yx = torch.autograd.grad(v_y, x, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
    u_xy = torch.autograd.grad(u_x, y, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    e1 = d_t + du_x + dv_y
    e2 = d * (u_t + u * u_x + v * u_y) + p_x - keci * (u_xx + u_yy) - (keci + yifu / 3.0) * (u_xx + v_yx)
    e3 = d * (v_t + u * v_x + v * v_y) + p_y - keci * (v_xx + v_yy) - (keci + yifu / 3.0) * (u_xy + v_yy)
    e4 = E_t + Fu_x + Fv_y

    return e1, e2, e3, e4
