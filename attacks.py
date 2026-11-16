"""White-box, untargeted L-infinity attacks. Inputs x are in [0, 1]; every adversarial
image stays within eps of x (per pixel) and inside [0, 1].

  FGSM     one step:      x' = x + eps * sign(grad)
  I-FGSM   T small steps of size alpha = eps/T, projected back into the eps-ball
  PGD      I-FGSM + random start inside the eps-ball, larger step (2.5*eps/T)
  MI-FGSM  I-FGSM with a momentum term on L1-normalized gradients (Dong et al., 2018)
"""
import torch
import torch.nn.functional as F


def _grad(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    (g,) = torch.autograd.grad(loss, x)
    return g


def _project(x_adv, x, eps):
    return torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0.0, 1.0).detach()


def fgsm(model, x, y, eps, **_):
    return (x + eps * _grad(model, x, y).sign()).clamp(0.0, 1.0).detach()


def ifgsm(model, x, y, eps, steps=10, alpha=None, **_):
    alpha = alpha if alpha is not None else eps / steps
    x_adv = x.clone().detach()
    for _ in range(steps):
        x_adv = _project(x_adv + alpha * _grad(model, x_adv, y).sign(), x, eps)
    return x_adv


def pgd(model, x, y, eps, steps=10, alpha=None, random_start=True, **_):
    alpha = alpha if alpha is not None else 2.5 * eps / steps
    x_adv = x.clone().detach()
    if random_start:
        x_adv = (x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)).clamp(0.0, 1.0)
    for _ in range(steps):
        x_adv = _project(x_adv + alpha * _grad(model, x_adv, y).sign(), x, eps)
    return x_adv


def mifgsm(model, x, y, eps, steps=10, alpha=None, mu=1.0, **_):
    alpha = alpha if alpha is not None else eps / steps
    x_adv = x.clone().detach()
    momentum = torch.zeros_like(x)
    for _ in range(steps):
        g = _grad(model, x_adv, y)
        g = g / (g.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-12)  # L1 normalization
        momentum = mu * momentum + g
        x_adv = _project(x_adv + alpha * momentum.sign(), x, eps)
    return x_adv


ATTACKS = {"fgsm": fgsm, "ifgsm": ifgsm, "pgd": pgd, "mifgsm": mifgsm}
ATTACK_NAMES = {"fgsm": "FGSM", "ifgsm": "I-FGSM", "pgd": "PGD", "mifgsm": "MI-FGSM"}
