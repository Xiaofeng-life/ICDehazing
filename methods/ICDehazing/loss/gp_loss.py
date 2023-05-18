import torch


def gradient_penalty(D, x_real, x_fake):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""

    # Compute loss for gradient penalty.
    alpha = torch.rand(x_real.size(0), 1, 1, 1).cuda()
    x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
    out_src = D(x_hat)

    weight = torch.ones(out_src.size()).cuda()
    dydx = torch.autograd.grad(outputs=out_src,
                               inputs=x_hat,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
    return torch.mean((dydx_l2norm - 1) ** 2)