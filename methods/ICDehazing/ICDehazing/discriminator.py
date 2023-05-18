import torch.nn as nn


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class Discriminator(nn.Module):
    def __init__(self, in_ch=3, ndf=64, n_layers=4):
        super(Discriminator, self).__init__()

        model = [nn.Conv2d(in_ch, ndf, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        for i in range(1, n_layers):
            model += [nn.Conv2d(ndf, ndf*2, 4, stride=2, padding=1),
                      nn.BatchNorm2d(ndf * 2),
                      nn.LeakyReLU(0.2, inplace=True)]
            ndf = ndf * 2

        # model += [nn.Conv2d(ndf, ndf * 8, 4, stride=1, padding=1),
        #           nn.BatchNorm2d(ndf * 8),
        #           nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(ndf, 1, 3, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == "__main__":
    import torch
    x = torch.rand(size=(1, 3, 256, 256))
    # dis = Discriminator(3)
    # pred = dis(x)
    # print(dis)