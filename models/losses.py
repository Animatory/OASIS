import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vggloss import VGG19


class LossesComputer:
    def __init__(self, opt):
        self.opt = opt
        if not opt.no_labelmix:
            self.labelmix_function = torch.nn.MSELoss()

    def loss(self, input, label, for_real):
        if for_real:
            # --- balancing classes ---
            weights = get_class_balancing(self.opt, label)
            target = torch.argmax(label, dim=1) + 1

            loss = F.cross_entropy(input, target, weight=weights,
                                   ignore_index=1 if self.opt.contain_dontcare_label else -100)
        else:
            target = torch.zeros_like(label[:, 0], requires_grad=False, dtype=torch.long)
            loss = F.cross_entropy(input, target)
        return loss

    def loss_segm(self, input, label, for_real):
        if for_real:
            # --- balancing classes ---
            loss = F.softplus(input[:, 0]).mean()
            pred_labels = input[:, 1:]
            weights = get_class_balancing(self.opt, label)
            target = torch.argmax(label, dim=1)

            loss += F.cross_entropy(pred_labels, target, weight=weights[1:],
                                    ignore_index=0 if self.opt.contain_dontcare_label else -100)
        else:
            loss = F.softplus(-input[:, 0]).mean()
        return loss

    def loss_labelmix(self, mask, output_D_mixed, output_D_fake, output_D_real):
        mixed_D_output = mask * output_D_real + (1 - mask) * output_D_fake
        return self.labelmix_function(mixed_D_output, output_D_mixed)

    def loss_ae(self, input, target):
        return F.mse_loss(input, target)

    def loss_features(self, input: torch.Tensor):
        mean, log_var = torch.chunk(input, 2, dim=1)
        kld_loss = -0.5 * torch.mean(torch.sum(1 + log_var - mean ** 2 - log_var.exp(), dim=1))
        return kld_loss * 0.1

    def loss_unsup(self, input):
        input = torch.softmax(input, 1)[:, 0]
        target = torch.zeros_like(input, requires_grad=False)
        loss = F.binary_cross_entropy(input, target)
        return loss


def get_class_balancing(opt, label):
    nc = label.shape[1]
    coefficients = torch.ones(nc + 1, dtype=label.dtype)
    if not opt.no_balancing_inloss:
        class_occurrence = torch.cat([torch.zeros(1, device=label.device, dtype=label.dtype),
                                      label.sum(dim=(0, 2, 3))])
        num_of_classes = torch.nonzero(class_occurrence, as_tuple=False).numel()
        coefficients = torch.reciprocal(class_occurrence) * (label.numel() / num_of_classes / nc)
        if opt.contain_dontcare_label:
            coefficients[0] = 0
            coefficients[1] = 0
    return coefficients


class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
