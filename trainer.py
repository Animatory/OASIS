import random

import torch
import torch.nn.functional as F
from apex import amp


class Trainer:
    def __init__(self, opt, model, optimizer_d, optimizer_g, loss_computer, visualizer_losses):
        self.opt = opt
        self.model = model
        self.optimizer_d = optimizer_d
        self.optimizer_g = optimizer_g
        self.loss_computer = loss_computer
        self.visualizer_losses = visualizer_losses

    # def forward_unsup(self, data):
    #     if 'image_unsup' in data:
    #         unsup_image = data['image_unsup']
    #         unsup_features, pred_unsup_labels, recs, part = self.model.netD(unsup_image, True)
    #         loss_d_unsup = F.softplus(pred_unsup_labels[:, 0]).mean()
    #         loss_d_unsup += self.loss_computer.loss_recons(recs, unsup_image, part)
    #         return loss_d_unsup
    #     else:
    #         return 0

    def forward_unsup(self, data):
        if 'image_unsup' in data:
            unsup_image = data['image_unsup']
            unsup_features, unsup_labels = self.model.netD(unsup_image)
            loss_d_unsup = F.softplus(1 + unsup_labels[:, 0]).mean()
            loss_d_unsup += F.softplus(1 + self.model.to_logit(unsup_features)).mean()
            return loss_d_unsup
        else:
            return 0

    def forward_labelmix(self, labels, fake_big_image, real_image, pred_fake_labels, pred_real_labels):
        mixed_inp, mask = self.model.generate_labelmix(labels, fake_big_image, real_image)
        _, pred_mixes_labels = self.model.forward_discriminator(mixed_inp)
        loss_d_lm = self.loss_computer.loss_labelmix(mask, pred_mixes_labels, pred_fake_labels, pred_real_labels)
        loss_d_lm *= self.opt.lambda_labelmix
        return loss_d_lm

    def mode1(self, data):
        real_image = data['image']
        labels = data['label']
        loss_computer = self.loss_computer

        # --- generator update ---#
        self.model.netG.zero_grad()
        fake_image = self.model.netG(labels)
        fake_pred_features, fake_pred_labels = self.model.netD(fake_image)
        loss_g_adv = loss_computer.loss(fake_pred_labels, labels, for_real=True)
        loss_g = loss_g_adv

        with amp.scale_loss(loss_g, self.optimizer_g, loss_id=1) as loss_g_scaled:
            loss_g_scaled.backward()
            self.optimizer_g.step()

        # --- discriminator update ---#
        self.model.netD.zero_grad()
        fake_image = fake_image.detach()

        real_features, pred_real_labels = self.model.netD(real_image)
        loss_d_real = loss_computer.loss(pred_real_labels, labels, True)
        loss_d_real += loss_computer.loss_features(real_features)

        fake_features, pred_fake_labels = self.model.netD(fake_image)
        loss_d_fake = loss_computer.loss(pred_fake_labels, labels, False)
        loss_d_lm = self.forward_labelmix(labels, fake_image, real_image, pred_fake_labels, pred_real_labels)
        loss_d = loss_d_real + loss_d_fake + loss_d_lm + self.forward_unsup(data)

        with amp.scale_loss(loss_d, self.optimizer_d, loss_id=1) as loss_d_scaled:
            loss_d_scaled.backward()
            self.optimizer_d.step()
        torch.cuda.empty_cache()
        return loss_d, loss_g

    def mode2(self, data):
        real_image = data['image']
        labels = data['label']
        loss_computer = self.loss_computer

        # --- discriminator update ---#
        self.model.netD.zero_grad()

        real_features, pred_real_labels = self.model.netD(real_image)
        loss_d_real = loss_computer.loss(pred_real_labels, labels, True)
        loss_d_real += loss_computer.loss_features(real_features)

        with_features = random.random() > 0.5
        fake_image = self.model.netG(labels=labels, features=real_features.detach() if with_features else None)
        fake_image_dt = fake_image.detach()

        fake_features, pred_fake_labels = self.model.netD(fake_image_dt)
        loss_d_fake = loss_computer.loss_segm(pred_fake_labels, labels, False)

        loss_d_lm = self.forward_labelmix(labels, fake_image_dt, real_image, pred_fake_labels, pred_real_labels)
        loss_d = loss_d_real + loss_d_fake + loss_d_lm + self.forward_unsup(data)
        with amp.scale_loss(loss_d, self.optimizer_d, loss_id=0) as loss_d_scaled:
            loss_d_scaled.backward()
            self.optimizer_d.step()

        # --- generator update ---#
        self.model.netG.zero_grad()
        pred_fake, segm_fake = self.model.netD(fake_image)
        loss_g = loss_computer.loss(segm_fake, labels, True)
        if with_features:
            loss_g += F.mse_loss(pred_fake, real_features.detach())

        with amp.scale_loss(loss_g, self.optimizer_g, loss_id=1) as loss_g_scaled:
            loss_g_scaled.backward()
            self.optimizer_g.step()
        # torch.cuda.empty_cache()
        return loss_d, loss_g

    def mode_alae(self, data):
        real_image = data['image']
        labels = data['label']
        loss_computer = self.loss_computer
        b, c, h, w = labels.shape

        # --- discriminator update ---#
        self.model.netD.zero_grad()
        self.model.to_logit.zero_grad()

        real_features, real_labels = self.model.netD(real_image)
        loss_d_real = loss_computer.loss(real_labels, labels, True)
        loss_d_real += F.softplus(1 + self.model.to_logit(real_features)).mean()

        from_real = random.random() > 0.5
        if from_real:
            features = real_features.detach()
        else:
            noise = torch.randn(b, self.opt.z_dim, dtype=labels.dtype,
                                device=labels.device, requires_grad=False)
            features = self.model.to_feature(noise)
        fake_image = self.model.netG(labels=labels, features=features)
        features_dt = features.detach()
        fake_image_dt = fake_image.detach()

        fake_features, fake_labels = self.model.netD(fake_image_dt)
        loss_d_fake = loss_computer.loss(fake_labels, labels, False)
        loss_d_fake += F.softplus(1 - self.model.to_logit(fake_features)).mean()
        loss_d_fake += F.mse_loss(fake_features, features_dt)
        if not from_real:
            loss_d_fake += F.softplus(1 - self.model.to_logit(features_dt)).mean()

        loss_d_lm = self.forward_labelmix(labels, fake_image_dt, real_image, fake_labels, real_labels)
        loss_d = loss_d_real + loss_d_fake + loss_d_lm + self.forward_unsup(data)
        with amp.scale_loss(loss_d, self.optimizer_d, loss_id=0) as loss_d_scaled:
            loss_d_scaled.backward()
            self.optimizer_d.step()

        # --- generator update ---#
        self.model.netG.zero_grad()
        self.model.to_feature.zero_grad()

        fake_features, fake_labels = self.model.netD(fake_image)
        loss_g = loss_computer.loss(fake_labels, labels, True)
        loss_g += self.model.to_logit(fake_features).mean()
        loss_g += F.mse_loss(fake_features, features_dt)
        if not from_real:
            loss_g += self.model.to_logit(features).mean()

        with amp.scale_loss(loss_g, self.optimizer_g, loss_id=1) as loss_g_scaled:
            loss_g_scaled.backward()
            self.optimizer_g.step()

        # # --- autoencoder update ---#
        # self.model.zero_grad()
        # #
        # features_dt = real_features.detach()
        # fake_image = self.model.netG(labels=labels, features=features_dt)
        # fake_features, fake_labels = self.model.netD(fake_image)
        # loss_ae = F.mse_loss(fake_features, features_dt)
        #
        # with amp.scale_loss(loss_ae, [self.optimizer_g, self.optimizer_d], loss_id=2) as loss_ae_scaled:
        #     loss_ae_scaled.backward()
        # self.optimizer_g.step()
        # self.optimizer_d.step()

        return loss_d, loss_g

    def train_step_fast_gan(self, data, step):
        # mod = step % 1
        # if mod == 0:
        #     loss_d, loss_g = self.mode1(data)
        # elif mod == 1:
        # loss_d, loss_g = self.mode2(data)
        # elif mod == 2:
        losses = self.mode_alae(data)

        # self.visualizer_losses(step, losses_list)
        return losses
