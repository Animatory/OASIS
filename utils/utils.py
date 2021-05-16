import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def plot_images(imgs, names=None, show=True, nrows=None, ncols=None, figsize=(16, 8), title=None):
    if not isinstance(imgs, list):
        imgs = [imgs]

    from math import ceil
    if nrows is None and ncols is None:
        nrows = 1
        ncols = len(imgs)
    elif nrows is None:
        nrows = ceil(len(imgs) / ncols)
    elif ncols is None:
        ncols = ceil(len(imgs) / nrows)

    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axs.imshow(imgs[0])
        axs.set_axis_off()
        if names and len(names) > 0:
            axs.set_title(names[0], fontsize=15)
    elif nrows == 1 or ncols == 1:
        for j, ax in enumerate(axs):
            ax.imshow(imgs[j])
            ax.set_axis_off()
            if names and j < len(names):
                ax.set_title(names[j], fontsize=15)
    else:
        for j, ax in enumerate(axs):
            for k, sub_ax in enumerate(ax):
                image_id = j * ncols + k
                sub_ax.set_axis_off()
                if image_id < len(imgs):
                    sub_ax.imshow(imgs[image_id])
                    if names and image_id < len(names):
                        sub_ax.set_title(names[image_id], fontsize=15)
    if show:
        plt.show()
    else:
        fig.tight_layout()
        plt.savefig(title)
        plt.close()


def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def get_start_iters(start_iter, dataset_size):
    if start_iter == 0:
        return 0, 0
    start_epoch, start_iter = divmod(start_iter + 1, dataset_size)
    return start_epoch, start_iter


class ResultsSaver:
    def __init__(self, opt):
        path = os.path.join(opt.results_dir, opt.name, opt.ckpt_iter)
        self.path_label = os.path.join(path, "label")
        self.path_image = os.path.join(path, "image")
        self.path_to_save = {"label": self.path_label, "image": self.path_image}
        os.makedirs(self.path_label, exist_ok=True)
        os.makedirs(self.path_image, exist_ok=True)
        self.num_cl = opt.label_nc + 2

    def __call__(self, label, generated, name):
        assert len(label) == len(generated)
        for i in range(len(label)):
            im = tens_to_lab(label[i], self.num_cl)
            self.save_im(im, "label", name[i])
            im = tens_to_im(generated[i]) * 255
            self.save_im(im, "image", name[i])

    def save_im(self, im, mode, name):
        im = Image.fromarray(im.astype(np.uint8))
        im.save(os.path.join(self.path_to_save[mode], name.split("/")[-1]).replace('.jpg', '.png'))


class Timer:
    def __init__(self, opt):
        self.prev_time = time.time()
        self.prev_epoch = 0
        self.num_epochs = opt.num_epochs
        self.file_name = Path(opt.checkpoints_dir) / opt.name / "progress.txt"

    def __call__(self, epoch, cur_iter):
        if cur_iter != 0:
            avg = (time.time() - self.prev_time) / (cur_iter - self.prev_epoch)
        else:
            avg = 0
        self.prev_time = time.time()
        self.prev_epoch = cur_iter

        msg = f'[epoch {epoch}/{self.num_epochs} - iter {cur_iter}], time:{avg:.3f} \n'
        with open(self.file_name, "a") as log_file:
            log_file.write(msg)
        print(msg)
        return avg


class LossesSaver:
    def __init__(self, opt):
        self.name_list = ["Generator", "Vgg", "D_fake", "D_real", "LabelMix"]
        self.opt = opt
        self.freq_smooth_loss = opt.freq_smooth_loss
        self.freq_save_loss = opt.freq_save_loss
        self.losses = dict()
        self.cur_estimates = np.zeros(len(self.name_list))
        self.path = Path(self.opt.checkpoints_dir, self.opt.name, "losses")
        self.is_first = True
        os.makedirs(self.path, exist_ok=True)
        for name in self.name_list:
            if opt.continue_train:
                self.losses[name] = np.load(self.path / "losses.npy", allow_pickle=True).item()[name]
            else:
                self.losses[name] = list()

    def __call__(self, epoch, losses):
        for i, loss in enumerate(losses):
            if loss is None:
                self.cur_estimates[i] = None
            else:
                self.cur_estimates[i] += loss.detach().cpu().numpy()
        if epoch % self.freq_smooth_loss == self.freq_smooth_loss - 1:
            for i, loss in enumerate(losses):
                if not self.cur_estimates[i] is None:
                    self.losses[self.name_list[i]].append(self.cur_estimates[i] / self.opt.freq_smooth_loss)
                    self.cur_estimates[i] = 0
        if epoch % self.freq_save_loss == self.freq_save_loss - 1:
            self.plot_losses()
            np.save(self.path / "losses", self.losses)

    def plot_losses(self):
        for curve in self.losses:
            fig, ax = plt.subplots(1)
            n = np.array(range(len(self.losses[curve]))) * self.opt.freq_smooth_loss
            plt.plot(n[1:], self.losses[curve][1:])
            plt.ylabel('loss')
            plt.xlabel('epochs')

            plt.savefig(self.path / f'{curve}.png', dpi=600)
            plt.close(fig)

        fig, ax = plt.subplots(1)
        for curve in self.losses:
            if np.isnan(self.losses[curve][0]):
                continue
            plt.plot(n[1:], self.losses[curve][1:], label=curve)
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.legend(loc="upper right")
        plt.savefig(self.path / 'combined.png', dpi=600)
        plt.close(fig)


class ImageSaver:
    def __init__(self, opt):
        self.cols = 4
        self.rows = 3
        self.grid = 5
        self.path = opt.checkpoints_dir / opt.name / "images"
        self.opt = opt
        self.num_cl = opt.label_nc + 2
        self.path.mkdir(exist_ok=True)

    def visualize_batch(self, model, image, label, cur_iter):
        self.save_images(label, "label", cur_iter, is_label=True)
        self.save_images(image, "real", cur_iter)
        with torch.no_grad():
            model.eval()
            fake = model(image, label, 'generate', is_ema=False)
            self.save_images(fake, "fake", cur_iter)
            if not self.opt.no_EMA:
                fake = model(image, label, 'generate', is_ema=True)
                self.save_images(fake, "fake_ema", cur_iter)
            model.train()

    def save_images(self, batch, name, cur_iter, is_label=False):
        fig = plt.figure()
        for i in range(min(self.rows * self.cols, len(batch))):
            if is_label:
                im = tens_to_lab(batch[i], self.num_cl)
            else:
                im = tens_to_im(batch[i])
            plt.axis("off")
            fig.add_subplot(self.rows, self.cols, i + 1)
            plt.axis("off")
            plt.imshow(im)
        fig.tight_layout()
        plt.savefig(self.path / f"{cur_iter}_{name}")
        plt.close()


def make_one_hot(labels, num_classes):
    labels = labels[:, None]
    one_hot = torch.zeros(labels.size(0), num_classes, labels.size(2), labels.size(3), device=labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target


def tens_to_im(tens):
    out = (tens + 1) / 2
    out.clamp(0, 1)
    out = out.detach().cpu().numpy()
    if tens.ndim == 3:
        out = np.transpose(out, (1, 2, 0))
    else:
        out = np.transpose(out, (0, 2, 3, 1))
    out = (out * 255).astype('uint8')
    return out


def tens_to_lab(tens, num_cl):
    label_tensor = colorize(tens, num_cl)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy


###############################################################################
# Code below from
# https://github.com/visinf/1-stage-wseg/blob/38130fee2102d3a140f74a45eec46063fcbeaaf8/datasets/utils.py
# Modified so it complies with the Cityscapes label map colors (fct labelcolormap)
###############################################################################

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def colorize(tens, num_cl):
    cmap = labelcolormap(num_cl)
    cmap = torch.from_numpy(cmap[:num_cl])
    c, h, w = tens.shape
    color_image = torch.zeros(3, h, w, dtype=torch.uint8)
    tens = torch.argmax(tens, dim=0, keepdim=True)

    for label in range(0, len(cmap)):
        mask = (label == tens[0]).cpu()
        color_image[0][mask] = cmap[label][0]
        color_image[1][mask] = cmap[label][1]
        color_image[2][mask] = cmap[label][2]
    return color_image


def labelcolormap(n):
    if n == 35:
        cmap = np.array([(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (111, 74, 0), (81, 0, 81),
                         (128, 64, 128), (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70),
                         (102, 102, 156), (190, 153, 153),
                         (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153),
                         (250, 170, 30), (220, 220, 0),
                         (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142),
                         (0, 0, 70),
                         (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142)],
                        dtype=np.uint8)
    else:
        cmap = np.zeros((n, 3), dtype=np.uint8)
        for i in range(n):
            r, g, b = 0, 0, 0
            idx = i + 1  # let's give 0 a color
            for j in range(7):
                str_id = uint82bin(idx)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                idx = idx >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap
