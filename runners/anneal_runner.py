import numpy as np
import tqdm
from losses.dsm import anneal_dsm_score_estimation
from losses.sliced_sm import anneal_sliced_score_estimation_vr
import torchvision.transforms.functional as F
import logging
import torch
import os
import shutil
import tensorboardX
import torch.optim as optim
from torchvision.datasets import MNIST, CIFAR10, SVHN
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from datasets.celeba import CelebA
from datasets.nyu import NYUv2
from models.cond_refinenet_dilated import CondRefineNetDilated
from torchvision.utils import save_image, make_grid
from PIL import Image


__all__ = ['AnnealRunner']


class AnnealRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def get_optimizer(self, parameters):
        if self.config.optim.optimizer == 'Adam':
            return optim.Adam(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                              betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad)
        elif self.config.optim.optimizer == 'RMSProp':
            return optim.RMSprop(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.optimizer == 'SGD':
            return optim.SGD(parameters, lr=self.config.optim.lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.config.optim.optimizer))

    def logit_transform(self, image, lam=1e-6):
        image = lam + (1 - 2 * lam) * image
        return torch.log(image) - torch.log1p(-image)

    def train(self):
        if self.config.data.random_flip is False:
            tran_transform = test_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])
        else:
            tran_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ])
            test_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])

        if self.config.data.dataset == 'CIFAR10':
            dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=True, download=True,
                              transform=tran_transform)
            test_dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10_test'), train=False, download=True,
                                   transform=test_transform)

        elif self.config.data.dataset == 'MNIST':
            dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist'), train=True, download=True,
                            transform=tran_transform)
            test_dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist_test'), train=False, download=True,
                                 transform=test_transform)

        elif self.config.data.dataset == 'CELEBA':
            if self.config.data.random_flip:
                dataset = CelebA(root=os.path.join(self.args.run, 'datasets', 'celeba'), split='train',
                                 transform=transforms.Compose([
                                     transforms.CenterCrop(140),
                                     transforms.Resize(self.config.data.image_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                 ]), download=True)
            else:
                dataset = CelebA(root=os.path.join(self.args.run, 'datasets', 'celeba'), split='train',
                                 transform=transforms.Compose([
                                     transforms.CenterCrop(140),
                                     transforms.Resize(self.config.data.image_size),
                                     transforms.ToTensor(),
                                 ]), download=True)

            test_dataset = CelebA(root=os.path.join(self.args.run, 'datasets', 'celeba_test'), split='test',
                                  transform=transforms.Compose([
                                      transforms.CenterCrop(140),
                                      transforms.Resize(self.config.data.image_size),
                                      transforms.ToTensor(),
                                  ]), download=True)

        elif self.config.data.dataset == 'SVHN':
            dataset = SVHN(os.path.join(self.args.run, 'datasets', 'svhn'), split='train', download=True,
                           transform=tran_transform)
            test_dataset = SVHN(os.path.join(self.args.run, 'datasets', 'svhn_test'), split='test', download=True,
                                transform=test_transform)

        elif self.config.data.dataset == 'NYUv2':
            if self.config.data.random_flip is False:
                nyu_train_transform = nyu_test_transform = transforms.Compose([
                    transforms.CenterCrop((400, 400)),
                    transforms.Resize(32),
                    transforms.ToTensor()
                ])
            else:
                nyu_train_transform = transforms.Compose([
                    transforms.CenterCrop((400, 400)),
                    transforms.Resize(32),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor()
                ])
                nyu_test_transform = transforms.Compose([
                    transforms.CenterCrop((400, 400)),
                    transforms.Resize(32),
                    transforms.ToTensor()
                ])

            dataset = NYUv2(os.path.join(self.args.run, 'datasets', 'nyuv2'), train=True, download=True,
                            rgb_transform=nyu_train_transform, depth_transform=nyu_train_transform)
            test_dataset = NYUv2(os.path.join(self.args.run, 'datasets', 'nyuv2'), train=False, download=True,
                                 rgb_transform=nyu_test_transform, depth_transform=nyu_test_transform)

        dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                num_workers=0)  # changed num_workers from 4 to 0
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=0, drop_last=True)  # changed num_workers from 4 to 0

        test_iter = iter(test_loader)
        self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels

        tb_path = os.path.join(self.args.run, 'tensorboard', self.args.doc)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)

        tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)
        score = CondRefineNetDilated(self.config).to(self.config.device)

        score = torch.nn.DataParallel(score)

        optimizer = self.get_optimizer(score.parameters())

        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'))
            score.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])

        step = 0

        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                               self.config.model.num_classes))).float().to(self.config.device)

        for epoch in range(self.config.training.n_epochs):
            for i, (X, y) in enumerate(dataloader):
                step += 1
                score.train()
                X = X.to(self.config.device)
                X = X / 256. * 255. + torch.rand_like(X) / 256.

                if self.config.data.logit_transform:
                    X = self.logit_transform(X)

                if self.config.data.dataset == 'NYUv2':
                    # concatenate depth map with image
                    y = y[0]
                    # code to see resized depth map
                    # input_gt_depth_image = y[0][0].data.cpu().numpy().astype(np.float32)
                    # plot.imsave('gt_depth_map_{}.png'.format(i), input_gt_depth_image,
                    #             cmap="viridis")
                    y = y.to(self.config.device)
                    X = torch.cat((X, y), 1)

                labels = torch.randint(0, len(sigmas), (X.shape[0],), device=X.device)

                if self.config.training.algo == 'dsm':
                    loss = anneal_dsm_score_estimation(score, X, labels, sigmas, self.config.training.anneal_power)
                elif self.config.training.algo == 'ssm':
                    loss = anneal_sliced_score_estimation_vr(score, X, labels, sigmas,
                                                             n_particles=self.config.training.n_particles)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tb_logger.add_scalar('loss', loss, global_step=step)
                logging.info("step: {}, loss: {}".format(step, loss.item()))

                if step >= self.config.training.n_iters:
                    return 0

                if step % 100 == 0:
                    score.eval()
                    try:
                        test_X, test_y = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_X, test_y = next(test_iter)

                    test_X = test_X.to(self.config.device)

                    test_X = test_X / 256. * 255. + torch.rand_like(test_X) / 256.
                    if self.config.data.logit_transform:
                        test_X = self.logit_transform(test_X)

                    if self.config.data.dataset == 'NYUv2':
                        test_y = test_y[0]
                        test_y = test_y.to(self.config.device)
                        test_X = torch.cat((test_X, test_y), 1)

                    test_labels = torch.randint(0, len(sigmas), (test_X.shape[0],), device=test_X.device)

                    with torch.no_grad():
                        test_dsm_loss = anneal_dsm_score_estimation(score, test_X, test_labels, sigmas,
                                                                    self.config.training.anneal_power)

                    tb_logger.add_scalar('test_dsm_loss', test_dsm_loss, global_step=step)

                if step % self.config.training.snapshot_freq == 0:
                    states = [
                        score.state_dict(),
                        optimizer.state_dict(),
                    ]
                    torch.save(states, os.path.join(self.args.log, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.log, 'checkpoint.pth'))

    def Langevin_dynamics(self, x_mod, scorenet, n_steps=200, step_lr=0.00005):
        images = []

        labels = torch.ones(x_mod.shape[0], device=x_mod.device) * 9
        labels = labels.long()

        with torch.no_grad():
            for _ in range(n_steps):
                images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
                noise = torch.randn_like(x_mod) * np.sqrt(step_lr * 2)
                grad = scorenet(x_mod, labels)
                x_mod = x_mod + step_lr * grad + noise
                x_mod = x_mod
                print("modulus of grad components: mean {}, max {}".format(grad.abs().mean(), grad.abs().max()))

            return images

    def anneal_Langevin_dynamics(self, x_mod, scorenet, sigmas, n_steps_each=100, step_lr=0.00002):
        images = []

        with torch.no_grad():
            for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc='annealed Langevin dynamics sampling'):
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
                labels = labels.long()
                step_size = step_lr * (sigma / sigmas[-1]) ** 2
                for s in range(n_steps_each):
                    images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
                    noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                    grad = scorenet(x_mod, labels)
                    x_mod = x_mod + step_size * grad + noise
                    # print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                    #                                                          grad.abs().max()))

            return images

    def test(self):
        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
        score = CondRefineNetDilated(self.config).to(self.config.device)
        score = torch.nn.DataParallel(score)

        score.load_state_dict(states[0])

        if not os.path.exists(self.args.image_folder):
            os.makedirs(self.args.image_folder)

        sigmas = np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                                    self.config.model.num_classes))

        score.eval()
        grid_size = 5

        imgs = []
        if self.config.data.dataset == 'MNIST':
            samples = torch.rand(grid_size ** 2, 1, 28, 28, device=self.config.device)
            all_samples = self.anneal_Langevin_dynamics(samples, score, sigmas, 100, 0.00002)

            for i, sample in enumerate(tqdm.tqdm(all_samples, total=len(all_samples), desc='saving images')):
                sample = sample.view(grid_size ** 2, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                image_grid = make_grid(sample, nrow=grid_size)
                if i % 10 == 0:
                    im = Image.fromarray(
                        image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                    imgs.append(im)

                save_image(image_grid, os.path.join(self.args.image_folder, 'image_{}.png'.format(i)))
                torch.save(sample, os.path.join(self.args.image_folder, 'image_raw_{}.pth'.format(i)))

        elif self.config.data.dataset == 'NYUv2':
            deps = []
            samples = torch.rand(grid_size ** 2, 4, 32, 32, device=self.config.device)

            all_samples = self.anneal_Langevin_dynamics(samples, score, sigmas, 100, 0.00002)

            for i, sample in enumerate(tqdm.tqdm(all_samples, total=len(all_samples), desc='saving images')):
                sample = sample.view(grid_size ** 2, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

                image_grid = make_grid(sample[:, :3, :, :], nrow=grid_size)
                # make grid always return 3 channels if 1 channel given it will repeat the same channel 3 times
                depth_grid = make_grid(sample[:, 3, :, :].unsqueeze(1), nrow=grid_size)
                if i % 10 == 0:
                    im = Image.fromarray(
                        image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                    imgs.append(im)

                    # dep = Image.fromarray((depth_grid[0, :, :].mul_(255)/torch.max(depth_grid[0, :, :])).permute(1, 2, 0).to('cpu', torch.uint8).numpy(), mode='L')
                    dep = F.to_pil_image(depth_grid[0, :, :])
                    deps.append(dep)

                save_image(image_grid, os.path.join(self.args.image_folder, 'image_{}.png'.format(i)))
                save_image(depth_grid, os.path.join(self.args.image_folder, 'depth_{}.png'.format(i)))
                # torch.save(sample, os.path.join(self.args.image_folder, 'image_raw_{}.pth'.format(i)))

            deps[0].save(os.path.join(self.args.image_folder, "movie_d.gif"), save_all=True, append_images=deps[1:],
                         duration=1, loop=0)

        else:
            samples = torch.rand(grid_size ** 2, 3, 32, 32, device=self.config.device)

            all_samples = self.anneal_Langevin_dynamics(samples, score, sigmas, 100, 0.00002)

            for i, sample in enumerate(tqdm.tqdm(all_samples, total=len(all_samples), desc='saving images')):
                sample = sample.view(grid_size ** 2, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                image_grid = make_grid(sample, nrow=grid_size)
                if i % 10 == 0:
                    im = Image.fromarray(
                        image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                    imgs.append(im)

                save_image(image_grid, os.path.join(self.args.image_folder, 'image_{}.png'.format(i)), nrow=10)
                torch.save(sample, os.path.join(self.args.image_folder, 'image_raw_{}.pth'.format(i)))

        imgs[0].save(os.path.join(self.args.image_folder, "movie.gif"), save_all=True, append_images=imgs[1:],
                     duration=1, loop=0)

    def anneal_Langevin_dynamics_inpainting(self, x_mod, refer_image, scorenet, sigmas, n_steps_each=100,
                                            step_lr=0.000008):
        images = []

        refer_image = refer_image.unsqueeze(1).expand(-1, x_mod.shape[1], -1, -1, -1)
        refer_image = refer_image.contiguous().view(-1, 3, 32, 32)
        x_mod = x_mod.view(-1, 3, 32, 32)
        half_refer_image = refer_image[..., :16]
        with torch.no_grad():
            for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc="annealed Langevin dynamics sampling"):
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
                labels = labels.long()
                step_size = step_lr * (sigma / sigmas[-1]) ** 2

                corrupted_half_image = half_refer_image + torch.randn_like(half_refer_image) * sigma
                x_mod[:, :, :, :16] = corrupted_half_image
                for s in range(n_steps_each):
                    images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
                    noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                    grad = scorenet(x_mod, labels)
                    x_mod = x_mod + step_size * grad + noise
                    x_mod[:, :, :, :16] = corrupted_half_image
                    # print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                    #                                                          grad.abs().max()))

            return images

    def anneal_Langevin_dynamics_prediction(self, x_mod, rgb_image, scorenet, sigmas, n_steps_each=100,
                                            step_lr=0.000008):
        """
        The function make prediction of depth map for a given rbg image with annealed langevin dynamics

        :param x_mod: an initial random sample with the shape [batch, repeat, channel, width, height]
        :param refer_image: the rgb image containing with shape [batch, channel, width, height]
        :param scorenet: the score network takes input of (rgb, depth) image and return the score estimation with same dimension
        :param sigmas: the noise levels of annealing and conditioning on score estimation
        :param n_steps_each: number of langevin steps
        :param step_lr: learning rate for the smallest noise level
        :return: list of sampled depth maps with shapes [batch * repeat , 1, width, height]
                 and converged sample as prediction depth map
        """

        depths = []
        batch = x_mod.shape[0]
        repeat = x_mod.shape[1]
        rgb_image = rgb_image.unsqueeze(1).expand(-1, repeat, -1, -1, -1)
        rgb_image = rgb_image.contiguous().view(-1, 3, 32, 32)
        x_mod = x_mod.view(-1, 4, 32, 32)

        with torch.no_grad():
            for c, sigma in tqdm.tqdm(enumerate(sigmas), total=len(sigmas), desc="annealed Langevin dynamics sampling"):
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
                labels = labels.long()
                step_size = step_lr * (sigma / sigmas[-1]) ** 2

                corrupted_rgb_image = rgb_image + torch.randn_like(rgb_image) * sigma

                # fix the known pixels with noise injected
                x_mod[:, :3, :, :] = corrupted_rgb_image
                for s in range(n_steps_each):
                    noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                    grad = scorenet(x_mod, labels)
                    x_mod = x_mod + step_size * grad + noise
                    x_mod[:, :3, :, :] = corrupted_rgb_image
                    depths.append(x_mod[:, 3, :, :].to('cpu'))
            depth_pred = x_mod[:, 3, :, :]
            depth_pred = depth_pred.unsqueeze(1).contiguous().view(repeat, batch, 1, 32, 32)
            depth_pred = depth_pred.mean(0).squeeze(0)

            return depths, depth_pred



    def test_inpainting(self):
        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
        score = CondRefineNetDilated(self.config).to(self.config.device)
        score = torch.nn.DataParallel(score)

        score.load_state_dict(states[0])

        if not os.path.exists(self.args.image_folder):
            os.makedirs(self.args.image_folder)

        sigmas = np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                                    self.config.model.num_classes))
        score.eval()

        imgs = []
        if self.config.data.dataset == 'CELEBA':
            dataset = CelebA(root=os.path.join(self.args.run, 'datasets', 'celeba'), split='test',
                             transform=transforms.Compose([
                                 transforms.CenterCrop(140),
                                 transforms.Resize(self.config.data.image_size),
                                 transforms.ToTensor(),
                             ]), download=True)

            dataloader = DataLoader(dataset, batch_size=20, shuffle=True,
                                    num_workers=0)  # changed num_workers from 4 to 0
            refer_image, _ = next(iter(dataloader))

            samples = torch.rand(20, 20, 3, self.config.data.image_size, self.config.data.image_size,
                                 device=self.config.device)

            all_samples = self.anneal_Langevin_dynamics_inpainting(samples, refer_image, score, sigmas, 100, 0.00002)
            torch.save(refer_image, os.path.join(self.args.image_folder, 'refer_image.pth'))

            for i, sample in enumerate(tqdm.tqdm(all_samples)):
                sample = sample.view(400, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                image_grid = make_grid(sample, nrow=20)
                if i % 10 == 0:
                    im = Image.fromarray(
                        image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                    imgs.append(im)

                save_image(image_grid, os.path.join(self.args.image_folder, 'image_completion_{}.png'.format(i)))
                torch.save(sample, os.path.join(self.args.image_folder, 'image_completion_raw_{}.pth'.format(i)))

        elif self.config.data.dataset == 'NYUv2':
            # TODO implement inpainting and MSE calculate for NYUv2
            nyu_transform = transforms.Compose([transforms.CenterCrop((400, 400)),
                                                transforms.Resize(32),
                                                transforms.ToTensor()])
            dataset = NYUv2(os.path.join(self.args.run, 'datasets', 'nyuv2'), train=False, download=True,
                            rgb_transform=nyu_transform, depth_transform=nyu_transform)

            dataloader = DataLoader(dataset, batch_size=20, shuffle=True,
                                    num_workers=0)

            data_iter = iter(dataloader)
            rgb_image, depth = next(data_iter)
            rgb_image = rgb_image.to(self.config.device)
            depth = depth[0].to(self.config.device)

            # MSE loss evaluation
            mse = torch.nn.MSELoss()

            rgb_image = rgb_image / 256. * 255. + torch.rand_like(rgb_image) / 256.

            # torch.save(rgb_image, os.path.join(self.args.image_folder, 'rgb_image.pth'))
            samples = torch.rand(20, 20, self.config.data.channels, self.config.data.image_size,
                                 self.config.data.image_size).to(self.config.device)

            all_depth_samples, depth_pred = self.anneal_Langevin_dynamics_prediction(samples, rgb_image, score, sigmas, 100, 0.00002)

            print("MSE loss is %5.4f" % (mse(depth_pred, depth)))

            for i, sample in enumerate(tqdm.tqdm(all_depth_samples)):
                sample = sample.view(400, self.config.data.channels - 3, self.config.data.image_size,
                                     self.config.data.image_size)

                sample = torch.cat((depth.to('cpu'), sample), 0)

                depth_grid = make_grid(sample, nrow=20)
                if i % 10 == 0:
                    # dep = Image.fromarray(depth_grid.to('cpu').numpy().astype(np.float32), mode='F')
                    dep = F.to_pil_image(depth_grid[0, :, :])
                    imgs.append(dep)

                save_image(depth_grid, os.path.join(self.args.image_folder, 'depth_prediction_{}.png'.format(i)))
                # torch.save(sample, os.path.join(self.args.image_folder, 'depth_prediction_raw_{}.pth'.format(i)))



        else:
            transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])

            if self.config.data.dataset == 'CIFAR10':
                dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=True, download=True,
                                  transform=transform)
            elif self.config.data.dataset == 'SVHN':
                dataset = SVHN(os.path.join(self.args.run, 'datasets', 'svhn'), split='train', download=True,
                               transform=transform)

            dataloader = DataLoader(dataset, batch_size=20, shuffle=True,
                                    num_workers=0)  # changed num_workers from 4 to 0
            data_iter = iter(dataloader)
            refer_image, _ = next(data_iter)

            torch.save(refer_image, os.path.join(self.args.image_folder, 'refer_image.pth'))
            samples = torch.rand(20, 20, self.config.data.channels, self.config.data.image_size,
                                 self.config.data.image_size).to(self.config.device)

            all_samples = self.anneal_Langevin_dynamics_inpainting(samples, refer_image, score, sigmas, 100, 0.00002)

            for i, sample in enumerate(tqdm.tqdm(all_samples)):
                sample = sample.view(400, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                image_grid = make_grid(sample, nrow=20)
                if i % 10 == 0:
                    im = Image.fromarray(
                        image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                    imgs.append(im)

                save_image(image_grid, os.path.join(self.args.image_folder, 'image_completion_{}.png'.format(i)))
                torch.save(sample, os.path.join(self.args.image_folder, 'image_completion_raw_{}.pth'.format(i)))

        imgs[0].save(os.path.join(self.args.image_folder, "movie.gif"), save_all=True, append_images=imgs[1:],
                     duration=1, loop=0)
