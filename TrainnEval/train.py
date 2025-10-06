import torch
from data.dataset import CustomDataset
import sys
from utils import *
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image
from Model.discriminator import Discriminator
from Model.generator import Generator

def train(disc_h, disc_z, gen_h, gen_z,
          loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):

    loop = tqdm(loader, leave=True)

    for idx, (z, h) in enumerate(loop):
        z = z.to(config.Device)
        h = h.to(config.Device)

        with torch.cuda.amp.autocast():
            fake_h = gen_h(z)
            disc_h_real = disc_h(h)
            disc_h_fake = disc_h(fake_h.detach())
            # h_real += disc_h_real.mean().item()
            # h_fake += disc_h_fake.mean().item()
            d_h_real_loss = mse(disc_h_real, torch.ones_like(disc_h_real))
            d_h_fake_loss = mse(disc_h_fake, torch.zeros_like(disc_h_fake))
            d_h_loss = d_h_real_loss + d_h_fake_loss

            fake_z = gen_z(h)
            disc_z_real = disc_z(z)
            disc_z_fake = disc_z(fake_z.detach())
            d_z_real_loss = mse(disc_z_real, torch.ones_like(disc_z_real))
            d_z_fake_loss = mse(disc_z_fake, torch.zeros_like(disc_z_fake))
            d_z_loss = d_z_real_loss + d_z_fake_loss

            # Complete Discriminator Loss
            d_loss = (d_h_loss + d_z_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(d_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        with torch.cuda.amp.autocast():
            # Adversarial Loss
            d_h_fake = disc_h(fake_h)
            d_z_fake = disc_z(fake_z)
            loss_g_h = mse(d_h_fake, torch.ones_like(d_h_fake))
            loss_g_z = mse(d_z_fake, torch.ones_like(d_z_fake))

            # Cycle Consistency Loss
            cycle_z = gen_z(fake_h)
            cycle_h = gen_h(fake_z)
            cycle_z_loss = l1(z, cycle_z)
            cycle_h_loss = l1(h, cycle_h)

            # Identity Loss
            identity_z = gen_z(z)
            identity_h = gen_h(h)
            identity_z_loss = l1(z, identity_z)
            identity_h_loss = l1(h, identity_h)

            # Complete Generator Loss
            g_loss = (
                loss_g_h + loss_g_z
                + cycle_z_loss * config.Lambda_cycle
                + cycle_h_loss * config.Lambda_cycle
                + identity_z_loss * config.Lambda_identity
                + identity_h_loss * config.Lambda_identity
            )

        opt_gen.zero_grad()
        g_scaler.scale(g_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_h * 0.5 + 0.5, f"saved_images/horse_{idx}.png")
            save_image(fake_z * 0.5 + 0.5, f"saved_images/zebra_{idx}.png")


def main():
    disc_h = Discriminator(in_channels=3).to(config.Device)
    disc_z = Discriminator(in_channels=3).to(config.Device)
    gen_h = Generator(img_channels=3, num_residuals=9).to(config.Device)
    gen_z = Generator(img_channels=3, num_residuals=9).to(config.Device)
    opt_gen = optim.Adam(list(gen_h.parameters()) + list(gen_z.parameters()),
                         lr=config.LR, betas=(0.5, 0.999)
                         )
    opt_disc = optim.Adam(list(disc_h.parameters()) + list(disc_z.parameters()),
                          lr=config.LR, betas=(0.5, 0.999)
                          )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.Load_model:
        load_checkpoint(config.Checkpoint_gen_h, opt_gen, gen_h, config.LR)
        load_checkpoint(config.Checkpoint_gen_z, opt_gen, gen_z, config.LR)
        load_checkpoint(config.Checkpoint_critic_h, disc_h, opt_disc, config.LR)
        load_checkpoint(config.Checkpoint_critic_z, disc_z, opt_disc, config.LR)

    dataset = CustomDataset(
        root_z=config.Train_dir + '/zebra',
        root_h=config.Train_dir + '/horse',
        transform=config.transforms
    )
    val_dataset = CustomDataset(
        root_z=config.Train_dir + '/zebra1',
        root_h=config.Train_dir + '/horse1',
        transform=config.transforms
    )
    loader = DataLoader(dataset, batch_size=config.Batch_size,
                        shuffle=True, num_workers=config.Num_workers,
                        pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=config.Batch_size,
                            shuffle=True, num_workers=config.Num_workers,
                            pin_memory=True)
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.Num_epochs):
        train(
            disc_z, disc_h, gen_z, gen_h, loader, opt_disc,
            opt_gen, L1, mse, d_scaler, g_scaler
        )

        if config.Save_model:
            save_checkpoints(gen_h, opt_gen, filename=config.Checkpoint_gen_h)
            save_checkpoints(gen_z, opt_gen, filename=config.Checkpoint_gen_z)
            save_checkpoints(disc_h, opt_disc, filename=config.Checkpoint_critic_h)
            save_checkpoints(disc_z, opt_disc, filename=config.Checkpoint_critic_z)

if __name__ == '__main__':
    main()