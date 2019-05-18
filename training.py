import numpy as np
import torch
import torchvision
import torch.optim as optim
from easydict import EasyDict as edict
import torchvision.transforms as transforms

from model import Gan3DposeNet
from utils.model_utils import init_weights, heuristic_loss
from transformation import *
from dataset import MPii2DPoseDataset


def create_dataset(mode='train', **kwargs):

    err = (
        f"Expected value 'train' or 'valid' for the parameter mode,"
        f"got {mode} instead"
    )
    assert mode in ['train', 'valid'], err

    transform = transforms.Compose([
            Normalize(),
            ToTensor()
        ])

    if mode == 'train':
        json_file = kwargs.train_json_file
    else:
        json_file = kwargs.valid_json_file


    data = MPii2DPoseDataset(json_file=kwargs.json_file,
                             transform=transform)
    return data


def build_loaders(data, **kwargs):
    loader = torch.utils.data.DataLoader(data,
                                         batch_size=kwargs.batch_size)
    return loader




def set_model(device, **kwargs):

    generator = Gan3DposeNet(n_inputs=kwargs.n_inputs,
                             n_unit=kwargs.n_unit,
                             mode='generator').to(device)

    discriminator = Gan3DposeNet(n_inputs=kwargs.n_inputs,
                                 n_unit=kwargs.n_unit,
                                 mode='discriminator').to(device)

    generator.apply(init_weights)
    discriminator.apply(init_weights)

    gen_optimizer = optim.Adam(params=generator.parameters(),
                               lr=kwargs.lr, betas=kwargs.betas,
                               weight_decay=kwargs.weight_decay)

    dis_optimizer = optim.Adam(params=discriminator.parameters(),
                               lr=kwargs.lr, betas=kwargs.betas,
                               weight_decay=kwargs.weight_decay)

    return generator, discriminator, gen_optimizer, dis_optimizer


def train(train_loader,
          generator, discriminator,
          gen_optimizer, dis_optimizer, device, **kwargs):

    n_epochs = kwargs.n_epochs
    batch_size = kwargs.batch_size
    heuristic_loss_weight = kwargs.heuristic_loss_weight
    model_path = kwargs.model_path

    for epoch in range(n_epochs):

        for xy_real in train_loader:
            # do things (preprocessing, GPU, etc...)
            # ...
            # ...

            z_pred = generator(xy_real)
            xyz = np.stack((xy_real, z_pred), axis=-1)


            # Random rotation
            theta = np.random.uniform(0, 2*np.pi, batch_size)
            cos_theta = np.cos(theta).reshape(-1, 1)
            sin_theta = np.sin(theta).reshape(-1, 1)

            # 2D projection - reminder: rotation around y
            x, y = xyz[:, :1], xyz[:,1:-1]
            new_x = x * cos_theta + z_pred * sin_theta
            xy_projected = np.stack((new_x, y), axis=-1).reshape(batch_size, -1)


            output_dis_real = discriminator(xy_real) #1
            output_dis_fake = discriminator(xy_projected) #0

            accuracy_real = np.sum(
                output_dis_real / np.ones(output_dis_real.shape)) / 2

            accuracy_fake = np.sum(
                output_dis_fake / np.zeros(output_dis_fake.shape)) / 2

            accuracy_dis = (accuracy_real + accuracy_fake) / 2

            # Compute generator loss
            loss_gen = F.softmax(-output_dis_fake, dim=0).sum() / batch_size
            loss_heuristic = heuristic_loss(xy_real, z_pred)
            loss_gen += loss_heuristic * heuristic_loss_weight

            gen_optimizer.zero_grad()

            # update on condition
            if accuracy_dis >= 0.1:
                loss_gen.backward()
                gen_optimizer.step()

            # unchain backward here "Not sure of this implementation"
            xy_projected.detach()

            # Compute discriminator loss
            loss_dis = F.softmax(-output_dis_real, dim=0).sum() / batch_size
            loss_dis += F.softmax(output_dis_real, dim=0).sum() / batch_size

            dis_optimizer.zero_grad()

            # update on condition
            if accuracy_dis <= 0.9:
                loss_dis.backward()
                dis_optimizer.update()

            print(f"Generator loss: {loss_gen} \tDiscriminator loss: {loss_dis}")




if __name__ == '__main__':
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    params = edict({
        'batch_size': 16,
        'num_workers': 4,
        'n_epochs': 50,
        'n_inputs': 34,
        'n_unit': 1024,
        'lr': 0.001,
        'alpha': 2e-4,
        'betas': (0.5, 0.999),
        'weight_decay': 1e-5,
        'heuristic_loss_weight': 1.0,
        'model_path_gen': 'saved_models/Gen.pt',
        'model_path_dis': 'saved_models/Dis.pt',
        'train_json_file': 'data/train.json',
        'valid_json_file': 'data/valid.json'
    })

    train_set = create_dataset(mode='train', **params)
    valid_set = create_dataset(mode='train', **params)

    train_loader = build_loaders(train_set, **kwargs)
    valid_loader = build_loaders(valid_set, **kwargs)
    
    #visualize data in batches

    generator, discriminator, gen_optimizer, dis_optimizer = set_model(device,
                                                                       **params)

    train(train_loader,
          generator, discriminator,
          gen_optimizer, dis_optimizer, device, **params)
    print("Training done.")

    torch.save(generator.state_dict(), model_path_gen)
    torch.save(discriminator.state_dict(), model_path_dis)
    print("\nModels Saved.")
        



























