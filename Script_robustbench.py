import numpy as np
import sys
import argparse
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchattacks import PGD
from robustbench import load_model
from utils import *
from tqdm import *
from ddpm import script_utils
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))

def load_Data(dataset, image_size, batch_size, data_path = "./dataset/"):
    if dataset == "cifar10":
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=data_path + "cifar10", train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Resize(image_size),
                             ])), batch_size=batch_size, shuffle=True)
        # Testing Dataset
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=data_path + "cifar10", train=False,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Resize(image_size),
                             ])), batch_size=batch_size, shuffle=True)
    elif dataset == "cifar100":
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root=data_path + "cifar100", train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Resize(image_size),
                              ])), batch_size=batch_size, shuffle=True)
        # Testing Dataset
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root=data_path + "cifar100", train=False,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Resize(image_size),
                              ])), batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

def main(args):
    device = args.device
    D_model = Discriminator(nc=3, nf=32).to(device)
    # train_loader, test_loader = load_Data(args.dataset, args.image_size, args.batch_size)
    diffusion = script_utils.get_diffusion_from_args(args).to(device)
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.iterations * len(train_loader),
            1,
            1e-8 / args.learning_rate))
    config_train = {
        'epsilon': 8 / 255,
        'num_steps': 10,
        'step_size': 2 / 255,
    }
    D_opt = torch.optim.SGD(D_model.parameters(), lr=3e-5, momentum=0.9, weight_decay=0.0005)
    # classifier = load_model(model_name=args.classifier_name, threat_model=args.norm_classifier, dataset= args.dataset, model_dir= "./defense/models").to(device)
    criteriaBCE = torch.nn.BCELoss()
    criteriaMSE = torch.nn.MSELoss()
    contra_loss = contrastive_loss(classifier)

    for iteration in tqdm(range(1, args.iterations + 1)):
        diffusion.train()
        sys.stdout.flush()
        for index, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            net = AttackPGD(classifier, config_train)
            _, adv_pertubation = net(x, y)
            D_G_output, _ = diffusion.get_predict(x, y, adv_pertubation)
            D_score_real = D_model(x - D_G_output)
            D_score_fake = D_model(x + adv_pertubation - D_G_output)
            label_real = torch.ones_like(D_score_real)
            label_fake = torch.ones_like(D_score_fake)
            D_opt.zero_grad()
            D_loss = criteriaBCE(D_score_real, label_real) + criteriaBCE(D_score_fake,label_fake)
            D_loss.backward()
            D_opt.step()

            G_output, _ = diffusion.get_predict(x, y, adv_pertubation)
            puried_exam = x + adv_pertubation - G_output
            G_score_fake = D_model(puried_exam)
            optimizer.zero_grad()
            mse = criteriaMSE(puried_exam, x)
            loss_con = contra_loss(puried_exam, x)
            loss = mse + loss_con + criteriaBCE(G_score_fake, label_real)

            loss.backward()
            optimizer.step()
            scheduler.step()
            diffusion.update_ema()
    model_filename = f"./{args.log_dir}/DDPM-{args.dataset}-{args.classifier_name}-{args.iterations}.pt"
    torch.save(diffusion, model_filename)


class contrastive_loss(torch.nn.Module):
    def __init__(self, classifier, T = 4, margin= 2.0, measure= torch.nn.L1Loss()):
        super(contrastive_loss, self).__init__()
        self.classifier = classifier
        self.margin = margin
        self.measure = measure
        self.T = T

    def forward(self, x_purified, x_roi):
        y_purified, y_ori = self.classifier(x_purified), self.classifier(x_roi)
        loss = self.measure(y_ori, y_purified)
        return loss

def create_argparser(iterations= 50, dataset= "cifar10", classifier_name= None, image_size= 32, batch_size= 128):

    # # Specify model parameters
    defaults = dict(
        image_size= image_size,
        batch_size= batch_size,
        iterations= iterations,
        dataset= dataset,
        classifier_name= classifier_name,
        device=device,
    )
    defaults.update(script_utils.diffusion_defaults())
    parser = argparse.ArgumentParser()
    script_utils.add_dict_to_argparser(parser, defaults)
    return parser

def ddpm_from_params(iterations=50, dataset= "cifar10", classifier_name= None):
    args = create_argparser(iterations, dataset, classifier_name).parse_args()
    diffusion = script_utils.get_diffusion_from_args(args).to(device)
    return diffusion


if __name__ == '__main__':
    # # parameters setting
    classifier_name = ["Sehwag2021Proxy_R18", "Engstrom2019Robustness", "Rade2021Helper_R18_ddpm", "Rebuffi2021Fixing_28_10_cutmix_ddpm"]
    dataset_list = ["cifar10", "cifar10", "cifar100", "cifar100"]
    norm = ["L2", "L2", "Linf", "Linf"]

    # # Train Phase
    for i in range(len(classifier_name)):
        args = create_argparser(iterations=50, dataset=dataset_list[i], classifier_name=classifier_name[i]).parse_args()
        train_loader, test_loader = load_Data(args.dataset, args.image_size, args.batch_size)
        classifier = load_model(model_name=args.classifier_name, threat_model=norm[i],
                                dataset=args.dataset, model_dir="./models").to(device)
        # main(args)

    # # Testing Phase
    for i in range(len(classifier_name)):
        args = create_argparser(iterations=50, dataset=dataset_list[i], classifier_name=classifier_name[i]).parse_args()
        train_loader, test_loader = load_Data(args.dataset, args.image_size, args.batch_size)
        classifier = load_model(model_name=args.classifier_name, threat_model=norm[i],
                                dataset=args.dataset, model_dir="./models").to(device)
        model_filename = f"./{args.log_dir}/DDPM-{args.dataset}-{args.classifier_name}-{args.iterations}.pt"
        diffusion = torch.load(model_filename)
        Attack_FASN(classifier, test_loader, Gen=diffusion, attack=None)

        Attack_FASN(classifier, test_loader, Gen=diffusion, attack=PGD(classifier, eps=8 / 255))







