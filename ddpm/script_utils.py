import argparse,sys
import torchvision
import torch.nn.functional as F


from .diffusion import (
    GaussianDiffusion,
    generate_linear_schedule,
    generate_cosine_schedule,
)


def cycle(dl):
    """
    https://github.com/lucidrains/denoising-diffusion-pytorch/
    """
    while True:
        for data in dl:
            yield data

def get_transform():
    class RescaleChannels(object):
        def __call__(self, sample):
            return 2 * sample - 1

    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        RescaleChannels(),
    ])


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def add_dict_to_argparser(parser, default_dict):
    """
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/script_util.py
    """
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def diffusion_defaults():
    defaults = dict(
        learning_rate=2e-4,
        num_timesteps=130,
        log_to_wandb=False,
        log_rate=1000,
        checkpoint_rate=1000,
        log_dir="./save_file",
        schedule_low=1e-4,
        schedule_high=0.02,
        model_checkpoint=None,
        optim_checkpoint=None,
        schedule="linear",
        loss_type="l2",
        use_labels=False,
        base_channels=128,
        channel_mults=(1, 2, 2, 2),
        num_res_blocks=2,
        time_emb_dim=128 * 4,
        norm="gn",
        dropout=0.1,
        activation="silu",
        attention_resolutions=(1,),

        ema_decay=0.9999,
        ema_update_rate=1,

    )
    return defaults


def get_diffusion_from_args(args):
    activations = {
        "relu": F.relu,
        "mish": F.mish,
        "silu": F.silu,
    }
    from .new_unet import UNet
    model = UNet(
        in_channel=6,
        out_channel=3,
        norm_groups=32,
        inner_channel=64,
        channel_mults=[1,2,4,8,8],
        attn_res=[16],
        res_blocks=2,
        dropout=0.2,
        image_size=args.image_size
    )

    if args.schedule == "cosine":
        betas = generate_cosine_schedule(args.num_timesteps)
    else:
        betas = generate_linear_schedule(
            args.num_timesteps,
            args.schedule_low * 1000 / args.num_timesteps,
            args.schedule_high * 1000 / args.num_timesteps,
        )

    diffusion = GaussianDiffusion(
        model, (args.image_size, args.image_size), 3, 10,
        betas,
        # classifier= Setup_Data.load_ModelParams(args.dataset, args.index),
        # classifier= load_model(model_name=args.classifier_name, threat_model=args.norm_classifier, dataset=args.dataset, model_dir="./defense/models"),
        ema_decay=args.ema_decay,
        ema_update_rate=args.ema_update_rate,
        ema_start=2000,
        # loss_type=args.loss_type,
        project_name = args.dataset
    )
    return diffusion