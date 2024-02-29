import argparse
import traceback
import shutil
import logging
import yaml
import sys
import torch
import numpy as np
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# import torch.utils.tensorboard as tb
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
from runners.statistical_translation import Diffusion

torch.set_printoptions(sci_mode=False)

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument('--train_path_a',
                        default='/Balvan_patches/fold1/patch_tlevel1/A/test',
                        help='path to training set for modality A')
    parser.add_argument('--train_path_b',
                        default='/Balvan_patches/fold1/patch_tlevel1/B/test',
                        help='path to training set for modality B')

    parser.add_argument('--val_path_a',
                        default='/Balvan_patches/fold1/patch_tlevel1/A/test',
                        help='path to val set for modality A')
    parser.add_argument('--val_path_b',
                        default='/Balvan_patches/fold1/patch_tlevel1/B/test',
                        help='path to val set for modality B')

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--doc",
        type=str,
        required=True,
        help="A string for documentation purpose. "
             "Will be the name of the log folder.",
    )
    parser.add_argument(
        "--comment", type=str, default="", help="A string for experiment comment"
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Whether to produce samples from the model",
    )
    parser.add_argument("--fid", action="store_true")
    parser.add_argument("--interpolation", action="store_true")

    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="images",
        help="The folder name of samples",
    )

    parser.add_argument("--use_pretrained", action="store_false")
    parser.add_argument(
        "--sample_type",
        type=str,
        default="generalized",
        help="sampling approach (generalized or ddpm_noisy)",
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="uniform",
        help="skip according to (uniform or quadratic)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="number of steps involved"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="eta used to control the variances of sigma",
    )
    parser.add_argument('--sample_step', type=int, default=3, help='Total sampling steps')
    parser.add_argument('--t', type=int, default=400, help='Sampling noise scale')
    parser.add_argument('--evaluation', action="store_true", help='Evaluation')
    parser.add_argument("--sequence", action="store_true")

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, "logs", args.doc)

    # parse config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    tb_path = os.path.join(args.exp, "tensorboard", args.doc)


    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("level {} not supported".format(args.verbose))

    os.makedirs(args.log_path.replace('logs', 'results_logs'), exist_ok=True)
    handler1 = logging.FileHandler(os.path.join(args.log_path.replace('logs', 'results_logs'), "sampling_logs.txt"))
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    if args.sample:
        os.makedirs(os.path.join(args.log_path, "image_samples"), exist_ok=True)
        args.image_folder = os.path.join(
            args.log_path, "image_samples", args.image_folder
        )
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            if not (args.fid or args.interpolation):
                overwrite = False
                if args.ni:
                    overwrite = True
                else:
                    response = input(
                        f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)"
                    )
                    if response.upper() == "Y":
                        overwrite = True

                if overwrite:
                    shutil.rmtree(args.image_folder)
                    os.makedirs(args.image_folder)
                else:
                    print("Output image folder exists. Program halted.")
                    sys.exit(0)
    else:
        print('Do not train the model')

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))

    try:
        runner = Diffusion(args, config)
        print('Sampling...')
        # if not args.evaluation:
        #     runner.image_editing_sample()
        # else:
        runner.evaluation_sample(logging)
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
