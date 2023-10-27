import torch
import sys
import argparse
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

from lama_inpaint import inpaint_img_with_lama
from utils import load_img_to_array, save_array_to_img, dilate_mask, show_mask, show_points, get_clicked_point


def setup_args(parser):
    parser.add_argument(
        "--input_img", type=str, required=True,
        help="Path to a single input img",
    )
    parser.add_argument(
        "--output_dir", type=str, required=False,
        default="./results",
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--point_coords", type=float, nargs='+', required=True,
        help="The coordinate of the point prompt, [coord_W coord_H].",
    )
    parser.add_argument(
        "--lama_config", type=str,
        default="./lama/configs/prediction/default.yaml",
        help="The path to the config file of lama model. "
             "Default: the config of big-lama",
    )
    parser.add_argument(
        "--lama_ckpt", type=str, required=False,
        default="./pretrained_models/big-lama",
        help="The path to the lama checkpoint.",
    )


if __name__ == "__main__":
    """Example usage:
    python remove_anything.py \
        --input_img FA_demo/FA1_dog.png \
        --coords_type key_in \
        --point_coords 750 500 \
        --point_labels 1 \
        --dilate_kernel_size 15 \
        --output_dir ./results \
        --sam_model_type "vit_h" \
        --sam_ckpt sam_vit_h_4b8939.pth \
        --lama_config lama/configs/prediction/default.yaml \
        --lama_ckpt big-lama 
    """
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img = load_img_to_array(args.input_img)

    img_stem = Path(args.input_img).stem
    out_dir = Path(args.output_dir) / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    img_inpainted_p = out_dir / f"inpainted_result_image.png"
    mask=0
    img_inpainted = inpaint_img_with_lama(img, mask, args.lama_config, args.lama_ckpt, device=device)
    save_array_to_img(img_inpainted, img_inpainted_p)
