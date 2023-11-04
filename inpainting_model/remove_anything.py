import torch
import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
import argparse
from pathlib import Path
import json

from inpainting_model.lama_inpaint import inpaint_img_with_lama
from inpainting_model.utils import load_img_to_array, save_array_to_img


def setup_args(parser):
    parser.add_argument(
        "--input_img", type=str, required=True,
        help="Path to a single input img",
    )
    parser.add_argument(
        "--point_coords", type=str, required=True,
        help="The coordinate of the point list prompt.",
    )


if __name__ == "__main__":
    """Example usage:
    python remove_anything.py \
        --input_img sign.jpg \
        --point_coords "[[[50,138], [396,154], [397,208], [48,194]],[[196,211], [383,218], [385,352], [193,350]]]"
    """
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_path = str(current_dir.parent) +'/inpainting_model/input_img/'+ args.input_img
    out_dir = str(current_dir.parent) + '/inpainting_model/results/' +  Path(img_path).stem +'_result.png'

    img = load_img_to_array(img_path)

    lama_config ="../inpainting_model/lama/configs/prediction/default.yaml"
    lama_ckpt = "../inpainting_model/pretrained_models/big-lama"

    point_coords = json.loads(args.point_coords)

    img_inpainted = inpaint_img_with_lama(img, lama_config, lama_ckpt, point_coords, device=device)
    save_array_to_img(img_inpainted, out_dir)
