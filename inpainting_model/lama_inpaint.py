import os
import sys
import numpy as np
import torch
import yaml
import glob
import argparse
from omegaconf import OmegaConf
from pathlib import Path
from typing import List
import cv2

# 경로설정
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parent / "lama"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "lama" / "models"))

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.data import pad_tensor_to_modulo
from .utils import load_img_to_array, save_array_to_img

# 데코레이터를 사용하여 해당 함수가 gradient를 계산하지 않도록 설정합니다.
@torch.no_grad()
def inpaint_img_with_lama(
        img: np.ndarray,
        config_p: str,
        ckpt_p: str,
        coord: List,
        mod=8,
        device="cuda",
): 
    # 인페인팅 함수 정의
    polygon_points_list = coord
    mask = create_mask_from_polygon(img.shape, polygon_points_list)
    # 마스크된 영역 확인용, 마스크 이미지 저장
    cv2.imwrite(str(current_dir.parent) + '/inpainting_model/results/mask_image.jpg', mask)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite(str(current_dir.parent) + '/inpainting_model/results/mask_image_origin.jpg', masked_img)

    # 마스크는 2차원이어야 함을 확인
    assert len(mask.shape) == 2
    if np.max(mask) == 1:
        # 마스크 값을 0과 255 사이로 스케일링
        mask = mask * 255
    img = torch.from_numpy(img).float().div(255.)
    mask = torch.from_numpy(mask).float()
    predict_config = OmegaConf.load(config_p)
    predict_config.model.path = ckpt_p
    device = torch.device(device)

    train_config_path = os.path.join(
        predict_config.model.path, 'config.yaml')

    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = os.path.join(
        predict_config.model.path, 'models',
        predict_config.model.checkpoint
    )
    model = load_checkpoint(
        train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    if not predict_config.get('refine', False):
        model.to(device)

    batch = {}
    # 이미지를 배치 형태로 변환
    batch['image'] = img.permute(2, 0, 1).unsqueeze(0)
    # 마스크를 배치 형태로 변환
    batch['mask'] = mask[None, None]
    unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
    batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
    batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)
    batch = move_to_device(batch, device)
    batch['mask'] = (batch['mask'] > 0) * 1

    batch = model(batch)
    
    cur_res = batch[predict_config.out_key][0].permute(1, 2, 0)
    cur_res = cur_res.detach().cpu().numpy()

    if unpad_to_size is not None:
        orig_height, orig_width = unpad_to_size
        cur_res = cur_res[:orig_height, :orig_width]

    cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
    return cur_res

def create_mask_from_polygon(image_shape, polygon_points_list):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for polygon_points in polygon_points_list:
        cv2.fillPoly(mask, [np.array(polygon_points, dtype=np.int32)], 255)
    return mask

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