import sys
from pathlib import Path
# 경로설정
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

from fastapi import FastAPI, HTTPException
from models import InpaintModel, InpaintResponse

import torch
from inpainting_model.lama_inpaint import inpaint_img_with_lama
from inpainting_model.utils import load_img_to_array, save_array_to_img

# FastAPI 인스턴스를 생성.
app = FastAPI()

@app.get("/")
def read_root():
    return {"Inpainting model API running..."}

@app.post("/api/inpaint/", response_model=InpaintResponse)
async def inpaint(InpaintModel:InpaintModel):

    # CUDA가 사용 가능여부 체크
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 입력 이미지의 경로를 설정.
    img_path = str(current_dir.parent) +'/inpainting_model/input_img/'+ InpaintModel.img
    # 결과 이미지의 저장 경로를 설정.
    out_dir = str(current_dir.parent) + '/inpainting_model/results/' +  Path(img_path).stem +'_result.png'

    # 입력 이미지 경로가 실제 파일이 아닌 경우를 확인.
    if not Path(img_path).is_file():
        raise HTTPException(status_code=400, detail="Image not found")
    # 주어진 경로에서 이미지를 배열로 로드.
    img = load_img_to_array(img_path)
    
    # LaMa 인페인팅 모델 설정 파일 경로를 정의.
    lama_config ="../inpainting_model/lama/configs/prediction/default.yaml"
    # 사전 훈련된 모델의 체크포인트 파일 경로를 정의.
    lama_ckpt = "../inpainting_model/pretrained_models/big-lama"
    # 인페인팅을 할 영역의 좌표를 받습니다.
    coord = InpaintModel.coord
    
     # 모델을 사용하여 이미지 인페인팅을 수행.
    img_inpainted = inpaint_img_with_lama(img, lama_config, lama_ckpt, coord, device=device)
    # 인페인팅 처리된 이미지를 저장.
    save_array_to_img(img_inpainted, out_dir)

    return InpaintResponse(status="success", result_path=str(out_dir))

