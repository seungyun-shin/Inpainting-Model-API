import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

from fastapi import FastAPI, HTTPException
from models import InpaintModel, InpaintResponse

import torch
from inpainting_model.lama_inpaint import inpaint_img_with_lama
from inpainting_model.utils import load_img_to_array, save_array_to_img

app = FastAPI()

@app.get("/")
def read_root():
    return {"Inpainting model API running..."}

@app.post("/api/inpaint/", response_model=InpaintResponse)
async def inpaint(InpaintModel:InpaintModel):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_path = str(current_dir.parent) +'/inpainting_model/input_img/'+ InpaintModel.img
    out_dir = str(current_dir.parent) + '/inpainting_model/results/' +  Path(img_path).stem +'_result.png'
    if not Path(img_path).is_file():
        raise HTTPException(status_code=400, detail="Image not found")
    img = load_img_to_array(img_path)
    
    lama_config ="../inpainting_model/lama/configs/prediction/default.yaml"
    lama_ckpt = "../inpainting_model/pretrained_models/big-lama"
    coord = InpaintModel.coord
    img_inpainted = inpaint_img_with_lama(img, lama_config, lama_ckpt, coord, device=device)
    save_array_to_img(img_inpainted, out_dir)

    return InpaintResponse(status="success", result_path=str(out_dir))

