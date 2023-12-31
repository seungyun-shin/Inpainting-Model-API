# Inpaint Model API: Image Inpainting Model
- 해당 프로젝트는 Fast API를 사용하여 만들어진 Inpainting 서비스 입니다. 이미지와 마름모 네변의 좌표를 제공하면, 해당 모델은 이미지에서 지정된 마름모 영역을 완벽히 지우고 자연스럽게 복원하는 기능을 제공합니다. 

## 📜 Installation
Requires `python>=3.8`
```bash
python -m pip install torch torchvision torchaudio
python -m pip install -r lama/requirements.txt 
```
In Windows, we recommend you to first install [miniconda](https://docs.conda.io/en/latest/miniconda.html) and 
open `Anaconda Powershell Prompt (miniconda3)` as administrator.
Then pip install [./lama_requirements_windows.txt](lama_requirements_windows.txt) instead of 
[./lama/requirements.txt](lama%2Frequirements.txt).

### 💡 Usage
- [해당 링크](https://drive.google.com/drive/folders/1wpY-upCo4GIW4wVPnlMh_ym779lLIG2A?usp=sharing)에서 `/big-lama` 모델 체크포인트 폴더를 다운로드해주세요.
- 다운로드한 체크포인트 폴더를 `./pretrained_models` 디렉토리에 넣어주세요.

Specify an image and a four points, and Model will remove the image area.
```bash
python remove_anything.py \
    --input_img sign.jpg \
    --point_coords "[[[50,138], [396,154], [397,208], [48,194]],[[196,211], [383,218], [385,352], [193,350]]]"
```

## 💡 API 
- `app/` 경로에서 Fast API 실행
```bash
uvicorn main:app --reload
```

## 💡 ENDPOINT
- Inpaint 요청 POST : `/api/inpaint/`
POST BODY 형식 예시
```bash
{
  "img": "sign.jpg",
  "coord": [
    [[50,138], [396,154], [397,208], [48,194]],
    [[196,211], [383,218], [385,352], [193,350]]
    ]
}
```

## 💡 DOCKER
- Dockerfile dir : `/inpainting_model/lama/docker/Dockerfile`
- Check if the gpu is available before
- If you can't use gpu, remove gpu parameter when running Docker container
```bash
$ docker build -t {image_name} .
$ docker run --gpus all -it -p 8000:8000 -v (pwd):/workspace --name {container_name} {image_name} 

# Runing Fast API server
$ uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 💡 Example Demo
<table>
  <tr>
    <td><img src="./inpainting_model/example/sign.jpg" width="100%"></td>
    <td><img src="./inpainting_model/example/sign_masked.jpg" width="100%"></td>
    <td><img src="./inpainting_model/example/sign_result.png" width="100%"></td>
  </tr>
</table>


