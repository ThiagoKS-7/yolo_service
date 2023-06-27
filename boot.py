import os
from yolo.conf.detect import YOLO_img_to_base64_response as yolo_b64
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
folder = os.path.abspath('.') + "/yolo/weights"

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/image")
async def image(images: UploadFile):
    return yolo_b64.predict(images)



if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=5000)
