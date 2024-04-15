from fastapi import FastAPI, UploadFile
from fastapi.exceptions import HTTPException
from redis import Redis
from rq import Queue, Worker, SimpleWorker
from model_job import *
from PIL import Image
import io
import requests
import time

# global sampler
# sampler = initial_point_e()
app = FastAPI()
redis_conn = Redis(host="localhost", port=6379)
task_queue = Queue("model_queue", connection=redis_conn)
# worker1 = Worker([task_queue], connection=redis_conn, name='model1')
# worker1.work()

def process_model(image, image_id):
    # pc = create_point_cloud(image, sampler)
    # voxel_grid, vsize = voxelization(pc)
    # write_vox_file(voxel_grid, image_id)
    data = requests.get("https://api.restful-api.dev/objects", id=image_id)
    time.sleep(20)
    print("filename: ", image.filename, "image_id: ",  image_id)
    print(data)


@app.get("/")
def read_main():
    return {"message": "This is your main app"}


@app.post("/new_model", status_code=202)
async def post_model(image: UploadFile, image_id: str):
    try:
        image_data = await image.read()
        img = Image.open(io.BytesIO(image_data))
        task_queue.enqueue(process_model, (img, image_id))
    except Exception:
        raise HTTPException(status_code=400)
    return {"message": "Request accepted"}



