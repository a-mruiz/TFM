from model import dataloader
import numpy as np
import cv2
import base64
import requests
import json

rgb,_=dataloader.read_rgb("test_data/middlebury/rgb/9_30.png")
gt,_=dataloader.read_depth("test_data/middlebury/gt/9_30_0.png")


DIAMOND_KERNEL_11 = np.asarray(
[
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
], dtype=np.uint8)         
DIAMOND_KERNEL_9 = np.asarray(
        [
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
        ], dtype=np.uint8)
mask=(gt==0).astype(np.uint8)

degraded_mask = cv2.dilate(mask, DIAMOND_KERNEL_9)

degraded_mask=np.expand_dims(degraded_mask,2)
degraded_depth=(1-degraded_mask)*gt     
   
rgb=cv2.imencode('.png', rgb)[1]
degraded_depth=cv2.imencode('.png', degraded_depth)[1]
gt=cv2.imencode('.png', gt)[1]
items = {"rgb": base64.b64encode(rgb).decode("utf-8"), "d": base64.b64encode(degraded_depth).decode("utf-8"), 'gt':base64.b64encode(gt).decode("utf-8")}


items = json.dumps(items)
#Create the requets to the server

endpoint="http://127.0.0.1:5001/score"
endpoint="https://tfm-endpoint.westeurope.inference.ml.azure.com/score"

newHeaders = {'Content-type': 'application/json', 'Accept': 'text/plain'}
print(f"Sending request to {endpoint}...")
response = requests.post(endpoint,
                         data=items,
                         headers=newHeaders)


print("Response:")
try:    
    print(response.json())
    print('{0} {1}'.format(response.status_code, json.loads(response.text)["message"]))             
except:
    pass




















