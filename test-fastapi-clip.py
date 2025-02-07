import requests

url = "http://127.0.0.1:8000/clip_predict"
image_path = "elephant.webp"
cls_options = [
    "an elephant", 
    "a dog", 
    "a cat"
]

with open(image_path, "rb") as img_file:
    files = {"file": img_file}
    data = {"cls_options": cls_options}
    
    response = requests.post(url, files=files, data=data)
    print(response.json())