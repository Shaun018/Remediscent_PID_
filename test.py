
import requests

url = "http://127.0.0.1:5000/predict"

image_path = (
    r"dataset\split\test\class_01_AMARYL_4MG_TABLETS"
    r"\!P31ERRM_M7RRSPBWHL80GEAAAA6WF.JPG"
)

with open(image_path, "rb") as image:
    response = requests.post(
        url,
        files={"image": image}
    )

print("Status:", response.status_code)
print("Response:")
print(response.json())
