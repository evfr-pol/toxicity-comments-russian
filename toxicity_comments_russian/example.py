import json

import requests

url = "http://127.0.0.1:5000/invocations"

payload = {
    "dataframe_split": {
        "columns": ["text"],
        "data": [["Сдохни, тварь"], ["Спасибо за отличный ответ"], ["Абоба"], ["Ты дурак"]],
    }
}

response = requests.post(
    url,
    headers={"Content-Type": "application/json"},
    data=json.dumps(payload),
)

print(response.json())
