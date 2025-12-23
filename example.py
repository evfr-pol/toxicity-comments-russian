import json

import requests

url = "http://127.0.0.1:5000/invocations"

payload = {
    "dataframe_split": {
        "columns": ["text"],
        "data": [["Ты ужасный человек"], ["Спасибо за отличный ответ"]],
    }
}

response = requests.post(
    url,
    headers={"Content-Type": "application/json"},
    data=json.dumps(payload),
)

print(response.json())
