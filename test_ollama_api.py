import requests

payload = {
    "model": "llama3.1:latest",
    "prompt": "hello",
    "stream": False
}

try:
    response = requests.post("http://localhost:11434/api/generate", json=payload)
    print("Status code:", response.status_code)
    print("Response:", response.text)
except Exception as e:
    print("Error:", e)
