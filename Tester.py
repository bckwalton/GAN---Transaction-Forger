import requests

res = requests.post('http://localhost:5000', json={'code': 'Gay', 'lang': 'java'})

if res.ok:
    print("POST Success")
    print(res.json())
