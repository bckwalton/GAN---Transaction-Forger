from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import ShowTell
import subprocess
import json

app = Flask(__name__)
CORS(app)


@app.route('/', methods=["POST", "OPTION"])
def decision():
    list = ShowTell.print().tolist()
    print(list)
    return json.dumps(list)

if __name__ == "__main__":
    app.run(host='0.0.0.0')
