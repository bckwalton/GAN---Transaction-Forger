from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import ShowTell
import subprocess

app = Flask(__name__)
CORS(app)


@app.route('/', methods=["POST", "OPTION"])
def decision():
    results = ShowTell.print()
    print(results)
    return jsonify(results)

if __name__ == "__main__":
    app.run(host='0.0.0.0')
