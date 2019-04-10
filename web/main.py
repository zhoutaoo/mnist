import numpy as np
import recognizing
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/api/mnist', methods=['post'])
def mnist():
    input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)
    output1 = recognizing.regression(input)
    output2 = recognizing.convolutional(input)
    return jsonify(results=[output1, output2])


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=8000)
