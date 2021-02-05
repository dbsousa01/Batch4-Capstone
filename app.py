from flask import Flask, request, jsonify

# Custom imports
import utils.modelling as md

app = Flask(__name__)

# Load the model
pipeline, columns, dtypes = md.load_model()


@app.route('/predict', methods=['POST'])
def predict():
    prediction = 0.5
    return jsonify({
        'prediction': prediction
    })


if __name__ == "__main__":
    app.run(debug=True)
