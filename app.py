import pandas as pd
from flask import Flask, request, jsonify

# Custom imports
import utils.modelling as md
import data_processing.processing as pc

app = Flask(__name__)

# Load the model
pipeline, columns, dtypes = md.load_model()


@app.route('/should_search', methods=['POST'])
def predict():
    payload = request.get_json()
    obs = pd.DataFrame([payload], columns=columns)
    obs_processed = pc.create_time_features(obs).astype(dtype=dtypes)

    prediction = pipeline.predict(obs_processed)[0]
    return jsonify({
        'outcome': bool(prediction)
    })


if __name__ == "__main__":
    app.run(debug=True)
