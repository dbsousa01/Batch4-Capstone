import pandas as pd
from flask import Flask, request, jsonify
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model, IntegerField,
    FloatField, TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict

# Custom imports
import utils.modelling as md
import data_processing.processing as pc


########################################
# Begin database stuff

DB = SqliteDatabase('predictions.db')


class Prediction(Model):
    observation_id = TextField(unique=True)
    observation = TextField()
    proba = FloatField()
    predict = IntegerField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

# End database stuff
########################################
# Load the model

pipeline, columns, dtypes = md.load_model()

########################################
# Begin webserver stuff
app = Flask(__name__)


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
