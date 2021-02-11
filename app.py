import os
import pandas as pd
from flask import Flask, request, jsonify
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model, IntegerField,
    FloatField, TextField, IntegrityError
)
from playhouse.db_url import connect

# Custom imports
import utils.modelling as md
import data_processing.processing as pc


########################################
# Begin database stuff

DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')


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


@app.route('/should_search/', methods=['POST'])
def predict():
    obs_dict = request.get_json()
    print("Req received: {}".format(obs_dict))
    _id = obs_dict["observation_id"]

    obs = pd.DataFrame([obs_dict], columns=columns)
    obs_processed = pc.create_time_features(obs).astype(dtype=dtypes)
    obs_processed = pc.build_features(obs_processed)

    prediction = pipeline.predict(obs_processed)[0]
    proba = pipeline.predict_proba(obs_processed)[0, 1]

    response = {}

    # Save the prediction in the DB
    p = Prediction(
        observation_id=_id,
        proba=proba,
        predict=prediction,
        observation=obs_dict
    )
    try:
        p.save()
    except IntegrityError as e:
        error_msg = "ERROR: Observation ID: '{}' already exists".format(_id)
        response["error"] = error_msg
        print(e)
        DB.rollback()
        return jsonify(response)

    response = {'outcome': bool(prediction)}
    print(response)
    return jsonify(response)


@app.route('/search_result/', methods=['POST'])
def update():
    obs = request.get_json()
    print("Req received: {}".format(obs))

    try:
        p = Prediction.get(Prediction.observation_id == obs['observation_id'])
        p.true_class = obs['outcome']
        p.save()
        response = {
            "observation_id": p.observation_id,
            "predicted_outcome": bool(p.predict),
            "outcome": bool(p.true_class)
        }
        print(response)
        return jsonify(response)

    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs['observation_id'])
        print(error_msg)
        return jsonify({'error': error_msg})


if __name__ == "__main__":
    app.run(debug=True)
