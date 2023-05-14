from flask import Flask,request,jsonify
from ML import pipeline_transformer, preprocess, predict_variety, FrequencyEncoder
import pickle
import joblib
import json
import lightgbm as lgb



app= Flask('app')

@app.route('/', methods=['GET'])
def home():
    return 'Running in homepage!!'


@app.route('/test', methods=['GET'])
def test():
    return 'Running!!'

@app.route('/predict', methods=['POST'])
def predict():
    wine= request.get_json()
    print(type(wine))
    
    # Load the model using the built-in LGBMClassifier constructor
    clf = lgb.LGBMClassifier()
     # Load the pre-trained model using joblib
    clf = joblib.load('./model_files/lgbm_model.joblib')
    
    predictions= json.dumps(int(predict_variety(wine, clf)))

    options = {
    0: 'Grüner Veltliner',
    1: 'Zinfandel',
    2: 'Tempranillo',
    3: 'Nebbiolo',
    4: 'Merlot',
    5: 'Sauvignon Blanc',
    6: 'Chardonnay',
    7: 'Sparkling Blend',
    8: 'Sangiovese',
    9: 'Pinot Noir',
    10: 'Rosé',
    11: 'Riesling',
    12: 'Malbec',
    13: 'Portuguese Red',
    14: 'Red Blend',
    15: 'Syrah',
    16: 'Cabernet Sauvignon',
    17: 'Gewürztraminer',
    18: 'Bordeaux-style White Blend',
    19: 'Champagne Blend',
    20: 'Cabernet Franc',
    21: 'Rhône-style Red Blend',
    22: 'White Blend',
    23: 'Bordeaux-style Red Blend',
    24: 'Pinot Gris',
    25: 'Portuguese White',
    26: 'Gamay',
    27: 'Pinot Grigio'
}


    result = {
        'variety_prediction': options[int(predictions)]
    }

    print(type(result))

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)    