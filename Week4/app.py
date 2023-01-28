import numpy as np
from flask import Flask, request,render_template
import pickle
import sklearn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    to_predict_list = request.form.to_dict()
    to_predict_list = list(int(float(x)) for x in to_predict_list.values())
    to_predict_list = list(map(int, to_predict_list))
    norm = MinMaxScaler().fit([to_predict_list])
    features = pd.DataFrame(norm.transform([to_predict_list]))
    final_features = np.array(features).reshape(1, -1)

    prediction = model.predict(final_features)

    if prediction[0] == 1:
        output = 'Not Cancelled'
    else:
        output = 'Cancelled'

    return render_template('index.html', prediction_text='Your booking status: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)