import numpy as np
from flask import Flask, request,render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('finalized_model.sav', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    int_features = [int(x) for x in request.form.values()]
    input = pd.DataFrame([int_features])
    prediction = model.predict(input)
    
    if prediction==1:
        output='survived'
        probab=model.predict_proba(input)[0][1]*100
    else:
        output='died'
        probab=model.predict_proba(input)[0][0]*100

    return render_template('index.html', prediction_text=f'This passenger had  {output} with chance of: {round(probab) }%' )

if __name__ == "__main__":
     app.run(debug=True)
