from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

#Load the model
with open('default_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html') #send index.html to the browser

#Prediction Fuction
def ValuePredictor(list_to_predict):
    to_predict = np.array(list_to_predict).reshape(1,-1)
    result = model.predict(to_predict)
    return result[0]

#Request Handling for Prediction
@app.route('/result', methods=['post'])
def result():
    if request.method == 'POST':
        list_to_predict = request.form.to_dict() #Convert form data into dictionary
        list_to_predict = list(list_to_predict.values()) #get only values
        list_to_predict = list(map(float, list_to_predict)) #convert to float

        result = ValuePredictor(list_to_predict) #Call predictor fucntion

        prediction = "❌ Customer is likely to default!" if result == 1 else "✅ Customer is financially stable."
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)