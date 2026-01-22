
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('Model/brand_recommender.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        float(request.form['Age']),
        float(request.form['AnnualIncome']),
        float(request.form['PurchaseAmount']),
        float(request.form['PurchaseFrequency']),
        float(request.form['LoyaltyScore'])
    ]

    prediction = model.predict([features])[0]

    if prediction == 1:
        result = "Customer is likely to recommend the brand"
    else:
        result = "Customer is unlikely to recommend the brand"

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
