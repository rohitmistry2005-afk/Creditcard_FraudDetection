from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load Model
with open("tm1.pkl", "rb") as model_file:
    model = pickle.load(model_file)


# Dashboard = Homepage
@app.route('/')
def dashboard():
    return render_template('dashboard.html')


# Transactions Page
@app.route('/transactions')
def transactions():
    return render_template('transactions.html')


# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():

    # Get form data
    amount = float(request.form['Transaction_Amount_INR'])
    international = float(request.form['Is_International'])

    # Create DataFrame (Fix sklearn warning)
    input_data = pd.DataFrame(
        [[amount, international]],
        columns=['Transaction_Amount_INR', 'Is_International']
    )

    # Model Prediction
    prediction = model.predict(input_data)[0]

    # Result mapping
    if prediction == 1:
        result = "Fraud Detected"
        risk = "High Risk"
    else:
        result = "Legitimate Transaction"
        risk = "Low Risk"

    print("Prediction:", prediction)
    print("Result:", result)

    # Send to results page
    return render_template(
        'results.html',
        prediction=prediction,   # send raw prediction
        result=result,
        risk=risk,
        amount=amount,
        transaction_type="International" if international == 1 else "Domestic"
    )


if __name__ == '__main__':
    app.run(debug=True)