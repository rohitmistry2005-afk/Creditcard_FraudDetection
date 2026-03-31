from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# -------------------------------
# Load & Train Model
# -------------------------------

df = pd.read_csv(r"C:\Users\User\Documents\GitHub\Creditcard_FraudDetection\indian_credit_card_fraud_dataset_5000.csv")

# Drop unwanted columns
if 't_id' in df.columns:
    df.drop('t_id', axis=1, inplace=True)

if 'Transaction_Time' in df.columns:
    df.drop('Transaction_Time', axis=1, inplace=True)

# Feature Engineering
df['cust_txn_count'] = df.groupby('c_id')['c_id'].transform('count')
df['cust_avg_amt'] = df.groupby('c_id')['Transaction_Amount_INR'].transform('mean')

df.drop('c_id', axis=1, inplace=True)

# Convert Yes/No
df['Is_International'] = df['Is_International'].map({
    'Yes': 1,
    'No': 0
})

# One Hot Encoding
df = pd.get_dummies(df, columns=[
    'Device_Type',
    'Transaction_City',
    'Merchant_Category',
    'Card_Type',
    'Bank'
], drop_first=True)

# Features & Target
X = df.drop('Is_Fraud', axis=1)
y = df['Is_Fraud']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save column structure
model_columns = X.columns


# -------------------------------
# Existing Routes (UNCHANGED)
# -------------------------------

@app.route('/')
def dashboard():
    return render_template('dashboard.html')


@app.route('/transactions')
def transactions():
    return render_template('transactions.html')


@app.route('/flaggedalerts')
def flaggedalerts():
    return render_template('flaggedalerts.html')


@app.route('/analytics')
def analytics():
    return render_template('analytics.html')


@app.route('/rules')
def rules():
    return render_template('rules.html')


@app.route('/settings')
def settings():
    return render_template('settings.html')


# -------------------------------
# Prediction Route (UPDATED)
# -------------------------------

@app.route('/predict', methods=['POST'])
def predict():

    try:

        data = request.form.to_dict()
        input_df = pd.DataFrame([data])

        # Convert numeric safely
        input_df['Transaction_Amount_INR'] = input_df['Transaction_Amount_INR'].astype(float)

        # Convert Yes/No
        input_df['Is_International'] = input_df['Is_International'].map({
            '1': 1,
            '0': 0,
            'Yes': 1,
            'No': 0
        }).astype(int)

        # Feature Engineering
        input_df['cust_txn_count'] = 1
        input_df['cust_avg_amt'] = input_df['Transaction_Amount_INR']

        # One Hot Encoding
        input_df = pd.get_dummies(input_df)

        # Add missing columns
        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[model_columns]

        # Scale
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        # Result Mapping
        if prediction == 1:
            result = "Fraud Detected"
            risk = "High Risk"
        else:
            result = "Legitimate Transaction"
            risk = "Low Risk"

        return render_template(
            'results.html',
            prediction=prediction,
            result=result,
            risk=risk,
            probability=round(probability * 100, 2),
            amount=float(input_df['Transaction_Amount_INR'].values[0])
        )

    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    app.run(debug=True)