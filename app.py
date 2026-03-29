from flask import Flask,request,render_template
import pickle
import numpy as np

app=Flask(__name__)

with open("tm1.pkl","rb") as model_file:
	ml_model=pickle.load(model_file)

@app.route("/")
def index():
	return render_template("index.html")

@app.route("/process",methods=['POST'])
def process():
	if request.method=='POST':
		features=[
			float(request.form['Transaction_Amount_INR']),
			float(request.form['Is_International'])

		]

		input_data=np.array(features).reshape(1, -1)

		result=ml_model.predict(input_data)[0]
		detection="Credit Card Fraud Detection:Yes" if result== 1 else "Credit Card Fraud Detection:No"
		print("Predicted Outcome=",detection)

		return render_template("result.html",prediction=detection)



if __name__=="__main__":
	app.run(debug=True)