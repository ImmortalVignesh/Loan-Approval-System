import numpy as np

from flask import Flask, request, render_template
import re
from sklearn.tree import DecisionTreeClassifier
import pickle as pickle

app = Flask(__name__)
model = pickle.load(open("LoanFinal.pickle", "rb"))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    print("IN predict")
    input_data = [float(x) for x in request.form.values()]
    final_features = [np.array(input_data)]
    prediction = model.predict(final_features)
    output = np.around(prediction)
    if output == 1:
        output = "Yes"
    elif output == 0:
        output = "No"
    print(output)
    return render_template('index.html', prediction_text='Loan Approval Status :{}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)
