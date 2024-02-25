from flask import  Flask,request,jsonify
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello World"

@app.route('/predict',methods=['POST'])
def predict():
    N = request.form.get('N')
    P = request.form.get('P')
    K = request.form.get('K')
    Temperature = request.form.get('Temperature')
    Humidity = request.form.get('Humidity')
    pH = request.form.get('pH')

    input_query =np.array([[N,P,K,Temperature,Humidity,pH]])

    result = model.predict(input_query)[0]

    return jsonify({'result':result})
# if __name__ == '__main__':
#     app.run(debug=True)