from flask import Flask,render_template,request
import numpy as np
import pickle

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/prediction',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        salary = float(request.form['salary'])
        data = np.array([[age,salary]])
        model = pickle.load(open('model.pkl','rb'))
        purchase = model.predict(data)
        if purchase[0] == 0:
            msg = "Customer won't purchase"
        else:
            msg = "Customer will purchase"
    return render_template('index.html',msg=msg)

if __name__ =='__main__':
    app.run()