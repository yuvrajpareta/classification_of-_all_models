import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_ngrok import run_with_ngrok
import pickle


app = Flask(__name__)
model_RF = pickle.load(open('/content/drive/My Drive/Colab Notebooks/RF_project-7(minor).pkl','rb'))
model_DT = pickle.load(open('/content/drive/My Drive/Colab Notebooks/DT_project7minor.pkl','rb'))
model_KNN = pickle.load(open('/content/drive/My Drive/Colab Notebooks/KNN_project7minor.pkl','rb'))
model_SVM = pickle.load(open('/content/drive/My Drive/Colab Notebooks/SVM_project7minor.pkl','rb'))
model_NB = pickle.load(open('/content/drive/My Drive/Colab Notebooks/NB_project7minor.pkl','rb'))
run_with_ngrok(app)

@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    
    cs = int(request.args.get('creditscore'))
    age = int(request.args.get('age'))
    ten = int(request.args.get('tenure'))
    bal = float(request.args.get('balance'))
    pro = int(request.args.get('products'))
    sal = float(request.args.get('salary'))
    con = (request.args.get('con'))

    if con=="France":
      con = 0
    elif con=="Spain":
      con = 1
    else:
      con = 2

    gen = (request.args.get('gen'))
    
    if gen=="Male":
      gen = 1
    else:
      gen = 0

    cr = (request.args.get('cr'))

    if cr=="Yes":
      cr = 1
    else:
      cr = 0

    active = (request.args.get('active'))

    if active=="Male":
      active = 1
    else:
      active = 0

# CreditScore	Geography	Gender	Age	Tenure	Balance	NumOfProducts	HasCrCard	IsActiveMember	EstimatedSalary	
    Model = (request.args.get('Model'))

    if Model=="Random Forest Classifier":
      prediction = model_RF.predict([[cs, con, gen, age, ten, bal, pro, cr, active, sal]])

    elif Model=="Decision Tree Classifier":
      prediction = model_DT.predict([[cs, con, gen, age, ten, bal, pro, cr, active, sal]])

    elif Model=="KNN Classifier":
      prediction = model_KNN.predict([[cs, con, gen, age, ten, bal, pro, cr, active, sal]])

    elif Model=="SVM Classifier":
      prediction = model_SVM.predict([[cs, con, gen, age, ten, bal, pro, cr, active, sal]])

    else:
      prediction = model_NB.predict([[cs, con, gen, age, ten, bal, pro, cr, active, sal]])

    
    if prediction == [0]:
      return render_template('index.html', prediction_text='The person exited', extra_text ="-> Prediction by " + Model)
    
    else:
      return render_template('index.html', prediction_text='The person not exited yet', extra_text ="-> Prediction by " + Model)

if__name__=="__main__":
  app.run(debug=True)


