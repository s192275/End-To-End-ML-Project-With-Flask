import numpy as np
from flask import Flask, request, render_template, url_for
import pickle
import pymysql

def sql_connector():
    conn = pymysql.connect(user = 'root', password = 'root', db = 'patient_infos', host = 'localhost')
    c = conn.cursor()
    return conn, c

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/info")
def user_info():
    return render_template("user_info.html")

@app.route("/predict", methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        conn, c = sql_connector()
        float_features = [float(x) for x in request.form.values()]
        features = [np.array(float_features)]
        prediction = model.predict(features)
        age = float_features[0]
        sex = float_features[1]
        cp = float_features[2]
        trestbps = float_features[3]
        chol = float_features[4]
        fbs = float_features[5]
        restecg = float_features[6]
        thalach = float_features[7]
        exang = float_features[8]
        oldpeak = float_features[9]
        slope = float_features[10]
        ca = float_features[11]
        thal = float_features[12]
        c.execute("INSERT INTO patients (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal))
        conn.commit()
        conn.close()
        c.close()
        if prediction == 1:
            a = "You probably have a heart disase!..."
            return render_template("user_info.html", prediction_text = a)
        if prediction == 0:
            b = "You probably don't have a heart disase!..."
            return render_template("user_info.html", prediction_text = b)
    return render_template("user_info.html")

if __name__ == "__main__":
    app.run(debug=True)