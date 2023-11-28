from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd

application = Flask(__name__)  #same as app = express()
app=application

#import ridge regressor model and standard scaler model
standard_scaler = pickle.load(open('models/scaler2.pkl', 'rb'))
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))

# route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predictdata", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        Temp = float(request.form.get("Temperature"))
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))

        #scaling the new data that we got
        new_data_scaled = standard_scaler.transform([[Temp, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        res = ridge_model.predict(new_data_scaled)

        return render_template("home.html", result = res[0])
    else:
        return render_template("home.html")
        
# entry point(server will start on 127.0.0.1:5000)
if __name__=="__main__":
    app.run(host="0.0.0.0")
