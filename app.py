from flask import Flask, render_template, request
import pickle
import numpy as np
from numpy.core.numeric import outer
app = Flask(__name__)

list_of_model_pickles = ['CHW_Pump_1_Speed(%)_MODEL.pkl', 'FSP_MODEL.pkl', 'PSP_MODEL.pkl', 'NOC_MODEL.pkl'] # add any model pickle file here

@app.route('/')
def helloworld():
    return render_template('home.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    prediction_text_all_pickles = "Prediction: " 
    for model_file in list_of_model_pickles:
        f_pickle = open(model_file, 'rb')

        model = pickle.load(f_pickle)
        float_features=[float(x) for x in request.form.values()]
        final = [np.array(float_features)]
        # final = [['1', '2', '3', '4', '5', '6', '7', '8', '9', '90']]
        prediction = model.predict(final)
        if model_file == "CHW_Pump_1_Speed(%)_MODEL.pkl":
            prediction_text_all_pickles += f"CHW_Pump_1_Speed Predicted: {prediction}, \n"
        if model_file == "FSP_MODEL.pkl":
            prediction_text_all_pickles += f"FSP Predicted: {prediction} , \n"
        if model_file == "PSP_MODEL.pkl":
            prediction_text_all_pickles += f"PSP Predicted: {prediction} , \n"
        if model_file == "NOC_MODEL.pkl":
            prediction_text_all_pickles += f"NOC Predicted: {prediction}. \n"
        # f"{prediction} For {model_file} . \n"
        f_pickle.close()
        
    return render_template('predict.html',  pred=prediction_text_all_pickles)


if __name__ == '__main__':
    app.run(debug=True)
