import tensorflow as tf
from tensorflow import keras
import cv2
import os
from flask import Flask,render_template,request,make_response
import h5py
import tensorflow_hub as hub
import json
from datetime import datetime
import pdfkit
import matplotlib.pyplot as plt

#import wkhtmltopdf
#import pathlib

app=Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")
    
@app.route("/covid")
def covid():
    return render_template("covid.html")

@app.route("/brain_tumor")
def brain():
    return render_template("brain_tumor.html")
 
@app.route("/pneumonia")
def pneumonia():
    return render_template("pneumonia.html")

@app.route("/skin_cancer")
def skin():
    return render_template("skin_cancer.html")
    
@app.route("/predict_covid",methods=["POST","GET"])
def predict_covid():
    if request.method=="POST":
        f=request.files["myfile"]
        pt=request.form["pt_id"]
        
        basepath=os.path.dirname(__file__)
        file_path=os.path.join(
            
           basepath,"uploads")
            
        f.save(file_path)
        
        #pt_id=input("Please Enter Patient ID")
        
        model=keras.models.load_model("model/covid")
        
        
        img_arr=cv2.imread(file_path)
        img_res=cv2.resize(img_arr,(224,224))
        x=img_res/255
        
        
        data=model.predict(x.reshape(1,224,224,3))
        
        if data[0,0]<0.5:
            test="negative"
        else:
            test="positive"
            
        date=datetime.now()
        
        prob_pos=data[0,0]
        prob_neg=1-data[0,0]
        
        
        res=render_template("pdf.html",data=["COVID-19",test,prob_pos,prob_neg,date,pt,"Dr. Ganesh","Radiologist"])
        config=pdfkit.configuration(wkhtmltopdf="wkhtmltopdf/bin/wkhtmltopdf.exe")
        options = {'enable-local-file-access': None}
        
        response_string=pdfkit.from_string(res,False,configuration=config)
        
        response=make_response(response_string)
        
        response.headers["Content-Type"]="application/pdf"
        
        response.headers["Content-Disposition"]="attachment;filename="+str(pt)+".pdf"
        
        return response
        
        
    else:
        return "something went wrong"

@app.route("/predict_brain_tumor",methods=["POST","GET"])
def predict_brain_tumor():

    if request.method=="POST":
        f=request.files["myfile"]
        pt=request.form["pt_id"]
        
        basepath=os.path.dirname(__file__)
        file_path=os.path.join(
            
           basepath,"uploads")
            
        f.save(file_path)
        
        model=keras.models.load_model("model/brain")
        
        
        img_arr=cv2.imread(file_path)
        img_res=cv2.resize(img_arr,(224,224))
        x=img_res/255
        
        data=model.predict(x.reshape(1,224,224,3))
        
        if data[0,0]<0.5:
            test="negative"
        else:
            test="positive"
            
        date=datetime.now()
        
        prob_pos=data[0,0]
        prob_neg=1-data[0,0]
        
        
        res=render_template("pdf.html",data=["BRAIN TUMOR",test,prob_pos,prob_neg,date,pt,"Dr. Santosh","Neurologist"])
        config=pdfkit.configuration(wkhtmltopdf="wkhtmltopdf/bin/wkhtmltopdf.exe")
        options = {'enable-local-file-access': None}
        
        response_string=pdfkit.from_string(res,False,configuration=config)
        
        response=make_response(response_string)
        
        response.headers["Content-Type"]="application/pdf"
        
        response.headers["Content-Disposition"]="attachment;filename="+str(pt)+".pdf"
        
        return response
        
    else:
        return "something went wrong"


@app.route("/predict_skin_cancer",methods=["POST","GET"])
def predict_skin_cancer():
    if request.method=="POST":
        f=request.files["myfile"]
        pt=request.form["pt_id"]
        
        basepath=os.path.dirname(__file__)
        file_path=os.path.join(
            
           basepath,"uploads")
            
        f.save(file_path)
        
        model=keras.models.load_model("model/skin_cancer")
        
        
        img_arr=cv2.imread(file_path)
        img_res=cv2.resize(img_arr,(224,224))
        x=img_res/255
        
        data=model.predict(x.reshape(1,224,224,3))
        
        if data[0,0]<0.5:
            test="Benign"
        else:
            test="Malignant"
            
        date=datetime.now()
        
        prob_pos=data[0,0]
        prob_neg=1-data[0,0]
        
        
        res=render_template("pdf.html",data=["SKIN CANCER",test,prob_pos,prob_neg,date,pt,"Dr. Gaurav","Dermatologist"])
        config=pdfkit.configuration(wkhtmltopdf="wkhtmltopdf/bin/wkhtmltopdf.exe")
        options = {'enable-local-file-access': None}
        
        response_string=pdfkit.from_string(res,False,configuration=config)
        
        response=make_response(response_string)
        
        response.headers["Content-Type"]="application/pdf"
        
        response.headers["Content-Disposition"]="attachment;filename="+str(pt)+".pdf"
        
        return response
    else:
        return "something went wrong"

@app.route("/predict_pneumonia",methods=["POST","GET"])
def predict_pneumonia():
    if request.method=="POST":
        f=request.files["myfile"]
        pt=request.form["pt_id"]
        
        basepath=os.path.dirname(__file__)
        file_path=os.path.join(
            
           basepath,"uploads")
            
        f.save(file_path)
        
        model=keras.models.load_model("model/pneumonia")
        
        
        img_arr=cv2.imread(file_path)
        img_res=cv2.resize(img_arr,(224,224))
        x=img_res/255
        
        data=model.predict(x.reshape(1,224,224,3))
        
        if data[0,0]<0.5:
            test="negative"
        else:
            test="positive"
            
        date=datetime.now()
        
        prob_pos=data[0,0]
        prob_neg=1-data[0,0]
        
        
        res=render_template("pdf.html",data=["PNEUMONIA",test,prob_pos,prob_neg,date,pt,"Dr. Ganesh","Radiologist"])
        config=pdfkit.configuration(wkhtmltopdf="wkhtmltopdf/bin/wkhtmltopdf.exe")
        options = {'enable-local-file-access': None}
        
        response_string=pdfkit.from_string(res,False,configuration=config)
        
        response=make_response(response_string)
        
        response.headers["Content-Type"]="application/pdf"
        
        response.headers["Content-Disposition"]="attachment;filename="+str(pt)+".pdf"
        
        return response
        
    else:
        return "something went wrong"

    
    
if __name__=="__main__":
    app.run("localhost","8080",use_reloader=False,debug=True)