#http://localhost:5000/<app route>

# Import Files
from flask import ( Flask, render_template, request )
import pandas as pd
import numpy as np
from sklearn.externals import joblib

# Create the application instance
app = Flask(__name__, template_folder="templates")

# Function to predict the iris flower
def pred_iris_flower(input_arr):
   lr = joblib.load('model.pkl')
   iris_prediction = lr.predict(input_arr)

   if ( iris_prediction == [0] ) :
      iris_prediction = 'SETOSA'
   elif ( iris_prediction == [1] ) :
      iris_prediction = 'VIRGINICA'
   elif ( iris_prediction == [2] ) :
      iris_prediction = 'VERSICOLOR'
   else:
      iris_prediction = 'NA'

   print('Result from function -->', iris_prediction)
   return iris_prediction;

# Create a URL route in our application for "/"
@app.route('/')
def student():
   return render_template('index.html')

# URL for showing the result "/"
@app.route('/result',methods = ['POST', 'GET'])
def result():

   if request.method == 'POST':
      #Receives the form data
      result = request.form
      values = [request.form[k] for k in request.form]

      #fields = [k for k in request.form]                                      
      #data = dict(zip(fields, values))

      input_arr = np.array(values).reshape(1, -1)

   """
    This function just responds to the browser ULR
    localhost:5000/results.html
    return: the rendered template 'result.html'
   """
    
   return render_template("result.html",result = result , pred = pred_iris_flower(input_arr))

# Run the application
if __name__ == '__main__':
   app.run(debug = True)



