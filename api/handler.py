import pandas as pd
import pickle
from flask             import Flask, request, Response
from rossmann.Rossmann import Rossmann
import json


#loading model
model = pickle.load( open( '/Users/Caetano/repos/Predict-Time-Series/model/model_rossmann.pkl', 'rb') )

app = Flask( __name__ )

@app.route( '/rossmann/predict/', methods=['POST'] )
def rossmann_predict():
    try:
        test_json = request.get_json()

        if test_json: #there is data 
            
            if isinstance( test_json, dict): #unique Example
                test_raw = pd.DataFrame( test_json, index[0] )

            else: # multiple examples
                test_raw = pd.DataFrame( test_json, columns=test_json[0].keys() )
            
            # Instantiate Rossmann class
            pipeline = Rossmann()
           
            #data cleaning
            df1 = pipeline.data_cleaning( test_raw)

            #feature engineering
            df2 = pipeline.feature_engineering( df1 )

            #data preparation
            df3 = pipeline.data_preparation( df2 )

            #prediction
            df_response = pipeline.get_prediction( model, test_raw, df3)

            return df_response

        else:
            return Response( '{}', status=404, mimetype='application/json' )
    except Exception as e:
        print("/rossamann/predict/ error: ", e)
    
if __name__ == '__main__':
    #app.run( 'localhost' )
    app.run( '0.0.0.0') 
    
