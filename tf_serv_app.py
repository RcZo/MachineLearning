import json 
import codecs 
import numpy as np 
import requests 
from flask import Flask, request, jsonify 
from flask_cors import cross_origin 
from waitress import serve 
from keras_bert import Tokenizer 

dict_path = "/home/vuser/app/vocab.txt" 
token_dict = {}

with codecs.open(dict_path, "r", "utf8") as reader:
  for line in reader:         
    token = line.strip()         
    token_dict[token] = len(token_dict)
    
class ForgeTokenizer(Tokenizer):
  def _tokenize(self, text):
    R = []
    for c in text:
      if c in self._token_dict:
        R.append(c)
      elif self._is_space(c):
        R.append("[unused1]") 
        # space替換為 [unused1]             
       else:
        R.append("[UNK]") 
        # 其他找不到的替換為 [UNK]         
    return R 
              
tokenizer = ForgeTokenizer(token_dict)

app = Flask(__name__)
# Uncomment this line if you are making a Cross domain request
# CORS(app)
# Testing URL
@app.route("/hello/", methods=["GET", "POST"])
def hello_world():
  return "Hello, Frank!"

@app.route("/ImpactClass/predict/", methods=["POST"])
@cross_origin()
def impact_class():
  user_request = json.loads(request.get_data())
  NewsText=user_request["ImpactClass"]["NewsText"]
  x1, x2 = tokenizer.encode(first=NewsText)
  # Creating payload for TensorFlow serving request
  payload = {
    "instances":[{
      "Input-Token":x1,
      "Input-Segment":[0]
    }]
  }
  # Making POST request
  req = requests.post("http://TF_SERVER:80/v1/models/ImpactClass:predict", json=payload)
  # Decoding results from TensorFlow Serving server
  pred = json.loads(req.content.decode("utf-8"))
  max_prediction=np.argmax(pred["predictions"])
  response={"ImpactClass":{"prediction": int(max_prediction-3) }}
  # Returning JSON response to the frontend
  return jsonify(response)

if __name__ == "__main__":
  serve(app, host="0.0.0.0", port=80, threads=2)
