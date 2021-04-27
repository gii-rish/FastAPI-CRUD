import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import pandas as pd

app = FastAPI()
pickle_in = open("classifier.pkl", "rb")
classifier = pickle.load(pickle_in)


#model creation
class BankNote(BaseModel):
	variance: float
	skewness: float
	curtosis: float
	entropy: float

@app.get("/")
def home():
	return {"message":"Welcome to Bank Note Prediction classifier"}

@app.post("/predict")
def predict_banknote(data: BankNote):
	data = data.dict()
	variance, skewness, curtosis, entropy = data['variance'], data['skewness'], data['curtosis'], data['entropy']

	prediction = classifier.predict([[variance, skewness, curtosis,entropy]])

	if not prediction[0] > 0.5:
		return {"prediction": "It's a Bank note!"}
	return {"prediction": "Fake Note"}

if __name__ == '__main__':
	uvicorn.run(main, host='127.0.0.1', port=8000)
