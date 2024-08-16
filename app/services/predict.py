from fastapi import APIRouter, HTTPException
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import sagemaker
from app.utils.utils import region_name
import boto3
import json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

load_dotenv()
router = APIRouter()

# Config SageMaker
session = boto3.Session(
    aws_access_key_id = os.getenv("ACCESS_KEY"),
    aws_secret_access_key = os.getenv("SECRET_ACCESS_KEY"),
    region_name=region_name
)
sagemaker_runtime = session.client('sagemaker-runtime')
endpoint_name = os.getenv("ENDPOINT_NAME")

genres=['Fiction', 'Novel', 'Adventure', 'Romance', 'History', 'Thriller', 'Horror', 'Biography', 'Fantasy', 'Other']

# Model input
class BookInput(BaseModel):
    title: str

# Helper function for convert probability to genre value 
def interpret_genre_prediction(prediction_result, genres=genres):
    probabilities = np.array(prediction_result['predictions'][0])
    predicted_index = np.argmax(probabilities)
    predicted_genre = genres[predicted_index]
    probability = probabilities[predicted_index]
    return predicted_genre, probability

@router.post("/predict_genre")
async def predict_genre(book: BookInput):
    try:
        num_words = 1000
        max_length = 100
        
        tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
        tokenizer.fit_on_texts([book.title])
        sequences = tokenizer.texts_to_sequences([book.title])
        padded_sequence = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
        
        # Input for model
        input_data = padded_sequence.tolist()
        input_json = json.dumps({"instances": input_data})
        
        # Call endpoint SageMaker
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=input_json
        )
        
        # Result
        result = json.loads(response['Body'].read().decode())
        predicted_genre, probability = interpret_genre_prediction(result)
        
        return {
            "status": 200,
            "data": {
                "book_title": book.title,
                "genre": predicted_genre,
                "confidence": f"{probability:.2%}"
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))