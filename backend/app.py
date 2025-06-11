import re
import mlflow
import joblib
import dagshub

import pandas as pd

from typing import List
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from youtube_comments_analyzer.config import (
    STOP_WORDS_TO_KEEP,
    MODELS_DIR,
    VECTORIZER_FILE_NAME,
)

dagshub.init(
    repo_owner="SushrutGaikwad",
    repo_name="youtube-comments-analyzer",
    mlflow=True,
)
client = MlflowClient()

stop_words = set(stopwords.words("english")) - STOP_WORDS_TO_KEEP
lemmatizer = WordNetLemmatizer()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CommentRequest(BaseModel):
    comments: List[str]


def preprocess_comment(lemmatizer, comment: str, stop_words: set = stop_words) -> str:
    try:
        comment = comment.lower().strip()
        comment = re.sub(r"\n", " ", comment)
        comment = re.sub(r"[^A-Za-z0-9\s!?.,]", "", comment)
        comment = " ".join([word for word in comment.split() if word not in stop_words])
        comment = " ".join([lemmatizer.lemmatize(word) for word in comment.split()])
        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment


def load_model_and_vectorizer(
    model_name: str, model_version: str, vectorizer_path: Path
):
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


@app.on_event("startup")
def startup_event():
    global model, vectorizer
    model, vectorizer = load_model_and_vectorizer(
        model_name="yt_chrome_plugin_model",
        model_version="2",
        vectorizer_path=MODELS_DIR / VECTORIZER_FILE_NAME,
    )


@app.post("/predict")
def predict(request: CommentRequest):
    comments = request.comments
    if not comments:
        raise HTTPException(status_code=400, detail="No comments provided")

    try:
        preprocessed_comments = [
            preprocess_comment(lemmatizer=lemmatizer, comment=comment)
            for comment in comments
        ]
        vectorized_comments = vectorizer.transform(preprocessed_comments)

        vectorized_df = pd.DataFrame(vectorized_comments.toarray())
        vectorized_df.columns = [str(i) for i in range(vectorized_df.shape[1])]

        predictions = model.predict(vectorized_df).tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    output = [{"comment": c, "sentiment": s} for c, s in zip(comments, predictions)]
    return output
