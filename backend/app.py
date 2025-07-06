import re
import io
import mlflow
import joblib
import dagshub
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from loguru import logger
from typing import List, Dict
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
from wordcloud import WordCloud

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from youtube_comments_analyzer.config import (
    STOP_WORDS_TO_KEEP,
    MODELS_DIR,
    VECTORIZER_FILE_NAME,
)

# DagsHub and MLflow setup
dagshub.init(
    repo_owner="SushrutGaikwad",
    repo_name="youtube-comments-analyzer",
    mlflow=True,
)
client = MlflowClient()

# Globals
stop_words = set(stopwords.words("english")) - STOP_WORDS_TO_KEEP
lemmatizer = WordNetLemmatizer()

# App initialization
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Preprocessing
def preprocess_comment(comment: str) -> str:
    try:
        comment = comment.lower().strip()
        comment = re.sub(r"\n", " ", comment)
        comment = re.sub(r"[^A-Za-z0-9\s!?.,]", "", comment)
        comment = " ".join([w for w in comment.split() if w not in stop_words])
        comment = " ".join([lemmatizer.lemmatize(w) for w in comment.split()])
        return comment
    except Exception as e:
        logger.error(f"Error preprocessing comment: {e}")
        return comment


# Load model + vectorizer
def load_model_and_vectorizer(
    model_name: str, model_version: str, vectorizer_path: Path
):
    try:
        logger.info(f"Loading model '{model_name}' version {model_version}")
        model_uri = f"models:/{model_name}/{model_version}"
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Loading vectorizer from {vectorizer_path}")
        vectorizer = joblib.load(vectorizer_path)
        logger.success("Model and vectorizer loaded successfully.")
        return model, vectorizer
    except Exception as e:
        logger.error(f"Failed to load model/vectorizer: {e}")
        raise


@app.on_event("startup")
def startup_event():
    global model, vectorizer
    model, vectorizer = load_model_and_vectorizer(
        model_name="yt_chrome_plugin_model",
        model_version="10",
        vectorizer_path=MODELS_DIR / VECTORIZER_FILE_NAME,
    )


@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    comments = data.get("comments")
    if not comments:
        raise HTTPException(status_code=400, detail="No comments provided")

    try:
        logger.info("Running /predict on received comments")
        preprocessed = [preprocess_comment(c) for c in comments]
        vec = vectorizer.transform(preprocessed)
        df = pd.DataFrame(vec.toarray())
        df.columns = [str(i) for i in range(df.shape[1])]
        predictions = [str(p) for p in model.predict(df).tolist()]
        return [{"comment": c, "sentiment": s} for c, s in zip(comments, predictions)]
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/predict_with_timestamps")
async def predict_with_timestamps(request: Request):
    data = await request.json()
    comments_data = data.get("comments")
    if not comments_data:
        logger.warning("No comments provided to /predict_with_timestamps")
        raise HTTPException(status_code=400, detail="No comments provided")

    try:
        logger.info("Running /predict_with_timestamps")
        comments = [item["text"] for item in comments_data]
        timestamps = [item["timestamp"] for item in comments_data]
        preprocessed = [preprocess_comment(c) for c in comments]
        vec = vectorizer.transform(preprocessed)
        df = pd.DataFrame(vec.toarray())
        df.columns = [str(i) for i in range(df.shape[1])]
        predictions = [str(p) for p in model.predict(df).tolist()]
        return [
            {"comment": c, "sentiment": s, "timestamp": t}
            for c, s, t in zip(comments, predictions, timestamps)
        ]
    except Exception as e:
        logger.error(f"Prediction with timestamps failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/generate_chart")
async def generate_chart(request: Request):
    data = await request.json()
    counts = data.get("sentiment_counts")
    if not counts:
        logger.warning("No sentiment counts provided to /generate_chart")
        raise HTTPException(status_code=400, detail="No sentiment counts provided")

    try:
        logger.info("Generating sentiment pie chart")
        labels = ["Positive", "Neutral", "Negative"]
        sizes = [
            int(counts.get("1", 0)),
            int(counts.get("0", 0)),
            int(counts.get("2", 0)),
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")

        colors = ["#36A2EB", "#C9CBCF", "#FF6384"]
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=140,
            textprops={"color": "w"},
        )
        plt.axis("equal")

        buf = io.BytesIO()
        plt.savefig(buf, format="PNG", transparent=True)
        buf.seek(0)
        plt.close()
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        logger.error(f"Chart generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chart generation failed: {e}")


@app.post("/generate_wordcloud")
async def generate_wordcloud(request: Request):
    data = await request.json()
    comments = data.get("comments")
    if not comments:
        logger.warning("No comments provided to /generate_wordcloud")
        raise HTTPException(status_code=400, detail="No comments provided")

    try:
        logger.info("Generating word cloud")
        preprocessed = [preprocess_comment(c) for c in comments]
        text = " ".join(preprocessed)
        wc = WordCloud(
            width=800,
            height=400,
            background_color="black",
            colormap="Blues",
            stopwords=set(stopwords.words("english")),
            collocations=False,
        ).generate(text)
        buf = io.BytesIO()
        wc.to_image().save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        logger.error(f"Word cloud generation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Word cloud generation failed: {e}"
        )


@app.post("/generate_trend_graph")
async def generate_trend_graph(request: Request):
    data = await request.json()
    sentiment_data = data.get("sentiment_data")
    if not sentiment_data:
        logger.warning("No sentiment data provided to /generate_trend_graph")
        raise HTTPException(status_code=400, detail="No sentiment data provided")

    try:
        logger.info("Generating sentiment trend graph")
        df = pd.DataFrame(sentiment_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df["sentiment"] = df["sentiment"].astype(int)

        sentiment_labels = {2: "Negative", 0: "Neutral", 1: "Positive"}
        monthly = df.resample("M")["sentiment"].value_counts().unstack(fill_value=0)
        total = monthly.sum(axis=1)
        percentages = (monthly.T / total).T * 100

        for s in [2, 0, 1]:
            if s not in percentages.columns:
                percentages[s] = 0
        percentages = percentages[[2, 0, 1]]

        plt.figure(figsize=(12, 6))
        colors = {2: "red", 0: "gray", 1: "green"}
        for s in [2, 0, 1]:
            plt.plot(
                percentages.index,
                percentages[s],
                marker="o",
                label=sentiment_labels[s],
                color=colors[s],
            )
        plt.title("Monthly Sentiment Percentage Over Time")
        plt.xlabel("Month")
        plt.ylabel("Percentage of Comments (%)")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        plt.legend()
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="PNG")
        buf.seek(0)
        plt.close()
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        logger.error(f"Trend graph generation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Trend graph generation failed: {e}"
        )
