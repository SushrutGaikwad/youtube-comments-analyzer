FROM python:3.11-slim

# Set working directory inside the container to where app.py is located
WORKDIR /app/backend

# Install system dependencies
RUN apt-get update && apt-get install -y libgomp1

# Copy the entire project (or only what's needed)
COPY backend/ /app/backend/
COPY models/ /app/models/

# Install Python dependencies from backend/requirements.txt
RUN pip install -r requirements.txt

# Download NLTK assets
RUN python -m nltk.downloader stopwords wordnet

# Expose FastAPI port
EXPOSE 5000

# Start the FastAPI app
CMD [ "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000" ]
