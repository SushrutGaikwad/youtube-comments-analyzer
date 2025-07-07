#!/bin/bash

# Log everything to start_docker.log
exec > /home/ubuntu/start_docker.log 2>&1

echo "Logging in to ECR..."
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 872515288060.dkr.ecr.us-east-2.amazonaws.com

echo "Pulling Docker image..."
docker pull 872515288060.dkr.ecr.us-east-2.amazonaws.com/youtube-comments-analyzer:latest

echo "Checking for existing container..."
if [ "$(docker ps -q -f name=youtube-comments-analyzer)" ]; then
    echo "Stopping existing container..."
    docker stop youtube-comments-analyzer
fi

if [ "$(docker ps -aq -f name=youtube-comments-analyzer)" ]; then
    echo "Removing existing container..."
    docker rm youtube-comments-analyzer
fi

echo "Starting new container..."
docker run -d -p 80:5000 --name youtube-comments-analyzer 872515288060.dkr.ecr.us-east-2.amazonaws.com/youtube-comments-analyzer:latest

echo "Container started successfully."
