#!/bin/bash

echo "Initializing HDFS directories for ML Pipeline..."

# Wait for HDFS to be ready
echo "Waiting for HDFS NameNode to be ready..."
sleep 30

# Create base directories
echo "Creating base directories..."
docker exec namenode hdfs dfsadmin -safemode leave 2>/dev/null || true
docker exec namenode hadoop fs -mkdir -p /models
docker exec namenode hadoop fs -mkdir -p /models/stock_prediction
docker exec namenode hadoop fs -mkdir -p /models/sentiment_analysis
docker exec namenode hadoop fs -mkdir -p /models/news_classification
docker exec namenode hadoop fs -mkdir -p /models/archived
docker exec namenode hadoop fs -mkdir -p /data
docker exec namenode hadoop fs -mkdir -p /data/input
docker exec namenode hadoop fs -mkdir -p /data/output
docker exec namenode hadoop fs -mkdir -p /data/checkpoint
docker exec namenode hadoop fs -mkdir -p /user/hive/warehouse

# Set permissions
echo "Setting permissions..."
docker exec namenode hadoop fs -chmod -R 777 /models
docker exec namenode hadoop fs -chmod -R 777 /data
docker exec namenode hadoop fs -chmod -R 777 /user

# List created directories
echo "HDFS directories created:"
docker exec namenode hadoop fs -ls /
docker exec namenode hadoop fs -ls /models

echo "HDFS initialization completed!"
