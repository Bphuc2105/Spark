#!/bin/bash

echo "=== HDFS Health Check ==="

# Check NameNode status
echo "1. Checking NameNode status..."
curl -s http://localhost:9870/jmx?qry=Hadoop:service=NameNode,name=NameNodeStatus | grep -o '"State":"[^"]*"' || echo "NameNode not accessible"

# Check DataNode status
echo "2. Checking DataNode status..."
curl -s http://localhost:9864/jmx?qry=Hadoop:service=DataNode,name=DataNodeInfo | grep -o '"State":"[^"]*"' || echo "DataNode not accessible"

# Check HDFS file system
echo "3. Checking HDFS filesystem..."
docker exec namenode hadoop fs -df -h / 2>/dev/null || echo "HDFS not accessible"

# List model directories
echo "4. Checking model directories..."
docker exec namenode hadoop fs -ls /models 2>/dev/null || echo "Models directory not accessible"

echo "=== Health Check Complete ==="
