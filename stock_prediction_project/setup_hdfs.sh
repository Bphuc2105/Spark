#!/bin/bash

# HDFS Setup Script for Spark ML Pipeline
# This script sets up the necessary directories and configuration for HDFS integration

echo "=== HDFS Setup Script for Spark ML Pipeline ==="

# Create hadoop configuration directory
echo "1. Creating Hadoop configuration directory..."
mkdir -p hadoop-config
mkdir -p hue-config

# Create core-site.xml
echo "2. Creating core-site.xml..."
cat > hadoop-config/core-site.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <property>
        <name>fs.defaultFS</name>
        <value>hdfs://namenode:9000</value>
    </property>
    <property>
        <name>hadoop.http.staticuser.user</name>
        <value>root</value>
    </property>
    <property>
        <name>hadoop.proxyuser.hue.hosts</name>
        <value>*</value>
    </property>
    <property>
        <name>hadoop.proxyuser.hue.groups</name>
        <value>*</value>
    </property>
</configuration>
EOF

# Create hdfs-site.xml
echo "3. Creating hdfs-site.xml..."
cat > hadoop-config/hdfs-site.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <property>
        <name>dfs.nameservices</name>
        <value>hdfscluster</value>
    </property>
    <property>
        <name>dfs.ha.namenodes.hdfscluster</name>
        <value>nn1</value>
    </property>
    <property>
        <name>dfs.namenode.rpc-address.hdfscluster.nn1</name>
        <value>namenode:9000</value>
    </property>
    <property>
        <name>dfs.namenode.http-address.hdfscluster.nn1</name>
        <value>namenode:9870</value>
    </property>
    <property>
        <name>dfs.webhdfs.enabled</name>
        <value>true</value>
    </property>
    <property>
        <name>dfs.permissions.enabled</name>
        <value>false</value>
    </property>
    <property>
        <name>dfs.replication</name>
        <value>1</value>
    </property>
</configuration>
EOF

# Create hue configuration (optional)
echo "4. Creating Hue configuration..."
cat > hue-config/hue.ini << 'EOF'
[desktop]
secret_key=hue_secret_key_for_spark_ml_pipeline
http_host=0.0.0.0
http_port=8888
time_zone=Asia/Ho_Chi_Minh

[hadoop]
[[hdfs_clusters]]
[[[default]]]
fs_defaultfs=hdfs://namenode:9000
webhdfs_url=http://namenode:9870/webhdfs/v1

[[yarn_clusters]]
[[[default]]]
resourcemanager_host=namenode
resourcemanager_port=8032
EOF

# Create initialization script for HDFS directories
echo "5. Creating HDFS initialization script..."
cat > init_hdfs_dirs.sh << 'EOF'
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
EOF

chmod +x init_hdfs_dirs.sh

# Create health check script
echo "6. Creating health check script..."
cat > check_hdfs_health.sh << 'EOF'
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
EOF

chmod +x check_hdfs_health.sh

# Create backup script
echo "7. Creating backup script..."
cat > backup_models.sh << 'EOF'
#!/bin/bash

BACKUP_DIR="./hdfs_backup/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "=== HDFS Models Backup ==="
echo "Backup directory: $BACKUP_DIR"

# Backup all models
echo "Backing up models from HDFS..."
docker exec namenode hadoop fs -copyToLocal /models "$BACKUP_DIR/" 2>/dev/null || echo "No models to backup"

# Create backup info file
cat > "$BACKUP_DIR/backup_info.txt" << EOL
Backup Date: $(date)
HDFS NameNode: namenode:9000
Source Path: hdfs://namenode:9000/models
Backup Tool: hadoop fs -copyToLocal
EOL

echo "Backup completed: $BACKUP_DIR"
ls -la "$BACKUP_DIR"
EOF

chmod +x backup_models.sh

# Create usage instructions
echo "8. Creating usage instructions..."
cat > HDFS_USAGE.md << 'EOF'
# HDFS Integration for Spark ML Pipeline

## Setup Instructions

1. **Build and start services:**
   ```bash
   docker-compose up --build -d
   ```

2. **Initialize HDFS directories:**
   ```bash
   ./init_hdfs_dirs.sh
   ```

3. **Check HDFS health:**
   ```bash
   ./check_hdfs_health.sh
   ```

## Accessing HDFS

- **NameNode Web UI:** http://localhost:9870
- **DataNode Web UI:** http://localhost:9864
- **HDFS File Browser (Hue):** http://localhost:8888

## Model Management

### Save Model to HDFS
```python
from your_model_utils import save_model

# Save model to HDFS
hdfs_path = "hdfs://namenode:9000/models/stock_prediction/model_20241201"
success = save_model(trained_model, hdfs_path)
```

### Load Model from HDFS
```python
from your_model_utils import load_model_from_hdfs

# Load model from HDFS
model = load_model_from_hdfs("hdfs://namenode:9000/models/stock_prediction/model_20241201")
```

### Model Versioning
```python
from your_model_utils import save_model_with_version, load_latest_model

# Save with automatic versioning
success, path = save_model_with_version(model, "hdfs://namenode:9000/models/stock_prediction")

# Load latest version
latest_model = load_latest_model("hdfs://namenode:9000/models/stock_prediction")
```

## HDFS Commands

### Basic Operations
```bash
# List files
docker exec namenode hadoop fs -ls /models

# Create directory
docker exec namenode hadoop fs -mkdir /models/new_model_type

# Copy file to HDFS
docker exec namenode hadoop fs -copyFromLocal /local/path /hdfs/path

# Copy file from HDFS
docker exec namenode hadoop fs -copyToLocal /hdfs/path /local/path

# Remove file/directory
docker exec namenode hadoop fs -rm -r /hdfs/path
```

### Monitoring
```bash
# Check HDFS health
docker exec namenode hdfs dfsadmin -report

# Check safe mode
docker exec namenode hdfs dfsadmin -safemode get

# Leave safe mode (if needed)
docker exec namenode hdfs dfsadmin -safemode leave
```

## Backup and Recovery

### Manual Backup
```bash
./backup_models.sh
```

### Restore from Backup
```bash
docker exec namenode hadoop fs -copyFromLocal ./hdfs_backup/20241201_120000/models /
```

## Troubleshooting

### Common Issues

1. **NameNode in Safe Mode:**
   ```bash
   docker exec namenode hdfs dfsadmin -safemode leave
   ```

2. **Permission Denied:**
   ```bash
   docker exec namenode hadoop fs -chmod -R 777 /models
   ```

3. **Connection Refused:**
   - Check if NameNode is running: `docker ps`
   - Check NameNode logs: `docker logs namenode`
   - Verify network connectivity: `docker network ls`

4. **Out of Space:**
   ```bash
   docker exec namenode hadoop fs -df -h
   ```

### Log Locations
- NameNode logs: `docker logs namenode`
- DataNode logs: `docker logs datanode`
- Spark logs: `docker logs spark-master` or `docker logs spark-worker`

## Best Practices

1. **Model Organization:**
   - Use descriptive paths: `/models/stock_prediction/lstm_v1`
   - Include timestamps: `/models/stock_prediction/model_20241201_120000`
   - Separate by model type: `/models/{prediction_type}/{algorithm}/{version}`

2. **Version Management:**
   - Always use versioning for production models
   - Keep latest version markers
   - Archive old models regularly

3. **Backup Strategy:**
   - Regular automated backups
   - Test restore procedures
   - Keep multiple backup generations

4. **Monitoring:**
   - Regular health checks
   - Monitor disk usage
   - Track model performance metrics
EOF

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Files created:"
echo "  📁 hadoop-config/"
echo "    📄 core-site.xml"
echo "    📄 hdfs-site.xml"
echo "  📁 hue-config/"
echo "    📄 hue.ini"
echo "  📄 init_hdfs_dirs.sh"
echo "  📄 check_hdfs_health.sh"
echo "  📄 backup_models.sh"
echo "  📄 HDFS_USAGE.md"
echo ""
echo "Next steps:"
echo "1. Start services: docker-compose up --build -d"
echo "2. Initialize HDFS: ./init_hdfs_dirs.sh"
echo "3. Check health: ./check_hdfs_health.sh"
echo "4. Read HDFS_USAGE.md for detailed instructions"
echo ""
echo "Web UIs will be available at:"
echo "  - NameNode: http://localhost:9870"
echo "  - DataNode: http://localhost:9864"
echo "  - Spark Master: http://localhost:8080"
echo "  - HDFS Browser: http://localhost:8888"
echo ""