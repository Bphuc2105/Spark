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
