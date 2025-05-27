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
