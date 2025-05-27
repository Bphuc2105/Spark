# Example usage of HDFS model saving and loading

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

from src.train import save_model, load_model_from_hdfs

# Initialize Spark with HDFS support
def create_spark_session_with_hdfs():
    """Tạo SparkSession với cấu hình HDFS"""
    spark = SparkSession.builder \
        .appName("MLModelWithHDFS") \
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
        .config("spark.sql.warehouse.dir", "hdfs://namenode:9000/user/hive/warehouse") \
        .config("spark.hadoop.dfs.client.use.datanode.hostname", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .getOrCreate()
    
    return spark

# Example: Train and save model to HDFS
def train_and_save_model_example():
    """Ví dụ huấn luyện và lưu model lên HDFS"""
    
    # Tạo SparkSession
    spark = create_spark_session_with_hdfs()
    
    # Tạo dữ liệu mẫu
    data = spark.createDataFrame([
        (1.0, 2.0, 3.0, 10.0),
        (2.0, 3.0, 4.0, 20.0),
        (3.0, 4.0, 5.0, 30.0),
        (4.0, 5.0, 6.0, 40.0),
        (5.0, 6.0, 7.0, 50.0)
    ], ["feature1", "feature2", "feature3", "label"])
    
    # Tạo pipeline
    assembler = VectorAssembler(
        inputCols=["feature1", "feature2", "feature3"],
        outputCol="features"
    )
    
    lr = LinearRegression(featuresCol="features", labelCol="label")
    
    pipeline = Pipeline(stages=[assembler, lr])
    
    # Huấn luyện model
    print("Đang huấn luyện model...")
    model = pipeline.fit(data)
    
    # Lưu model lên HDFS
    hdfs_model_path = "hdfs://namenode:9000/models/stock_prediction_model"
    print(f"Đang lưu model lên HDFS: {hdfs_model_path}")
    
    success = save_model(model, hdfs_model_path)
    if success:
        print("Model đã được lưu thành công lên HDFS!")
    else:
        print("Có lỗi xảy ra khi lưu model lên HDFS!")
    
    return success, hdfs_model_path

# Example: Load model from HDFS and make predictions
def load_and_predict_example(hdfs_model_path):
    """Ví dụ tải model từ HDFS và thực hiện dự đoán"""
    
    # Tạo SparkSession
    spark = create_spark_session_with_hdfs()
    
    # Tải model từ HDFS
    print(f"Đang tải model từ HDFS: {hdfs_model_path}")
    model = load_model_from_hdfs(hdfs_model_path)
    
    if model is None:
        print("Không thể tải model từ HDFS!")
        return False
    
    # Tạo dữ liệu test
    test_data = spark.createDataFrame([
        (6.0, 7.0, 8.0),
        (7.0, 8.0, 9.0),
        (8.0, 9.0, 10.0)
    ], ["feature1", "feature2", "feature3"])
    
    # Thực hiện dự đoán
    print("Đang thực hiện dự đoán...")
    predictions = model.transform(test_data)
    
    # Hiển thị kết quả
    print("Kết quả dự đoán:")
    predictions.select("feature1", "feature2", "feature3", "prediction").show()
    
    return True

# Example: Manage HDFS directories for models
def manage_hdfs_model_directories():
    """Ví dụ quản lý thư mục models trên HDFS"""
    
    from pyspark.sql import SparkSession
    
    spark = SparkSession.getActiveSession()
    if spark is None:
        spark = create_spark_session_with_hdfs()
    
    # Lấy Hadoop Configuration
    hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
    
    # Import Java classes
    from py4j.java_gateway import java_import
    java_import(spark.sparkContext._gateway.jvm, "org.apache.hadoop.fs.FileSystem")
    java_import(spark.sparkContext._gateway.jvm, "org.apache.hadoop.fs.Path")
    
    FileSystem = spark.sparkContext._gateway.jvm.org.apache.hadoop.fs.FileSystem
    Path = spark.sparkContext._gateway.jvm.org.apache.hadoop.fs.Path
    
    fs = FileSystem.get(hadoop_conf)
    
    # Tạo thư mục chính cho models
    models_dir = Path("hdfs://namenode:9000/models")
    if not fs.exists(models_dir):
        print("Tạo thư mục /models trên HDFS...")
        fs.mkdirs(models_dir)
        print("Đã tạo thư mục /models")
    else:
        print("Thư mục /models đã tồn tại trên HDFS")
    
    # Tạo các thư mục con cho các loại model khác nhau
    subdirs = [
        "hdfs://namenode:9000/models/stock_prediction",
        "hdfs://namenode:9000/models/sentiment_analysis", 
        "hdfs://namenode:9000/models/news_classification",
        "hdfs://namenode:9000/models/archived"
    ]
    
    for subdir in subdirs:
        subdir_path = Path(subdir)
        if not fs.exists(subdir_path):
            print(f"Tạo thư mục {subdir}...")
            fs.mkdirs(subdir_path)
        else:
            print(f"Thư mục {subdir} đã tồn tại")
    
    # Liệt kê tất cả các model hiện có
    print("\n--- Danh sách models trên HDFS ---")
    list_hdfs_models(fs, models_dir)

def list_hdfs_models(fs, models_dir):
    """Liệt kê tất cả models trong thư mục HDFS"""
    
    try:
        file_status_list = fs.listStatus(models_dir)
        if len(file_status_list) == 0:
            print("Không có model nào trong thư mục")
            return
        
        for file_status in file_status_list:
            path = file_status.getPath()
            name = path.getName()
            is_dir = file_status.isDirectory()
            size = file_status.getLen()
            modification_time = file_status.getModificationTime()
            
            # Chuyển đổi timestamp thành định dạng đọc được
            import datetime
            mod_time_str = datetime.datetime.fromtimestamp(modification_time / 1000).strftime('%Y-%m-%d %H:%M:%S')
            
            if is_dir:
                print(f"📁 {name}/ - Thư mục - {mod_time_str}")
                # Đệ quy liệt kê nội dung thư mục con
                sub_files = fs.listStatus(path)
                for sub_file in sub_files[:3]:  # Chỉ hiển thị 3 file đầu
                    sub_name = sub_file.getPath().getName()
                    sub_size = sub_file.getLen()
                    print(f"  📄 {sub_name} ({sub_size} bytes)")
                if len(sub_files) > 3:
                    print(f"  ... và {len(sub_files) - 3} file khác")
            else:
                print(f"📄 {name} - {size} bytes - {mod_time_str}")
                
    except Exception as e:
        print(f"Lỗi khi liệt kê models: {e}")

# Example: Backup and restore models
def backup_model_to_local(hdfs_model_path, local_backup_path):
    """Sao lưu model từ HDFS về local filesystem"""
    
    from pyspark.sql import SparkSession
    import subprocess
    import os
    
    spark = SparkSession.getActiveSession()
    if spark is None:
        spark = create_spark_session_with_hdfs()
    
    try:
        # Tạo thư mục backup local nếu chưa tồn tại
        os.makedirs(os.path.dirname(local_backup_path), exist_ok=True)
        
        # Sử dụng hadoop fs command để copy
        cmd = f"hadoop fs -copyToLocal {hdfs_model_path} {local_backup_path}"
        print(f"Đang thực hiện backup: {cmd}")
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"THÀNH CÔNG: Model đã được backup từ {hdfs_model_path} về {local_backup_path}")
            return True
        else:
            print(f"LỖI: Không thể backup model. Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Lỗi xảy ra khi backup model: {e}")
        return False

def restore_model_from_local(local_model_path, hdfs_restore_path):
    """Khôi phục model từ local filesystem lên HDFS"""
    
    from pyspark.sql import SparkSession
    import subprocess
    import os
    
    spark = SparkSession.getActiveSession()
    if spark is None:
        spark = create_spark_session_with_hdfs()
    
    try:
        # Kiểm tra file local có tồn tại không
        if not os.path.exists(local_model_path):
            print(f"LỖI: File local {local_model_path} không tồn tại!")
            return False
        
        # Sử dụng hadoop fs command để copy
        cmd = f"hadoop fs -copyFromLocal {local_model_path} {hdfs_restore_path}"
        print(f"Đang thực hiện restore: {cmd}")
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"THÀNH CÔNG: Model đã được restore từ {local_model_path} lên {hdfs_restore_path}")
            return True
        else:
            print(f"LỖI: Không thể restore model. Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Lỗi xảy ra khi restore model: {e}")
        return False

# Main execution example
if __name__ == "__main__":
    print("=== HDFS Model Management Example ===\n")
    
    # Bước 1: Thiết lập thư mục HDFS
    print("1. Thiết lập thư mục models trên HDFS...")
    manage_hdfs_model_directories()
    
    # Bước 2: Huấn luyện và lưu model
    print("\n2. Huấn luyện và lưu model...")
    success, model_path = train_and_save_model_example()
    
    if success:
        # Bước 3: Tải và sử dụng model
        print("\n3. Tải model và thực hiện dự đoán...")
        load_and_predict_example(model_path)
        
        # Bước 4: Backup model (optional)
        print("\n4. Backup model về local...")
        backup_model_to_local(model_path, "/tmp/backup/stock_model")
        
    print("\n=== Hoàn tất ===")

# Utility functions for model versioning
def save_model_with_version(model, base_path, version=None):
    """Lưu model với versioning trên HDFS"""
    
    import datetime
    
    if version is None:
        # Tự động tạo version based on timestamp
        version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    versioned_path = f"{base_path}_v{version}"
    
    print(f"Lưu model với version: {version}")
    success = save_model(model, versioned_path)
    
    if success:
        # Tạo symlink tới latest version
        latest_path = f"{base_path}_latest"
        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.getActiveSession()
            hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
            
            from py4j.java_gateway import java_import
            java_import(spark.sparkContext._gateway.jvm, "org.apache.hadoop.fs.FileSystem")
            java_import(spark.sparkContext._gateway.jvm, "org.apache.hadoop.fs.Path")
            
            FileSystem = spark.sparkContext._gateway.jvm.org.apache.hadoop.fs.FileSystem
            Path = spark.sparkContext._gateway.jvm.org.apache.hadoop.fs.Path
            
            fs = FileSystem.get(hadoop_conf)
            latest_path_obj = Path(latest_path)
            
            # Xóa latest cũ nếu có
            if fs.exists(latest_path_obj):
                fs.delete(latest_path_obj, True)
            
            # Tạo một file marker để đánh dấu latest version
            version_marker_path = f"{base_path}_latest_version"
            version_marker_obj = Path(version_marker_path)
            
            if fs.exists(version_marker_obj):
                fs.delete(version_marker_obj, True)
            
            # Ghi version info vào file marker
            output_stream = fs.create(version_marker_obj)
            output_stream.write(version.encode('utf-8'))
            output_stream.close()
            
            print(f"Model version {version} đã được đánh dấu là latest")
            
        except Exception as e:
            print(f"Không thể tạo latest marker: {e}")
    
    return success, versioned_path

def load_latest_model(base_path):
    """Tải latest version của model từ HDFS"""
    
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.getActiveSession()
        
        # Đọc latest version từ marker file
        version_marker_path = f"{base_path}_latest_version"
        
        hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
        
        from py4j.java_gateway import java_import
        java_import(spark.sparkContext._gateway.jvm, "org.apache.hadoop.fs.FileSystem")
        java_import(spark.sparkContext._gateway.jvm, "org.apache.hadoop.fs.Path")
        
        FileSystem = spark.sparkContext._gateway.jvm.org.apache.hadoop.fs.FileSystem
        Path = spark.sparkContext._gateway.jvm.org.apache.hadoop.fs.Path
        
        fs = FileSystem.get(hadoop_conf)
        marker_path_obj = Path(version_marker_path)
        
        if fs.exists(marker_path_obj):
            # Đọc version từ marker file
            input_stream = fs.open(marker_path_obj)
            version = input_stream.read().decode('utf-8')
            input_stream.close()
            
            versioned_path = f"{base_path}_v{version}"
            print(f"Tải latest model version: {version}")
            return load_model_from_hdfs(versioned_path)
        else:
            print("Không tìm thấy latest version marker")
            return None
            
    except Exception as e:
        print(f"Lỗi khi tải latest model: {e}")
        return None