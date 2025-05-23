# src/predict.py

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.sql.functions import col, to_date, lit, current_timestamp
import traceback

# Import cấu hình sử dụng absolute import để đảm bảo import đúng module
try:
    from src import config
    ES_PREDICTION_INDEX = config.ES_PREDICTION_INDEX
    ES_NODES = config.ES_NODES
    ES_PORT = config.ES_PORT
except ImportError as e:
    print(f"Lỗi import config trong predict.py: {e}")
    print("Cảnh báo: Không thể import cấu hình Elasticsearch trong predict.py, sử dụng giá trị mặc định.")
    ES_PREDICTION_INDEX = "stock_predictions_fallback" # Fallback
    ES_NODES = "localhost" # Fallback
    ES_PORT = "9200" # Fallback


def load_prediction_model(model_path):
    """
    Tải PipelineModel đã huấn luyện từ đường dẫn được chỉ định.
    """
    try:
        print(f"Đang tải mô hình từ: {model_path}")
        model = PipelineModel.load(model_path)
        print("Tải mô hình thành công.")
        return model
    except Exception as e:
        print(f"Lỗi khi tải mô hình từ {model_path}: {e}")
        traceback.print_exc()
        return None

def make_predictions(model, data_df):
    """
    Thực hiện dự đoán trên DataFrame đầu vào bằng mô hình đã cho.
    Hàm này hoạt động với cả Batch và Streaming DataFrame.
    """
    if model is None or data_df is None:
        print("Mô hình hoặc dữ liệu đầu vào là None. Không thể thực hiện dự đoán.")
        return None
    try:
        print("Thực hiện dự đoán trên dữ liệu...")
        predictions_df = model.transform(data_df)
        print("Áp dụng mô hình hoàn tất.")
        return predictions_df
    except Exception as e:
        print(f"Lỗi trong quá trình áp dụng mô hình: {e}")
        traceback.print_exc()
        return None

# --- Hàm mới: Ghi stream ra Elasticsearch ---

def write_stream_to_elasticsearch(streaming_df, es_index, es_host, es_port):
    """
    Ghi DataFrame streaming ra Elasticsearch.

    Args:
        streaming_df (DataFrame): DataFrame streaming chứa kết quả dự đoán.
        es_index (str): Tên index trong Elasticsearch.
        es_host (str): Địa chỉ host của Elasticsearch.
        es_port (str): Port của Elasticsearch.

    Returns:
        StreamingQuery: Đối tượng StreamingQuery.
                        Trả về None nếu có lỗi cấu hình hoặc bắt đầu query.
    """
    print(f"Đang cấu hình ghi stream ra Elasticsearch index: {es_index} tại {es_host}:{es_port}")
    try:
        es_options = {
            "es.nodes": es_host,
            "es.port": es_port,
            "es.resource": es_index,
            "es.nodes.wan.only": "true",
            "es.write.operation": "index",
        }

        checkpoint_dir = "/app/checkpoint/predictions_to_es"
        print(f"Sử dụng checkpoint location: {checkpoint_dir}")

        streaming_query = streaming_df \
            .writeStream \
            .format("org.elasticsearch.spark.sql") \
            .outputMode("append") \
            .options(**es_options) \
            .option("checkpointLocation", checkpoint_dir) \
            .start()

        print("Streaming query để ghi ra Elasticsearch đã được khởi tạo.")
        return streaming_query

    except Exception as e:
        print(f"Lỗi khi cấu hình hoặc bắt đầu ghi stream ra Elasticsearch: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("--- Bỏ qua Test Standalone cho predict.py khi tích hợp Kafka/ES ---")
    print("Vui lòng chạy qua docker-compose và kiểm tra luồng end-to-end.")
