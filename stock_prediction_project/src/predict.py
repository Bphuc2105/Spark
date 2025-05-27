from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.sql.functions import col, to_date, lit, current_timestamp
import traceback

try:
    from src import config
    ES_PREDICTION_INDEX = config.ES_PREDICTION_INDEX
    ES_NODES = config.ES_NODES
    ES_PORT = config.ES_PORT
except ImportError as e:
    print(f"Lỗi import config trong predict.py: {e}")
    print("Cảnh báo: Không thể import cấu hình Elasticsearch trong predict.py, sử dụng giá trị mặc định.")
    ES_PREDICTION_INDEX = "stock_predictions_fallback" 
    ES_NODES = "localhost" 
    ES_PORT = "9200" 

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

def write_dataframe_to_elasticsearch(df, es_index, es_host, es_port):
    """
    Ghi DataFrame ra Elasticsearch một lần.
    
    Args:
        df (DataFrame): DataFrame chứa dữ liệu cần ghi.
        es_index (str): Tên index trong Elasticsearch.
        es_host (str): Địa chỉ host của Elasticsearch.
        es_port (str): Port của Elasticsearch.
    
    Returns:
        bool: True nếu ghi thành công, False nếu có lỗi.
    """
    from pyspark.sql.functions import col, to_date, to_timestamp
    print(f"Đang cấu hình ghi DataFrame ra Elasticsearch index: {es_index} tại {es_host}:{es_port}")
    
    try:
        
        columns = df.columns
        if 'date' in columns:
            print("Tìm thấy trường 'date', đang format thành kiểu date...")

            date_type = str(df.schema['date'].dataType)
            print(f"Kiểu dữ liệu hiện tại của trường 'date': {date_type}")
            
            if 'long' in date_type.lower() or 'bigint' in date_type.lower():
                print("Trường 'date' là timestamp, đang convert...")
                
                df = df.withColumn('date', 
                    to_timestamp(col('date') / 1000).cast('string'))
            else:
                
                df = df.withColumn('date', 
                    to_date(col('date')).cast('string'))
            
            print("Đã format trường 'date' thành công!")
        else:
            print("Không tìm thấy trường 'date' trong DataFrame")
        es_options = {
            "es.nodes": es_host,
            "es.port": es_port,
            "es.resource": es_index,
            "es.nodes.wan.only": "true",
            "es.write.operation": "index",
        }
        
        print("Bắt đầu ghi Spark DataFrame vào Elasticsearch...")

        df.write \
        .format("org.elasticsearch.spark.sql") \
        .mode("append") \
        .options(**es_options) \
        .save()
        
        print(f"Đã ghi thành công Spark DataFrame vào Elasticsearch index: {es_index}")
        
        print("Ghi DataFrame vào Elasticsearch thành công!")
        return True
        
    except Exception as e:
        print(f"Lỗi khi ghi DataFrame vào Elasticsearch: {str(e)}")
        return False

if __name__ == "__main__":
    from data_loader import get_spark_session, configure_elasticsearch_connection, load_raw_data
    from preprocessing import create_preprocessing_pipeline
    
    ES_HOST = "localhost"
    ES_PORT = "9200"
    ES_USER = None  
    ES_PASSWORD = None  
    ES_SSL = False  
    
    spark = get_spark_session("ElasticsearchDataLoading")
    configure_elasticsearch_connection(spark, ES_HOST, ES_PORT, ES_USER, ES_PASSWORD, ES_SSL)
    raw_data_df = load_raw_data(spark, ES_HOST, ES_PORT)
    if raw_data_df:
        print("\nDữ liệu thô sau khi join từ Elasticsearch:")
        raw_data_df.show(5, truncate=True)
        raw_data_df.printSchema()
        pipeline = create_preprocessing_pipeline(
                text_input_col="full_article_text",
                numerical_input_cols=["open_price", "close_price"],
                output_features_col="features",
                output_label_col="label"
        )
        
        columns_to_check_null = ["date", "symbol", "open_price", "close_price", "full_article_text"]
        cleaned_data_df = raw_data_df.na.drop(subset=columns_to_check_null)

        if cleaned_data_df.count() == 0:
            print("Không có dữ liệu sau khi loại bỏ các hàng null. Không thể fit pipeline.")
        else:
            print(f"Số lượng mẫu sau khi làm sạch null: {cleaned_data_df.count()}")
            print("\nFitting preprocessing pipeline...")
            
            pipeline_model = pipeline.fit(cleaned_data_df)

            
            print("\nTransforming data using the fitted pipeline...")
            processed_df = pipeline_model.transform(cleaned_data_df)
            print("\nDữ liệu sau khi qua pipeline tiền xử lý:")
            processed_df.printSchema()
            
            processed_df.select("date", "symbol", "label", "features").show(10, truncate=50)

            if processed_df.count() > 0:
                num_features_in_vector = len(processed_df.select("features").first()[0])
                print(f"\nSố lượng đặc trưng trong vector 'features': {num_features_in_vector}")
            model = load_prediction_model("models/stock_prediction_pipeline_model_regression")
            print("Đang áp dụng mô hình dự đoán lên stream dữ liệu...")

            predictions_stream_df = make_predictions(model, processed_df)

            if predictions_stream_df is None:
                print("Áp dụng mô hình dự đoán lên stream thất bại. Kết thúc quy trình.")

            print("Stream DataFrame sau khi áp dụng mô hình dự đoán:")
            predictions_stream_df.printSchema()
