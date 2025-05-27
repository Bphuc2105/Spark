# main.py

import argparse
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.sql.functions import col, to_date, lit, current_timestamp
from pyspark.sql import Window
from pyspark.sql.functions import lead
import os
import sys
import traceback

# Import các module và cấu hình từ thư mục src
try:
    # Sử dụng absolute import để đảm bảo import đúng module config
    from src import config
    from src.utils import get_logger
    # Chỉ import những hàm cần thiết cho chế độ train (CSV)
    from src.data_loader import load_raw_data, read_stream_from_kafka, configure_elasticsearch_connection     # Import hàm mới để đọc từ Kafka cho chế độ predict
    from src.preprocessing import create_preprocessing_pipeline
    from src.train import train_regression_model, save_model
    from src.predict import load_prediction_model, make_predictions, write_dataframe_to_elasticsearch
    
except ImportError as e:
    print(f"LỖI IMPORT TRONG MAIN.PY: {e}")
    print("Hãy đảm bảo cấu trúc thư mục đúng (main.py và src/...) và PYTHONPATH được thiết lập chính xác.")
    print("Nếu chạy bằng spark-submit /app/main.py trong Docker, thư mục /app cần chứa cả main.py và thư mục src.")
    # Fallback cơ bản cho logger nếu import thất bại
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    def get_logger(name): return logging.getLogger(name)
    # Thoát chương trình nếu import các module chính thất bại
    sys.exit("Không thể import các module cần thiết. Vui lòng kiểm tra cấu trúc thư mục và PYTHONPATH.")


logger = get_logger(__name__)

def run_training_pipeline(spark):
    logger.info("Bắt đầu quy trình huấn luyện mô hình HỒI QUY (đọc từ CSV)...")

    configure_elasticsearch_connection(spark, config.ES_NODES, config.ES_PORT)
    
    date_format_prices = getattr(config, 'DATE_FORMAT_PRICES', "yyyy-MM-dd")
    raw_joined_df = load_raw_data(spark, config.ES_NODES, config.ES_PORT, config.ES_PRICES_INDEX, 
                                  config.ES_ARTICLES_INDEX, config.ARTICLE_SEPARATOR, date_format_prices)

    if raw_joined_df is None:
        logger.error("Không thể kết hợp dữ liệu. Kết thúc quy trình huấn luyện.")
        return

    logger.info("Dữ liệu đã join để huấn luyện (5 dòng đầu):")
    raw_joined_df.show(5, truncate=True)

    logger.info("Đang tạo pipeline tiền xử lý cho hồi quy...")
    output_label_col_name = getattr(config, 'REGRESSION_LABEL_OUTPUT_COLUMN', 'percentage_change')
    logger.info(f"Sử dụng cột nhãn: '{output_label_col_name}'")

    text_feature_col = getattr(config, 'TEXT_INPUT_COLUMN', 'full_article_text')
    numerical_feature_cols = getattr(config, 'NUMERICAL_INPUT_COLUMNS', ['open_price'])
    features_output_col = getattr(config, 'FEATURES_OUTPUT_COLUMN', 'features')

    preprocessing_pipeline_obj = create_preprocessing_pipeline(
        text_input_col=text_feature_col,
        numerical_input_cols=numerical_feature_cols,
        output_features_col=features_output_col,
        output_label_col=output_label_col_name
    )
    preprocessing_stages = preprocessing_pipeline_obj.getStages()

    logger.info("Bắt đầu huấn luyện mô hình HỒI QUY...")
    trained_model = train_regression_model(spark, raw_joined_df, preprocessing_stages, label_col_name=output_label_col_name)

    if trained_model:
        model_save_path = getattr(config, 'SAVED_REGRESSION_MODEL_PATH', os.path.join(config.MODELS_DIR, "stock_prediction_pipeline_model_regression"))
        logger.info(f"Đang lưu mô hình hồi quy vào: {model_save_path}")

        save_successful = save_model(trained_model, model_save_path)
        if save_successful:
            logger.info(f"Quy trình huấn luyện hồi quy hoàn tất và mô hình đã được LƯU THÀNH CÔNG vào {model_save_path}.")
        else:
            logger.error(f"LƯU MÔ HÌNH THẤT BẠI vào {model_save_path}. Kiểm tra lỗi chi tiết trong log và quyền ghi thư mục.")
    else:
        logger.error("Huấn luyện mô hình hồi quy thất bại. Không có mô hình nào được tạo để lưu.")


def run_prediction_pipeline(spark):
    logger.info("Bắt đầu quy trình dự đoán với mô hình HỒI QUY (đọc từ Kafka, ghi ra Elasticsearch)...")

    # --- Bước 1: Tải mô hình đã huấn luyện ---
    model_load_path = getattr(config, 'SAVED_REGRESSION_MODEL_PATH', os.path.join(config.MODELS_DIR, "stock_prediction_pipeline_model_regression"))
    logger.info(f"Đang tải mô hình hồi quy từ: {model_load_path}")
    prediction_model = load_prediction_model(model_load_path)

    if not prediction_model:
        logger.error(f"Không thể tải mô hình hồi quy từ {model_load_path}. Kết thúc quy trình dự đoán.")
        return

    # --- Bước 2: Đọc dữ liệu từ elasticsearch ---
    # --- Cấu hình kết nối Elasticsearch ---
    configure_elasticsearch_connection(spark, config.ES_NODES, config.ES_PORT)
    
    # --- Tải dữ liệu từ Elasticsearch ---
    raw_data_df = load_raw_data(spark, config.ES_NODES, config.ES_PORT)
    if raw_data_df:
        logger.info("\nDữ liệu thô sau khi join từ Elasticsearch:")
        raw_data_df.show(5, truncate=True)
        raw_data_df.printSchema()
    else:
        print("Fail")
        return

    predictions_df = make_predictions(prediction_model, raw_data_df)

    if predictions_df is None:
        logger.error("Áp dụng mô hình dự đoán lên stream thất bại. Kết thúc quy trình.")
        return

    logger.info("Stream DataFrame sau khi áp dụng mô hình dự đoán:")
    predictions_df.printSchema()
    
    predictions_df.select("date", "symbol", "full_article_text", "percentage_change", "prediction").show(10, truncate=10)
    # Thêm cột timestamp khi dự đoán được tạo ra
    # Define a window partitioned by symbol and ordered by date
    window_spec = Window.partitionBy("symbol").orderBy("date")
    # Alternatively, if you want to shift across the entire DataFrame without partitioning:
    # window_spec = Window.orderBy("date")

    # Add prediction_timestamp as the date from the next row
    predictions_df = predictions_df.withColumn(
        "prediction_timestamp",
        lead("date").over(window_spec)
    )

    # Filter out rows where prediction_timestamp is null (this removes the last row in each partition)
    predictions_df = predictions_df.filter(predictions_df.prediction_timestamp.isNotNull())

    # Show the result to verify
    predictions_df.select("date", "symbol", "prediction", "prediction_timestamp").show(10, truncate=10)

    # Chọn các cột cần thiết để ghi vào Elasticsearch
    output_cols_for_es = [
        "id",
        "date",
        "symbol",
        "full_article_text",
        "open_price",
        "prediction",
        "prediction_timestamp"
    ]

    existing_output_cols = [c for c in output_cols_for_es if c in predictions_df.columns]
    if not existing_output_cols:
        logger.error("Không có cột nào được chọn để ghi vào Elasticsearch tồn tại trong stream DataFrame.")
        return

    final_predictions_df = predictions_df.select(existing_output_cols)

    # --- Bước 4: Ghi kết quả streaming ra Elasticsearch ---
    es_host = getattr(config, 'ELASTICSEARCH_HOST', 'elasticsearch')
    es_port = getattr(config, 'ELASTICSEARCH_PORT', '9200')
    es_prediction_index = getattr(config, 'ES_PREDICTION_INDEX', 'stock_predictions')

    logger.info(f"Đang ghi stream kết quả dự đoán ra Elasticsearch index: {es_prediction_index} tại {es_host}:{es_port}")

    write_dataframe_to_elasticsearch(
        final_predictions_df,
        es_prediction_index,
        es_host,
        es_port
    )


def main():
    parser = argparse.ArgumentParser(description="Chạy pipeline dự đoán giá cổ phiếu bằng PySpark (Hồi quy).")
    parser.add_argument(
        "mode",
        choices=["train", "predict"],
        help="Chế độ hoạt động: 'train' để huấn luyện mô hình (từ CSV), 'predict' để thực hiện dự đoán (từ Kafka)."
    )
    args = parser.parse_args()

    spark = None
    try:
        spark_builder = SparkSession.builder \
            .appName(config.SPARK_APP_NAME + "_Regression") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
            .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1,org.elasticsearch:elasticsearch-spark-30_2.12:8.11.0") \
            .config("spark.es.nodes", config.ES_NODES) \
            .config("spark.es.port", config.ES_PORT) \
            .config("spark.es.nodes.wan.only", "true")

        spark = spark_builder.getOrCreate()

        logger.info(f"SparkSession đã được khởi tạo với tên ứng dụng: {spark.conf.get('spark.app.name')}")
        logger.info(f"Spark packages: {spark.conf.get('spark.jars.packages')}")
        logger.info(f"Spark ES nodes: {spark.conf.get('spark.es.nodes')}")


        if args.mode == "train":
            run_training_pipeline(spark)
        elif args.mode == "predict":
            run_prediction_pipeline(spark)

    except Exception as e:
        logger.error(f"Đã xảy ra lỗi trong quá trình thực thi chính: {e}", exc_info=True)
    finally:
        if spark is not None:
            logger.info("Đang dừng SparkSession.")
            # Đối với streaming, awaitTermination() sẽ giữ SparkSession chạy.
            # Lệnh stop() này có thể không cần thiết hoặc chỉ chạy khi có lỗi trước khi start stream.
            # Nếu streaming query đã start và bạn nhấn Ctrl+C, nó sẽ tự dừng.
            # spark.stop() # Bỏ comment nếu cần stop SparkSession trong trường hợp không chạy streaming

if __name__ == "__main__":
    main()
