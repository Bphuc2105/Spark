# main.py

import argparse
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.sql.functions import col, to_date, lit, current_timestamp
import os
import sys
import traceback

# Import các module và cấu hình từ thư mục src
try:
    # Sử dụng absolute import để đảm bảo import đúng module config
    from src import config
    from src.utils import get_logger
    # Chỉ import những hàm cần thiết cho chế độ train (CSV)
    from src.data_loader import load_stock_prices, load_news_articles, join_data, read_stream_from_kafka     # Import hàm mới để đọc từ Kafka cho chế độ predict
    from src.preprocessing import create_preprocessing_pipeline
    from src.train import train_regression_model, save_model # Đảm bảo save_model cũng được import nếu bạn dùng nó ở main
    from src.predict import load_prediction_model, make_predictions, write_stream_to_elasticsearch

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
    logger.info("Bắt đầu quy trình huấn luyện mô hình HỒI QUY (đọc từ CSV, lưu lên HDFS)...")

    date_format_prices = getattr(config, 'DATE_FORMAT_PRICES', "yyyy-MM-dd")

    logger.info(f"Đang tải dữ liệu giá từ: {config.TRAIN_PRICES_FILE} với định dạng ngày: {date_format_prices}")
    prices_df = load_stock_prices(spark, config.TRAIN_PRICES_FILE, date_format=date_format_prices)

    logger.info(f"Đang tải dữ liệu bài báo từ: {config.TRAIN_ARTICLES_FILE} (Spark tự nhận diện định dạng ngày)")
    articles_df = load_news_articles(spark, config.TRAIN_ARTICLES_FILE)

    if prices_df is None or articles_df is None:
        logger.error("Không thể tải dữ liệu huấn luyện (giá hoặc bài báo là None). Kết thúc quy trình huấn luyện.")
        return

    logger.info("Đang kết hợp dữ liệu giá và bài báo...")
    article_separator_to_use = getattr(config, 'ARTICLE_SEPARATOR', " --- ")
    raw_joined_df = join_data(prices_df, articles_df, article_separator=article_separator_to_use)

    if raw_joined_df is None:
        logger.error("Không thể kết hợp dữ liệu. Kết thúc quy trình huấn luyện.")
        return

    logger.info("Dữ liệu đã join để huấn luyện (5 dòng đầu):")
    raw_joined_df.show(5, truncate=True)

    logger.info("Đang tạo pipeline tiền xử lý cho hồi quy...")
    output_label_col_name = getattr(config, 'REGRESSION_LABEL_OUTPUT_COLUMN', 'percentage_change')
    text_feature_col = getattr(config, 'TEXT_INPUT_COLUMN', 'full_article_text')
    numerical_feature_cols = getattr(config, 'NUMERICAL_INPUT_COLUMNS', ['open_price'])
    features_output_col = getattr(config, 'FEATURES_OUTPUT_COLUMN', 'features') # Đây là "features"

    logger.info(f"Sử dụng cột nhãn: '{output_label_col_name}', cột features: '{features_output_col}'")

    preprocessing_pipeline_obj = create_preprocessing_pipeline(
        text_input_col=text_feature_col,
        numerical_input_cols=numerical_feature_cols,
        output_features_col=features_output_col, # output_features_col được đặt là "features"
        output_label_col=output_label_col_name
    )
    
    # === SỬA ĐỔI QUAN TRỌNG ===
    # Lấy danh sách các giai đoạn (stages) từ đối tượng Pipeline tiền xử lý
    actual_preprocessing_stages = preprocessing_pipeline_obj.getStages()
    # ==========================

    logger.info("Bắt đầu huấn luyện mô hình HỒI QUY...")
    # Truyền danh sách các giai đoạn đã lấy được vào hàm huấn luyện
    trained_model = train_regression_model(
        spark,
        raw_joined_df,
        actual_preprocessing_stages, # Truyền danh sách các stages
        label_col_name=output_label_col_name
        # File train.py của bạn hardcode featuresCol="features" trong GBTRegressor,
        # nên không cần truyền features_col_name ở đây.
    )

    if trained_model:
        model_save_path_hdfs = getattr(config, 'HDFS_MODEL_SAVE_PATH', None)
        model_save_path_local = getattr(config, 'LOCAL_SAVED_REGRESSION_MODEL_PATH', os.path.join(config.MODELS_DIR, "stock_prediction_pipeline_model_regression_local_fallback"))

        save_path_to_use = None
        is_hdfs = False

        if model_save_path_hdfs and model_save_path_hdfs.startswith("hdfs://"):
            logger.info(f"Sẽ lưu mô hình lên HDFS: {model_save_path_hdfs}")
            save_path_to_use = model_save_path_hdfs
            is_hdfs = True
        else:
            logger.warning(f"Đường dẫn HDFS_MODEL_SAVE_PATH ('{model_save_path_hdfs}') không hợp lệ hoặc không được cấu hình. Sẽ thử lưu cục bộ.")
            logger.info(f"Sẽ lưu mô hình cục bộ tại: {model_save_path_local}")
            save_path_to_use = model_save_path_local
            is_hdfs = False
            local_model_dir = os.path.dirname(model_save_path_local)
            if not os.path.exists(local_model_dir):
                try:
                    os.makedirs(local_model_dir, exist_ok=True)
                    logger.info(f"Đã tạo thư mục cục bộ: {local_model_dir}")
                except Exception as e_mkdir:
                    logger.error(f"Không thể tạo thư mục cục bộ {local_model_dir}: {e_mkdir}")
                    save_path_to_use = None

        if save_path_to_use:
            # Giả sử hàm save_model trong train.py đã được cập nhật để xử lý is_hdfs_path
            save_successful = save_model(trained_model, save_path_to_use, is_hdfs_path=is_hdfs if hasattr(save_model, 'is_hdfs_path') else False)
            if save_successful:
                logger.info(f"Quy trình huấn luyện hoàn tất và mô hình đã được LƯU THÀNH CÔNG vào {save_path_to_use}.")
            else:
                logger.error(f"LƯU MÔ HÌNH THẤT BẠI vào {save_path_to_use}. Kiểm tra lỗi chi tiết trong log và quyền ghi.")
        else:
            logger.error("Không có đường dẫn hợp lệ để lưu mô hình.")
    else:
        logger.error("Huấn luyện mô hình hồi quy thất bại. Không có mô hình nào được tạo để lưu.")


def run_prediction_pipeline(spark):
    logger.info("Bắt đầu quy trình dự đoán với mô hình HỒI QUY (đọc từ Kafka, ghi ra Elasticsearch)...")

    model_load_path_hdfs = getattr(config, 'HDFS_MODEL_SAVE_PATH', None)
    model_load_path_local = getattr(config, 'LOCAL_SAVED_REGRESSION_MODEL_PATH', os.path.join(config.MODELS_DIR, "stock_prediction_pipeline_model_regression_local_fallback"))
    model_load_path = None

    if model_load_path_hdfs and model_load_path_hdfs.startswith("hdfs://"):
        logger.info(f"Sẽ tải mô hình từ HDFS: {model_load_path_hdfs}")
        model_load_path = model_load_path_hdfs
    else:
        logger.warning(f"Đường dẫn HDFS_MODEL_SAVE_PATH ('{model_load_path_hdfs}') không hợp lệ hoặc không được cấu hình. Sẽ thử tải từ cục bộ.")
        logger.info(f"Sẽ tải mô hình từ cục bộ: {model_load_path_local}")
        model_load_path = model_load_path_local


    if not model_load_path:
        logger.error("Không có đường dẫn hợp lệ để tải mô hình. Kết thúc quy trình dự đoán.")
        return

    prediction_model = load_prediction_model(model_load_path)

    if not prediction_model:
        logger.error(f"Không thể tải mô hình hồi quy từ {model_load_path}. Kết thúc quy trình dự đoán.")
        return

    kafka_broker = getattr(config, 'KAFKA_BROKER', 'kafka:9092')
    news_articles_topic = getattr(config, 'NEWS_ARTICLES_TOPIC', 'news_articles')
    logger.info(f"Đang đọc stream từ Kafka broker: {kafka_broker}, topic: {news_articles_topic}")

    streaming_input_df = read_stream_from_kafka(spark, kafka_broker, news_articles_topic)

    if streaming_input_df is None:
        logger.error("Không thể khởi tạo stream từ Kafka. Kết thúc quy trình dự đoán.")
        return

    logger.info("Stream DataFrame từ Kafka đã sẵn sàng.")
    streaming_input_df.printSchema()

    logger.info("Đang áp dụng mô hình dự đoán lên stream dữ liệu...")
    predictions_stream_df = make_predictions(prediction_model, streaming_input_df)


    if predictions_stream_df is None:
        logger.error("Áp dụng mô hình dự đoán lên stream thất bại. Kết thúc quy trình.")
        return

    logger.info("Stream DataFrame sau khi áp dụng mô hình dự đoán:")
    predictions_stream_df.printSchema()

    predictions_stream_df = predictions_stream_df.withColumn("prediction_timestamp", current_timestamp())

    output_cols_for_es = [
        "id", 
        "date",
        "symbol",
        "full_article_text",
        "open_price",
        "prediction", 
        "prediction_timestamp"
    ]
    
    final_output_cols = [c for c in output_cols_for_es if c in predictions_stream_df.columns]
    if not final_output_cols:
         logger.error("Không có cột nào được chọn để ghi vào Elasticsearch tồn tại trong stream DataFrame.")
         return
    
    final_predictions_stream_df = predictions_stream_df.select(final_output_cols)

    es_host = getattr(config, 'ELASTICSEARCH_HOST', 'elasticsearch')
    es_port = str(getattr(config, 'ELASTICSEARCH_PORT', '9200')) 
    es_prediction_index = getattr(config, 'ES_PREDICTION_INDEX', 'stock_predictions')

    logger.info(f"Đang ghi stream kết quả dự đoán ra Elasticsearch index: {es_prediction_index} tại {es_host}:{es_port}")

    streaming_query = write_stream_to_elasticsearch(
        final_predictions_stream_df,
        es_prediction_index,
        es_host,
        es_port
    )

    if streaming_query:
        logger.info("Streaming query để ghi ra Elasticsearch đã bắt đầu.")
        logger.info("Đang chờ stream kết thúc (Ctrl+C để dừng)...")
        try:
            streaming_query.awaitTermination()
        except KeyboardInterrupt:
            logger.info("Đã nhận tín hiệu dừng (KeyboardInterrupt). Đang dừng streaming query...")
        finally:
            logger.info("Streaming query đã dừng.")
    else:
        logger.error("Không thể bắt đầu streaming query để ghi ra Elasticsearch.")


def main():
    parser = argparse.ArgumentParser(description="Chạy pipeline dự đoán giá cổ phiếu bằng PySpark (Hồi quy).")
    parser.add_argument(
        "mode",
        choices=["train", "predict"],
        help="Chế độ hoạt động: 'train' để huấn luyện mô hình, 'predict' để thực hiện dự đoán."
    )
    args = parser.parse_args()

    spark = None
    try:
        es_nodes_conf = getattr(config, 'ES_NODES', 'elasticsearch')
        es_port_conf = str(getattr(config, 'ES_PORT', '9200'))

        spark_builder = SparkSession.builder \
            .appName(config.SPARK_APP_NAME + "_Regression") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
            .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1,org.elasticsearch:elasticsearch-spark-30_2.12:8.11.0") \
            .config("spark.es.nodes", es_nodes_conf) \
            .config("spark.es.port", es_port_conf) \
            .config("spark.es.nodes.wan.only", "true")

        spark = spark_builder.getOrCreate()

        logger.info(f"SparkSession đã được khởi tạo với tên ứng dụng: {spark.conf.get('spark.app.name')}")
        logger.info(f"Spark UI: http://localhost:4040 (nếu chạy local) hoặc UI của Spark Master")
        logger.info(f"Spark packages: {spark.conf.get('spark.jars.packages')}")
        logger.info(f"Spark ES nodes: {spark.conf.get('spark.es.nodes')}:{spark.conf.get('spark.es.port')}")


        if args.mode == "train":
            run_training_pipeline(spark)
        elif args.mode == "predict":
            run_prediction_pipeline(spark)

    except Exception as e:
        logger.error(f"Đã xảy ra lỗi trong quá trình thực thi chính: {e}", exc_info=True)
        traceback.print_exc() 
    finally:
        if spark is not None and args.mode == "train": 
            logger.info("Đang dừng SparkSession sau khi huấn luyện.")
            spark.stop()
        elif spark is not None and args.mode == "predict":
             logger.info("SparkSession sẽ tiếp tục chạy cho streaming predict. Dừng thủ công nếu cần.")

if __name__ == "__main__":
    main()