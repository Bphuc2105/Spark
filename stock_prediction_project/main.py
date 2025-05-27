# main.py
import argparse
from pyspark.sql import SparkSession
# --- ĐẢM BẢO CÁC IMPORT NÀY CÓ Ở ĐẦU FILE ---
from pyspark.sql.functions import current_timestamp, lit, col, date_format 
from pyspark.sql.types import StringType, DateType, TimestampType # StringType và các kiểu khác được import ở đây
# ---------------------------------------------
import os
import sys
import traceback

try:
    from src import config
    from src.utils import get_logger
    from src.data_loader import load_stock_prices, load_news_articles, join_data
    from src.preprocessing import create_preprocessing_pipeline
    from src.train import train_regression_model, save_model
    from src.predict import load_prediction_model, make_predictions, write_batch_to_elasticsearch

except ImportError as e:
    print(f"LỖI IMPORT TRONG MAIN.PY: {e}")
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    def get_logger(name): return logging.getLogger(name)
    sys.exit("Không thể import các module cần thiết.")

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

    if raw_joined_df is None or raw_joined_df.rdd.isEmpty(): 
        logger.error("Không thể kết hợp dữ liệu hoặc dữ liệu sau khi join rỗng. Kết thúc quy trình huấn luyện.")
        return

    logger.info("Dữ liệu đã join để huấn luyện (5 dòng đầu):")
    raw_joined_df.show(5, truncate=True)

    logger.info("Đang tạo pipeline tiền xử lý cho hồi quy...")
    output_label_col_name = getattr(config, 'REGRESSION_LABEL_OUTPUT_COLUMN', 'percentage_change')
    text_feature_col = getattr(config, 'TEXT_INPUT_COLUMN', 'full_article_text')
    numerical_feature_cols = getattr(config, 'NUMERICAL_INPUT_COLUMNS', ['open_price'])
    features_output_col = getattr(config, 'FEATURES_OUTPUT_COLUMN', 'features')
    symbol_col_name = "symbol" 

    logger.info(f"Sử dụng cột nhãn: '{output_label_col_name}', cột features: '{features_output_col}'")

    preprocessing_pipeline_obj = create_preprocessing_pipeline(
        text_input_col=text_feature_col,
        numerical_input_cols=numerical_feature_cols,
        symbol_col=symbol_col_name, 
        output_features_col=features_output_col,
        output_label_col=output_label_col_name
    )
    actual_preprocessing_stages = preprocessing_pipeline_obj.getStages()

    logger.info("Bắt đầu huấn luyện mô hình HỒI QUY...")
    trained_model = train_regression_model(
        spark,
        raw_joined_df,
        actual_preprocessing_stages,
        label_col_name=output_label_col_name,
        features_col_name=features_output_col
    )

    if trained_model:
        model_save_path_hdfs = getattr(config, 'HDFS_MODEL_SAVE_PATH', None)
        model_save_path_local = getattr(config, 'LOCAL_SAVED_REGRESSION_MODEL_PATH', None)
        save_path_to_use = None
        is_hdfs = False

        if model_save_path_hdfs and model_save_path_hdfs.startswith("hdfs://"):
            logger.info(f"Sẽ lưu mô hình lên HDFS: {model_save_path_hdfs}")
            save_path_to_use = model_save_path_hdfs
            is_hdfs = True
        elif model_save_path_local:
            logger.info(f"Không có cấu hình HDFS hợp lệ. Sẽ lưu mô hình cục bộ tại: {model_save_path_local}")
            save_path_to_use = model_save_path_local
            is_hdfs = False
            local_model_dir = os.path.dirname(model_save_path_local)
            if not os.path.exists(local_model_dir) and local_model_dir : 
                try:
                    os.makedirs(local_model_dir, exist_ok=True)
                    logger.info(f"Đã tạo thư mục cục bộ: {local_model_dir}")
                except Exception as e_mkdir:
                    logger.error(f"Không thể tạo thư mục cục bộ {local_model_dir}: {e_mkdir}")
                    save_path_to_use = None
        else:
            logger.error("Không có đường dẫn hợp lệ (HDFS hoặc local) để lưu mô hình.")


        if save_path_to_use:
            save_successful = save_model(trained_model, save_path_to_use, is_hdfs_path=is_hdfs)
            if save_successful:
                logger.info(f"Quy trình huấn luyện hoàn tất và mô hình đã được LƯU THÀNH CÔNG vào {save_path_to_use}.")
            else:
                logger.error(f"LƯU MÔ HÌNH THẤT BẠI vào {save_path_to_use}.")
    else:
        logger.error("Huấn luyện mô hình hồi quy thất bại. Không có mô hình nào được tạo để lưu.")


def run_prediction_pipeline(spark):
    logger.info("Bắt đầu quy trình dự đoán với mô hình HỒI QUY (đọc từ CSV, lưu kết quả ra Elasticsearch)...")

    model_load_path_hdfs = getattr(config, 'HDFS_MODEL_SAVE_PATH', None)
    model_load_path_local = getattr(config, 'LOCAL_SAVED_REGRESSION_MODEL_PATH', None)
    model_load_path = None

    if model_load_path_hdfs and model_load_path_hdfs.startswith("hdfs://"):
        logger.info(f"Sẽ tải mô hình từ HDFS: {model_load_path_hdfs}")
        model_load_path = model_load_path_hdfs
    elif model_load_path_local:
        logger.info(f"Không tìm thấy cấu hình HDFS hợp lệ, sẽ tải mô hình từ cục bộ: {model_load_path_local}")
        model_load_path = model_load_path_local
    else:
        logger.error("Không có đường dẫn hợp lệ (HDFS hoặc local) để tải mô hình. Kết thúc quy trình dự đoán.")
        return

    prediction_model = load_prediction_model(model_load_path)

    if not prediction_model:
        logger.error(f"Không thể tải mô hình hồi quy từ {model_load_path}. Kết thúc quy trình dự đoán.")
        return

    date_format_prices = getattr(config, 'DATE_FORMAT_PRICES', "yyyy-MM-dd")
    predict_prices_file = getattr(config, 'PREDICT_PRICES_FILE', None)
    predict_articles_file = getattr(config, 'PREDICT_ARTICLES_FILE', None)
    
    path_inside_container_prices = os.path.join("/app", predict_prices_file) if predict_prices_file else None
    path_inside_container_articles = os.path.join("/app", predict_articles_file) if predict_articles_file else None

    if not path_inside_container_prices or not os.path.exists(path_inside_container_prices):
        logger.error(f"File giá dự đoán '{path_inside_container_prices}' không được cấu hình hoặc không tồn tại bên trong container.")
        return
    if not path_inside_container_articles or not os.path.exists(path_inside_container_articles):
        logger.error(f"File bài báo dự đoán '{path_inside_container_articles}' không được cấu hình hoặc không tồn tại bên trong container.")
        return

    logger.info(f"Đang tải dữ liệu giá dự đoán từ: {predict_prices_file} (path trong container: {path_inside_container_prices}) với định dạng ngày: {date_format_prices}")
    new_prices_df = load_stock_prices(spark, path_inside_container_prices, date_format=date_format_prices)

    logger.info(f"Đang tải dữ liệu bài báo dự đoán từ: {predict_articles_file} (path trong container: {path_inside_container_articles})")
    new_articles_df = load_news_articles(spark, path_inside_container_articles)


    if new_prices_df is None or new_articles_df is None:
        logger.error("Không thể tải dữ liệu dự đoán (giá hoặc bài báo là None). Kết thúc quy trình.")
        return

    logger.info("Đang kết hợp dữ liệu giá và bài báo cho dự đoán...")
    article_separator_to_use = getattr(config, 'ARTICLE_SEPARATOR', " --- ")
    
    if "close_price" not in new_prices_df.columns:
        logger.warning("Cột 'close_price' không có trong new_prices_df cho dự đoán. Thêm cột giả giá trị 0.0.")
        new_prices_df = new_prices_df.withColumn("close_price", lit(0.0))

    input_data_df = join_data(new_prices_df, new_articles_df, article_separator=article_separator_to_use)

    if input_data_df is None or input_data_df.rdd.isEmpty():
        logger.error("Không có dữ liệu đầu vào sau khi join để thực hiện dự đoán. Kết thúc quy trình.")
        if input_data_df is not None : input_data_df.show(5)
        return

    logger.info("Dữ liệu đầu vào cho dự đoán (5 dòng đầu):")
    input_data_df.show(5, truncate=True)

    logger.info("Đang áp dụng mô hình dự đoán lên dữ liệu batch...")
    predictions_df = make_predictions(prediction_model, input_data_df)

    if predictions_df is None:
        logger.error("Thực hiện dự đoán trên dữ liệu batch thất bại. Kết thúc quy trình.")
        return

    predictions_df_with_ts = predictions_df.withColumn("prediction_timestamp", current_timestamp())

    output_cols_for_es = [
        "date", "symbol", "open_price", "full_article_text",
        "prediction", "prediction_timestamp"
    ]
    if "id" in predictions_df_with_ts.columns:
         output_cols_for_es.insert(0, "id")
    else:
        logger.warning("Không tìm thấy cột 'id' trong DataFrame dự đoán. Document ID trong Elasticsearch có thể sẽ tự động được tạo.")

    final_predictions_df_for_es = predictions_df_with_ts.select(
        [col_name for col_name in output_cols_for_es if col_name in predictions_df_with_ts.columns]
    )
    
    # --- ĐỊNH DẠNG LẠI CÁC CỘT NGÀY THÁNG TRƯỚC KHI GHI VÀO ELASTICSEARCH ---
    if "date" in final_predictions_df_for_es.columns:
        # Đảm bảo cột 'date' là kiểu DateType trước khi format
        # Hàm load_stock_prices/load_news_articles đã chuyển nó thành DateType
        logger.info("Chuyển đổi cột 'date' sang StringType 'yyyy-MM-dd' cho Elasticsearch.")
        final_predictions_df_for_es = final_predictions_df_for_es.withColumn("date", date_format(col("date"), "yyyy-MM-dd"))
    
    if "prediction_timestamp" in final_predictions_df_for_es.columns:
        # Chuyển TimestampType thành chuỗi ISO 8601 (Elasticsearch thường hiểu được điều này cho kiểu 'date')
        logger.info("Chuyển đổi cột 'prediction_timestamp' sang StringType (ISO 8601) cho Elasticsearch.")
        final_predictions_df_for_es = final_predictions_df_for_es.withColumn("prediction_timestamp", col("prediction_timestamp").cast(StringType()))
    # --------------------------------------------------------------------
    
    logger.info("Kết quả dự đoán sẽ được ghi vào Elasticsearch (5 dòng đầu sau khi định dạng date):")
    final_predictions_df_for_es.show(5, truncate=False)
    final_predictions_df_for_es.printSchema() # Kiểm tra schema cuối cùng trước khi ghi

    es_host = getattr(config, 'ELASTICSEARCH_HOST', 'elasticsearch')
    es_port = str(getattr(config, 'ELASTICSEARCH_PORT', '9200'))
    es_prediction_index = getattr(config, 'ES_PREDICTION_INDEX', 'stock_predictions_batch')

    logger.info(f"Đang ghi batch kết quả dự đoán ra Elasticsearch index: {es_prediction_index} tại {es_host}:{es_port}")
    
    write_successful = write_batch_to_elasticsearch(
        final_predictions_df_for_es, # DataFrame đã được định dạng lại ngày tháng
        es_prediction_index,
        es_nodes=es_host,
        es_port=es_port
    )

    if write_successful:
        logger.info(f"Đã ghi thành công kết quả dự đoán vào Elasticsearch index: {es_prediction_index}")
    else:
        logger.error(f"Ghi kết quả dự đoán vào Elasticsearch THẤT BẠI. Kiểm tra log chi tiết.")

    logger.info("Quy trình dự đoán batch hoàn tất.")

def main():
    parser = argparse.ArgumentParser(description="Chạy pipeline dự đoán giá cổ phiếu bằng PySpark (Hồi quy).")
    parser.add_argument(
        "mode",
        choices=["train", "predict"],
        help="Chế độ hoạt động: 'train' để huấn luyện mô hình, 'predict' để thực hiện dự đoán (batch từ CSV)."
    )
    args = parser.parse_args()

    spark = None
    try:
        spark_builder = SparkSession.builder \
            .appName(config.SPARK_APP_NAME + "_Regression_" + args.mode.upper()) \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
            
        # Thêm packages và config cho ES nếu chế độ predict cần ghi vào ES
        if args.mode == "predict" or getattr(config, 'ALWAYS_INCLUDE_ES_PACKAGE', True): # Mặc định là True để luôn có ES package
            spark_builder = spark_builder.config("spark.jars.packages", "org.elasticsearch:elasticsearch-spark-30_2.12:8.11.0") \
                                .config("spark.es.nodes", getattr(config, 'ES_NODES', 'elasticsearch')) \
                                .config("spark.es.port", str(getattr(config, 'ES_PORT', '9200'))) \
                                .config("spark.es.nodes.wan.only", "true")
            
        spark = spark_builder.getOrCreate()

        logger.info(f"SparkSession đã được khởi tạo với tên ứng dụng: {spark.conf.get('spark.app.name')}")
        
        packages_conf = spark.conf.get("spark.jars.packages", None)
        if packages_conf is not None:
            logger.info(f"Spark packages: {packages_conf}")

        es_nodes_conf = spark.conf.get("spark.es.nodes", None)
        if es_nodes_conf is not None:
            es_port_conf = spark.conf.get("spark.es.port", "N/A") 
            logger.info(f"Spark ES nodes: {es_nodes_conf}:{es_port_conf}")

        if args.mode == "train":
            run_training_pipeline(spark)
        elif args.mode == "predict":
            run_prediction_pipeline(spark)

    except Exception as e:
        logger.error(f"Đã xảy ra lỗi trong quá trình thực thi chính: {e}", exc_info=True)
        traceback.print_exc() 
    finally:
        if spark is not None:
            logger.info("Đang dừng SparkSession.")
            spark.stop()

if __name__ == "__main__":
    main()
