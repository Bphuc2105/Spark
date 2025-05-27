# main.py

import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, lit # Added lit for potential use in run_prediction_pipeline
import os
import sys
import traceback

# Import các module và cấu hình từ thư mục src
try:
    from src import config
    from src.utils import get_logger
    from src.data_loader import load_stock_prices, load_news_articles, join_data
    from src.preprocessing import create_preprocessing_pipeline
    from src.train import train_regression_model, save_model
    from src.predict import load_prediction_model, make_predictions
    # Các import không còn dùng cho predict mode mới (batch)
    # from src.data_loader import read_stream_from_kafka
    # from src.predict import write_stream_to_elasticsearch

except ImportError as e:
    print(f"LỖI IMPORT TRONG MAIN.PY: {e}")
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    def get_logger(name): return logging.getLogger(name)
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

    if raw_joined_df is None or raw_joined_df.count() == 0:
        logger.error("Không thể kết hợp dữ liệu hoặc dữ liệu sau khi join rỗng. Kết thúc quy trình huấn luyện.")
        if raw_joined_df is not None: raw_joined_df.show()
        return

    logger.info("Dữ liệu đã join để huấn luyện (5 dòng đầu):")
    raw_joined_df.show(5, truncate=True)

    logger.info("Đang tạo pipeline tiền xử lý cho hồi quy...")
    output_label_col_name = getattr(config, 'REGRESSION_LABEL_OUTPUT_COLUMN', 'percentage_change')
    text_feature_col = getattr(config, 'TEXT_INPUT_COLUMN', 'full_article_text')
    numerical_feature_cols = getattr(config, 'NUMERICAL_INPUT_COLUMNS', ['open_price'])
    features_output_col = getattr(config, 'FEATURES_OUTPUT_COLUMN', 'features')

    logger.info(f"Sử dụng cột nhãn: '{output_label_col_name}', cột features: '{features_output_col}'")

    preprocessing_pipeline_obj = create_preprocessing_pipeline(
        text_input_col=text_feature_col,
        numerical_input_cols=numerical_feature_cols,
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
            if not os.path.exists(local_model_dir) and local_model_dir : # Ensure local_model_dir is not empty
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
        # else: # Đã log ở trên nếu save_path_to_use là None
            # logger.error("Không có đường dẫn hợp lệ để lưu mô hình.")
    else:
        logger.error("Huấn luyện mô hình hồi quy thất bại. Không có mô hình nào được tạo để lưu.")


def run_prediction_pipeline(spark):
    logger.info("Bắt đầu quy trình dự đoán với mô hình HỒI QUY (đọc từ CSV, lưu kết quả ra CSV)...")

    # --- Bước 1: Tải mô hình đã huấn luyện ---
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

    # --- Bước 2: Đọc dữ liệu batch từ file CSV ---
    date_format_prices = getattr(config, 'DATE_FORMAT_PRICES', "yyyy-MM-dd")
    predict_prices_file = getattr(config, 'PREDICT_PRICES_FILE', None)
    predict_articles_file = getattr(config, 'PREDICT_ARTICLES_FILE', None)

    # Kiểm tra sự tồn tại của file trên hệ thống tệp mà Spark có thể truy cập (bên trong container)
    # Đối với Docker, đường dẫn này là /app/data/... nếu WORKDIR là /app
    # và ./data được mount vào /app/data
    # Cần đảm bảo os.path.exists kiểm tra đúng đường dẫn bên trong container
    # Hoặc để Spark tự báo lỗi nếu file không tìm thấy khi đọc
    # Tạm thời bỏ qua os.path.exists vì nó có thể không phản ánh đúng nếu đường dẫn không được xử lý tuyệt đối
    
    if not predict_prices_file: # or not os.path.exists(predict_prices_file):
        logger.error(f"File giá dự đoán '{predict_prices_file}' không được cấu hình.")
        return
    if not predict_articles_file: # or not os.path.exists(predict_articles_file):
        logger.error(f"File bài báo dự đoán '{predict_articles_file}' không được cấu hình.")
        return

    logger.info(f"Đang tải dữ liệu giá dự đoán từ: {predict_prices_file} với định dạng ngày: {date_format_prices}")
    new_prices_df = load_stock_prices(spark, predict_prices_file, date_format=date_format_prices)

    logger.info(f"Đang tải dữ liệu bài báo dự đoán từ: {predict_articles_file}")
    new_articles_df = load_news_articles(spark, predict_articles_file)

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
        if input_data_df is not None : input_data_df.show(5) # Show if not None but empty
        return

    logger.info("Dữ liệu đầu vào cho dự đoán (5 dòng đầu):")
    input_data_df.show(5, truncate=True)

    # --- Bước 3: Áp dụng mô hình dự đoán ---
    logger.info("Đang áp dụng mô hình dự đoán lên dữ liệu batch...")
    predictions_df = make_predictions(prediction_model, input_data_df)

    if predictions_df is None:
        logger.error("Thực hiện dự đoán trên dữ liệu batch thất bại. Kết thúc quy trình.")
        return

    logger.info("DataFrame sau khi áp dụng mô hình dự đoán:")
    predictions_df.printSchema()
    
    predictions_df_with_ts = predictions_df.withColumn("prediction_timestamp", current_timestamp())

    output_cols_for_csv = [
        "date", "symbol", "open_price", "full_article_text",
        "prediction", 
        "prediction_timestamp"
    ]
    if "id" in predictions_df_with_ts.columns:
        output_cols_for_csv.insert(0, "id")

    final_predictions_df = predictions_df_with_ts.select([col_name for col_name in output_cols_for_csv if col_name in predictions_df_with_ts.columns])

    logger.info("Kết quả dự đoán (5 dòng đầu):")
    final_predictions_df.show(5, truncate=False)

    # --- Bước 4: Lưu kết quả dự đoán ra thư mục CSV ---
    # Sử dụng PREDICTION_OUTPUT_DIR_CSV từ config
    prediction_output_dir = getattr(config, 'PREDICTION_OUTPUT_DIR_CSV', None)
    if prediction_output_dir:
        logger.info(f"Đang lưu kết quả dự đoán vào thư mục CSV: {prediction_output_dir}")
        try:
            final_predictions_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(prediction_output_dir)
            logger.info(f"Đã lưu kết quả dự đoán vào thư mục: {prediction_output_dir}. File CSV sẽ có dạng part-*.csv bên trong.")
            logger.info("Lưu ý: Đường dẫn này là bên trong container Spark. Nếu thư mục data được mount, bạn sẽ thấy nó trên host.")
        except Exception as e:
            logger.error(f"Lỗi khi lưu kết quả dự đoán ra thư mục CSV: {e}", exc_info=True)
    else:
        logger.warning("Không có đường dẫn PREDICTION_OUTPUT_DIR_CSV được cấu hình. Kết quả chỉ hiển thị trên console.")

    logger.info("Quy trình dự đoán batch hoàn tất.")

# ... (hàm main() giữ nguyên như đã sửa lỗi AttributeError) ...
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