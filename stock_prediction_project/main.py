# main.py

import argparse
from pyspark.sql import SparkSession

# Import các module và cấu hình từ thư mục src
from src import config
from src.utils import get_logger
from src.data_loader import load_stock_prices, load_news_articles, join_data
from src.preprocessing import create_preprocessing_pipeline
from src.train import train_model, save_model
from src.predict import load_prediction_model, make_predictions

logger = get_logger(__name__)

def run_training_pipeline(spark):
    logger.info("Bắt đầu quy trình huấn luyện mô hình...")
    logger.info(f"Đang tải dữ liệu giá từ: {config.TRAIN_PRICES_FILE}")
    prices_df = load_stock_prices(spark, config.TRAIN_PRICES_FILE)
    logger.info(f"Đang tải dữ liệu bài báo từ: {config.TRAIN_ARTICLES_FILE}")
    articles_df = load_news_articles(spark, config.TRAIN_ARTICLES_FILE)

    if not prices_df or not articles_df:
        logger.error("Không thể tải dữ liệu huấn luyện. Kết thúc quy trình huấn luyện.")
        return

    logger.info("Đang kết hợp dữ liệu giá và bài báo...")
    raw_joined_df = join_data(prices_df, articles_df, article_separator=config.ARTICLE_SEPARATOR)

    if not raw_joined_df:
        logger.error("Không thể kết hợp dữ liệu. Kết thúc quy trình huấn luyện.")
        return
    
    logger.info("Dữ liệu đã join để huấn luyện:")
    raw_joined_df.show(5, truncate=True)

    logger.info("Đang tạo pipeline tiền xử lý...")
    preprocessing_pipeline_obj = create_preprocessing_pipeline(
        text_input_col=config.TEXT_INPUT_COLUMN,
        numerical_input_cols=config.NUMERICAL_INPUT_COLUMNS,
        output_features_col=config.FEATURES_OUTPUT_COLUMN,
        output_label_col=config.LABEL_OUTPUT_COLUMN
    )
    preprocessing_stages = preprocessing_pipeline_obj.getStages()

    logger.info("Bắt đầu huấn luyện mô hình...")
    trained_pipeline_model = train_model(spark, raw_joined_df, preprocessing_stages)

    if trained_pipeline_model:
        logger.info(f"Đang lưu mô hình vào: {config.SAVED_MODEL_PATH}")
        save_model(trained_pipeline_model, config.SAVED_MODEL_PATH)
        logger.info("Quy trình huấn luyện hoàn tất và mô hình đã được lưu.")
    else:
        logger.error("Huấn luyện mô hình thất bại. Không có mô hình nào được lưu.")

def run_prediction_pipeline(spark):
    logger.info("Bắt đầu quy trình dự đoán...")
    logger.info(f"Đang tải mô hình từ: {config.SAVED_MODEL_PATH}")
    prediction_model = load_prediction_model(config.SAVED_MODEL_PATH)

    if not prediction_model:
        logger.error("Không thể tải mô hình. Kết thúc quy trình dự đoán.")
        return

    logger.info(f"Đang tải dữ liệu giá mới từ: {config.PREDICT_PRICES_FILE}")
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType
    from pyspark.sql.functions import col, to_date, lit

    predict_price_schema = StructType([
        StructField("date_str", StringType(), True),
        StructField("open_price", DoubleType(), True),
        StructField("symbol", StringType(), True)
    ])
    new_prices_df = spark.read.csv(config.PREDICT_PRICES_FILE, header=True, schema=predict_price_schema)
    new_prices_df = new_prices_df.withColumn("date", to_date(col("date_str"), "yyyy-MM-dd")).drop("date_str")
    new_prices_df = new_prices_df.select("date", "symbol", "open_price")
    if "close_price" not in new_prices_df.columns:
             new_prices_df = new_prices_df.withColumn("close_price", lit(None).cast(DoubleType()))

    logger.info(f"Đang tải dữ liệu bài báo mới từ: {config.PREDICT_ARTICLES_FILE}")
    new_articles_df = load_news_articles(spark, config.PREDICT_ARTICLES_FILE)

    if not new_prices_df or not new_articles_df:
        logger.error("Không thể tải dữ liệu mới để dự đoán. Kết thúc quy trình.")
        return

    logger.info("Đang kết hợp dữ liệu mới...")
    input_data_df = join_data(new_prices_df, new_articles_df, article_separator=config.ARTICLE_SEPARATOR)

    if not input_data_df:
        logger.error("Không thể kết hợp dữ liệu mới. Kết thúc quy trình dự đoán.")
        return
    
    logger.info("Dữ liệu đầu vào cho dự đoán sau khi xử lý:")
    input_data_df.show(truncate=False)

    logger.info("Đang thực hiện dự đoán...")
    results_df = make_predictions(prediction_model, input_data_df)

    if results_df:
        logger.info("--- Kết quả dự đoán ---")
        results_df.show(truncate=False)
        logger.info("Quy trình dự đoán hoàn tất.")
    else:
        logger.error("Dự đoán thất bại.")

def main():
    parser = argparse.ArgumentParser(description="Chạy pipeline dự đoán giá cổ phiếu bằng PySpark.")
    parser.add_argument(
        "mode",
        choices=["train", "predict"],
        help="Chế độ hoạt động: 'train' để huấn luyện mô hình, 'predict' để thực hiện dự đoán."
    )
    args = parser.parse_args()

    try:
        # Khởi tạo SparkSession
        # URL Master sẽ được lấy từ lệnh spark-submit trong docker-compose.yml
        # Do đó, không cần gọi .master() ở đây nữa.
        spark_builder = SparkSession.builder \
            .appName(config.SPARK_APP_NAME) \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        
        # Nếu bạn có các cấu hình Spark cụ thể khác muốn thêm từ config.py,
        # bạn có thể thêm chúng ở đây, ví dụ:
        # if hasattr(config, 'SPARK_DRIVER_MEMORY'):
        #     spark_builder = spark_builder.config("spark.driver.memory", config.SPARK_DRIVER_MEMORY)

        spark = spark_builder.getOrCreate()
        
        logger.info(f"SparkSession đã được khởi tạo với tên ứng dụng: {config.SPARK_APP_NAME}")

        if args.mode == "train":
            run_training_pipeline(spark)
        elif args.mode == "predict":
            run_prediction_pipeline(spark)

    except Exception as e:
        logger.error(f"Đã xảy ra lỗi trong quá trình thực thi chính: {e}", exc_info=True)
    finally:
        if 'spark' in locals() and spark is not None:
            logger.info("Đang dừng SparkSession.")
            spark.stop()

if __name__ == "__main__":
    main()
