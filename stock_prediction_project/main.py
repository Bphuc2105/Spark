# main.py

import argparse
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType # Thêm các import cần thiết
from pyspark.sql.functions import col, to_date, lit # Thêm các import cần thiết

# Import các module và cấu hình từ thư mục src
# Đảm bảo các import này là chính xác và các tệp tồn tại
try:
    from src import config
    from src.utils import get_logger # Giả sử bạn có get_logger
    from src.data_loader import load_stock_prices, load_news_articles, join_data
    from src.preprocessing import create_preprocessing_pipeline
    # SỬA ĐỔI Ở ĐÂY: Import đúng tên hàm huấn luyện hồi quy
    from src.train import train_regression_model, save_model
    from src.predict import load_prediction_model, make_predictions
except ImportError as e:
    print(f"Lỗi import trong main.py: {e}")
    # Fallback hoặc thoát nếu không thể import
    # Điều này giúp xác định rõ hơn lỗi import nếu có
    # Ví dụ:
    # if 'config' not in globals(): print("Không thể import src.config")
    # if 'get_logger' not in globals(): print("Không thể import src.utils.get_logger")
    # ... và cứ thế
    raise # Ném lại lỗi để dừng chương trình nếu import thất bại


logger = get_logger(__name__) # Đảm bảo get_logger hoạt động

def run_training_pipeline(spark):
    logger.info("Bắt đầu quy trình huấn luyện mô hình HỒI QUY...")
    logger.info(f"Đang tải dữ liệu giá từ: {config.TRAIN_PRICES_FILE}")
    prices_df = load_stock_prices(spark, config.TRAIN_PRICES_FILE)
    logger.info(f"Đang tải dữ liệu bài báo từ: {config.TRAIN_ARTICLES_FILE}")
    articles_df = load_news_articles(spark, config.TRAIN_ARTICLES_FILE)

    if not prices_df or not articles_df:
        logger.error("Không thể tải dữ liệu huấn luyện. Kết thúc quy trình huấn luyện.")
        return

    logger.info("Đang kết hợp dữ liệu giá và bài báo...")
    # Sử dụng ARTICLE_SEPARATOR từ config nếu có, nếu không thì dùng giá trị mặc định của hàm
    article_separator_to_use = getattr(config, 'ARTICLE_SEPARATOR', " --- ") # Lấy từ config hoặc mặc định
    raw_joined_df = join_data(prices_df, articles_df, article_separator=article_separator_to_use)


    if not raw_joined_df:
        logger.error("Không thể kết hợp dữ liệu. Kết thúc quy trình huấn luyện.")
        return
    
    logger.info("Dữ liệu đã join để huấn luyện:")
    raw_joined_df.show(5, truncate=True)

    logger.info("Đang tạo pipeline tiền xử lý cho hồi quy...")
    # Đảm bảo các tên cột này khớp với cấu hình và những gì create_preprocessing_pipeline mong đợi
    # Đặc biệt là output_label_col cho bài toán hồi quy
    output_label_col_name = getattr(config, 'REGRESSION_LABEL_OUTPUT_COLUMN', 'percentage_change')

    preprocessing_pipeline_obj = create_preprocessing_pipeline(
        text_input_col=config.TEXT_INPUT_COLUMN,
        numerical_input_cols=config.NUMERICAL_INPUT_COLUMNS, # Ví dụ: ["open_price"]
        output_features_col=config.FEATURES_OUTPUT_COLUMN,
        output_label_col=output_label_col_name # Nhãn cho hồi quy
    )
    preprocessing_stages = preprocessing_pipeline_obj.getStages()

    logger.info("Bắt đầu huấn luyện mô hình HỒI QUY...")
    # SỬA ĐỔI Ở ĐÂY: Gọi đúng hàm huấn luyện hồi quy
    trained_model = train_regression_model(spark, raw_joined_df, preprocessing_stages)

    if trained_model:
        # Cân nhắc đổi tên đường dẫn lưu model để phân biệt với model phân loại cũ (nếu có)
        model_save_path = getattr(config, 'SAVED_REGRESSION_MODEL_PATH', config.SAVED_MODEL_PATH + "_regression")
        logger.info(f"Đang lưu mô hình hồi quy vào: {model_save_path}")
        save_model(trained_model, model_save_path)
        logger.info("Quy trình huấn luyện hồi quy hoàn tất và mô hình đã được lưu.")
    else:
        logger.error("Huấn luyện mô hình hồi quy thất bại. Không có mô hình nào được lưu.")

def run_prediction_pipeline(spark):
    logger.info("Bắt đầu quy trình dự đoán với mô hình HỒI QUY...")
    # Đường dẫn tới mô hình hồi quy đã lưu
    model_load_path = getattr(config, 'SAVED_REGRESSION_MODEL_PATH', config.SAVED_MODEL_PATH + "_regression")
    logger.info(f"Đang tải mô hình hồi quy từ: {model_load_path}")
    prediction_model = load_prediction_model(model_load_path)

    if not prediction_model:
        logger.error("Không thể tải mô hình hồi quy. Kết thúc quy trình dự đoán.")
        return

    logger.info(f"Đang tải dữ liệu giá mới từ: {config.PREDICT_PRICES_FILE}")
    # Schema cho dữ liệu dự đoán thường không có close_price
    predict_price_schema = StructType([
        StructField("date_str", StringType(), True),
        StructField("open_price", DoubleType(), True),
        StructField("symbol", StringType(), True)
        # Không có close_price ở đây
    ])
    try:
        new_prices_df = spark.read.csv(config.PREDICT_PRICES_FILE, header=True, schema=predict_price_schema)
        new_prices_df = new_prices_df.withColumn("date", to_date(col("date_str"), "yyyy-MM-dd")).drop("date_str")
        # Thêm cột close_price là null để join_data hoạt động nếu nó yêu cầu
        # Hoặc điều chỉnh join_data để xử lý trường hợp thiếu close_price
        if "close_price" not in new_prices_df.columns:
             new_prices_df = new_prices_df.withColumn("close_price", lit(None).cast(DoubleType()))
        new_prices_df = new_prices_df.select("date", "symbol", "open_price", "close_price")

    except Exception as e:
        logger.error(f"Lỗi khi tải dữ liệu giá mới: {config.PREDICT_PRICES_FILE}. Lỗi: {e}")
        return


    logger.info(f"Đang tải dữ liệu bài báo mới từ: {config.PREDICT_ARTICLES_FILE}")
    try:
        new_articles_df = load_news_articles(spark, config.PREDICT_ARTICLES_FILE)
    except Exception as e:
        logger.error(f"Lỗi khi tải dữ liệu bài báo mới: {config.PREDICT_ARTICLES_FILE}. Lỗi: {e}")
        return


    if not new_prices_df or not new_articles_df:
        logger.error("Không thể tải dữ liệu mới để dự đoán. Kết thúc quy trình.")
        return

    logger.info("Đang kết hợp dữ liệu mới...")
    article_separator_to_use = getattr(config, 'ARTICLE_SEPARATOR', " --- ")
    input_data_df = join_data(new_prices_df, new_articles_df, article_separator=article_separator_to_use)


    if not input_data_df:
        logger.error("Không thể kết hợp dữ liệu mới. Kết thúc quy trình dự đoán.")
        return
    
    logger.info("Dữ liệu đầu vào cho dự đoán (hồi quy) sau khi xử lý:")
    input_data_df.show(truncate=False)

    logger.info("Đang thực hiện dự đoán (hồi quy)...")
    results_df = make_predictions(prediction_model, input_data_df) # make_predictions vẫn dùng được

    if results_df:
        logger.info("--- Kết quả dự đoán (Hồi quy) ---")
        # Cột 'prediction' bây giờ sẽ là giá trị phần trăm thay đổi dự kiến
        results_df.select("date", "symbol", "open_price", "full_article_text", "prediction").show(truncate=False)
        logger.info("Quy trình dự đoán (hồi quy) hoàn tất.")
        # Ví dụ: lưu kết quả
        # results_df.write.mode("overwrite").parquet(config.PREDICTIONS_OUTPUT_PATH + "_regression")
    else:
        logger.error("Dự đoán (hồi quy) thất bại.")

def main():
    parser = argparse.ArgumentParser(description="Chạy pipeline dự đoán giá cổ phiếu bằng PySpark (Hồi quy).")
    parser.add_argument(
        "mode",
        choices=["train", "predict"],
        help="Chế độ hoạt động: 'train' để huấn luyện mô hình, 'predict' để thực hiện dự đoán."
    )
    args = parser.parse_args()

    spark = None # Khởi tạo spark là None
    try:
        spark_builder = SparkSession.builder \
            .appName(config.SPARK_APP_NAME + "_Regression") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        
        spark = spark_builder.getOrCreate()
        
        logger.info(f"SparkSession đã được khởi tạo với tên ứng dụng: {spark.conf.get('spark.app.name')}")

        if args.mode == "train":
            run_training_pipeline(spark)
        elif args.mode == "predict":
            run_prediction_pipeline(spark)

    except Exception as e:
        logger.error(f"Đã xảy ra lỗi trong quá trình thực thi chính: {e}", exc_info=True)
    finally:
        if spark is not None: # Kiểm tra spark trước khi stop
            logger.info("Đang dừng SparkSession.")
            spark.stop()

if __name__ == "__main__":
    main()
