# main.py

import argparse
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType 
from pyspark.sql.functions import col, to_date, lit 
import os # Thêm os để tạo file mẫu

# Import các module và cấu hình từ thư mục src
try:
    from src import config
    from src.utils import get_logger 
    from src.data_loader import load_stock_prices, load_news_articles, join_data
    from src.preprocessing import create_preprocessing_pipeline
    from src.train import train_regression_model, save_model
    from src.predict import load_prediction_model, make_predictions
except ImportError as e:
    print(f"LỖI IMPORT TRONG MAIN.PY: {e}")
    print("Hãy đảm bảo tất cả các tệp trong thư mục 'src' đều tồn tại và sử dụng relative imports (ví dụ: from .data_loader import ...)")
    raise 


logger = get_logger(__name__) 

def run_training_pipeline(spark):
    logger.info("Bắt đầu quy trình huấn luyện mô hình HỒI QUY...")
    
    date_format_prices = getattr(config, 'DATE_FORMAT_PRICES', "yyyy-MM-dd")
    # Bỏ date_format_articles vì load_news_articles sẽ tự parse ISO timestamp

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
    logger.info(f"Sử dụng cột nhãn: '{output_label_col_name}'")

    # Lấy các cột features từ config
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
    logger.info("Bắt đầu quy trình dự đoán với mô hình HỒI QUY...")
    model_load_path = getattr(config, 'SAVED_REGRESSION_MODEL_PATH', os.path.join(config.MODELS_DIR, "stock_prediction_pipeline_model_regression"))
    logger.info(f"Đang tải mô hình hồi quy từ: {model_load_path}")
    prediction_model = load_prediction_model(model_load_path)

    if not prediction_model:
        logger.error(f"Không thể tải mô hình hồi quy từ {model_load_path}. Kết thúc quy trình dự đoán.")
        return

    predict_prices_file = getattr(config, 'PREDICT_PRICES_FILE', os.path.join(config.DATA_DIR, 'new_prices_sample.csv'))
    predict_articles_file = getattr(config, 'PREDICT_ARTICLES_FILE', os.path.join(config.DATA_DIR, 'new_articles_sample.csv'))
    date_format_predict = getattr(config, 'DATE_FORMAT_PREDICT', "yyyy-MM-dd")

    logger.info(f"Đang tải dữ liệu giá mới từ: {predict_prices_file}")
    
    # Schema cho dữ liệu dự đoán thường không có close_price ban đầu
    # Header của file new_prices_sample.csv mẫu là: date,open,symbol
    predict_price_schema = StructType([
        StructField("date", StringType(), True), 
        StructField("open", DoubleType(), True), 
        StructField("symbol", StringType(), True)
    ])
    try:
        new_prices_df = spark.read.csv(predict_prices_file, header=True, schema=predict_price_schema)
        if "open" in new_prices_df.columns:
            new_prices_df = new_prices_df.withColumnRenamed("open", "open_price")

        new_prices_df = new_prices_df.withColumn("parsed_date_col", to_date(col("date"), date_format_predict)).drop("date")
        new_prices_df = new_prices_df.withColumnRenamed("parsed_date_col", "date")
        
        if "close_price" not in new_prices_df.columns:
            new_prices_df = new_prices_df.withColumn("close_price", lit(None).cast(DoubleType()))
        
        new_prices_df = new_prices_df.select("date", "symbol", "open_price", "close_price")

    except Exception as e:
        logger.error(f"Lỗi khi tải dữ liệu giá mới: {predict_prices_file}. Lỗi: {e}", exc_info=True)
        return

    logger.info(f"Đang tải dữ liệu bài báo mới từ: {predict_articles_file}")
    try:
        new_articles_df = load_news_articles(spark, predict_articles_file)
    except Exception as e:
        logger.error(f"Lỗi khi tải dữ liệu bài báo mới: {predict_articles_file}. Lỗi: {e}", exc_info=True)
        return

    if new_prices_df is None or new_articles_df is None:
        logger.error("Không thể tải dữ liệu mới để dự đoán (giá hoặc bài báo là None). Kết thúc quy trình.")
        return

    logger.info("Đang kết hợp dữ liệu mới...")
    article_separator_to_use = getattr(config, 'ARTICLE_SEPARATOR', " --- ")
    input_data_df = join_data(new_prices_df, new_articles_df, article_separator=article_separator_to_use)

    if input_data_df is None:
        logger.error("Không thể kết hợp dữ liệu mới. Kết thúc quy trình dự đoán.")
        return
    
    logger.info("Dữ liệu đầu vào cho dự đoán (hồi quy) sau khi xử lý (5 dòng đầu):")
    input_data_df.show(5, truncate=False)

    logger.info("Đang thực hiện dự đoán (hồi quy)...")
    results_df = make_predictions(prediction_model, input_data_df) 

    if results_df:
        logger.info("--- Kết quả dự đoán (Hồi quy) ---")
        cols_to_show_results = [c for c in ["date", "symbol", "open_price", "full_article_text", "prediction"] if c in results_df.columns]
        if cols_to_show_results: # Chỉ show nếu có cột để show
            results_df.select(cols_to_show_results).show(truncate=False)
        else:
            logger.warning("Không có cột nào được chọn để hiển thị từ kết quả dự đoán. Hiển thị toàn bộ DataFrame:")
            results_df.show(truncate=False)
        logger.info("Quy trình dự đoán (hồi quy) hoàn tất.")
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

    spark = None 
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
            # Tạo file mẫu nếu chưa có để chạy predict
            data_dir = getattr(config, 'DATA_DIR', 'data')
            predict_prices_file = getattr(config, 'PREDICT_PRICES_FILE', os.path.join(data_dir, 'new_prices_sample.csv'))
            predict_articles_file = getattr(config, 'PREDICT_ARTICLES_FILE', os.path.join(data_dir, 'new_articles_sample.csv'))

            if not os.path.exists(data_dir):
                logger.info(f"Thư mục {data_dir} không tồn tại, đang tạo...")
                os.makedirs(data_dir, exist_ok=True) # exist_ok=True để không báo lỗi nếu thư mục đã tồn tại
            
            if not os.path.exists(predict_prices_file):
                logger.info(f"Tạo file mẫu {predict_prices_file} cho dự đoán.")
                with open(predict_prices_file, "w") as f:
                    f.write("date,open,symbol\n") 
                    f.write("2024-01-10,85.0,FPT\n")
                    f.write("2024-01-10,60.0,VCB\n")
            
            if not os.path.exists(predict_articles_file):
                logger.info(f"Tạo file mẫu {predict_articles_file} cho dự đoán.")
                with open(predict_articles_file, "w") as f:
                    f.write("id,date,link,title,article_text,symbol\n") # Header khớp với schema trong load_news_articles
                    f.write("pred_id1,2024-01-10T10:00:00+07:00,link1,title1,FPT news good,FPT\n")
                    f.write("pred_id2,2024-01-10T11:00:00+07:00,link2,title2,VCB stock rally,VCB\n")

            run_prediction_pipeline(spark)

    except Exception as e:
        logger.error(f"Đã xảy ra lỗi trong quá trình thực thi chính: {e}", exc_info=True)
    finally:
        if spark is not None: 
            logger.info("Đang dừng SparkSession.")
            spark.stop()

if __name__ == "__main__":
    main()
