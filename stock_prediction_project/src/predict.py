# src/predict.py

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.sql.functions import col, to_date, lit

# Giả sử data_loader.py nằm trong cùng thư mục src hoặc có thể import được
try:
    from .data_loader import get_spark_session, load_news_articles, join_data
    # load_stock_prices không thực sự cần thiết ở đây nếu chúng ta định nghĩa schema thủ công
    # Hoặc, nếu PREDICT_PRICES_FILE có cấu trúc giống TRAIN_PRICES_FILE, bạn có thể dùng lại
    # from .data_loader import load_stock_prices 
except ImportError:
    print("Cảnh báo: Không thể import .data_loader trong predict.py.")
    # Cung cấp một giải pháp thay thế đơn giản nếu không tìm thấy
    def get_spark_session(app_name="DefaultApp"):
        # Đây là một fallback rất cơ bản
        from pyspark.sql import SparkSession
        print("Cảnh báo: get_spark_session không được import đúng cách, sử dụng fallback.")
        return SparkSession.builder.appName(app_name).master("local[*]").getOrCreate()
    # Các hàm load_... và join_data sẽ cần được định nghĩa hoặc import đúng cách nếu fallback

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
        return None

def make_predictions(model, data_df):
    """
    Thực hiện dự đoán trên DataFrame đầu vào bằng mô hình đã cho.
    """
    if model is None or data_df is None:
        print("Mô hình hoặc dữ liệu đầu vào là None. Không thể thực hiện dự đoán.")
        return None
    try:
        print("Thực hiện dự đoán trên dữ liệu mới...")
        predictions_df = model.transform(data_df)
        print("Dự đoán hoàn tất.")
        # Chọn các cột quan trọng để hiển thị
        # Tên cột thực tế có thể thay đổi tùy theo mô hình và dữ liệu của bạn
        # Ví dụ, nếu là mô hình hồi quy, bạn sẽ quan tâm đến cột 'prediction' chứa giá trị số.
        # Nếu mô hình có các cột trung gian như 'features', 'rawPrediction', 'probability',
        # bạn có thể chọn hiển thị chúng nếu cần.
        # Giả sử các cột cơ bản và cột dự đoán là quan trọng.
        cols_to_select = ["date", "symbol", "open_price", "full_article_text", "prediction"]
        
        # Kiểm tra xem các cột này có tồn tại trong predictions_df không
        existing_cols = [c for c in cols_to_select if c in predictions_df.columns]
        if not existing_cols:
            print("Cảnh báo: Không có cột nào trong danh sách ['date', 'symbol', 'open_price', 'full_article_text', 'prediction'] tồn tại trong kết quả dự đoán.")
            return predictions_df # Trả về toàn bộ nếu không có cột nào khớp

        result_df = predictions_df.select(existing_cols)
        return result_df
    except Exception as e:
        print(f"Lỗi trong quá trình dự đoán: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    spark = get_spark_session(app_name="StockPrediction_Predict_Standalone")

    # Đường dẫn này cần được cập nhật cho phù hợp với vị trí lưu mô hình hồi quy
    saved_model_path = "../models/stock_regression_gbt_pipeline_model" # Ví dụ

    # Đường dẫn đến dữ liệu mới cần dự đoán (tạo file mẫu nếu chưa có)
    # Giả sử thư mục data nằm ở cấp trên của src
    new_prices_path = "../data/new_prices_sample.csv"
    new_articles_path = "../data/new_articles_sample.csv"

    import os
    data_dir = "../data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(new_prices_path):
        with open(new_prices_path, "w") as f:
            f.write("date_str,open_price,symbol\n")
            f.write("2023-01-03,103.5,AAPL\n")
            f.write("2023-01-03,150.0,MSFT\n")

    if not os.path.exists(new_articles_path):
        with open(new_articles_path, "w") as f:
            f.write("date_str,article_text,symbol\n")
            f.write("2023-01-03,\"Apple is expected to announce new chip technology today.\",AAPL\n")
            f.write("2023-01-03,\"Microsoft faces new challenges in the cloud market.\",MSFT\n")

    # Tải dữ liệu giá mới
    predict_price_schema = StructType([
        StructField("date_str", StringType(), True),
        StructField("open_price", DoubleType(), True),
        StructField("symbol", StringType(), True)
    ])
    new_prices_df = spark.read.csv(new_prices_path, header=True, schema=predict_price_schema)
    new_prices_df = new_prices_df.withColumn("date", to_date(col("date_str"), "yyyy-MM-dd")).drop("date_str")
    # Thêm cột close_price là null để join_data (nếu nó yêu cầu)
    if "close_price" not in new_prices_df.columns:
         new_prices_df = new_prices_df.withColumn("close_price", lit(None).cast(DoubleType()))
    new_prices_df = new_prices_df.select("date", "symbol", "open_price", "close_price")


    # Tải dữ liệu bài báo mới
    # Giả sử hàm load_news_articles từ .data_loader đã được import đúng
    new_articles_df_loaded = load_news_articles(spark, new_articles_path)

    if new_prices_df and new_articles_df_loaded:
        # Sử dụng hàm join_data từ .data_loader
        input_data_df = join_data(new_prices_df, new_articles_df_loaded)

        if input_data_df:
            print("\nDữ liệu đầu vào cho dự đoán (standalone test):")
            input_data_df.show(truncate=False)

            prediction_model_loaded = load_prediction_model(saved_model_path)

            if prediction_model_loaded:
                results = make_predictions(prediction_model_loaded, input_data_df)
                if results:
                    print("\n--- Kết quả dự đoán (standalone test) ---")
                    results.show(truncate=False)
        else:
            print("Không thể tạo dữ liệu đầu vào cho dự đoán (standalone test).")
    else:
        print("Không thể tải dữ liệu giá hoặc bài báo mới (standalone test).")

    spark.stop()
