# src/predict.py

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
# Giả sử data_loader.py nằm trong cùng thư mục src hoặc có thể import được
try:
    from data_loader import get_spark_session, load_stock_prices, load_news_articles, join_data
except ImportError:
    print("Hãy đảm bảo data_loader.py nằm trong thư mục src/ và có thể import.")
    # Cung cấp một giải pháp thay thế đơn giản nếu không tìm thấy
    def get_spark_session(app_name="DefaultApp"):
        return SparkSession.builder.appName(app_name).master("local[*]").getOrCreate()
    # Các hàm load_... và join_data sẽ cần được định nghĩa hoặc import đúng cách

def load_prediction_model(model_path):
    """
    Tải PipelineModel đã huấn luyện từ đường dẫn được chỉ định.

    Args:
        model_path (str): Đường dẫn đến thư mục chứa mô hình đã lưu.

    Returns:
        PipelineModel: Mô hình PipelineModel đã được tải.
                       Trả về None nếu có lỗi.
    """
    try:
        print(f"Đang tải mô hình từ: {model_path}")
        model = PipelineModel.load(model_path)
        print("Tải mô hình thành công.")
        return model
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return None

def make_predictions(model, data_df):
    """
    Thực hiện dự đoán trên DataFrame đầu vào bằng mô hình đã cho.

    Args:
        model (PipelineModel): PipelineModel đã huấn luyện.
        data_df (DataFrame): DataFrame chứa dữ liệu mới cần dự đoán.
                             DataFrame này phải có các cột mà mô hình mong đợi
                             (ví dụ: 'full_article_text', 'open_price').

    Returns:
        DataFrame: DataFrame với các cột dự đoán được thêm vào (thường là 'prediction').
                   Trả về None nếu có lỗi.
    """
    if model is None or data_df is None:
        print("Mô hình hoặc dữ liệu đầu vào là None. Không thể thực hiện dự đoán.")
        return None
    try:
        print("Thực hiện dự đoán trên dữ liệu mới...")
        predictions_df = model.transform(data_df)
        print("Dự đoán hoàn tất.")
        # Chọn các cột quan trọng để hiển thị, bao gồm cả cột 'prediction'
        # Các cột thực tế có thể khác nhau tùy thuộc vào dữ liệu và mô hình của bạn
        # Ví dụ: 'date', 'symbol', 'open_price', 'full_article_text', 'prediction'
        result_df = predictions_df.select("date", "symbol", "open_price", "full_article_text", "prediction")
        return result_df
    except Exception as e:
        print(f"Lỗi trong quá trình dự đoán: {e}")
        return None

if __name__ == "__main__":
    # Khởi tạo SparkSession
    # Tên ứng dụng có thể được lấy từ config.py nếu bạn có
    spark = get_spark_session(app_name="StockPrediction_Predict")

    # --- Cấu hình đường dẫn ---
    # Đường dẫn đến mô hình đã lưu (thay đổi nếu cần)
    # Giả sử mô hình được lưu trong thư mục gốc của project là 'models/stock_prediction_pipeline_model'
    saved_model_path = "../models/stock_prediction_pipeline_model"

    # Đường dẫn đến dữ liệu mới cần dự đoán
    # Giả sử bạn có các tệp CSV tương tự như lúc huấn luyện
    # Ví dụ: dữ liệu cho một ngày mới
    new_prices_path = "../data/new_prices_sample.csv" # Dữ liệu giá mới
    new_articles_path = "../data/new_articles_sample.csv" # Dữ liệu bài báo mới

    # --- Tạo dữ liệu mẫu cho việc dự đoán (chỉ để kiểm thử) ---
    import os
    if not os.path.exists("../data"):
        os.makedirs("../data")

    # Dữ liệu giá mẫu cho dự đoán (không cần 'close_price' vì đó là thứ ta muốn dự đoán xu hướng của nó)
    # Mô hình sẽ sử dụng 'open_price' và 'full_article_text'
    if not os.path.exists(new_prices_path):
        with open(new_prices_path, "w") as f:
            f.write("date_str,open_price,symbol\n") # Không có close_price
            f.write("2023-01-03,103.5,AAPL\n")
            f.write("2023-01-03,150.0,MSFT\n")

    if not os.path.exists(new_articles_path):
        with open(new_articles_path, "w") as f:
            f.write("date_str,article_text,symbol\n")
            f.write("2023-01-03,\"Apple is expected to announce new chip technology today.\",AAPL\n")
            f.write("2023-01-03,\"Microsoft faces new challenges in the cloud market.\",MSFT\n")
            f.write("2023-01-03,\"Another article about Apple's upcoming event.\",AAPL\n")


    # --- Tải dữ liệu mới ---
    # Sử dụng các hàm từ data_loader.py
    # Lưu ý: load_stock_prices có thể cần điều chỉnh nếu nó bắt buộc phải có 'close_price'
    # Trong ví dụ này, chúng ta giả định load_stock_prices có thể xử lý việc thiếu cột close_price
    # hoặc chúng ta tạo một phiên bản load_prediction_prices.
    # Để đơn giản, giả sử file new_prices_sample.csv chỉ có các cột cần thiết.
    
    # Tạm thời, chúng ta sẽ định nghĩa schema một cách linh hoạt hơn cho dữ liệu giá dự đoán
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType
    from pyspark.sql.functions import col, to_date

    predict_price_schema = StructType([
        StructField("date_str", StringType(), True),
        StructField("open_price", DoubleType(), True),
        StructField("symbol", StringType(), True)
    ])
    new_prices_df = spark.read.csv(new_prices_path, header=True, schema=predict_price_schema)
    new_prices_df = new_prices_df.withColumn("date", to_date(col("date_str"), "yyyy-MM-dd")).drop("date_str")
    new_prices_df = new_prices_df.select("date", "symbol", "open_price")


    new_articles_df = load_news_articles(spark, new_articles_path) # Giữ nguyên hàm này

    if new_prices_df and new_articles_df:
        # Kết hợp dữ liệu giá và bài báo mới
        # Hàm join_data sẽ tổng hợp các bài báo và join với giá
        # Nó sẽ tạo ra cột 'full_article_text'
        # Nếu new_prices_df không có cột 'close_price', join_data vẫn hoạt động
        # nhưng cột 'close_price' trong DataFrame kết quả sẽ là null hoặc không tồn tại
        # Điều này không sao vì PipelineModel đã huấn luyện sẽ không sử dụng 'close_price' làm feature.
        
        # Để join_data hoạt động chính xác khi prices_df thiếu cột close_price,
        # chúng ta cần đảm bảo nó không cố gắng select cột đó nếu không tồn tại.
        # Hoặc, chúng ta có thể thêm cột close_price giả (null) vào new_prices_df
        # để khớp schema mà join_data mong đợi từ prices_df thông thường.
        from pyspark.sql.functions import lit
        if "close_price" not in new_prices_df.columns:
             new_prices_df = new_prices_df.withColumn("close_price", lit(None).cast(DoubleType()))
        
        # Bây giờ new_prices_df có các cột: date, symbol, open_price, close_price (null)
        # Và new_articles_df có: date, symbol, article_text
        
        # Hàm join_data từ data_loader.py sẽ được sử dụng.
        # Nó sẽ nhóm các bài báo và join.
        # Kết quả sẽ có: date, symbol, open_price, close_price (null), full_article_text
        input_data_df = join_data(new_prices_df, new_articles_df)

        if input_data_df:
            print("\nDữ liệu đầu vào cho dự đoán sau khi xử lý:")
            input_data_df.show(truncate=False)

            # --- Tải mô hình ---
            prediction_model = load_prediction_model(saved_model_path)

            if prediction_model:
                # --- Thực hiện dự đoán ---
                results = make_predictions(prediction_model, input_data_df)

                if results:
                    print("\n--- Kết quả dự đoán ---")
                    results.show(truncate=False)
                    # Ở đây bạn có thể lưu kết quả vào file, DB, etc.
                    # Ví dụ: results.write.mode("overwrite").parquet("../data/predictions_output")
        else:
            print("Không thể tạo dữ liệu đầu vào cho dự đoán.")
    else:
        print("Không thể tải dữ liệu giá hoặc bài báo mới.")

    # Dừng SparkSession
    spark.stop()
