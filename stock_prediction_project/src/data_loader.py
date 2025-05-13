# src/data_loader.py

from pyspark.sql import SparkSession
# from pyspark.sql.types import StructType, StructField, StringType, DoubleType, DateType, IntegerType # Tạm thời không dùng schema cứng
from pyspark.sql.functions import col, to_date
from pyspark.sql import functions as F

def get_spark_session(app_name="StockPredictionApp"):
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()
    return spark

def load_stock_prices(spark, file_path, date_format="yyyy-MM-dd"):
    # Đường dẫn tuyệt đối bên trong container, vì WORKDIR là /app
    # và volume mount là .:/app, nên data/prices.csv trên host sẽ là /app/data/prices.csv
    absolute_file_path = f"/app/{file_path}" 
    try:
        print(f"Đang thử tải dữ liệu giá từ (với inferSchema, đường dẫn tuyệt đối): {absolute_file_path}")
        prices_df = spark.read.csv(absolute_file_path, header=True, inferSchema=True)

        required_cols = ["date_str", "open_price", "close_price", "symbol"]
        missing_cols = [r_col for r_col in required_cols if r_col not in prices_df.columns]
        if missing_cols:
            print(f"LƯU Ý: Các cột sau bị thiếu trong {absolute_file_path} sau khi inferSchema: {missing_cols}")
            print("Schema được suy luận:")
            prices_df.printSchema()

        if "date_str" in prices_df.columns:
            prices_df = prices_df.withColumn("date", to_date(col("date_str"), date_format)) \
                                 .drop("date_str")
        else:
            print(f"CẢNH BÁO: Không tìm thấy cột 'date_str' để chuyển đổi trong {absolute_file_path}")

        final_cols_prices = []
        if "date" in prices_df.columns: final_cols_prices.append("date")
        if "symbol" in prices_df.columns: final_cols_prices.append("symbol")
        if "open_price" in prices_df.columns: final_cols_prices.append("open_price")
        if "close_price" in prices_df.columns: final_cols_prices.append("close_price")
        
        if final_cols_prices:
            prices_df = prices_df.select(*final_cols_prices)
        else:
            print(f"CẢNH BÁO: Không có cột nào phù hợp để chọn từ {absolute_file_path}")

        print(f"Đã tải dữ liệu giá từ: {absolute_file_path}")
        prices_df.printSchema()
        print(f"Số lượng dòng trong prices_df: {prices_df.count()}") 
        prices_df.show(5, truncate=False)
        return prices_df
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu giá từ {absolute_file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_news_articles(spark, file_path, date_format="yyyy-MM-dd"):
    absolute_file_path = f"/app/{file_path}"
    try:
        print(f"Đang thử tải dữ liệu bài báo từ (với inferSchema, đường dẫn tuyệt đối): {absolute_file_path}")
        articles_df = spark.read.csv(absolute_file_path, header=True, inferSchema=True, multiLine=True, escape="\"")

        required_cols_articles = ["date_str", "article_text", "symbol"]
        missing_cols_articles = [r_col for r_col in required_cols_articles if r_col not in articles_df.columns]
        if missing_cols_articles:
            print(f"LƯU Ý: Các cột sau bị thiếu trong {absolute_file_path} sau khi inferSchema: {missing_cols_articles}")
            print("Schema được suy luận:")
            articles_df.printSchema()

        if "date_str" in articles_df.columns:
            articles_df = articles_df.withColumn("date", to_date(col("date_str"), date_format)) \
                                     .drop("date_str")
        else:
            print(f"CẢNH BÁO: Không tìm thấy cột 'date_str' để chuyển đổi trong {absolute_file_path}")

        final_cols_articles = []
        if "date" in articles_df.columns: final_cols_articles.append("date")
        if "symbol" in articles_df.columns: final_cols_articles.append("symbol")
        if "article_text" in articles_df.columns: final_cols_articles.append("article_text")

        if final_cols_articles:
            articles_df = articles_df.select(*final_cols_articles)
        else:
            print(f"CẢNH BÁO: Không có cột nào phù hợp để chọn từ {absolute_file_path}")

        print(f"Đã tải dữ liệu bài báo từ: {absolute_file_path}")
        articles_df.printSchema()
        print(f"Số lượng dòng trong articles_df: {articles_df.count()}") 
        articles_df.show(5, truncate=True)
        return articles_df
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu bài báo từ {absolute_file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def join_data(prices_df, articles_df, article_separator=" --- "):
    if prices_df is None or articles_df is None:
        print("Không thể kết hợp dữ liệu do một trong các DataFrame đầu vào là None.")
        return None
    
    if prices_df.rdd.isEmpty():
        print("CẢNH BÁO: prices_df trống trước khi join.")
    if articles_df.rdd.isEmpty():
        print("CẢNH BÁO: articles_df trống trước khi join.")

    try:
        print("Tổng hợp các bài báo theo ngày và mã cổ phiếu...")
        # Kiểm tra cột trước khi groupBy và agg
        if "date" not in articles_df.columns or "symbol" not in articles_df.columns or "article_text" not in articles_df.columns:
            print("Lỗi: articles_df thiếu các cột 'date', 'symbol', hoặc 'article_text' để tổng hợp.")
            articles_df.printSchema()
            return None
        articles_aggregated_df = articles_df.groupBy("date", "symbol") \
                                            .agg(F.concat_ws(article_separator, F.collect_list(col("article_text"))).alias("full_article_text"))

        print("Thực hiện join dữ liệu giá và dữ liệu bài báo đã tổng hợp...")
        if "date" not in prices_df.columns or "symbol" not in prices_df.columns:
            print("Lỗi: prices_df thiếu cột 'date' hoặc 'symbol' để join.")
            prices_df.printSchema()
            return None
        if "date" not in articles_aggregated_df.columns or "symbol" not in articles_aggregated_df.columns:
            print("Lỗi: articles_aggregated_df thiếu cột 'date' hoặc 'symbol' để join.")
            articles_aggregated_df.printSchema()
            return None
            
        joined_df = prices_df.join(articles_aggregated_df, ["date", "symbol"], "inner")
        
        print("Dữ liệu sau khi join và tổng hợp bài báo:")
        joined_df.printSchema()
        print(f"Số lượng dòng trong joined_df: {joined_df.count()}") 
        joined_df.show(5, truncate=True)
        return joined_df

    except Exception as e:
        print(f"Lỗi khi kết hợp dữ liệu: {e}")
        import traceback
        traceback.print_exc()
        return None
