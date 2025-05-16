# src/data_loader.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, concat_ws, collect_list, lit # Thêm lit
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, DateType, TimestampType

# Import cấu hình sử dụng relative import
try:
    from .config import ARTICLE_SEPARATOR
except ImportError:
    print("Cảnh báo: Không thể import .config trong data_loader.py, sử dụng giá trị ARTICLE_SEPARATOR mặc định.")
    ARTICLE_SEPARATOR = " --- "


def get_spark_session(app_name="StockPredictionPySpark"):
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()
    return spark

def load_stock_prices(spark, file_path, date_format="yyyy-MM-dd"):
    try:
        print(f"DEBUG: Bắt đầu tải dữ liệu giá từ: {file_path}")
        
        # Schema phản ánh đúng các cột và thứ tự trong prices.csv
        # Log trước đó cho thấy: id,date,close,volume,source,symbol,open
        # Đảm bảo tên cột trong schema khớp với header CSV của bạn.
        price_schema = StructType([
            StructField("id", StringType(), True),         
            StructField("date", StringType(), True),       # Cột ngày tháng (dạng string)
            StructField("close", DoubleType(), True),      # Sử dụng 'close' nếu đó là header
            StructField("volume", StringType(), True),     
            StructField("source", StringType(), True),     
            StructField("symbol", StringType(), True),
            StructField("open", DoubleType(), True)        # Sử dụng 'open' nếu đó là header
        ])

        prices_df = spark.read.csv(file_path, header=True, schema=price_schema)
        
        # Đổi tên cột 'open' và 'close' thành 'open_price' và 'close_price' để nhất quán với phần còn lại của code
        if "open" in prices_df.columns:
            prices_df = prices_df.withColumnRenamed("open", "open_price")
        if "close" in prices_df.columns:
            prices_df = prices_df.withColumnRenamed("close", "close_price")

        count_before_date_parse = prices_df.count()
        print(f"DEBUG: Số dòng prices_df tải được (sau khi áp schema và đổi tên cột nếu có): {count_before_date_parse}")
        
        if count_before_date_parse == 0:
            print(f"CẢNH BÁO: Không có dòng nào được đọc từ {file_path} với schema đã định nghĩa.")
            return None

        print(f"DEBUG: Giá trị cột 'date' (từ CSV) mẫu trước khi parse cho {file_path}:")
        prices_df.select("date").show(5, truncate=False)

        print(f"DEBUG: Chuyển đổi cột 'date' (từ CSV) sang 'date_parsed_temp' với định dạng '{date_format}' cho {file_path}")
        prices_df_with_date = prices_df.withColumn("date_parsed_temp", to_date(col("date"), date_format))
        
        print(f"DEBUG: Hiển thị một vài dòng sau khi thử parse date (cột 'date_parsed_temp') cho {file_path}:")
        prices_df_with_date.select(col("date").alias("original_date_from_csv"), "date_parsed_temp").show(5, truncate=False)

        prices_df_filtered = prices_df_with_date.filter(col("date_parsed_temp").isNotNull())
        count_after_date_filter = prices_df_filtered.count()
        print(f"DEBUG: Số dòng prices_df sau khi lọc các ngày không hợp lệ (date_parsed_temp isNotNull): {count_after_date_filter}")

        if count_after_date_filter == 0:
            print(f"Cảnh báo: Không có dữ liệu giá nào được tải từ {file_path} sau khi lọc ngày. Tất cả các giá trị trong cột 'date' (từ CSV) có thể không hợp lệ theo định dạng '{date_format}' hoặc là null.")
            return None
        
        # Chọn các cột cần thiết và đổi tên cột ngày đã parse thành "date"
        # Đảm bảo 'open_price' và 'close_price' tồn tại sau khi rename
        required_price_cols = ["symbol", "open_price", "close_price"]
        missing_cols = [c for c in required_price_cols if c not in prices_df_filtered.columns]
        if missing_cols:
            print(f"CẢNH BÁO NGHIÊM TRỌNG: Các cột giá cần thiết bị thiếu sau khi lọc: {missing_cols} trong file {file_path}")
            return None

        final_prices_df = prices_df_filtered.select(
            col("date_parsed_temp").alias("date"), 
            "symbol", 
            "open_price", 
            "close_price"
        )
        
        print(f"Tải thành công {final_prices_df.count()} dòng dữ liệu giá từ {file_path}.")
        return final_prices_df
    except Exception as e:
        print(f"Lỗi nghiêm trọng khi tải dữ liệu giá từ {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_news_articles(spark, file_path):
    try:
        print(f"DEBUG: Bắt đầu tải dữ liệu bài báo từ: {file_path}")

        # Schema dự kiến: id, date, link, title, article_text, symbol
        # Vui lòng xác nhận schema này khớp với tệp articles.csv mới của bạn
        article_schema = StructType([
            StructField("id", StringType(), True),
            StructField("date", StringType(), True), 
            StructField("link", StringType(), True),
            StructField("title", StringType(), True),
            StructField("text", StringType(), True), # Cột này sẽ được đổi tên thành 'text'
            StructField("symbol", StringType(), True) 
        ])
        
        articles_df_raw = spark.read \
            .option("header", "true") \
            .option("multiLine", "true") \
            .option("escape", "\"") \
            .schema(article_schema) \
            .csv(file_path)
            
        count_before_date_parse = articles_df_raw.count()
        print(f"DEBUG: Số dòng articles_df_raw tải được (sau khi áp schema, multiLine=true): {count_before_date_parse}")

        if count_before_date_parse == 0:
            print(f"CẢNH BÁO: Không có dòng nào được đọc từ {file_path} với schema và tùy chọn multiLine.")
            return None
        
        print(f"DEBUG: Giá trị cột 'date' (từ CSV) mẫu trước khi parse cho {file_path}:")
        articles_df_raw.select("date").show(5, truncate=False)

        articles_df_renamed = articles_df_raw
        if "article_text" in articles_df_raw.columns and "text" not in articles_df_raw.columns:
            articles_df_renamed = articles_df_raw.withColumnRenamed("article_text", "text")
        elif "article_text" not in articles_df_raw.columns and "text" not in articles_df_raw.columns:
            print(f"CẢNH BÁO: Không tìm thấy cột 'article_text' hoặc 'text' trong {file_path}. Sẽ tạo cột 'text' rỗng.")
            articles_df_renamed = articles_df_raw.withColumn("text", lit("").cast(StringType())) # Tạo cột text rỗng
        
        print(f"DEBUG: Chuyển đổi cột 'date' (từ CSV) sang 'date_parsed_temp' cho {file_path}. Spark sẽ cố gắng tự nhận diện định dạng.")
        articles_df_with_date = articles_df_renamed.withColumn("date_parsed_temp", to_date(col("date")))

        print(f"DEBUG: Hiển thị một vài dòng sau khi thử parse date (cột 'date_parsed_temp') cho {file_path}:")
        articles_df_with_date.select(col("date").alias("original_date_from_csv"), "date_parsed_temp").show(5, truncate=False)

        articles_df_filtered = articles_df_with_date.filter(col("date_parsed_temp").isNotNull())
        count_after_date_filter = articles_df_filtered.count()
        print(f"DEBUG: Số dòng articles_df sau khi lọc các ngày không hợp lệ (date_parsed_temp isNotNull): {count_after_date_filter}")

        if count_after_date_filter == 0:
            print(f"Cảnh báo: Không có dữ liệu bài báo nào được tải từ {file_path} sau khi lọc ngày.")
            return None
        
        cols_to_select_articles = []
        cols_to_select_articles.append(col("date_parsed_temp").alias("date"))
        
        if "symbol" in articles_df_filtered.columns:
             cols_to_select_articles.append("symbol")
        else:
            print(f"CẢNH BÁO: Cột 'symbol' không có trong articles_df_filtered cho file {file_path}. Sẽ bỏ qua cột symbol.")
            # Nếu symbol là bắt buộc, bạn có thể muốn trả về None ở đây hoặc xử lý khác

        if "text" in articles_df_filtered.columns:
            cols_to_select_articles.append("text")
        else:
            # Điều này không nên xảy ra nếu logic rename/tạo cột ở trên hoạt động đúng
            print(f"CẢNH BÁO NGHIÊM TRỌNG: Cột 'text' không có trong articles_df_filtered cho file {file_path}.")
            # Để tránh lỗi, chúng ta đảm bảo cột 'text' tồn tại, dù là rỗng
            articles_df_filtered = articles_df_filtered.withColumn("text", lit("").cast(StringType()))
            cols_to_select_articles.append("text")

        final_articles_df = articles_df_filtered.select(*cols_to_select_articles)

        print(f"Tải thành công {final_articles_df.count()} dòng dữ liệu bài báo từ {file_path}.")
        return final_articles_df
    except Exception as e:
        print(f"Lỗi nghiêm trọng khi tải dữ liệu bài báo từ {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def join_data(prices_df, articles_df, article_separator=None):
    if prices_df is None:
        print("Dữ liệu giá là None. Không thể join.")
        return None
    if articles_df is None:
        print("Dữ liệu bài báo là None. Sẽ trả về dữ liệu giá với cột full_article_text rỗng.")
        return prices_df.withColumn("full_article_text", lit("").cast(StringType()))

    separator_to_use = article_separator if article_separator is not None else ARTICLE_SEPARATOR
    try:
        # Kiểm tra các cột cần thiết trong articles_df trước khi join
        if "text" not in articles_df.columns:
            print("CẢNH BÁO: Cột 'text' không tồn tại trong articles_df khi join_data. Sẽ tạo cột full_article_text rỗng.")
            return prices_df.withColumn("full_article_text", lit("").cast(StringType()))
        if "symbol" not in articles_df.columns:
            print("CẢNH BÁO: Cột 'symbol' không tồn tại trong articles_df khi join_data. Không thể join theo symbol. Sẽ tạo cột full_article_text rỗng.")
            return prices_df.withColumn("full_article_text", lit("").cast(StringType()))
        if "date" not in articles_df.columns:
            print("CẢNH BÁO: Cột 'date' không tồn tại trong articles_df khi join_data. Không thể join theo date. Sẽ tạo cột full_article_text rỗng.")
            return prices_df.withColumn("full_article_text", lit("").cast(StringType()))

        aggregated_articles_df = articles_df \
            .groupBy("date", "symbol") \
            .agg(concat_ws(separator_to_use, collect_list("text")).alias("full_article_text"))
        
        joined_df = prices_df.join(aggregated_articles_df, on=["date", "symbol"], how="left")
        joined_df = joined_df.na.fill({"full_article_text": ""}) # Điền rỗng nếu không có bài báo

        print(f"Join dữ liệu thành công. Số dòng sau khi join: {joined_df.count()}")
        return joined_df
    except Exception as e:
        print(f"Lỗi trong quá trình join dữ liệu: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    spark_session = get_spark_session("DataLoaderTestStandalone")
    try:
        from .config import TRAIN_PRICES_FILE, TRAIN_ARTICLES_FILE
        test_prices_path = TRAIN_PRICES_FILE
        test_articles_path = TRAIN_ARTICLES_FILE
    except ImportError:
        base_path = "../data/" 
        test_prices_path = base_path + "prices.csv" # Đảm bảo file này tồn tại để test
        test_articles_path = base_path + "articles.csv" # Đảm bảo file này tồn tại để test

    print(f"--- Chạy Test Standalone cho data_loader.py ---")
    print(f"Đường dẫn thử nghiệm cho prices: {test_prices_path}")
    df_prices = load_stock_prices(spark_session, test_prices_path, date_format="yyyy-MM-dd")
    if df_prices:
        print("--- Kết quả test load_stock_prices ---")
        df_prices.printSchema()
        df_prices.show(5)

    print(f"\nĐường dẫn thử nghiệm cho articles: {test_articles_path}")
    df_articles = load_news_articles(spark_session, test_articles_path)
    if df_articles:
        print("--- Kết quả test load_news_articles ---")
        df_articles.printSchema()
        df_articles.show(5, truncate=50)

    if df_prices and df_articles:
        print("\n--- Test join_data ---")
        df_joined = join_data(df_prices, df_articles)
        if df_joined:
            print("Schema của df_joined:")
            df_joined.printSchema()
            df_joined.filter(col("full_article_text").isNotNull() & (col("full_article_text") != "")).show(5, truncate=70)
    elif df_prices:
        print("\n--- Test join_data (chỉ có prices_df) ---")
        df_joined_only_prices = join_data(df_prices, None)
        if df_joined_only_prices:
            df_joined_only_prices.show(5)


    spark_session.stop()
