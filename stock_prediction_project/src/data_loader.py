# src/data_loader.py
# src/data_loader.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, concat_ws, collect_list, lit, from_json, current_timestamp
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, DateType, TimestampType
import json
import traceback

# Import cấu hình sử dụng absolute import để đảm bảo import đúng module
try:
    from src import config
    ARTICLE_SEPARATOR = config.ARTICLE_SEPARATOR
    KAFKA_BROKER = config.KAFKA_BROKER
    NEWS_ARTICLES_TOPIC = config.NEWS_ARTICLES_TOPIC
    STOCK_PRICES_TOPIC = config.STOCK_PRICES_TOPIC
except ImportError as e:
    print(f"Lỗi import config trong data_loader.py: {e}")
    print("Cảnh báo: Không thể import config từ src, sử dụng giá trị mặc định và fallback cho Kafka/ES.")
    ARTICLE_SEPARATOR = " --- " # Fallback
    KAFKA_BROKER = "localhost:9092" # Fallback
    NEWS_ARTICLES_TOPIC = "news_articles" # Fallback
    STOCK_PRICES_TOPIC = "stock_prices" # Fallback


def get_spark_session(app_name="StockPredictionPySpark"):
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()
    return spark

# Giữ nguyên hàm load_stock_prices và load_news_articles để dùng cho chế độ training từ CSV
def load_stock_prices(spark, file_path, date_format="yyyy-MM-dd"):
    try:
        print(f"DEBUG: Bắt đầu tải dữ liệu giá từ: {file_path}")

        price_schema = StructType([
            StructField("id", StringType(), True),
            StructField("date", StringType(), True),
            StructField("close", DoubleType(), True),
            StructField("volume", StringType(), True),
            StructField("source", StringType(), True),
            StructField("symbol", StringType(), True),
            StructField("open", DoubleType(), True)
        ])

        prices_df = spark.read.csv(file_path, header=True, schema=price_schema)

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
        traceback.print_exc()
        return None


def load_news_articles(spark, file_path):
    try:
        print(f"DEBUG: Bắt đầu tải dữ liệu bài báo từ: {file_path}")

        article_schema = StructType([
            StructField("id", StringType(), True),
            StructField("date", StringType(), True),
            StructField("link", StringType(), True),
            StructField("title", StringType(), True),
            StructField("text", StringType(), True),
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
            print(f"CẢNH BẢO: Không tìm thấy cột 'article_text' hoặc 'text' trong {file_path}. Sẽ tạo cột 'text' rỗng.")
            articles_df_renamed = articles_df_raw.withColumn("text", lit("").cast(StringType()))

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

        if "text" in articles_df_filtered.columns:
            cols_to_select_articles.append("text")
        else:
            print(f"CẢNH BÁO NGHIÊM TRỌNG: Cột 'text' không có trong articles_df_filtered cho file {file_path}.")
            articles_df_filtered = articles_df_filtered.withColumn("text", lit("").cast(StringType()))
            cols_to_select_articles.append("text")

        final_articles_df = articles_df_filtered.select(*cols_to_select_articles)

        print(f"Tải thành công {final_articles_df.count()} dòng dữ liệu bài báo từ {file_path}.")
        return final_articles_df
    except Exception as e:
        print(f"Lỗi nghiêm trọng khi tải dữ liệu bài báo từ {file_path}: {e}")
        traceback.print_exc()
        return None

# Giữ nguyên hàm join_data để dùng cho chế độ training từ CSV
def join_data(prices_df, articles_df, article_separator=None):
    if prices_df is None:
        print("Dữ liệu giá là None. Không thể join.")
        return None
    if articles_df is None:
        print("Dữ liệu bài báo là None. Sẽ trả về dữ liệu giá với cột full_article_text rỗng.")
        return prices_df.withColumn("full_article_text", lit("").cast(StringType()))

    separator_to_use = article_separator if article_separator is not None else config.ARTICLE_SEPARATOR
    try:
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
        joined_df = joined_df.na.fill({"full_article_text": ""})

        print(f"Join dữ liệu thành công. Số dòng sau khi join: {joined_df.count()}")
        return joined_df
    except Exception as e:
        print(f"Lỗi trong quá trình join dữ liệu: {e}")
        traceback.print_exc()
        return None

# --- Hàm mới: Đọc dữ liệu từ Kafka bằng Spark Structured Streaming ---

kafka_data_schema = StructType([
    StructField("id", StringType(), True),
    StructField("date", StringType(), True),
    StructField("link", StringType(), True),
    StructField("title", StringType(), True),
    StructField("text", StringType(), True),
    StructField("symbol", StringType(), True),
    StructField("open_price", DoubleType(), True)
])


def read_stream_from_kafka(spark, kafka_broker, topic):
    """
    Đọc dữ liệu streaming từ Kafka topic.

    Args:
        spark (SparkSession): Đối tượng SparkSession.
        kafka_broker (str): Địa chỉ của Kafka broker (ví dụ: 'kafka:9092').
        topic (str): Tên của Kafka topic.

    Returns:
        DataFrame: Một DataFrame streaming đọc từ Kafka, đã được parse từ JSON.
                   Trả về None nếu có lỗi cấu hình hoặc kết nối.
    """
    print(f"Bắt đầu đọc stream từ Kafka broker: {kafka_broker}, topic: {topic}")
    try:
        kafka_stream_df = spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", kafka_broker) \
            .option("subscribe", topic) \
            .option("startingOffsets", "latest") \
            .load()

        print("Đã kết nối với Kafka stream.")
        print("Schema của Kafka stream thô:")
        kafka_stream_df.printSchema()

        parsed_stream_df = kafka_stream_df.selectExpr("CAST(value AS STRING)") \
            .select(from_json(col("value"), kafka_data_schema).alias("data")) \
            .select("data.*")

        print("Schema của Kafka stream sau khi parse JSON:")
        parsed_stream_df.printSchema()

        required_cols = ["date", "symbol", "text", "open_price"]
        for col_name in required_cols:
            if col_name not in parsed_stream_df.columns:
                print(f"LỖI: Cột '{col_name}' không tồn tại trong dữ liệu Kafka đã parse.")
                return None

        stream_df_for_pipeline = parsed_stream_df.select(
            col("date"),
            col("symbol"),
            col("text").alias("full_article_text"),
            col("open_price")
        )

        print("Schema của stream DataFrame sẵn sàng cho pipeline tiền xử lý:")
        stream_df_for_pipeline.printSchema()

        return stream_df_for_pipeline

    except Exception as e:
        print(f"Lỗi khi đọc stream từ Kafka: {e}")
        traceback.print_exc()
        return None


if __name__ == '__main__':
    print("--- Bỏ qua Test Standalone cho data_loader.py khi tích hợp Kafka ---")
    print("Vui lòng chạy qua docker-compose và kiểm tra luồng end-to-end.")
