# src/data_loader.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, concat_ws, collect_list, lit, from_json, expr # Thêm expr
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, DateType, TimestampType, BooleanType # Thêm BooleanType
import json
import traceback

# Import cấu hình
try:
    from . import config 
except ImportError as e:
    print(f"Lỗi import config trong data_loader.py: {e}")
    class FallbackConfigDataLoader:
        ARTICLE_SEPARATOR = " --- "
        KAFKA_BROKER = "localhost:9092"
        NEWS_ARTICLES_TOPIC = "news_articles"
        # STOCK_PRICES_TOPIC = "stock_prices" # Không thấy dùng trực tiếp trong file này
        DATE_FORMAT_PRICES = "yyyy-MM-dd"
    config = FallbackConfigDataLoader()

# Schema cho dữ liệu từ Kafka (nếu bạn dùng lại cho Kafka sau này)
# Dữ liệu từ Kafka producer (đã join) sẽ có các trường này
kafka_data_schema = StructType([
    StructField("id", StringType(), True), # ID của bài báo
    StructField("date", StringType(), True), # Ngày ở dạng chuỗi, cần parse
    StructField("link", StringType(), True),
    StructField("title", StringType(), True),
    StructField("text", StringType(), True), # Nội dung bài báo gốc
    StructField("symbol", StringType(), True),
    StructField("open_price", DoubleType(), True) # Giá mở cửa từ file giá
    # Thêm các trường khác nếu producer gửi
])


def get_spark_session(app_name="DefaultApp"):
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()
    return spark

def load_stock_prices(spark, file_path, date_format=None):
    try:
        effective_date_format = date_format or getattr(config, 'DATE_FORMAT_PRICES', "yyyy-MM-dd")
        print(f"DEBUG: Bắt đầu tải dữ liệu giá từ: {file_path} với encoding UTF-8 và định dạng ngày: {effective_date_format}")

        price_schema = StructType([
            StructField("id", StringType(), True),
            StructField("date", StringType(), True), 
            StructField("close", StringType(), True), # Đọc là string để xử lý lỗi tốt hơn
            StructField("volume", StringType(), True), 
            StructField("source", StringType(), True),
            StructField("symbol", StringType(), True),
            StructField("open", StringType(), True)    # Đọc là string
        ])

        prices_df = spark.read.csv(file_path, header=True, schema=price_schema, encoding="UTF-8")

        # Chuyển đổi 'open' và 'close' sang DoubleType, xử lý lỗi nếu có
        prices_df = prices_df.withColumn("open_price_temp", col("open").cast(DoubleType()))
        prices_df = prices_df.withColumn("close_price_temp", col("close").cast(DoubleType()))

        # Đổi tên cột
        prices_df = prices_df.withColumnRenamed("open_price_temp", "open_price")
        prices_df = prices_df.withColumnRenamed("close_price_temp", "close_price")
        
        prices_df_with_date = prices_df.withColumn("parsed_date_col", to_date(col("date"), effective_date_format))
        
        prices_df_filtered = prices_df_with_date.filter(col("parsed_date_col").isNotNull())
        
        null_open_count = prices_df_filtered.filter(col("open_price").isNull()).count()
        if null_open_count > 0:
            print(f"Cảnh báo: Tìm thấy {null_open_count} dòng có open_price là null sau khi cast sang double trong file giá.")
        
        null_close_count = prices_df_filtered.filter(col("close_price").isNull()).count()
        if null_close_count > 0:
            print(f"Cảnh báo: Tìm thấy {null_close_count} dòng có close_price là null sau khi cast sang double trong file giá.")

        # Bỏ qua các dòng có open_price hoặc close_price là null sau khi cast
        prices_df_filtered = prices_df_filtered.na.drop(subset=["open_price", "close_price"])


        count_after_filter = prices_df_filtered.count()
        if count_after_filter == 0:
            print(f"Cảnh báo: Không có dữ liệu giá nào được tải từ {file_path} sau khi lọc ngày không hợp lệ hoặc giá trị số không hợp lệ.")
            return None

        final_prices_df = prices_df_filtered.select(
            col("parsed_date_col").alias("date"), 
            "symbol",
            "open_price",
            "close_price",
            "id" 
        )
        print(f"Tải thành công {final_prices_df.count()} dòng dữ liệu giá từ {file_path}.")
        return final_prices_df
    except Exception as e:
        print(f"Lỗi nghiêm trọng khi tải dữ liệu giá từ {file_path}: {e}")
        traceback.print_exc()
        return None


def load_news_articles(spark, file_path):
    try:
        print(f"DEBUG: Bắt đầu tải dữ liệu bài báo từ: {file_path} với encoding UTF-8")
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
            .option("encoding", "UTF-8") \
            .schema(article_schema) \
            .csv(file_path)

        articles_df_with_date = articles_df_raw.withColumn("parsed_date_col", to_date(col("date"))) # Spark cố gắng tự parse
        
        articles_df_filtered = articles_df_with_date.filter(col("parsed_date_col").isNotNull())
        
        # Bỏ qua các dòng có text hoặc symbol là null
        articles_df_filtered = articles_df_filtered.na.drop(subset=["text", "symbol"])


        count_after_filter = articles_df_filtered.count()
        if count_after_filter == 0:
            print(f"Cảnh báo: Không có dữ liệu bài báo nào được tải từ {file_path} sau khi lọc ngày/text/symbol không hợp lệ.")
            return None
        
        final_articles_df = articles_df_filtered.select(
            col("parsed_date_col").alias("date"),
            "symbol",
            col("text").alias("full_article_text"), 
            "id", 
            "link", "title" 
        )
        print(f"Tải thành công {final_articles_df.count()} dòng dữ liệu bài báo từ {file_path}.")
        return final_articles_df
    except Exception as e:
        print(f"Lỗi nghiêm trọng khi tải dữ liệu bài báo từ {file_path}: {e}")
        traceback.print_exc()
        return None

def join_data(prices_df, articles_df, article_separator=None):
    if prices_df is None or articles_df is None:
        print("Dữ liệu giá hoặc bài báo là None. Không thể join.")
        return None
    if prices_df.rdd.isEmpty() or articles_df.rdd.isEmpty():
        print("Một trong hai DataFrame (giá hoặc bài báo) rỗng. Không thể join.")
        return None

    # Đảm bảo cột 'date' trong cả hai DataFrame là kiểu DateType
    if str(prices_df.schema["date"].dataType) != "DateType()":
        prices_df = prices_df.withColumn("date", to_date(col("date"), getattr(config, 'DATE_FORMAT_PRICES', "yyyy-MM-dd")))
    
    if str(articles_df.schema["date"].dataType) != "DateType()":
        articles_df = articles_df.withColumn("date", to_date(col("date"))) # Spark tự suy luận format

    # Loại bỏ các dòng có date là null sau khi chuyển đổi
    prices_df = prices_df.filter(col("date").isNotNull())
    articles_df = articles_df.filter(col("date").isNotNull())

    if prices_df.rdd.isEmpty() or articles_df.rdd.isEmpty():
        print("Một trong hai DataFrame rỗng sau khi lọc date null. Không thể join.")
        return None

    separator_to_use = article_separator if article_separator is not None else getattr(config, 'ARTICLE_SEPARATOR', " --- ")
    try:
        # Gom các bài báo theo ngày và mã cổ phiếu, nối các text lại
        aggregated_articles_df = articles_df \
            .groupBy("date", "symbol") \
            .agg(concat_ws(separator_to_use, collect_list("full_article_text")).alias("full_article_text"))
            # Nếu muốn giữ lại id, link, title, cần chiến lược khác (ví dụ: lấy first/last)
            # .agg(
            #     concat_ws(separator_to_use, collect_list(col("full_article_text"))).alias("full_article_text"),
            #     first("id").alias("article_id"), # Ví dụ
            #     first("link").alias("link"),
            #     first("title").alias("title")
            # )


        # Join prices_df với aggregated_articles_df
        # Điều này đảm bảo mỗi (date, symbol) trong prices_df sẽ được ghép với text (nếu có)
        joined_df = prices_df.join(aggregated_articles_df, on=["date", "symbol"], how="left")

        # Xử lý trường hợp không có bài báo nào cho một ngày/cổ phiếu cụ thể
        # full_article_text có thể là null. Các bước sau trong pipeline cần xử lý điều này.
        # Ví dụ, StockChunkExtractor nên trả về chuỗi rỗng nếu input text là null.
        joined_df = joined_df.na.fill({"full_article_text": ""}) # Điền rỗng nếu không có bài báo

        count = joined_df.count()
        if count > 0:
            print(f"Join dữ liệu thành công. Số dòng sau khi join: {count}")
            return joined_df
        else:
            print("Không có dữ liệu nào sau khi join. Kiểm tra lại logic join và dữ liệu đầu vào.")
            prices_df.show(5, truncate=False)
            articles_df.show(5, truncate=False)
            aggregated_articles_df.show(5, truncate=False)
            return None

    except Exception as e:
        print(f"Lỗi trong quá trình join dữ liệu: {e}")
        traceback.print_exc()
        return None

def read_stream_from_kafka(spark, kafka_broker, topic):
    """
    Đọc dữ liệu streaming từ Kafka topic và parse JSON.
    """
    try:
        print(f"Bắt đầu đọc stream từ Kafka broker: {kafka_broker}, topic: {topic}")
        kafka_df_raw = spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", kafka_broker) \
            .option("subscribe", topic) \
            .option("startingOffsets", "latest") \
            .option("failOnDataLoss", "false") \
            .load()
        
        print("Đã kết nối với Kafka stream.")
        print("Schema của Kafka stream thô:")
        kafka_df_raw.printSchema()

        # Chuyển đổi value từ binary sang string, sau đó parse JSON
        kafka_df_parsed = kafka_df_raw.select(
            from_json(col("value").cast("string"), kafka_data_schema).alias("data")
        ).select("data.*") # Lấy tất cả các trường từ struct 'data'

        print("Schema của Kafka stream sau khi parse JSON:")
        kafka_df_parsed.printSchema()

        # Đảm bảo các cột cần thiết cho pipeline tiền xử lý tồn tại
        # Pipeline tiền xử lý (sau khi bỏ SQLTransformer tạo label) sẽ cần:
        # - full_article_text (từ 'text' trong Kafka message)
        # - symbol
        # - open_price
        # - date (để join hoặc tham khảo, nhưng không trực tiếp vào VectorAssembler nếu không phải numerical)
        
        # Đổi tên cột 'text' thành 'full_article_text' nếu cần
        if "text" in kafka_df_parsed.columns and "full_article_text" not in kafka_df_parsed.columns:
            kafka_df_parsed = kafka_df_parsed.withColumnRenamed("text", "full_article_text")

        # Chọn các cột cần thiết cho các bước tiếp theo
        # Cột 'date' từ Kafka có thể là string, cần được xử lý nếu model yêu cầu DateType
        # Tuy nhiên, cho dự đoán, 'date' có thể chỉ dùng để hiển thị hoặc join, không phải input trực tiếp cho model features
        # mà không qua xử lý.
        # Hàm make_predictions sẽ xử lý các cột này.
        
        # Chuyển đổi 'date' string từ Kafka sang DateType nếu cần thiết cho logic sau này,
        # nhưng đảm bảo nó được chuyển lại thành string đúng format trước khi ghi ES.
        # Hoặc giữ nó là string và để Elasticsearch parse.
        # Hiện tại, giữ là string vì hàm load_stock_prices/articles cũng đọc là string rồi mới parse.
        # if "date" in kafka_df_parsed.columns:
        #     kafka_df_parsed = kafka_df_parsed.withColumn("date", to_date(col("date")))


        required_cols_for_pipeline = ["full_article_text", "symbol", "open_price", "date"]
        
        # Tạo các cột còn thiếu với giá trị mặc định nếu chúng không có trong stream
        # (điều này không nên xảy ra nếu producer gửi đúng schema)
        for r_col in required_cols_for_pipeline:
            if r_col not in kafka_df_parsed.columns:
                print(f"Cảnh báo: Cột '{r_col}' không có trong stream từ Kafka. Sẽ thêm cột với giá trị null/default.")
                if r_col == "open_price":
                    kafka_df_parsed = kafka_df_parsed.withColumn(r_col, lit(0.0).cast(DoubleType()))
                else:
                    kafka_df_parsed = kafka_df_parsed.withColumn(r_col, lit(None).cast(StringType()))


        final_stream_df = kafka_df_parsed.select(
            "date", # Giữ lại để có thể join hoặc hiển thị
            "symbol",
            "full_article_text",
            "open_price"
            # Thêm 'id' nếu có và cần thiết cho es.mapping.id
            # "id" 
        )
        if "id" in kafka_df_parsed.columns: # Nếu producer gửi 'id'
            final_stream_df = kafka_df_parsed.select("id", *final_stream_df.columns)


        print("Schema của stream DataFrame sẵn sàng cho pipeline tiền xử lý:")
        final_stream_df.printSchema()
        return final_stream_df

    except Exception as e:
        print(f"Lỗi khi đọc stream từ Kafka: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Phần này để test nhanh các hàm load dữ liệu nếu chạy file này trực tiếp
    spark_session = get_spark_session("DataLoaderTest")

    # Test load_stock_prices
    # Tạo file prices_sample.csv mẫu trong thư mục data/
    sample_prices_content = """id,date,close,volume,source,symbol,open
p1,2023-01-01,10.5,1000,src,SYM1,10.0
p2,2023-01-01,20.5,2000,src,SYM2,20.0
p3,bad-date,20.5,2000,src,SYM2,20.0
p4,2023-01-02,11,1200,src,SYM1,10.2
p5,2023-01-02,null,1200,src,SYM1,10.2
"""
    prices_sample_path = os.path.join(getattr(config, 'DATA_DIR', 'data'), "prices_sample_test.csv")
    with open(prices_sample_path, "w") as f:
        f.write(sample_prices_content)
    
    print(f"\n--- Testing load_stock_prices from {prices_sample_path} ---")
    df_prices = load_stock_prices(spark_session, prices_sample_path, date_format="yyyy-MM-dd")
    if df_prices:
        df_prices.show()
        df_prices.printSchema()

    # Test load_news_articles
    sample_articles_content = """id,date,link,title,text,symbol
a1,2023-01-01,link1,title1,"Good news for SYM1",SYM1
a2,2023-01-01,link2,title2,"SYM2 stock is up",SYM2
a3,2023-01-02,link3,title3,"Another news for SYM1",SYM1
a4,2023-01-02,link4,title4,null,SYM2 
""" # Dòng cuối có text là null
    articles_sample_path = os.path.join(getattr(config, 'DATA_DIR', 'data'), "articles_sample_test.csv")
    with open(articles_sample_path, "w") as f:
        f.write(sample_articles_content)

    print(f"\n--- Testing load_news_articles from {articles_sample_path} ---")
    df_articles = load_news_articles(spark_session, articles_sample_path)
    if df_articles:
        df_articles.show(truncate=False)
        df_articles.printSchema()

    # Test join_data
    if df_prices and df_articles:
        print("\n--- Testing join_data ---")
        joined_df_test = join_data(df_prices, df_articles)
        if joined_df_test:
            joined_df_test.show(truncate=False)
            joined_df_test.printSchema()

    spark_session.stop()
