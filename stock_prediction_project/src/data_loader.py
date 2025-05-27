# src/data_loader.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, from_json, to_timestamp
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, DateType, TimestampType
from pyspark.sql import functions as F
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


def get_spark_session(app_name="StockPredictionApp"):
    """
    Khởi tạo và trả về một SparkSession với Elasticsearch connector.
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .config("spark.jars.packages", "org.elasticsearch:elasticsearch-spark-30_2.12:8.11.0") \
        .getOrCreate()
    return spark

def load_stock_prices(spark, es_host="localhost", es_port="9200", es_index="prices", 
                     date_format_in_file="yyyy-MM-dd HH:mm:ssX"):
    """
    Tải dữ liệu giá cổ phiếu từ Elasticsearch index.
    Xử lý các tên cột và chuyển đổi cột ngày.
    Schema mong đợi: id, create_date (hoặc time), close, volume, source, symbol, open
    """
    try:
        print(f"Đang thử tải dữ liệu giá từ Elasticsearch: {es_host}:{es_port}/{es_index}")
        
        # Cấu hình Elasticsearch
        es_options = {
            "es.nodes": es_host,
            "es.port": es_port,
            "es.resource": es_index,
            "es.read.field.as.array.include": "tags",
            "es.nodes.wan.only": "true"
        }
        
        # Đọc dữ liệu từ Elasticsearch
        prices_df = spark.read.format("org.elasticsearch.spark.sql") \
                              .options(**es_options) \
                              .load()

        print("Schema của prices_df sau khi đọc từ Elasticsearch:")
        prices_df.printSchema()

        # Đổi tên cột 'open' và 'close' nếu chúng tồn tại và chưa đúng tên
        if "open" in prices_df.columns and "open_price" not in prices_df.columns:
            prices_df = prices_df.withColumnRenamed("open", "open_price")
        if "close" in prices_df.columns and "close_price" not in prices_df.columns:
            prices_df = prices_df.withColumnRenamed("close", "close_price")

        date_col_source = None
        date_col_final_name = "date"

        # Ưu tiên các tên cột ngày có thể có trong Elasticsearch index
        if "create_date" in prices_df.columns:
            date_col_source = "create_date"
            print(f"Sử dụng cột '{date_col_source}' để tạo cột '{date_col_final_name}' cho prices_df.")
            prices_df = prices_df.withColumn(date_col_final_name, to_date(to_timestamp(col(date_col_source), date_format_in_file)))
        elif "time" in prices_df.columns: 
            date_col_source = "time"
            print(f"Sử dụng cột '{date_col_source}' để tạo cột '{date_col_final_name}' cho prices_df.")
            prices_df = prices_df.withColumn(date_col_final_name, to_date(to_timestamp(col(date_col_source), date_format_in_file)))
        elif "date_str" in prices_df.columns: 
            date_col_source = "date_str"
            prices_df = prices_df.withColumn(date_col_final_name, to_date(col(date_col_source), "yyyy-MM-dd")) 
        elif "date" in prices_df.columns and str(prices_df.schema["date"].dataType) != "DateType()":
            date_col_source = "date" 
            print(f"Cột 'date' trong Elasticsearch index không phải DateType, đang thử chuyển đổi từ string...")
            prices_df = prices_df.withColumn("date_temp_col", to_date(col("date").cast("string"), "yyyy-MM-dd"))
            if "date_temp_col" in prices_df.columns:
                prices_df = prices_df.drop("date").withColumnRenamed("date_temp_col", date_col_final_name)
            else: 
                print(f"LỖI: Không thể chuyển đổi cột 'date' sang DateType trong prices_df.")
                return None
        elif "date" in prices_df.columns and str(prices_df.schema["date"].dataType) == "DateType()":
            print("Cột 'date' trong prices_df đã là DateType.")
            if "date" != date_col_final_name : 
                 prices_df = prices_df.withColumnRenamed("date", date_col_final_name) 
        else:
             print(f"LỖI: Không tìm thấy cột ngày phù hợp ('create_date', 'time', 'date_str', hoặc 'date') trong Elasticsearch index {es_index}")
             prices_df.printSchema()
             return None
        
        if date_col_source and date_col_source != date_col_final_name and date_col_source in prices_df.columns: 
            prices_df = prices_df.drop(date_col_source)
        
        required_cols = ["date", "symbol", "open_price", "close_price"]
        if not all(c in prices_df.columns for c in required_cols):
            print(f"LỖI: prices_df thiếu một hoặc nhiều cột cần thiết ({required_cols}) sau khi xử lý. Các cột hiện có: {prices_df.columns}")
            prices_df.printSchema()
            return None
            
        prices_df = prices_df.select(*required_cols) 
        
        print(f"Đã tải và xử lý dữ liệu giá từ Elasticsearch index: {es_index}")
        prices_df.printSchema()
        print(f"Số lượng dòng trong prices_df: {prices_df.count()}") 
        prices_df.show(5, truncate=False)
        return prices_df
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu giá từ Elasticsearch index {es_index}: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_news_articles(spark, es_host="localhost", es_port="9200", es_index="articles", 
                      date_format_in_file="yyyy-MM-dd"):
    """
    Tải dữ liệu bài báo từ Elasticsearch index.
    Xử lý cột ngày và cột text. Cột 'symbol' không còn được yêu cầu.
    Schema mong đợi: id, date, link, article_text, title
    """
    try:
        print(f"Đang thử tải dữ liệu bài báo từ Elasticsearch: {es_host}:{es_port}/{es_index}")
        
        # Cấu hình Elasticsearch
        es_options = {
            "es.nodes": es_host,
            "es.port": es_port,
            "es.resource": es_index,
            "es.read.field.as.array.include": "tags",
            "es.nodes.wan.only": "true"
        }
        
        # Đọc dữ liệu từ Elasticsearch
        articles_df = spark.read.format("org.elasticsearch.spark.sql") \
                                .options(**es_options) \
                                .load()
        
        print("Schema của articles_df sau khi đọc từ Elasticsearch:")
        articles_df.printSchema()

        date_col_final_name = "date"
        # Xử lý cột ngày:
        if date_col_final_name in articles_df.columns:
            if str(articles_df.schema[date_col_final_name].dataType) != "DateType()":
                print(f"Cột '{date_col_final_name}' trong Elasticsearch index là StringType, đang chuyển đổi sang DateType...")
                # Giả sử định dạng ngày trong cột 'date' của Elasticsearch index là một timestamp string có thể parse được
                articles_df = articles_df.withColumn(date_col_final_name, to_date(to_timestamp(col(date_col_final_name), date_format_in_file))) 
            else: # Đã là DateType
                print(f"Cột '{date_col_final_name}' trong Elasticsearch index đã là DateType.")
                # Đảm bảo tên cột là 'date' nếu nó đã là DateType nhưng có tên khác (ít khả năng)
                if date_col_final_name != "date": 
                    articles_df = articles_df.withColumnRenamed(date_col_final_name, "date")
        elif "create_date" in articles_df.columns: # Fallback nếu có create_date
             print("Sử dụng cột 'create_date' để tạo cột 'date' cho articles_df.")
             articles_df = articles_df.withColumn(date_col_final_name, to_date(to_timestamp(col("create_date"), date_format_in_file)))
             if "create_date" != date_col_final_name: articles_df = articles_df.drop("create_date")
        else: # Nếu không có cả 'date' lẫn 'create_date'
            print(f"LỖI: Không tìm thấy cột ngày phù hợp ('date' hoặc 'create_date') trong Elasticsearch index {es_index}.")
            return None 

        text_col_final_name = "article_text"
        if text_col_final_name not in articles_df.columns:
            if "text" in articles_df.columns: 
                articles_df = articles_df.withColumnRenamed("text", text_col_final_name)
                print("Đã đổi tên cột 'text' thành 'article_text'.")
            else:
                 print(f"LỖI: Không tìm thấy cột '{text_col_final_name}' hoặc 'text' trong Elasticsearch index {es_index}.")
                 return None
        else: # Cột article_text đã tồn tại
            if "article_text" != text_col_final_name: # Đảm bảo tên cuối cùng
                 articles_df = articles_df.withColumnRenamed("article_text", text_col_final_name)
            print(f"Cột '{text_col_final_name}' đã tồn tại.")

        # Chỉ chọn cột date và article_text
        final_selected_cols = [date_col_final_name, text_col_final_name]
        
        missing_final_cols = [c for c in final_selected_cols if c not in articles_df.columns]
        if missing_final_cols:
            print(f"LỖI: articles_df thiếu các cột cuối cùng mong đợi cho join theo ngày: {missing_final_cols}. Các cột hiện có: {articles_df.columns}")
            articles_df.printSchema()
            return None

        articles_df = articles_df.select(*final_selected_cols)

        print(f"Đã tải và xử lý dữ liệu bài báo từ Elasticsearch index: {es_index}")
        articles_df.printSchema()
        print(f"Số lượng dòng trong articles_df (sau xử lý): {articles_df.count()}") 
        articles_df.show(5, truncate=True)
        return articles_df
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu bài báo từ Elasticsearch index {es_index}: {e}")
        import traceback
        traceback.print_exc()
        return None

def join_data(prices_df, articles_df, article_separator="<s>"):
    """
    Kết hợp dữ liệu giá cổ phiếu và bài báo đã được tải từ Elasticsearch.
    """
    if prices_df is None or articles_df is None:
        print("Không thể kết hợp dữ liệu do một trong các DataFrame đầu vào là None.")
        return None
    
    required_prices_join_cols = ["date", "symbol", "open_price", "close_price"]
    if not all(c in prices_df.columns for c in required_prices_join_cols):
        print(f"Lỗi: prices_df thiếu các cột cần thiết: {required_prices_join_cols}. Các cột hiện có: {prices_df.columns}")
        return None

    required_articles_agg_cols = ["date", "article_text"]
    if not all(c in articles_df.columns for c in required_articles_agg_cols):
        print(f"Lỗi: articles_df thiếu các cột cần thiết cho tổng hợp: {required_articles_agg_cols}. Các cột hiện có: {articles_df.columns}")
        return None

    if prices_df.rdd.isEmpty():
        print("CẢNH BÁO: prices_df trống trước khi join.")
    if articles_df.rdd.isEmpty():
        print("CẢNH BÁO: articles_df trống trước khi join.")

    try:
        print("Tổng hợp các bài báo theo ngày...")
        articles_aggregated_df = articles_df.groupBy("date") \
                                            .agg(F.concat_ws(article_separator, F.collect_list(col("article_text"))).alias("full_article_text"))
        
        print("Schema của articles_aggregated_df sau khi groupBy('date'):")
        articles_aggregated_df.printSchema()

        print("Thực hiện join dữ liệu giá và dữ liệu bài báo đã tổng hợp chỉ bằng cột 'date'...")
        joined_df = prices_df.join(articles_aggregated_df, "date", "inner") 
        
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

def load_raw_data(spark, es_host="localhost", es_port="9200", stock_index="prices", article_index="articles",
                article_separator="<s>", date_format = "yyyy-MM-dd"):
    prices_df = load_stock_prices(spark, es_host=es_host, es_port=es_port, es_index=stock_index
                                , date_format_in_file=date_format)
    articles_df = load_news_articles(spark,  es_host=es_host, es_port=es_port, es_index=article_index
                                    , date_format_in_file=date_format)

    if prices_df is None or articles_df is None:
        print("Không thể tải dữ liệu huấn luyện ")
        return

    joined_df = join_data(prices_df, articles_df, article_separator=article_separator)

    if joined_df is None:
        print("Không thể kết hợp dữ liệu. Kết thúc quy trình huấn luyện.")
        return
    return joined_df

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

def configure_elasticsearch_connection(spark, es_host="localhost", es_port="9200", 
                                     es_user=None, es_password=None, es_ssl=False):
    """
    Cấu hình kết nối Elasticsearch cho Spark session.
    """
    spark.conf.set("es.nodes", es_host)
    spark.conf.set("es.port", es_port)
    
    # FIX: Remove or set es.nodes.wan.only to "false" for Docker networking.
    # This setting is for cloud/WAN environments, not local Docker networks.
    spark.conf.set("es.nodes.wan.only", "false") 
    
    if es_user and es_password:
        spark.conf.set("es.net.http.auth.user", es_user)
        spark.conf.set("es.net.http.auth.pass", es_password)
    
    if es_ssl:
        spark.conf.set("es.net.ssl", "true")
        # You might need this if you use self-signed certificates for ES
        # spark.conf.set("es.net.ssl.cert.allow.self.signed", "true") 
    
    print(f"Đã cấu hình kết nối Elasticsearch: {es_host}:{es_port}")

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
