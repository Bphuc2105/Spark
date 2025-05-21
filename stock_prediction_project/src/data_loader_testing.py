
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, to_timestamp
from pyspark.sql import functions as F
import os

def get_spark_session_with_nlp(app_name="StockPredictionApp"):
    """
    Khởi tạo và trả về một SparkSession có cấu hình Spark NLP.
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.1.0") \
        .getOrCreate()
    return spark

def get_spark_session(app_name="StockPredictionApp"):
    """
    Khởi tạo và trả về một SparkSession.
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()
    return spark

def load_stock_prices(spark, file_path, date_format_in_file="yyyy-MM-dd HH:mm:ssX"):
    """
    Tải dữ liệu giá cổ phiếu từ tệp CSV.
    Xử lý các tên cột và chuyển đổi cột ngày.
    Tệp CSV đầu vào được mong đợi có header.
    Schema suy luận từ log: id, create_date (hoặc time), close, volume, source, symbol, open
    """
    # Get the directory of the current script (data_loader.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Get the project root directory (parent of 'src')
    project_root = os.path.dirname(script_dir)
    
    absolute_file_path = os.path.join(project_root, file_path)
    try:
        print(f"Đang thử tải dữ liệu giá từ (với inferSchema, đường dẫn tuyệt đối): {absolute_file_path}")
        prices_df = spark.read.csv(absolute_file_path, header=True, inferSchema=True, escape="\"")

        print("Schema của prices_df sau khi inferSchema:")
        prices_df.printSchema()

        # Đổi tên cột 'open' và 'close' nếu chúng tồn tại và chưa đúng tên
        if "open" in prices_df.columns and "open_price" not in prices_df.columns:
            prices_df = prices_df.withColumnRenamed("open", "open_price")
        if "close" in prices_df.columns and "close_price" not in prices_df.columns:
            prices_df = prices_df.withColumnRenamed("close", "close_price")

        date_col_source = None
        date_col_final_name = "date"

        # Ưu tiên các tên cột ngày có thể có trong file prices.csv
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
            print(f"Cột 'date' trong {absolute_file_path} không phải DateType, đang thử chuyển đổi từ string...")
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
             print(f"LỖI: Không tìm thấy cột ngày phù hợp ('create_date', 'time', 'date_str', hoặc 'date') trong {absolute_file_path}")
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
        
        print(f"Đã tải và xử lý dữ liệu giá từ: {absolute_file_path}")
        prices_df.printSchema()
        print(f"Số lượng dòng trong prices_df: {prices_df.count()}") 
        prices_df.show(5, truncate=False)
        return prices_df
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu giá từ {absolute_file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_news_articles(spark, file_path, date_format_in_file="yyyy-MM-dd"):
    """
    Tải dữ liệu bài báo từ tệp CSV.
    Xử lý cột ngày và cột text. Cột 'symbol' không còn được yêu cầu.
    Tệp CSV đầu vào được mong đợi có header.
    Schema suy luận từ log: id, date, link, article_text, title
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Get the project root directory (parent of 'src')
    project_root = os.path.dirname(script_dir)
    
    absolute_file_path = os.path.join(project_root, file_path)
    try:
        print(f"Đang thử tải dữ liệu bài báo từ (với inferSchema, đường dẫn tuyệt đối): {absolute_file_path}")
        articles_df = spark.read.csv(absolute_file_path, header=True, inferSchema=True, multiLine=True, escape="\"")
        
        print("Schema của articles_df sau khi inferSchema:")
        articles_df.printSchema()

        date_col_final_name = "date"
        # Xử lý cột ngày:
        # Tệp CSV của bạn dường như đã có cột 'date' (nhưng là string)
        if date_col_final_name in articles_df.columns:
            if str(articles_df.schema[date_col_final_name].dataType) != "DateType()":
                print(f"Cột '{date_col_final_name}' trong articles.csv là StringType, đang chuyển đổi sang DateType...")
                # Giả sử định dạng ngày trong cột 'date' của articles.csv là một timestamp string có thể parse được
                articles_df = articles_df.withColumn(date_col_final_name, to_date(to_timestamp(col(date_col_final_name), date_format_in_file))) 
            else: # Đã là DateType
                print(f"Cột '{date_col_final_name}' trong articles.csv đã là DateType.")
                # Đảm bảo tên cột là 'date' nếu nó đã là DateType nhưng có tên khác (ít khả năng)
                if date_col_final_name != "date": 
                    articles_df = articles_df.withColumnRenamed(date_col_final_name, "date")
        elif "create_date" in articles_df.columns: # Fallback nếu có create_date
             print("Sử dụng cột 'create_date' để tạo cột 'date' cho articles_df.")
             articles_df = articles_df.withColumn(date_col_final_name, to_date(to_timestamp(col("create_date"), date_format_in_file)))
             if "create_date" != date_col_final_name: articles_df = articles_df.drop("create_date")
        else: # Nếu không có cả 'date' lẫn 'create_date'
            print(f"LỖI: Không tìm thấy cột ngày phù hợp ('date' hoặc 'create_date') trong {absolute_file_path}.")
            return None 

        text_col_final_name = "article_text"
        if text_col_final_name not in articles_df.columns:
            if "text" in articles_df.columns: 
                articles_df = articles_df.withColumnRenamed("text", text_col_final_name)
                print("Đã đổi tên cột 'text' thành 'article_text'.")
            else:
                 print(f"LỖI: Không tìm thấy cột '{text_col_final_name}' hoặc 'text' trong {absolute_file_path}.")
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

        print(f"Đã tải và xử lý dữ liệu bài báo từ: {absolute_file_path}")
        articles_df.printSchema()
        print(f"Số lượng dòng trong articles_df (sau xử lý): {articles_df.count()}") 
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
    
if __name__ == "__main__":
    # --- Cấu hình đường dẫn (giống như trong data_loader.py) ---
    spark = get_spark_session("DataLoading")
    prices_path = "data/prices.csv"
    articles_path = "data/articles.csv"

    # --- Tải dữ liệu ---
    prices_df = load_stock_prices(spark, prices_path)
    articles_df = load_news_articles(spark, articles_path)

    if prices_df and articles_df:
        # Kết hợp dữ liệu
        # Hàm join_data từ data_loader.py sẽ tạo cột 'full_article_text'
        raw_data_df = join_data(prices_df, articles_df)
        if raw_data_df:
            print("\nDữ liệu thô sau khi join:")
            raw_data_df.show(5, truncate=True)
            raw_data_df.printSchema()