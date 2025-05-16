# src/preprocessing.py

from pyspark.ml import Pipeline
from pyspark.ml.feature import SQLTransformer, VectorAssembler
from pyspark.sql.functions import col, when, udf
from pyspark.sql.types import ArrayType, FloatType
from sentence_transformers import SentenceTransformer
import numpy as np

# Global sentence transformer model instance (defined at module level)
# This will be initialized once and reused for all transformations
SENTENCE_MODEL = None

def initialize_sentence_transformer():
    """
    Initialize Sentence Transformer model for Vietnamese.
    Returns model instance.
    """
    global SENTENCE_MODEL
    
    if SENTENCE_MODEL is None:
        print("Initializing Sentence Transformer model...")
        # Load a Vietnamese-compatible Sentence Transformer model
        # For Vietnamese, there are a few good options:
        # - "vinai/phobert-base-v2" - PhoBERT wrapped in Sentence Transformers framework
        # - "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base" - Optimized for Vietnamese
        # - "distiluse-base-multilingual-cased-v2" - Multilingual model that works with Vietnamese
        model = SentenceTransformer("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
        
        SENTENCE_MODEL = model
        print("Sentence Transformer initialization completed.")
    
    return SENTENCE_MODEL

def create_label(input_df, open_col="open_price", close_col="close_price", label_col="label"):
    """
    Tạo cột nhãn dựa trên việc giá đóng cửa có cao hơn giá mở cửa hay không.
    1 nếu close_price > open_price, ngược lại là 0.

    Args:
        input_df (DataFrame): DataFrame đầu vào chứa cột giá mở cửa và giá đóng cửa.
        open_col (str): Tên cột giá mở cửa.
        close_col (str): Tên cột giá đóng cửa.
        label_col (str): Tên cột nhãn sẽ được tạo.

    Returns:
        DataFrame: DataFrame với cột nhãn đã được thêm vào.
    """
    print(f"Tạo cột nhãn '{label_col}'...")
    # Đảm bảo các cột cần thiết tồn tại
    if open_col not in input_df.columns or close_col not in input_df.columns:
        raise ValueError(f"Các cột '{open_col}' và/hoặc '{close_col}' không tồn tại trong DataFrame đầu vào.")

    # Xử lý trường hợp giá trị null trong open_price hoặc close_price nếu cần
    # Ví dụ: input_df = input_df.na.drop(subset=[open_col, close_col])

    labeled_df = input_df.withColumn(label_col,
                                     when(col(close_col) > col(open_col), 1.0)
                                     .otherwise(0.0))
    print(f"Số lượng mẫu theo nhãn:")
    labeled_df.groupBy(label_col).count().show()
    return labeled_df

def text_to_sentence_embedding(text):
    """
    Convert Vietnamese text to sentence embedding vector using Sentence Transformers.
    
    Args:
        text (str): Input text to convert.
        
    Returns:
        list: Sentence embedding as a list of floats.
    """
    # Initialize Sentence Transformer if not already initialized
    model = initialize_sentence_transformer()
    
    # Handle empty or None text
    if not text or text.strip() == "":
        # Return a zero vector of the same size as the model embeddings
        # (usually 768 for base models, 384 for some distilled models)
        embedding_size = model.get_sentence_embedding_dimension()
        return [0.0] * embedding_size
    
    # Generate embedding directly - Sentence Transformers handles tokenization internally
    embedding = model.encode(text)
    
    # Convert to regular Python list for PySpark UDF
    return embedding.tolist()

def create_preprocessing_pipeline(text_input_col="full_article_text",
                                  numerical_input_cols=["open_price"],
                                  output_features_col="features",
                                  output_label_col="label",
                                  embedding_output_col="text_features"):
    """
    Tạo một Spark ML Pipeline để tiền xử lý dữ liệu sử dụng Sentence Transformers.
    Pipeline bao gồm:
    1. Tạo nhãn (sử dụng SQLTransformer cho tính linh hoạt trong pipeline).
    2. Chuyển đổi văn bản thành embedding vector sử dụng Sentence Transformers.
    3. VectorAssembler: Kết hợp các đặc trưng văn bản (từ Sentence Transformers) và đặc trưng số.

    Args:
        text_input_col (str): Tên cột chứa văn bản đầu vào (ví dụ: 'full_article_text').
        numerical_input_cols (list): Danh sách tên các cột đặc trưng số (ví dụ: ['open_price']).
        output_features_col (str): Tên cột chứa vector đặc trưng kết hợp cuối cùng.
        output_label_col (str): Tên cột nhãn.
        embedding_output_col (str): Tên cột chứa embedding vector từ Sentence Transformers.

    Returns:
        pyspark.ml.Pipeline: Đối tượng Pipeline đã được cấu hình.
        function: UDF để tạo embedding
        VectorAssembler: Bộ kết hợp vector
        str: Tên cột chứa embedding
    """
    print("Đang tạo pipeline tiền xử lý với Sentence Transformers...")

    # Bước 1: Tạo nhãn (sử dụng SQLTransformer để tích hợp vào pipeline)
    sql_transformer_label = SQLTransformer(
        statement=f"SELECT *, CAST((CASE WHEN close_price > open_price THEN 1.0 ELSE 0.0 END) AS DOUBLE) AS {output_label_col} FROM __THIS__"
    )

    # Bước 2: Chuyển đổi văn bản thành sentence embedding
    # Đăng ký UDF để chuyển đổi văn bản
    text_to_embedding_udf = udf(text_to_sentence_embedding, ArrayType(FloatType()))
    
    # Bước 3: VectorAssembler để kết hợp đặc trưng văn bản và đặc trưng số
    assembler_input_cols = [embedding_output_col] + numerical_input_cols
    vector_assembler = VectorAssembler(inputCols=assembler_input_cols,
                                      outputCol=output_features_col)

    # Tạo Pipeline với bước tạo nhãn
    # UDF sẽ được áp dụng riêng khi transform dữ liệu
    preprocessing_pipeline = Pipeline(stages=[
        sql_transformer_label
    ])

    print("Pipeline tiền xử lý đã được tạo.")
    return preprocessing_pipeline, text_to_embedding_udf, vector_assembler, embedding_output_col

if __name__ == "__main__":
    # Phần này dùng để kiểm thử pipeline tiền xử lý
    # Bạn cần có data_loader.py để chạy phần này
    try:
        from data_loader import get_spark_session, load_stock_prices, load_news_articles, join_data
    except ImportError:
        print("Không thể import data_loader. Vui lòng đảm bảo nó tồn tại và có thể truy cập.")
        exit()

    spark = get_spark_session("PreprocessingTest")

    # --- Cấu hình đường dẫn (giống như trong data_loader.py) ---
    prices_path = "../data/prices.csv"
    articles_path = "../data/articles.csv"

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

            # --- Tạo và kiểm thử pipeline tiền xử lý ---
            embedding_output_col = "text_features"
            pipeline, text_to_embedding_udf, vector_assembler, embedding_col = create_preprocessing_pipeline(
                text_input_col="full_article_text",
                numerical_input_cols=["open_price"],
                output_features_col="features",
                output_label_col="label",
                embedding_output_col=embedding_output_col
            )

            # Loại bỏ các hàng có giá trị null trong các cột quan trọng trước khi fit
            columns_to_check_null = ["full_article_text", "open_price", "close_price"]
            cleaned_data_df = raw_data_df.na.drop(subset=columns_to_check_null)

            if cleaned_data_df.count() == 0:
                print("Không có dữ liệu sau khi loại bỏ các hàng null. Không thể fit pipeline.")
            else:
                print(f"Số lượng mẫu sau khi làm sạch null: {cleaned_data_df.count()}")
                
                # Áp dụng label transformer (phần cơ bản của pipeline)
                basic_transformed_df = pipeline.fit(cleaned_data_df).transform(cleaned_data_df)
                
                # Áp dụng Sentence Transformer UDF để tạo embedding
                print("\nÁp dụng Sentence Transformer để tạo embedding vectors...")
                # Lấy mẫu nhỏ nếu dữ liệu quá lớn để kiểm thử
                sample_df = basic_transformed_df.limit(10)  # Giới hạn số lượng mẫu để kiểm thử
                
                with_embedding_df = sample_df.withColumn(embedding_col, 
                                                        text_to_embedding_udf(col("full_article_text")))
                
                # Áp dụng VectorAssembler sau khi có embedding
                processed_df = vector_assembler.transform(with_embedding_df)
                
                print("\nDữ liệu sau khi qua pipeline tiền xử lý (với Sentence Transformer):")
                processed_df.printSchema()
                
                # Hiển thị các cột quan trọng
                processed_df.select("date", "symbol", "open_price", "close_price", "label", "features").show(5, truncate=True)
                
                # Kiểm tra kích thước vector đặc trưng
                if processed_df.count() > 0:
                    num_features_in_vector = len(processed_df.select("features").first()[0])
                    # Sử dụng get_sentence_embedding_dimension() để hiển thị kích thước chính xác
                    model = initialize_sentence_transformer()
                    embedding_size = model.get_sentence_embedding_dimension()
                    print(f"\nSố lượng đặc trưng trong vector 'features': {num_features_in_vector}")
                    print(f"({embedding_size} từ Sentence Transformer + {len(numerical_input_cols)} từ các đặc trưng số)")
        else:
            print("Không thể join dữ liệu.")
    else:
        print("Không thể tải dữ liệu giá hoặc bài báo.")

    spark.stop()