# src/preprocessing.py

from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, VectorAssembler, SQLTransformer
from pyspark.sql.functions import col, when

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

def create_preprocessing_pipeline(text_input_col="full_article_text",
                                  numerical_input_cols=["open_price"], # Danh sách các cột số
                                  output_features_col="features",
                                  output_label_col="label"):
    """
    Tạo một Spark ML Pipeline để tiền xử lý dữ liệu.
    Pipeline bao gồm:
    1. Tạo nhãn (sử dụng SQLTransformer cho tính linh hoạt trong pipeline).
    2. Tokenizer: Tách văn bản thành các từ.
    3. StopWordsRemover: Loại bỏ các từ dừng phổ biến.
    4. HashingTF: Chuyển đổi các từ thành vector tần số thô.
    5. IDF: Tính toán Inverse Document Frequency.
    6. VectorAssembler: Kết hợp các đặc trưng văn bản (TF-IDF) và đặc trưng số.

    Args:
        text_input_col (str): Tên cột chứa văn bản đầu vào (ví dụ: 'full_article_text').
        numerical_input_cols (list): Danh sách tên các cột đặc trưng số (ví dụ: ['open_price']).
        output_features_col (str): Tên cột chứa vector đặc trưng kết hợp cuối cùng.
        output_label_col (str): Tên cột nhãn.

    Returns:
        pyspark.ml.Pipeline: Đối tượng Pipeline đã được cấu hình.
    """
    print("Đang tạo pipeline tiền xử lý...")

    # Bước 1: Tạo nhãn (sử dụng SQLTransformer để tích hợp vào pipeline)
    # Câu lệnh SQL này giả định cột 'open_price' và 'close_price' tồn tại trong DataFrame đầu vào của pipeline
    # Lưu ý: Cột 'label' sẽ được tạo bởi SQLTransformer này.
    # Nếu bạn đã tạo nhãn trước đó bằng hàm create_label, bạn có thể bỏ qua bước này
    # hoặc đảm bảo nó không xung đột. Trong thiết kế này, pipeline sẽ tự tạo nhãn.
    sql_transformer_label = SQLTransformer(
        statement=f"SELECT *, CAST((CASE WHEN close_price > open_price THEN 1.0 ELSE 0.0 END) AS DOUBLE) AS {output_label_col} FROM __THIS__"
    )

    # Bước 2: Tokenizer cho văn bản
    tokenizer = Tokenizer(inputCol=text_input_col, outputCol="words")

    # Bước 3: StopWordsRemover
    # Bạn có thể tùy chỉnh danh sách từ dừng hoặc sử dụng danh sách mặc định cho tiếng Anh
    # stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words", locale="en_US")
    # Đối với tiếng Việt, bạn cần cung cấp danh sách từ dừng tiếng Việt
    # Ví dụ (cần một danh sách đầy đủ hơn):
    vietnamese_stopwords = ["và", "là", "có", "của", "trong", "cho", "đến", "khi", "thì", "mà", "ở", "tại", "này", "đó", "các", "những", "một", "hai", "ba", "được", "bị", "rằng", "để", "không", "có_thể", "cũng", "với", "như", "về", "sau", "trước", "trên", "dưới"] # Cần danh sách đầy đủ hơn
    stopwords_remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(),
                                         outputCol="filtered_words",
                                         stopWords=vietnamese_stopwords) # Sử dụng danh sách từ dừng tiếng Việt

    # Bước 4: HashingTF
    # numFeatures là số lượng đặc trưng (kích thước của vector đầu ra)
    hashing_tf = HashingTF(inputCol=stopwords_remover.getOutputCol(),
                           outputCol="raw_features",
                           numFeatures=10000) # Có thể điều chỉnh số lượng features

    # Bước 5: IDF
    idf = IDF(inputCol=hashing_tf.getOutputCol(), outputCol="text_features")

    # Bước 6: VectorAssembler để kết hợp đặc trưng văn bản và đặc trưng số
    # Đầu vào là cột text_features (từ IDF) và các cột số được chỉ định
    assembler_input_cols = [idf.getOutputCol()] + numerical_input_cols
    vector_assembler = VectorAssembler(inputCols=assembler_input_cols,
                                       outputCol=output_features_col)

    # Tạo Pipeline với tất cả các bước
    # Lưu ý thứ tự của các bước là quan trọng
    preprocessing_pipeline = Pipeline(stages=[
        sql_transformer_label, # Tạo nhãn trước
        tokenizer,
        stopwords_remover,
        hashing_tf,
        idf,
        vector_assembler
    ])

    print("Pipeline tiền xử lý đã được tạo.")
    return preprocessing_pipeline

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
    prices_path = "../data/prices_sample.csv"
    articles_path = "../data/articles_sample.csv"

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
            # Giả sử cột văn bản là 'full_article_text' và cột số là 'open_price'
            # Cột nhãn sẽ được pipeline tạo ra là 'label'
            # Cột đặc trưng cuối cùng sẽ là 'features'
            pipeline = create_preprocessing_pipeline(
                text_input_col="full_article_text",
                numerical_input_cols=["open_price"], # Phải là một list
                output_features_col="features",
                output_label_col="label"
            )

            # Huấn luyện pipeline tiền xử lý trên dữ liệu (chỉ các transformer không yêu cầu huấn luyện trước)
            # Đối với các Estimator như IDF, chúng cần được fit.
            print("\nFitting preprocessing pipeline...")
            # Loại bỏ các hàng có giá trị null trong các cột quan trọng trước khi fit
            # Ví dụ: cột 'full_article_text', 'open_price', 'close_price'
            # Cột 'close_price' cần thiết cho SQLTransformer để tạo nhãn
            columns_to_check_null = ["full_article_text", "open_price", "close_price"]
            cleaned_data_df = raw_data_df.na.drop(subset=columns_to_check_null)

            if cleaned_data_df.count() == 0:
                print("Không có dữ liệu sau khi loại bỏ các hàng null. Không thể fit pipeline.")
            else:
                print(f"Số lượng mẫu sau khi làm sạch null: {cleaned_data_df.count()}")
                pipeline_model = pipeline.fit(cleaned_data_df)

                # Áp dụng pipeline đã fit để biến đổi dữ liệu
                print("\nTransforming data using the fitted pipeline...")
                processed_df = pipeline_model.transform(cleaned_data_df)

                print("\nDữ liệu sau khi qua pipeline tiền xử lý:")
                processed_df.printSchema()
                # Hiển thị các cột quan trọng: nhãn và vector đặc trưng
                processed_df.select("date", "symbol", "open_price", "close_price", "full_article_text", "label", "features").show(5, truncate=True)

                # Kiểm tra số lượng đặc trưng trong vector 'features'
                if processed_df.count() > 0:
                    num_features_in_vector = len(processed_df.select("features").first()[0])
                    print(f"\nSố lượng đặc trưng trong vector 'features': {num_features_in_vector}")
        else:
            print("Không thể join dữ liệu.")
    else:
        print("Không thể tải dữ liệu giá hoặc bài báo.")

    spark.stop()
