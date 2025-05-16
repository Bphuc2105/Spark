# src/preprocessing.py

from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, VectorAssembler, SQLTransformer, RegexTokenizer
from pyspark.ml import Pipeline
from pyspark.sql.types import DoubleType

# Import cấu hình sử dụng relative import
try:
    from .config import HASHING_TF_NUM_FEATURES, VIETNAMESE_STOPWORDS
except ImportError:
    print("Cảnh báo: Không thể import .config trong preprocessing.py, sử dụng giá trị mặc định.")
    HASHING_TF_NUM_FEATURES = 10000
    # Ví dụ rút gọn, đảm bảo VIETNAMESE_STOPWORDS được định nghĩa nếu config không import được
    VIETNAMESE_STOPWORDS = [
        "và", "là", "có", "của", "trong", "cho", "đến", "khi", "thì", "mà", "ở", "tại",
        "này", "đó", "các", "những", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín", "mười",
        "được", "bị", "do", "vì", "nên", "nhưng", "nếu", "thế", "đã", "sẽ", "đang", "rằng", "vẫn",
        "để", "không", "có_thể", "cũng", "với", "như", "về", "sau", "trước", "trên", "dưới",
        "ông", "bà", "anh", "chị", "em", "tôi", "chúng_tôi", "bạn", "họ", "ai", "gì",
        "ngày", "tháng", "năm", "theo", "tuy_nhiên", "tuyệt_vời", "bao_gồm", "thực_sự",
        "vào", "ra", "lên", "xuống", "qua", "lại", "từ", "chỉ", "còn", "mới", "rất", "quá",
        "điều", "việc", "người", "cách", "khác", "phải", "luôn", "bao_giờ", "hơn", "nhất"
    ]


def create_preprocessing_pipeline(
    text_input_col="full_article_text",
    numerical_input_cols=["open_price"],
    output_features_col="features",
    output_label_col="label"
):
    """
    Tạo một Pipeline tiền xử lý cho cả dữ liệu văn bản và dữ liệu số.
    Nhãn sẽ được tạo dưới dạng phần trăm thay đổi giá.

    Args:
        text_input_col (str): Tên cột chứa văn bản đầu vào.
        numerical_input_cols (list): Danh sách tên các cột số đầu vào.
        output_features_col (str): Tên cột chứa vector đặc trưng kết hợp đầu ra.
        output_label_col (str): Tên cột chứa nhãn hồi quy (phần trăm thay đổi giá).

    Returns:
        pyspark.ml.Pipeline: Đối tượng Pipeline chứa các stage tiền xử lý.
    """
    regex_tokenizer = RegexTokenizer(
        inputCol=text_input_col,
        outputCol="tokens",
        pattern="\\W"
    )

    stopwords_remover = StopWordsRemover(
        inputCol="tokens",
        outputCol="filtered_tokens",
        stopWords=VIETNAMESE_STOPWORDS # Đảm bảo biến này được định nghĩa
    )

    hashing_tf = HashingTF(
        inputCol="filtered_tokens",
        outputCol="raw_text_features",
        numFeatures=HASHING_TF_NUM_FEATURES # Đảm bảo biến này được định nghĩa
    )

    idf = IDF(
        inputCol="raw_text_features",
        outputCol="text_features_idf"
    )

    numerical_assembler = VectorAssembler(
        inputCols=numerical_input_cols,
        outputCol="numerical_vector_features",
        handleInvalid="skip"
    )

    label_creator = SQLTransformer(
        statement=f"""
            SELECT
                *,
                CASE
                    WHEN open_price IS NOT NULL AND open_price != 0 AND close_price IS NOT NULL THEN
                        CAST(( (close_price - open_price) / open_price ) * 100.0 AS DOUBLE)
                    ELSE
                        NULL
                END AS {output_label_col}
            FROM __THIS__
        """
    )

    assembler_input_cols = []
    if text_input_col:
        assembler_input_cols.append("text_features_idf")
    if numerical_input_cols:
        assembler_input_cols.append("numerical_vector_features")

    if not assembler_input_cols:
        raise ValueError("Không có cột đặc trưng nào được chọn để kết hợp.")

    final_assembler = VectorAssembler(
        inputCols=assembler_input_cols,
        outputCol=output_features_col,
        handleInvalid="skip"
    )

    preprocessing_stages = []
    preprocessing_stages.append(label_creator)
    if text_input_col:
        preprocessing_stages.extend([regex_tokenizer, stopwords_remover, hashing_tf, idf])
    if numerical_input_cols:
        preprocessing_stages.append(numerical_assembler)
    preprocessing_stages.append(final_assembler)

    preprocessing_pipeline = Pipeline(stages=preprocessing_stages)

    return preprocessing_pipeline


if __name__ == "__main__":
    from pyspark.sql import SparkSession
    # Khi chạy trực tiếp, relative import có thể gây lỗi nếu không chạy bằng `python -m src.preprocessing`
    # Đoạn test này có thể cần điều chỉnh hoặc bỏ qua nếu chỉ tập trung vào việc module được import đúng
    try:
        from config import HASHING_TF_NUM_FEATURES, VIETNAMESE_STOPWORDS
    except ImportError: # Fallback for direct execution if .config fails
        print("Running __main__ in preprocessing.py: Falling back on config import for testing.")
        # Định nghĩa lại các biến config cần thiết cho test nếu không import được
        HASHING_TF_NUM_FEATURES = 10000
        VIETNAMESE_STOPWORDS = ["và", "là", "có"]


    spark = SparkSession.builder.appName("PreprocessingTestRegression").master("local[*]").getOrCreate()

    sample_data = [
        (1, "2023-01-01", "AAPL", "Giá Apple tăng mạnh sau tin tức tốt", 150.0, 155.0),
        (2, "2023-01-01", "MSFT", "Microsoft công bố lợi nhuận quý", 250.0, 252.0),
        (3, "2023-01-02", "AAPL", "Apple đối mặt khó khăn", 154.0, 153.0),
        (6, "2023-01-03", "TSLA", "Tesla giảm giá", 0.0, 110.0),
    ]
    columns = ["id", "date_str", "symbol", "full_article_text", "open_price", "close_price"]
    data_df = spark.createDataFrame(sample_data, columns)

    preprocessing_pipeline_obj = create_preprocessing_pipeline(
        text_input_col="full_article_text",
        numerical_input_cols=["open_price"],
        output_features_col="features",
        output_label_col="percentage_change"
    )

    transformed_df = preprocessing_pipeline_obj.fit(data_df).transform(data_df)
    print("Dữ liệu sau khi tiền xử lý (cho hồi quy) từ __main__ preprocessing.py:")
    transformed_df.select(
        "id", "symbol", "open_price", "close_price",
        "percentage_change",
        "features"
    ).show(truncate=False)
    spark.stop()

