# src/config.py

# --- Cấu hình chung cho ứng dụng Spark ---
SPARK_APP_NAME = "StockPricePredictionPySpark"

# --- ĐỊNH NGHĨA CÁC THƯ MỤC CHÍNH TRƯỚC ---
DATA_DIR = "data" # Đường dẫn đến thư mục data, tương đối với /app
MODELS_DIR = "models" # Đường dẫn đến thư thư mục models, tương đối với /app

# --- Đường dẫn dữ liệu (bên trong container, tương đối với WORKDIR /app) ---
TRAIN_PRICES_FILE = f"{DATA_DIR}/prices.csv"
TRAIN_ARTICLES_FILE = f"{DATA_DIR}/articles.csv"

PREDICT_PRICES_FILE = f"{DATA_DIR}/new_prices.csv"
PREDICT_ARTICLES_FILE = f"{DATA_DIR}/new_articles.csv"

HDFS_MODEL_SAVE_PATH = "hdfs://namenode:9000/user/stock_models/stock_prediction_pipeline_model_regression"
LOCAL_SAVED_REGRESSION_MODEL_PATH = f"{MODELS_DIR}/stock_prediction_pipeline_model_regression_local"

PREDICTION_OUTPUT_DIR_CSV = f"{DATA_DIR}/batch_predictions_output"

# --- Cấu hình tiền xử lý ---
TEXT_INPUT_COLUMN = "full_article_text"
NUMERICAL_INPUT_COLUMNS = ["open_price"]
FEATURES_OUTPUT_COLUMN = "features"
REGRESSION_LABEL_OUTPUT_COLUMN = "percentage_change"

HASHING_TF_NUM_FEATURES = 10000

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
# --- THÊM LẠI ARTICLE_SEPARATOR NẾU BỊ THIẾU ---
ARTICLE_SEPARATOR = " --- "
# ---------------------------------------------
DATE_FORMAT_PRICES = "yyyy-MM-dd"

# --- Cấu hình huấn luyện mô hình ---
TRAIN_TEST_SPLIT_RATIO = [0.8, 0.2]
RANDOM_SEED = 42

# --- Cấu hình Kafka ---
KAFKA_BROKER = "kafka:9092"
NEWS_ARTICLES_TOPIC = "news_articles"
STOCK_PRICES_TOPIC = "stock_prices" # Biến này được tham chiếu trong data_loader.py, nên cần giữ lại

# --- Cấu hình Elasticsearch ---
ELASTICSEARCH_HOST = "elasticsearch"
ELASTICSEARCH_PORT = "9200"
ES_PREDICTION_INDEX = "stock_prediction_batch"
ES_NODES = ELASTICSEARCH_HOST
# ES_PORT đã có, đảm bảo nhất quán

if __name__ == "__main__":
    print(f"Tên ứng dụng Spark: {SPARK_APP_NAME}")
    print(f"Thư mục dữ liệu (DATA_DIR): {DATA_DIR}")
    print(f"Đường dẫn lưu kết quả dự đoán CSV (thư mục): {PREDICTION_OUTPUT_DIR_CSV}") # Sửa ở đây
    print(f"Article Separator: {ARTICLE_SEPARATOR}")