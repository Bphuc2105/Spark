# src/config.py

# --- Cấu hình chung cho ứng dụng Spark ---
SPARK_APP_NAME = "StockPricePredictionPySpark"

# --- Đường dẫn dữ liệu (bên trong container, tương đối với WORKDIR /app) ---
DATA_DIR = "data" # Đường dẫn đến thư mục data, tương đối với /app
MODELS_DIR = "models" # Đường dẫn đến thư thư mục models, tương đối với /app

# Đường dẫn cụ thể cho các tệp dữ liệu huấn luyện/kiểm thử (vẫn dùng CSV cho training)
TRAIN_PRICES_FILE = f"{DATA_DIR}/prices.csv"
TRAIN_ARTICLES_FILE = f"{DATA_DIR}/articles.csv"

# Đường dẫn cụ thể cho các tệp dữ liệu dự đoán (Sẽ đọc từ Kafka thay vì file tĩnh này)
# Tuy nhiên, vẫn giữ biến này cho mục đích tham khảo hoặc fallback
PREDICT_PRICES_FILE = f"{DATA_DIR}/new_prices.csv"
PREDICT_ARTICLES_FILE = f"{DATA_DIR}/new_articles.csv"

# Đường dẫn để lưu/tải mô hình đã huấn luyện
HDFS_MODEL_SAVE_PATH = "hdfs://namenode:9000/user/stock_models/stock_prediction_pipeline_model_regression"

# --- Cấu hình tiền xử lý ---
TEXT_INPUT_COLUMN = "full_article_text"
NUMERICAL_INPUT_COLUMNS = ["open_price", "close_price"]
FEATURES_OUTPUT_COLUMN = "features" 
LABEL_OUTPUT_COLUMN = "label" 

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
ARTICLE_SEPARATOR = "<s>" 

# --- Cấu hình huấn luyện mô hình (vẫn dùng cho chế độ train) ---
TRAIN_TEST_SPLIT_RATIO = [0.8, 0.2]
RANDOM_SEED = 42

LOGISTIC_REGRESSION_MAX_ITER = 100
LOGISTIC_REGRESSION_REG_PARAM = 0.01

# --- Cấu hình đánh giá (vẫn dùng cho chế độ train) ---
EVALUATOR_METRIC_NAME_ROC = "areaUnderROC"
EVALUATOR_METRIC_NAME_PR = "areaUnderPR"

# --- Cấu hình Kafka ---
KAFKA_BROKER = "kafka:9092" # Tên service 'kafka' trong docker-compose và port mặc định
NEWS_ARTICLES_TOPIC = "news_articles" # Topic Kafka cho dữ liệu bài báo mới
STOCK_PRICES_TOPIC = "stock_prices" # Topic Kafka cho dữ liệu giá mới (tùy chọn, nếu stream cả giá)

# --- Cấu hình Elasticsearch ---
ELASTICSEARCH_HOST = "elasticsearch" # Tên service 'elasticsearch' trong docker-compose
ELASTICSEARCH_PORT = "9200" # Port mặc định của Elasticsearch
ES_PREDICTION_INDEX = "stock_predictions" # Index trong Elasticsearch để lưu kết quả dự đoán
# Cấu hình cho Spark-Elasticsearch connector (cần cho Spark để ghi dữ liệu)
ES_NODES = ELASTICSEARCH_HOST # Hoặc "http://elasticsearch:9200" tùy cấu hình connector
ES_PORT = ELASTICSEARCH_PORT

if __name__ == "__main__":
    print(f"Tên ứng dụng Spark: {SPARK_APP_NAME}")
    print(f"Đường dẫn tệp giá huấn luyện (trong container): {TRAIN_PRICES_FILE}")
    print(f"Đường dẫn lưu mô hình (trong container): {SAVED_REGRESSION_MODEL_PATH}")
    print(f"Số lượng features cho HashingTF: {HASHING_TF_NUM_FEATURES}")
    print(f"Tỷ lệ chia train/test: {TRAIN_TEST_SPLIT_RATIO}")
    print(f"Một vài từ dừng tiếng Việt đầu tiên: {VIETNAMESE_STOPWORDS[:5]}")
    print(f"Kafka Broker: {KAFKA_BROKER}")
    print(f"Kafka Topic Bài báo: {NEWS_ARTICLES_TOPIC}")
    print(f"Elasticsearch Host: {ELASTICSEARCH_HOST}")
    print(f"Elasticsearch Port: {ELASTICSEARCH_PORT}")
    print(f"Elasticsearch Prediction Index: {ES_PREDICTION_INDEX}")
