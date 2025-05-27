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
SAVED_REGRESSION_MODEL_PATH = f"{MODELS_DIR}/stock_prediction_pipeline_model_regression" # Đã sửa tên biến cho rõ ràng

# --- Cấu hình tiền xử lý ---
TEXT_INPUT_COLUMN = "full_article_text"
NUMERICAL_INPUT_COLUMNS = ["open_price", "close_price"]
FEATURES_OUTPUT_COLUMN = "features" 
LABEL_OUTPUT_COLUMN = "label" 

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
ES_NODES = "elasticsearch" # Hoặc "http://elasticsearch:9200" tùy cấu hình connector
ES_PORT = "9200"
ES_USER = None  # Đặt username nếu cần xác thực
ES_PASSWORD = None  # Đặt password nếu cần xác thực
ES_SSL = False  # Đặt True nếu sử dụng HTTPS

# index chứa data trong elasticsearch
ES_PRICES_INDEX = "prices"
ES_ARTICLES_INDEX = "articles"

# index lưu data predict được
ES_PREDICTION_INDEX = "prices"

if __name__ == "__main__":
    print(f"Tên ứng dụng Spark: {SPARK_APP_NAME}")
    print(f"Đường dẫn tệp giá huấn luyện (trong container): {TRAIN_PRICES_FILE}")
    print(f"Đường dẫn lưu mô hình (trong container): {SAVED_REGRESSION_MODEL_PATH}")
    print(f"Tỷ lệ chia train/test: {TRAIN_TEST_SPLIT_RATIO}")
    print(f"Kafka Broker: {KAFKA_BROKER}")
    print(f"Kafka Topic Bài báo: {NEWS_ARTICLES_TOPIC}")
    print(f"Elasticsearch Host: {ES_NODES}")
    print(f"Elasticsearch Port: {ES_PORT}")
    print(f"Elasticsearch Prediction Index: {ES_PREDICTION_INDEX}")
