# src/config.py

# --- Cấu hình chung cho ứng dụng Spark ---
SPARK_APP_NAME = "StockPricePredictionPySpark"
# SPARK_MASTER không còn cần thiết ở đây nếu bạn luôn chạy qua docker-compose
# và master được chỉ định trong lệnh spark-submit.
# SPARK_MASTER = "local[*]" 

# --- Đường dẫn dữ liệu (bên trong container, tương đối với WORKDIR /app) ---
DATA_DIR = "data" # Đường dẫn đến thư mục data, tương đối với /app
MODELS_DIR = "models" # Đường dẫn đến thư mục models, tương đối với /app

# Đường dẫn cụ thể cho các tệp dữ liệu huấn luyện/kiểm thử
# Bây giờ sẽ là "data/prices_sample.csv" và "data/articles_sample.csv" bên trong container
TRAIN_PRICES_FILE = f"{DATA_DIR}/prices.csv"
TRAIN_ARTICLES_FILE = f"{DATA_DIR}/articles.csv"

# Đường dẫn cụ thể cho các tệp dữ liệu dự đoán (nếu có)
PREDICT_PRICES_FILE = f"{DATA_DIR}/new_prices.csv"
PREDICT_ARTICLES_FILE = f"{DATA_DIR}/new_articles.csv"

# Đường dẫn để lưu/tải mô hình đã huấn luyện
SAVED_MODEL_PATH = f"{MODELS_DIR}/stock_prediction_pipeline_model"

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

# --- Cấu hình huấn luyện mô hình ---
TRAIN_TEST_SPLIT_RATIO = [0.8, 0.2] 
RANDOM_SEED = 42 

LOGISTIC_REGRESSION_MAX_ITER = 100
LOGISTIC_REGRESSION_REG_PARAM = 0.01 

# --- Cấu hình đánh giá ---
EVALUATOR_METRIC_NAME_ROC = "areaUnderROC"
EVALUATOR_METRIC_NAME_PR = "areaUnderPR"

if __name__ == "__main__":
    print(f"Tên ứng dụng Spark: {SPARK_APP_NAME}")
    print(f"Đường dẫn tệp giá huấn luyện (trong container): {TRAIN_PRICES_FILE}") # Sẽ là data/prices_sample.csv
    print(f"Đường dẫn lưu mô hình (trong container): {SAVED_MODEL_PATH}") # Sẽ là models/stock_prediction_pipeline_model
    print(f"Số lượng features cho HashingTF: {HASHING_TF_NUM_FEATURES}")
    print(f"Tỷ lệ chia train/test: {TRAIN_TEST_SPLIT_RATIO}")
    print(f"Một vài từ dừng tiếng Việt đầu tiên: {VIETNAMESE_STOPWORDS[:5]}")
