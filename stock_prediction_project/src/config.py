SPARK_APP_NAME = "StockPricePredictionPySpark"
DATA_DIR = "data" 
MODELS_DIR = "models" 

TRAIN_PRICES_FILE = f"{DATA_DIR}/prices.csv"
TRAIN_ARTICLES_FILE = f"{DATA_DIR}/articles.csv"

PREDICT_PRICES_FILE = f"{DATA_DIR}/new_prices.csv"
PREDICT_ARTICLES_FILE = f"{DATA_DIR}/new_articles.csv"

HDFS_MODEL_SAVE_PATH = "hdfs://namenode:9000/user/stock_models/stock_prediction_pipeline_model_regression"

TEXT_INPUT_COLUMN = "full_article_text"
NUMERICAL_INPUT_COLUMNS = ["open_price", "close_price"]
FEATURES_OUTPUT_COLUMN = "features" 
LABEL_OUTPUT_COLUMN = "label" 

ARTICLE_SEPARATOR = "<s>" 

TRAIN_TEST_SPLIT_RATIO = [0.8, 0.2]
RANDOM_SEED = 42

LOGISTIC_REGRESSION_MAX_ITER = 100
LOGISTIC_REGRESSION_REG_PARAM = 0.01

EVALUATOR_METRIC_NAME_ROC = "areaUnderROC"
EVALUATOR_METRIC_NAME_PR = "areaUnderPR"

KAFKA_BROKER = "kafka:9092" 
NEWS_ARTICLES_TOPIC = "news_articles" 
STOCK_PRICES_TOPIC = "stock_prices" 

ES_NODES = "elasticsearch" 
ES_PORT = "9200"
ES_USER = None  
ES_PASSWORD = None  
ES_SSL = False  

ES_PRICES_INDEX = "prices"
ES_ARTICLES_INDEX = "articles"

ES_PREDICTION_INDEX = "predict_prices"

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
