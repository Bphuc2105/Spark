# src/__init__.py

# Dòng này thường được thêm vào để thông báo rằng đây là một package.
# print(f"Initializing package: {__name__}")

# Bạn có thể để trống tệp này.
# Hoặc, bạn có thể import các thành phần quan trọng từ các module con
# để chúng có thể được truy cập dễ dàng hơn từ package 'src'.

# Ví dụ (tùy chọn, và phụ thuộc vào cách bạn muốn cấu trúc import):
# from .data_loader import get_spark_session, load_stock_prices, load_news_articles, join_data
# from .preprocessing import create_preprocessing_pipeline
# from .train import train_model, save_model
# from .predict import load_prediction_model, make_predictions
# from .utils import get_logger, load_config
# from . import config # Import toàn bộ module config

# Việc import như trên cho phép bạn làm điều này từ bên ngoài thư mục src (nếu src nằm trong PYTHONPATH):
# import src
# spark = src.get_spark_session()
# logger = src.get_logger("my_app")

# Hoặc nếu bạn muốn giữ namespace rõ ràng:
# from src import data_loader, config
# spark = data_loader.get_spark_session(config.SPARK_APP_NAME)

# Đối với dự án này, việc để trống hoặc chỉ có một vài bình luận là đủ.
# Các module sẽ được import trực tiếp khi cần, ví dụ:
# from src.data_loader import get_spark_session
