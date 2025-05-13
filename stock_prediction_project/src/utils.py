# src/utils.py

import logging
import yaml # Bạn có thể cần cài đặt thư viện PyYAML: pip install PyYAML

# --- Cấu hình Logging cơ bản ---
# Bạn có thể tùy chỉnh cấu hình này cho phức tạp hơn nếu cần
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def get_logger(name):
    """
    Lấy một đối tượng logger với tên được chỉ định.

    Args:
        name (str): Tên của logger (thường là __name__ của module gọi nó).

    Returns:
        logging.Logger: Đối tượng logger.
    """
    return logging.getLogger(name)

# --- Ví dụ về hàm tải cấu hình từ tệp YAML ---
# Bạn sẽ cần tạo một tệp config.yaml nếu muốn sử dụng hàm này.
# Ví dụ:
# config.yaml:
# app_name: "StockPredictionApp"
# data_paths:
#   prices: "../data/prices_data.csv"
#   articles: "../data/articles_data.csv"
# model_paths:
#   output_dir: "../models/my_stock_model"

def load_config(config_path="config.yaml"):
    """
    Tải cấu hình từ một tệp YAML.

    Args:
        config_path (str): Đường dẫn đến tệp cấu hình YAML.

    Returns:
        dict: Một dictionary chứa các cấu hình.
              Trả về None nếu có lỗi.
    """
    logger = get_logger(__name__)
    try:
        with open(config_path, 'r') as stream:
            config = yaml.safe_load(stream)
        logger.info(f"Đã tải cấu hình thành công từ: {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Lỗi: Không tìm thấy tệp cấu hình tại '{config_path}'.")
        return None
    except yaml.YAMLError as exc:
        logger.error(f"Lỗi khi phân tích cú pháp tệp YAML '{config_path}': {exc}")
        return None
    except Exception as e:
        logger.error(f"Một lỗi không mong muốn xảy ra khi tải cấu hình: {e}")
        return None

# --- Các hàm tiện ích khác có thể được thêm vào đây ---

# Ví dụ: một hàm kiểm tra đường dẫn
def check_path_exists(path_to_check):
    """
    Kiểm tra xem một đường dẫn có tồn tại không.

    Args:
        path_to_check (str): Đường dẫn cần kiểm tra.

    Returns:
        bool: True nếu đường dẫn tồn tại, False nếu không.
    """
    import os
    logger = get_logger(__name__)
    exists = os.path.exists(path_to_check)
    if exists:
        logger.debug(f"Đường dẫn '{path_to_check}' tồn tại.")
    else:
        logger.warning(f"Đường dẫn '{path_to_check}' không tồn tại.")
    return exists


if __name__ == "__main__":
    # Ví dụ cách sử dụng các hàm trong utils.py
    logger = get_logger("UtilsTest")

    logger.info("Đây là một thông điệp info từ UtilsTest.")
    logger.warning("Đây là một thông điệp warning.")
    logger.error("Đây là một thông điệp error.")

    # Kiểm tra hàm load_config
    # Tạo một tệp config_sample.yaml để kiểm thử
    sample_config_content = """
    app_name: "MySampleApp"
    version: "1.0"
    paths:
      input: "/data/input"
      output: "/data/output"
    """
    sample_config_path = "config_sample.yaml"
    with open(sample_config_path, "w") as f:
        f.write(sample_config_content)

    config_data = load_config(sample_config_path)
    if config_data:
        logger.info(f"Tên ứng dụng từ config: {config_data.get('app_name')}")
        logger.info(f"Đường dẫn input từ config: {config_data.get('paths', {}).get('input')}")

    # Xóa tệp config mẫu sau khi kiểm thử
    import os
    if os.path.exists(sample_config_path):
        os.remove(sample_config_path)

    # Kiểm tra hàm check_path_exists
    check_path_exists("../data") # Giả sử thư mục data tồn tại ở cấp trên
    check_path_exists("../non_existent_folder")
