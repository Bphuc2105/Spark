# src/predict.py

from pyspark.ml import PipelineModel
from pyspark.ml.feature import SQLTransformer # Cần thiết để kiểm tra isinstance
# from pyspark.sql.functions import current_timestamp # current_timestamp được dùng trong main.py

# Import cấu hình và logger một cách nhất quán
try:
    from . import config # Sử dụng config cho các hằng số nếu cần
    from .utils import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    # Fallback logger nếu import thất bại
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)
    # Fallback config nếu cần
    class FallbackConfigPredict:
        # REGRESSION_LABEL_OUTPUT_COLUMN = "percentage_change" # Ví dụ nếu cần
        pass
    config = FallbackConfigPredict()


def load_prediction_model(model_path):
    """
    Tải PipelineModel đã huấn luyện từ đường dẫn được chỉ định.
    """
    try:
        logger.info(f"Đang tải mô hình từ: {model_path}")
        model = PipelineModel.load(model_path)
        logger.info("Tải mô hình thành công.")
        return model
    except Exception as e:
        logger.error(f"Lỗi khi tải mô hình từ {model_path}: {e}", exc_info=True)
        return None

def make_predictions(model, data_df):
    """
    Thực hiện dự đoán trên DataFrame đầu vào bằng mô hình đã cho.
    Hàm này sẽ cố gắng loại bỏ SQLTransformer tạo nhãn (nếu có) khỏi model đã tải.
    """
    if model is None or data_df is None:
        logger.error("Mô hình hoặc dữ liệu đầu vào là None. Không thể thực hiện dự đoán.")
        return None
    try:
        logger.info("Hàm make_predictions: Chuẩn bị áp dụng mô hình cho dự đoán...")
        
        original_stages = model.stages
        prediction_model_to_use = model 

        if original_stages and isinstance(original_stages[0], SQLTransformer):
            logger.info(f"Hàm make_predictions: Mô hình đã tải có {len(original_stages)} stages. Stage đầu tiên là: {type(original_stages[0])}.")
            # Kiểm tra thêm xem có phải là SQLTransformer tạo label không (ví dụ dựa vào outputCol)
            # label_col_name_from_config = getattr(config, 'REGRESSION_LABEL_OUTPUT_COLUMN', 'percentage_change')
            # if original_stages[0].getOutputCol() == label_col_name_from_config:
            logger.info("Hàm make_predictions: Loại bỏ stage đầu tiên (giả định là SQLTransformer tạo nhãn) cho pipeline dự đoán.")
            stages_for_prediction = original_stages[1:] 
            if not stages_for_prediction:
                logger.error("Hàm make_predictions: Lỗi - Không còn stage nào sau khi loại bỏ stage đầu tiên.")
                return None
            prediction_model_to_use = PipelineModel(stages=stages_for_prediction)
            logger.info(f"Hàm make_predictions: Sử dụng pipeline mới cho dự đoán với {len(stages_for_prediction)} stages.")
            # else:
            #     logger.warning(f"Hàm make_predictions: Stage đầu tiên là SQLTransformer nhưng outputCol ('{original_stages[0].getOutputCol()}') không khớp với cột nhãn dự kiến ('{label_col_name_from_config}'). Sử dụng mô hình gốc.")
        else:
            logger.warning("Hàm make_predictions: Stage đầu tiên của mô hình đã tải không phải là SQLTransformer hoặc không có stages nào. Sử dụng mô hình gốc.")
            logger.info("Hàm make_predictions: Điều này có thể gây lỗi nếu 'close_price' vẫn được yêu cầu và dữ liệu không có.")

        logger.info("Hàm make_predictions: Đang thực hiện transform trên dữ liệu đầu vào...")
        predictions_df = prediction_model_to_use.transform(data_df)
        logger.info("Hàm make_predictions: Áp dụng mô hình (transform) hoàn tất.")
        return predictions_df
        
    except Exception as e:
        logger.error(f"Hàm make_predictions: Lỗi trong quá trình áp dụng mô hình: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    logger.info("predict.py được chạy trực tiếp (thường chỉ để import).")
