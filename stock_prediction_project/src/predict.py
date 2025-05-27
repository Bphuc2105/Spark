# src/predict.py

from pyspark.ml import PipelineModel
from pyspark.ml.feature import SQLTransformer
import os 

try:
    from . import config
    from .utils import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)
    class FallbackConfigPredict:
        # ES_NODES = "localhost" # Ví dụ
        # ES_PORT = "9200"
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
            # Thêm kiểm tra cẩn thận hơn về outputCol của SQLTransformer nếu cần
            # label_col_name_from_config = getattr(config, 'REGRESSION_LABEL_OUTPUT_COLUMN', 'percentage_change')
            # if hasattr(original_stages[0], 'getOutputCol') and original_stages[0].getOutputCol() == label_col_name_from_config:
            logger.info("Hàm make_predictions: Loại bỏ stage đầu tiên (giả định là SQLTransformer tạo nhãn) cho pipeline dự đoán.")
            stages_for_prediction = original_stages[1:] 
            if not stages_for_prediction:
                logger.error("Hàm make_predictions: Lỗi - Không còn stage nào sau khi loại bỏ stage đầu tiên.")
                return None
            prediction_model_to_use = PipelineModel(stages=stages_for_prediction)
            logger.info(f"Hàm make_predictions: Sử dụng pipeline mới cho dự đoán với {len(stages_for_prediction)} stages.")
            # else:
            #     logger.warning(f"Hàm make_predictions: Stage đầu tiên là SQLTransformer nhưng không phải là stage tạo nhãn dự kiến. Sử dụng mô hình gốc.")
        else:
            logger.warning("Hàm make_predictions: Stage đầu tiên của mô hình đã tải không phải là SQLTransformer hoặc không có stages nào. Sử dụng mô hình gốc.")

        logger.info("Hàm make_predictions: Đang thực hiện transform trên dữ liệu đầu vào...")
        predictions_df = prediction_model_to_use.transform(data_df)
        logger.info("Hàm make_predictions: Áp dụng mô hình (transform) hoàn tất.")
        return predictions_df
        
    except Exception as e:
        logger.error(f"Hàm make_predictions: Lỗi trong quá trình áp dụng mô hình: {e}", exc_info=True)
        return None

def write_batch_to_elasticsearch(batch_df, es_index, es_nodes, es_port, es_write_operation="index"):
    """
    Ghi một Batch DataFrame vào Elasticsearch.

    Args:
        batch_df (DataFrame): DataFrame batch chứa dữ liệu cần ghi.
        es_index (str): Tên index trong Elasticsearch.
        es_nodes (str): Địa chỉ host(s) của Elasticsearch.
        es_port (str): Port của Elasticsearch.
        es_write_operation (str): Thao tác ghi (index, create, update, upsert).
    """
    if batch_df is None or batch_df.rdd.isEmpty():
        logger.warning("DataFrame rỗng, không có gì để ghi vào Elasticsearch.")
        return True 

    logger.info(f"Chuẩn bị ghi batch DataFrame vào Elasticsearch index: {es_index} tại {es_nodes}:{es_port}")
    try:
        es_options = {
            "es.nodes": str(es_nodes),
            "es.port": str(es_port),
            "es.resource": str(es_index),
            "es.nodes.wan.only": "true", 
            "es.write.operation": es_write_operation,
        }
        
        if "id" in batch_df.columns: 
            es_options["es.mapping.id"] = "id"
            logger.info("Sử dụng cột 'id' làm document ID trong Elasticsearch.")

        batch_df.write \
            .format("org.elasticsearch.spark.sql") \
            .options(**es_options) \
            .mode("append") \
            .save() 

        logger.info(f"Ghi thành công batch DataFrame vào Elasticsearch index: {es_index}")
        return True
    except Exception as e:
        logger.error(f"Lỗi khi ghi batch DataFrame vào Elasticsearch: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    logger.info("predict.py được chạy trực tiếp (thường chỉ để import).")
