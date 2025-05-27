

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
import traceback
import os
try:
    from . import config 
    from .utils import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)
    
    class FallbackConfig:
        TEXT_INPUT_COLUMN = "full_article_text"
        NUMERICAL_INPUT_COLUMNS = ["open_price"]
        RANDOM_SEED = 42
    config = FallbackConfig()

def train_regression_model(spark, training_data_df, preprocessing_pipeline_stages, label_col_name="percentage_change", features_col_name="features"):
    """
    Huấn luyện mô hình hồi quy sử dụng pipeline tiền xử lý và dữ liệu huấn luyện.
    Quy trình này sẽ fit riêng pipeline tiền xử lý, transform dữ liệu, rồi fit GBTRegressor.
    """
    if training_data_df is None:
        logger.error("Dữ liệu huấn luyện là None. Không thể huấn luyện mô hình.")
        return None
    if not preprocessing_pipeline_stages:
        logger.error("Danh sách các giai đoạn tiền xử lý rỗng. Không thể huấn luyện.")
        return None

    try:
        logger.info(f"Bắt đầu quy trình huấn luyện GBTRegressor với cột nhãn '{label_col_name}' và cột features '{features_col_name}'.")
        logger.info("Đang kiểm tra các cột đầu vào cần thiết cho tiền xử lý...")
        
        required_input_cols_for_preprocessing = [
            getattr(config, 'TEXT_INPUT_COLUMN', 'full_article_text'),
            'open_price', 
            'close_price' 
        ]
        
        required_input_cols_for_preprocessing.extend(getattr(config, 'NUMERICAL_INPUT_COLUMNS', []))
        required_input_cols_for_preprocessing = list(set(required_input_cols_for_preprocessing)) 

        actual_cols_to_check_na = [c for c in required_input_cols_for_preprocessing if c in training_data_df.columns]
        logger.info(f"Các cột sẽ được kiểm tra NA trước khi fit pipeline tiền xử lý: {actual_cols_to_check_na}")
        cleaned_input_df = training_data_df.na.drop(subset=actual_cols_to_check_na)
        
        if cleaned_input_df.count() == 0:
            logger.error(f"Không còn dữ liệu sau khi loại bỏ NA cho các cột đầu vào tiền xử lý: {actual_cols_to_check_na}. DataFrame gốc có {training_data_df.count()} dòng.")
            logger.error("Vui lòng kiểm tra dữ liệu đầu vào prices.csv và articles.csv, đặc biệt là các cột được sử dụng để tạo nhãn và đặc trưng.")
            training_data_df.show(5) 
            return None
        logger.info(f"Số dòng sau khi làm sạch NA cho đầu vào tiền xử lý: {cleaned_input_df.count()}")
        
        logger.info("Đang fit pipeline tiền xử lý...")
        temp_preprocessing_pipeline = Pipeline(stages=preprocessing_pipeline_stages)
        fitted_preprocessing_model = temp_preprocessing_pipeline.fit(cleaned_input_df)
        logger.info("Pipeline tiền xử lý đã được fit.")
        logger.info("Đang transform dữ liệu bằng pipeline tiền xử lý đã fit...")
        
        processed_df = fitted_preprocessing_model.transform(cleaned_input_df)
        
        logger.info(f"Schema của DataFrame sau khi tiền xử lý (trước khi lọc null cho nhãn '{label_col_name}'):")
        processed_df.printSchema()
        logger.info(f"Các cột có sẵn: {processed_df.columns}")

        if features_col_name not in processed_df.columns:
            logger.error(f"Lỗi nghiêm trọng: Cột features '{features_col_name}' KHÔNG được tạo ra bởi pipeline tiền xử lý.")
            return None
        if label_col_name not in processed_df.columns:
            logger.error(f"Lỗi nghiêm trọng: Cột nhãn '{label_col_name}' KHÔNG được tạo ra bởi pipeline tiền xử lý (SQLTransformer).")
            logger.error("Kiểm tra lại SQLTransformer trong preprocessing.py và dữ liệu (open_price, close_price).")
            processed_df.select("open_price", "close_price", label_col_name).show(10, truncate=False)
            return None
        
        logger.info(f"Đang lọc các hàng có giá trị null trong cột nhãn '{label_col_name}'...")
        data_for_gbt_training = processed_df.filter(col(label_col_name).isNotNull())
        
        count_after_label_filter = data_for_gbt_training.count()
        if count_after_label_filter == 0:
            logger.error(f"Không có dữ liệu nào còn lại để huấn luyện GBTRegressor sau khi lọc null cho cột nhãn '{label_col_name}'.")
            logger.info("Xem xét dữ liệu đầu vào (open_price, close_price) và logic tạo nhãn trong SQLTransformer.")
            logger.info(f"Dữ liệu trước khi lọc null nhãn (cột open_price, close_price, {label_col_name}):")
            processed_df.select("open_price", "close_price", label_col_name).show(20, truncate=False)
            return None
        logger.info(f"Số lượng mẫu sau khi lọc null cho nhãn: {count_after_label_filter}")
        (train_df_for_gbt, test_df_for_gbt) = data_for_gbt_training.randomSplit([0.8, 0.2], seed=getattr(config, 'RANDOM_SEED', 42))
        logger.info(f"Số lượng mẫu huấn luyện cho GBT: {train_df_for_gbt.cache().count()}, mẫu kiểm tra cho GBT: {test_df_for_gbt.cache().count()}") 

        logger.info(f"Đang khởi tạo GBTRegressor với featuresCol='{features_col_name}' và labelCol='{label_col_name}'...")
        gbt = GBTRegressor(featuresCol=features_col_name, labelCol=label_col_name, maxIter=20) 

        logger.info(f"Bắt đầu huấn luyện GBTRegressor trên {train_df_for_gbt.count()} mẫu...")
        gbt_model = gbt.fit(train_df_for_gbt)
        logger.info("GBTRegressor đã được huấn luyện.")
        if test_df_for_gbt.count() > 0:
            logger.info("Đang đánh giá GBTRegressor trên tập kiểm tra...")
            predictions_gbt_test = gbt_model.transform(test_df_for_gbt)
            
            logger.info("Một vài dự đoán trên tập kiểm tra GBT:")
            cols_to_show_eval = [label_col_name, "prediction", features_col_name]
            
            existing_cols_to_show = [c for c in cols_to_show_eval if c in predictions_gbt_test.columns]
            if existing_cols_to_show:
                 predictions_gbt_test.select(existing_cols_to_show).show(5, truncate=False)
            else:
                logger.warning("Không có cột nào trong danh sách yêu cầu để hiển thị kết quả đánh giá.")
            evaluator_rmse = RegressionEvaluator(labelCol=label_col_name, predictionCol="prediction", metricName="rmse")
            rmse = evaluator_rmse.evaluate(predictions_gbt_test)
            logger.info(f"GBT - Root Mean Squared Error (RMSE) trên tập kiểm tra: {rmse:.4f}")

            evaluator_r2 = RegressionEvaluator(labelCol=label_col_name, predictionCol="prediction", metricName="r2")
            r2 = evaluator_r2.evaluate(predictions_gbt_test)
            logger.info(f"GBT - R-squared (R2) trên tập kiểm tra: {r2:.4f}")
        else:
            logger.warning("Tập kiểm tra cho GBT rỗng, bỏ qua bước đánh giá GBT.")
            
        complete_pipeline_model = PipelineModel(stages=fitted_preprocessing_model.stages + [gbt_model])
        logger.info("Đã tạo PipelineModel hoàn chỉnh.")
        return complete_pipeline_model

    except Exception as e:
        logger.error(f"Lỗi trong quá trình huấn luyện mô hình GBTRegressor: {e}", exc_info=True)
        return None

def save_model(model, path, is_hdfs_path=False):
    if model is None:
        logger.error("Mô hình là None. Không thể lưu.")
        return False
    if not isinstance(model, PipelineModel): 
        logger.error(f"Lỗi: Đối tượng được cung cấp để lưu không phải là PipelineModel. Loại đối tượng: {type(model)}")
        return False
            
    try:
        logger.info(f"\nThông tin lưu mô hình:")
        logger.info(f"- Đường dẫn lưu: {path}")
        logger.info(f"- Lưu vào HDFS: {is_hdfs_path}")
        logger.info(f"- Loại mô hình: {type(model)}")
        if hasattr(model, 'stages'):
            logger.info(f"- Số lượng stages: {len(model.stages)}")
        model.write().overwrite().save(path)
        logger.info(f"Mô hình đã được lưu thành công tại: {path}")
        
        if not is_hdfs_path and os.path.exists(path):
            logger.info(f"Kiểm tra cục bộ: Thư mục mô hình tồn tại tại {path}")
        elif is_hdfs_path:
            logger.info("Lưu ý: Để kiểm tra sự tồn tại trên HDFS, bạn cần sử dụng các lệnh HDFS (ví dụ: hdfs dfs -ls path).")
        return True
    except Exception as e:
        logger.error(f"Lỗi khi lưu mô hình vào {path}: {e}", exc_info=True)
        return False