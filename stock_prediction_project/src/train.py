# src/train.py

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
import traceback # Thêm import này

# Import các module cần thiết từ project sử dụng relative import
try:
    from .data_loader import get_spark_session # Chỉ import những gì cần thiết trực tiếp
    # from .preprocessing import create_preprocessing_pipeline # Không cần thiết ở đây nếu stages được truyền vào
except ImportError as e:
    print(f"Lỗi import trong src/train.py: {e}")
    if 'get_spark_session' not in locals(): 
        def get_spark_session(app_name="DefaultApp"):
            from pyspark.sql import SparkSession
            print("Cảnh báo: get_spark_session không được import đúng cách, sử dụng fallback.")
            return SparkSession.builder.appName(app_name).master("local[*]").getOrCreate()


def train_regression_model(spark, training_data_df, preprocessing_pipeline_stages, label_col_name="percentage_change"):
    """
    Huấn luyện mô hình hồi quy sử dụng pipeline tiền xử lý và dữ liệu huấn luyện.

    Args:
        spark (SparkSession): Đối tượng SparkSession.
        training_data_df (DataFrame): DataFrame chứa dữ liệu đã được join.
        preprocessing_pipeline_stages (list): Danh sách các stage của pipeline tiền xử lý.
        label_col_name (str): Tên của cột nhãn.

    Returns:
        pyspark.ml.PipelineModel: Mô hình Pipeline đã được huấn luyện.
                                  Trả về None nếu có lỗi.
    """
    if training_data_df is None:
        print("Dữ liệu huấn luyện là None. Không thể huấn luyện mô hình.")
        return None

    try:
        print(f"DEBUG train_regression_model: Sử dụng cột nhãn '{label_col_name}'")
        # Loại bỏ các hàng có giá trị null trong các cột quan trọng cho features
        # và cột close_price (cần để tính label bởi SQLTransformer trong preprocessing).
        # Cột label sẽ được tạo trong pipeline, nên ta sẽ lọc null trên label sau khi transform.
        columns_to_check_for_feature_input = ["full_article_text", "open_price", "close_price"]
        cleaned_df_for_features = training_data_df.na.drop(subset=columns_to_check_for_feature_input)

        if cleaned_df_for_features.count() == 0:
            print("Không có dữ liệu sau khi loại bỏ null ban đầu cho các cột features đầu vào.")
            return None

        (train_df_raw, test_df_raw) = cleaned_df_for_features.randomSplit([0.8, 0.2], seed=42)
        print(f"Số lượng mẫu huấn luyện thô (trước tiền xử lý): {train_df_raw.count()}")
        print(f"Số lượng mẫu kiểm tra thô (trước tiền xử lý): {test_df_raw.count()}")
        
        gbt = GBTRegressor(featuresCol="features", labelCol=label_col_name, maxIter=20) # Sử dụng label_col_name

        # Tạo pipeline chỉ chứa các bước tiền xử lý để biến đổi dữ liệu
        temp_preprocessing_pipeline_model = Pipeline(stages=preprocessing_pipeline_stages).fit(train_df_raw) # Fit trên train_df_raw
        
        print("\nÁp dụng tiền xử lý để tạo nhãn và features cho tập huấn luyện...")
        processed_train_df = temp_preprocessing_pipeline_model.transform(train_df_raw)
        
        print(f"Schema của processed_train_df trước khi lọc null label '{label_col_name}':")
        processed_train_df.printSchema()
        
        final_train_df = processed_train_df.filter(col(label_col_name).isNotNull())
        final_train_df_count = final_train_df.count() # Cache count
        print(f"Số lượng mẫu huấn luyện sau khi lọc label '{label_col_name}' NULL: {final_train_df_count}")

        if final_train_df_count == 0:
            print(f"Không có dữ liệu huấn luyện sau khi lọc các hàng có nhãn '{label_col_name}' NULL.")
            return None
            
        print(f"\nBắt đầu huấn luyện mô hình GBTRegressor với labelCol='{label_col_name}'...")
        gbt_model = gbt.fit(final_train_df) # final_train_df đã có 'features' và label_col_name không null
        print("Huấn luyện GBTRegressor hoàn tất.")

        # Kết hợp các stages tiền xử lý đã fit với mô hình GBT đã huấn luyện
        complete_pipeline_model = PipelineModel(stages=temp_preprocessing_pipeline_model.stages + [gbt_model])
        print("Đã tạo PipelineModel hoàn chỉnh.")

        print("\nĐánh giá mô hình trên tập kiểm tra...")
        processed_test_df = temp_preprocessing_pipeline_model.transform(test_df_raw) # Dùng lại temp_preprocessing_pipeline_model
        final_test_df = processed_test_df.filter(col(label_col_name).isNotNull())
        final_test_df_count = final_test_df.count() # Cache count
        print(f"Số lượng mẫu kiểm tra sau khi lọc label '{label_col_name}' NULL: {final_test_df_count}")

        if final_test_df_count == 0:
            print(f"Không có dữ liệu kiểm tra sau khi lọc các hàng có nhãn '{label_col_name}' NULL. Không thể đánh giá.")
            return complete_pipeline_model 

        predictions_df = gbt_model.transform(final_test_df)

        print("\nMột vài dự đoán trên tập kiểm tra:")
        # Đảm bảo các cột này tồn tại trong predictions_df trước khi select
        cols_to_show = [c for c in ["date", "symbol", "open_price", "close_price", label_col_name, "prediction"] if c in predictions_df.columns]
        predictions_df.select(cols_to_show).show(10, truncate=True)

        evaluator_rmse = RegressionEvaluator(labelCol=label_col_name, predictionCol="prediction", metricName="rmse")
        rmse = evaluator_rmse.evaluate(predictions_df)
        print(f"Root Mean Squared Error (RMSE) trên tập kiểm tra: {rmse:.4f}")

        evaluator_mae = RegressionEvaluator(labelCol=label_col_name, predictionCol="prediction", metricName="mae")
        mae = evaluator_mae.evaluate(predictions_df)
        print(f"Mean Absolute Error (MAE) trên tập kiểm tra: {mae:.4f}")
        
        evaluator_r2 = RegressionEvaluator(labelCol=label_col_name, predictionCol="prediction", metricName="r2")
        r2 = evaluator_r2.evaluate(predictions_df)
        print(f"R-squared (R2) trên tập kiểm tra: {r2:.4f}")

        return complete_pipeline_model

    except Exception as e:
        print(f"Lỗi trong quá trình huấn luyện mô hình hồi quy: {e}")
        traceback.print_exc()
        return None

def save_model(model, path):
    """
    Lưu PipelineModel đã huấn luyện.
    Returns:
        bool: True if saving was successful, False otherwise.
    """
    if model is None:
        print("Mô hình là None. Không thể lưu.")
        return False
    try:
        print(f"\nĐang lưu mô hình vào: {path}")
        model.write().overwrite().save(path)
        print("Lưu mô hình thành công.")
        return True # Trả về True nếu thành công
    except Exception as e:
        print(f"Lỗi khi lưu mô hình vào {path}: {e}")
        traceback.print_exc() # In đầy đủ traceback để gỡ lỗi
        return False # Trả về False nếu thất bại

# Khối if __name__ == "__main__": giữ nguyên như trước để test độc lập nếu cần
# (nhưng cần đảm bảo các import và đường dẫn file đúng với môi trường test đó)
