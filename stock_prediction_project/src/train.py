# src/train.py

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
import traceback # Thêm import này
import os
import zipfile
import tempfile
import shutil

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
    Lưu PipelineModel đã huấn luyện vào file zip.
    Returns:
        bool: True if saving was successful, False otherwise.
    """
    if model is None:
        print("Mô hình là None. Không thể lưu.")
        return False
    try:
        # Chuyển đổi đường dẫn thành đường dẫn tuyệt đối
        abs_path = os.path.abspath(path)
        
        # Thêm phần mở rộng .zip nếu chưa có
        if not abs_path.endswith('.zip'):
            abs_path = abs_path + '.zip'
            
        model_dir = os.path.dirname(abs_path)
        
        print(f"\nThông tin lưu mô hình:")
        print(f"- Đường dẫn tương đối: {path}")
        print(f"- Đường dẫn tuyệt đối: {abs_path}")
        print(f"- Thư mục chứa: {model_dir}")
        
        print(f"\nĐang tạo thư mục: {model_dir}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Kiểm tra quyền ghi
        if not os.access(model_dir, os.W_OK):
            print(f"Không có quyền ghi vào thư mục: {model_dir}")
            return False
            
        print(f"\nĐang lưu mô hình...")
        print(f"Loại mô hình: {type(model)}")
        print(f"Số lượng stages: {len(model.stages)}")
        
        # Tạo thư mục tạm thời để lưu mô hình
        with tempfile.TemporaryDirectory() as temp_dir:
            # Lưu mô hình vào thư mục tạm thời
            temp_model_path = os.path.join(temp_dir, "model")
            model.write().overwrite().save(temp_model_path)
            
            # Tạo file zip
            with zipfile.ZipFile(abs_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Thêm tất cả các file từ thư mục tạm thời vào zip
                for root, dirs, files in os.walk(temp_model_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, temp_model_path)
                        zipf.write(file_path, arcname)
        
        # Kiểm tra xem file zip đã được tạo thành công chưa
        if os.path.exists(abs_path):
            print(f"Mô hình đã được lưu thành công tại: {abs_path}")
            print(f"Kích thước file: {os.path.getsize(abs_path) / 1024 / 1024:.2f} MB")
        else:
            print(f"Lỗi: Không tìm thấy file zip tại {abs_path}")
            return False
            
        return True
    except Exception as e:
        print(f"Lỗi khi lưu mô hình vào {path}: {e}")
        traceback.print_exc()
        return False

# Khối if __name__ == "__main__": giữ nguyên như trước để test độc lập nếu cần
# (nhưng cần đảm bảo các import và đường dẫn file đúng với môi trường test đó)
