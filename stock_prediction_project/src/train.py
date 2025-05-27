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
    Lưu PipelineModel đã huấn luyện vào file zip một cách an toàn, tránh race conditions.

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    if model is None:
        print("Mô hình là None. Không thể lưu.")
        return False
    if not isinstance(model, PipelineModel):
        print(f"Đối tượng cung cấp không phải là PipelineModel (type: {type(model)}). Không thể lưu.")
        return False

    # Chuyển đổi đường dẫn thành đường dẫn tuyệt đối cho file zip cuối cùng
    abs_zip_path = os.path.abspath(path)
    if not abs_zip_path.endswith('.zip'):
        abs_zip_path = abs_zip_path + '.zip'

    # Tạo một thư mục tạm thời CỐ ĐỊNH để Spark lưu model vào.
    # Chúng ta sẽ tự xóa nó sau khi hoàn tất.
    model_dir_container = os.path.dirname(abs_zip_path)
    temp_save_dir = os.path.join(model_dir_container, "temp_spark_model_save")

    print(f"\n--- Bắt đầu quy trình lưu model an toàn ---")
    print(f"- Đường dẫn file zip cuối cùng: {abs_zip_path}")
    print(f"- Thư mục lưu model tạm thời: {temp_save_dir}")

    # Xóa thư mục tạm thời cũ nếu nó tồn tại để bắt đầu mới
    if os.path.exists(temp_save_dir):
        print(f"Xóa thư mục tạm thời cũ: {temp_save_dir}")
        shutil.rmtree(temp_save_dir)

    try:
        # Bước 1: Spark lưu model vào thư mục cố định.
        # Hành động này là blocking và sẽ đợi cho đến khi các job của Spark hoàn tất.
        print(f"\nBước 1: Spark đang lưu model vào thư mục tạm thời...")
        model.write().overwrite().save(temp_save_dir)
        print("Spark đã hoàn tất việc ghi model.")

        # Bước 2: Kiểm tra để chắc chắn rằng thư mục không rỗng.
        print(f"\nBước 2: Kiểm tra nội dung của thư mục tạm thời...")
        if not os.path.exists(temp_save_dir) or not os.listdir(temp_save_dir):
            print(f"LỖI: Thư mục lưu model tạm thời '{temp_save_dir}' rỗng hoặc không tồn tại sau khi Spark lưu!")
            return False
        
        # In ra một vài file để xác nhận
        print("Nội dung ví dụ trong thư mục tạm thời:")
        for item in os.listdir(temp_save_dir)[:5]:
            print(f"  - {item}")


        # Bước 3: Nén thư mục đã được lưu hoàn chỉnh.
        print(f"\nBước 3: Nén nội dung vào file zip '{abs_zip_path}'...")
        with zipfile.ZipFile(abs_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(temp_save_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_save_dir)
                    zipf.write(file_path, arcname)
        print("Nén file zip hoàn tất.")

        # Bước 4: Kiểm tra file zip cuối cùng
        if os.path.exists(abs_zip_path) and os.path.getsize(abs_zip_path) > 1024: # Kiểm tra > 1KB
            print(f"\nTHÀNH CÔNG: Mô hình đã được lưu tại: {abs_zip_path}")
            print(f"Kích thước file: {os.path.getsize(abs_zip_path) / (1024 * 1024):.2f} MB")
            return True
        else:
            print(f"\nLỖI: File zip cuối cùng bị rỗng hoặc không tạo được.")
            print(f"Kích thước file: {os.path.getsize(abs_zip_path) if os.path.exists(abs_zip_path) else 'N/A'}")
            return False

    except Exception as e:
        print(f"Lỗi xảy ra trong quá trình lưu model: {e}")
        traceback.print_exc()
        return False
    finally:
        # Bước 5: Dọn dẹp thư mục tạm thời.
        # Khối finally này sẽ luôn chạy, dù có lỗi hay không.
        if os.path.exists(temp_save_dir):
            print(f"\nBước 5: Dọn dẹp, xóa thư mục tạm thời '{temp_save_dir}'.")
            shutil.rmtree(temp_save_dir)
        print("--- Kết thúc quy trình lưu model ---")

if __name__ == "__main__":
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import Tokenizer
    from pyspark.sql import SparkSession

    from data_loader import get_spark_session
    
    # Create a Spark session
    spark = get_spark_session("Test")
    # Dummy data
    df = spark.createDataFrame([(1, "Hello world")], ["id", "text"])

    # Minimal pipeline
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    pipeline = Pipeline(stages=[tokenizer])
    model = pipeline.fit(df)
    result = save_model(model, "test_model.zip")
    print("Save result:", result)
        