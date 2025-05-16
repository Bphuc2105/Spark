# src/train.py

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col

# Import các module cần thiết từ project sử dụng relative import
try:
    from .data_loader import get_spark_session, load_stock_prices, load_news_articles, join_data
    from .preprocessing import create_preprocessing_pipeline
    # Nếu train.py cần config trực tiếp:
    # from .config import SOME_TRAIN_CONFIG_PARAM
except ImportError as e:
    print(f"Lỗi import trong src/train.py: {e}")
    # Cung cấp giải pháp thay thế đơn giản nếu không tìm thấy (chỉ cho mục đích gỡ lỗi)
    # Trong môi trường thực tế, lỗi import ở đây nên được coi là nghiêm trọng
    if 'get_spark_session' not in locals(): # Kiểm tra một hàm cụ thể
        def get_spark_session(app_name="DefaultApp"):
            # Đây là một fallback rất cơ bản, không nên dựa vào nó
            from pyspark.sql import SparkSession
            print("Cảnh báo: get_spark_session không được import đúng cách, sử dụng fallback.")
            return SparkSession.builder.appName(app_name).master("local[*]").getOrCreate()
    # raise # Ném lại lỗi để dừng nếu cần thiết


def train_regression_model(spark, training_data_df, preprocessing_pipeline_stages):
    """
    Huấn luyện mô hình hồi quy sử dụng pipeline tiền xử lý và dữ liệu huấn luyện.
    """
    if training_data_df is None:
        print("Dữ liệu huấn luyện là None. Không thể huấn luyện mô hình.")
        return None

    # Tên cột nhãn thực tế phụ thuộc vào output_label_col trong create_preprocessing_pipeline
    # Giả sử nó được truyền vào hoặc lấy từ config, ví dụ 'percentage_change'
    # Để đơn giản, chúng ta sẽ giả định tên cột nhãn là 'percentage_change' nếu không có cách lấy khác
    # Trong create_preprocessing_pipeline, output_label_col được truyền vào.
    # Hàm này nhận preprocessing_pipeline_stages, không phải tên cột nhãn trực tiếp.
    # Chúng ta cần tìm tên cột nhãn từ stages hoặc giả định nó.
    # Cách tốt nhất là hàm create_preprocessing_pipeline trả về cả stages và tên cột nhãn.
    # Hoặc, chúng ta tìm SQLTransformer và lấy outputCol của nó.
    label_col_name = None
    for stage in reversed(preprocessing_pipeline_stages): # Tìm từ cuối lên
        if hasattr(stage, 'getOutputCol') and hasattr(stage, 'getStatement'): # Heuristic for SQLTransformer
             # Kiểm tra xem statement có tạo ra cột nhãn không
            if "AS " in stage.getStatement().upper() and "FROM __THIS__" in stage.getStatement().upper():
                 # Lấy tên cột output của SQLTransformer đầu tiên (giả định là label creator)
                 # Đây là một cách suy đoán, tốt hơn là truyền tên cột nhãn vào.
                 # Giả sử output_label_col của create_preprocessing_pipeline là tên cột nhãn.
                 # Cách an toàn hơn: lấy từ featuresCol của GBTRegressor sau này.
                 # Hiện tại, chúng ta sẽ dựa vào việc GBTRegressor được cấu hình đúng labelCol.
                 pass # Sẽ lấy từ GBTRegressor sau
    
    # Nếu không tìm được, đặt một giá trị mặc định và hy vọng GBTRegressor có labelCol đúng
    if label_col_name is None:
        print("Cảnh báo: Không thể tự động xác định tên cột nhãn từ preprocessing_stages. Sẽ dựa vào cấu hình của GBTRegressor.")
        # label_col_name = "percentage_change" # Hoặc lấy từ config

    try:
        cleaned_df_for_features = training_data_df.na.drop(subset=["full_article_text", "open_price", "close_price"])

        if cleaned_df_for_features.count() == 0:
            print("Không có dữ liệu sau khi loại bỏ null ban đầu cho các cột features.")
            return None

        (train_df_raw, test_df_raw) = cleaned_df_for_features.randomSplit([0.8, 0.2], seed=42)
        
        # Lấy tên cột nhãn từ cấu hình của mô hình GBTRegressor sẽ được thêm vào pipeline
        # Điều này an toàn hơn là cố gắng suy đoán từ các stages tiền xử lý.
        # Giả sử GBTRegressor sẽ được cấu hình với labelCol chính xác.
        # Trong ví dụ này, GBTRegressor sẽ dùng labelCol là 'percentage_change' (mặc định hoặc từ config)

        # Tạo mô hình GBTRegressor
        # Tên cột nhãn phải khớp với những gì create_preprocessing_pipeline tạo ra
        # và những gì được cấu hình trong main.py (ví dụ: output_label_col_name)
        # Chúng ta sẽ dùng một tên cố định ở đây, ví dụ 'percentage_change',
        # và đảm bảo nó nhất quán.
        gbt_label_col = "percentage_change" # Đảm bảo tên này nhất quán!
        gbt = GBTRegressor(featuresCol="features", labelCol=gbt_label_col, maxIter=20)

        temp_preprocessing_pipeline = Pipeline(stages=preprocessing_pipeline_stages)
        
        print("\nÁp dụng tiền xử lý để tạo nhãn và features cho tập huấn luyện...")
        # Fit pipeline tiền xử lý trên dữ liệu huấn luyện thô
        fitted_preprocessing_model = temp_preprocessing_pipeline.fit(train_df_raw)
        processed_train_df = fitted_preprocessing_model.transform(train_df_raw)
        
        final_train_df = processed_train_df.filter(col(gbt_label_col).isNotNull())
        print(f"Số lượng mẫu huấn luyện sau khi lọc label NULL ({gbt_label_col}): {final_train_df.count()}")

        if final_train_df.count() == 0:
            print(f"Không có dữ liệu huấn luyện sau khi lọc các hàng có nhãn {gbt_label_col} NULL.")
            return None
            
        print(f"\nBắt đầu huấn luyện mô hình GBTRegressor với labelCol='{gbt_label_col}'...")
        gbt_model = gbt.fit(final_train_df)
        print("Huấn luyện GBTRegressor hoàn tất.")

        # Kết hợp các stages tiền xử lý đã fit với mô hình GBT đã huấn luyện
        complete_pipeline_model = PipelineModel(stages=fitted_preprocessing_model.stages + [gbt_model])
        print("Đã tạo PipelineModel hoàn chỉnh.")

        print("\nĐánh giá mô hình trên tập kiểm tra...")
        processed_test_df = fitted_preprocessing_model.transform(test_df_raw) # Dùng lại fitted_preprocessing_model
        final_test_df = processed_test_df.filter(col(gbt_label_col).isNotNull())
        
        if final_test_df.count() == 0:
            print(f"Không có dữ liệu kiểm tra sau khi lọc các hàng có nhãn {gbt_label_col} NULL. Không thể đánh giá.")
            return complete_pipeline_model

        predictions_df = gbt_model.transform(final_test_df)

        print("\nMột vài dự đoán trên tập kiểm tra:")
        predictions_df.select("date", "symbol", "open_price", "close_price", gbt_label_col, "prediction").show(10, truncate=True)

        evaluator_rmse = RegressionEvaluator(labelCol=gbt_label_col, predictionCol="prediction", metricName="rmse")
        rmse = evaluator_rmse.evaluate(predictions_df)
        print(f"Root Mean Squared Error (RMSE) trên tập kiểm tra: {rmse:.4f}")

        evaluator_mae = RegressionEvaluator(labelCol=gbt_label_col, predictionCol="prediction", metricName="mae")
        mae = evaluator_mae.evaluate(predictions_df)
        print(f"Mean Absolute Error (MAE) trên tập kiểm tra: {mae:.4f}")
        
        evaluator_r2 = RegressionEvaluator(labelCol=gbt_label_col, predictionCol="prediction", metricName="r2")
        r2 = evaluator_r2.evaluate(predictions_df)
        print(f"R-squared (R2) trên tập kiểm tra: {r2:.4f}")

        return complete_pipeline_model

    except Exception as e:
        print(f"Lỗi trong quá trình huấn luyện mô hình hồi quy: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_model(model, path):
    if model is None:
        print("Mô hình là None. Không thể lưu.")
        return
    try:
        print(f"\nĐang lưu mô hình vào: {path}")
        model.write().overwrite().save(path)
        print("Lưu mô hình thành công.")
    except Exception as e:
        print(f"Lỗi khi lưu mô hình: {e}")

if __name__ == "__main__":
    # Khi chạy trực tiếp, relative import có thể gây lỗi nếu không chạy bằng `python -m src.train`
    # Khối __main__ này chủ yếu để test nhanh, cần cẩn thận với imports.
    try:
        # Thử import lại với context là đang chạy file trực tiếp (không phải module)
        from data_loader import get_spark_session, load_stock_prices, load_news_articles, join_data
        from preprocessing import create_preprocessing_pipeline
        # from config import TRAIN_PRICES_FILE, TRAIN_ARTICLES_FILE # Nếu cần
    except ImportError:
        print("Running __main__ in train.py: Gặp lỗi khi import data_loader/preprocessing trực tiếp cho test.")
        print("Điều này có thể xảy ra nếu chạy 'python src/train.py' thay vì 'python -m src.train'")
        # Không thể tiếp tục test nếu các module cơ bản không load được
        exit(1)


    spark = get_spark_session(app_name="StockPrediction_Train_Regression_Standalone")
    
    # Cần định nghĩa các đường dẫn này hoặc import từ config
    # Ví dụ:
    # from config import TRAIN_PRICES_FILE, TRAIN_ARTICLES_FILE, SAVED_REGRESSION_MODEL_PATH
    # Giả sử các file nằm ở thư mục data tương đối với thư mục gốc của project (nơi chứa src)
    prices_path = "../data/prices.csv"
    articles_path = "../data/articles.csv"
    model_output_path = "../models/stock_regression_gbt_pipeline_model_standalone"

    loaded_prices_df = load_stock_prices(spark, prices_path)
    loaded_articles_df = load_news_articles(spark, articles_path)

    if loaded_prices_df and loaded_articles_df:
        raw_joined_data_df = join_data(loaded_prices_df, loaded_articles_df)

        if raw_joined_data_df:
            output_label_col_name_test = "percentage_change"
            preprocessing_pipeline_obj_test = create_preprocessing_pipeline(
                text_input_col="full_article_text",
                numerical_input_cols=["open_price"],
                output_features_col="features",
                output_label_col=output_label_col_name_test
            )
            preprocessing_stages_test = preprocessing_pipeline_obj_test.getStages()
            
            trained_model_test = train_regression_model(spark, raw_joined_data_df, preprocessing_stages_test)

            if trained_model_test:
                save_model(trained_model_test, model_output_path)
            else:
                print("Huấn luyện mô hình (standalone test) thất bại.")
        else:
            print("Không thể join dữ liệu (standalone test).")
    else:
        print("Không thể tải dữ liệu (standalone test).")

    spark.stop()
