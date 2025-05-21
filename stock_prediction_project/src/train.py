# src/train.py

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression # Ví dụ: sử dụng Logistic Regression
# Bạn có thể thử các mô hình khác như:
# from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, NaiveBayes
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col

# Import các module cần thiết từ project
try:
    from src.data_loader import get_spark_session, load_stock_prices, load_news_articles, join_data
    from preprocessing import create_preprocessing_pipeline # Sử dụng pipeline đã tạo
except ImportError:
    print("Lỗi: Không thể import các module data_loader hoặc preprocessing.")
    print("Hãy đảm bảo các tệp này tồn tại trong thư mục src/ và có thể truy cập.")
    # Cung cấp giải pháp thay thế đơn giản nếu không tìm thấy
    from pyspark.sql import SparkSession
    def get_spark_session(app_name="DefaultApp"):
        return SparkSession.builder.appName(app_name).master("local[*]").getOrCreate()
    # Các hàm khác sẽ cần được định nghĩa hoặc import đúng cách

def train_model(spark, training_data_df, preprocessing_pipeline_stages):
    """
    Huấn luyện mô hình phân loại sử dụng pipeline tiền xử lý và dữ liệu huấn luyện.

    Args:
        spark (SparkSession): Đối tượng SparkSession.
        training_data_df (DataFrame): DataFrame chứa dữ liệu đã được join (giá và bài báo).
                                      Cần có cột 'open_price', 'close_price', 'full_article_text'.
        preprocessing_pipeline_stages (list): Danh sách các stage của pipeline tiền xử lý
                                             (từ create_preprocessing_pipeline).

    Returns:
        pyspark.ml.PipelineModel: Mô hình Pipeline đã được huấn luyện.
                                  Trả về None nếu có lỗi.
    """
    if training_data_df is None:
        print("Dữ liệu huấn luyện là None. Không thể huấn luyện mô hình.")
        return None

    try:
        # --- 1. Chuẩn bị dữ liệu huấn luyện và kiểm tra ---
        # Loại bỏ các hàng có giá trị null trong các cột quan trọng
        # Cột 'close_price' cần thiết cho SQLTransformer (trong preprocessing_pipeline) để tạo nhãn.
        # 'full_article_text' và 'open_price' là các feature đầu vào.
        columns_to_check_null = ["full_article_text", "open_price", "close_price"]
        cleaned_df = training_data_df.na.drop(subset=columns_to_check_null)

        if cleaned_df.count() == 0:
            print("Không có dữ liệu huấn luyện sau khi loại bỏ các hàng null.")
            return None

        print(f"Số lượng mẫu sau khi làm sạch null để huấn luyện: {cleaned_df.count()}")

        # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
        # (Ví dụ: 80% huấn luyện, 20% kiểm tra)
        (train_df, test_df) = cleaned_df.randomSplit([0.8, 0.2], seed=42)
        print(f"Số lượng mẫu huấn luyện: {train_df.count()}")
        print(f"Số lượng mẫu kiểm tra: {test_df.count()}")

        # --- 2. Định nghĩa mô hình học máy ---
        # Ví dụ sử dụng Logistic Regression
        # Cột nhãn đã được tạo bởi SQLTransformer trong preprocessing_pipeline (tên là 'label')
        # Cột đặc trưng đã được tạo bởi VectorAssembler (tên là 'features')
        lr = LogisticRegression(featuresCol="features", labelCol="label")
        # Bạn có thể thử các mô hình khác:
        # rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=100)
        # gbt = GBTClassifier(featuresCol="features", labelCol="label", maxIter=10)
        # nb = NaiveBayes(featuresCol="features", labelCol="label", modelType="multinomial")

        # --- 3. Tạo Pipeline hoàn chỉnh ---
        # Kết hợp các bước tiền xử lý với mô hình học máy
        # preprocessing_pipeline_stages là một list các stages đã được tạo từ create_preprocessing_pipeline
        # Chúng ta sẽ thêm mô hình học máy (lr) làm stage cuối cùng
        full_pipeline = Pipeline(stages=preprocessing_pipeline_stages + [lr])
        # Nếu preprocessing_pipeline_stages đã là một đối tượng Pipeline, bạn có thể lấy stages của nó:
        # full_pipeline = Pipeline(stages=preprocessing_pipeline_stages.getStages() + [lr])


        # --- 4. Huấn luyện Pipeline ---
        print("\nBắt đầu huấn luyện pipeline hoàn chỉnh...")
        pipeline_model = full_pipeline.fit(train_df)
        print("Huấn luyện pipeline hoàn tất.")

        # --- 5. Đánh giá mô hình trên tập kiểm tra ---
        print("\nĐánh giá mô hình trên tập kiểm tra...")
        predictions_df = pipeline_model.transform(test_df)

        # Hiển thị một vài dự đoán
        print("\nMột vài dự đoán trên tập kiểm tra:")
        predictions_df.select("date", "symbol", "open_price", "close_price", "full_article_text", "label", "rawPrediction", "probability", "prediction").show(10, truncate=True)

        # Sử dụng BinaryClassificationEvaluator để đánh giá
        # Mặc định là AUC ROC nếu không chỉ định metricName
        evaluator_roc_auc = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label", metricName="areaUnderROC")
        roc_auc = evaluator_roc_auc.evaluate(predictions_df)
        print(f"Area Under ROC (AUC) trên tập kiểm tra: {roc_auc:.4f}")

        evaluator_pr_auc = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label", metricName="areaUnderPR")
        pr_auc = evaluator_pr_auc.evaluate(predictions_df)
        print(f"Area Under PR (Precision-Recall) trên tập kiểm tra: {pr_auc:.4f}")

        # Tính toán Accuracy (Độ chính xác)
        correct_predictions = predictions_df.filter(col("label") == col("prediction")).count()
        total_predictions = predictions_df.count()
        accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
        print(f"Accuracy trên tập kiểm tra: {accuracy:.2f}%")

        return pipeline_model

    except Exception as e:
        print(f"Lỗi trong quá trình huấn luyện mô hình: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_model(model, path):
    """
    Lưu PipelineModel đã huấn luyện.

    Args:
        model (PipelineModel): Mô hình PipelineModel cần lưu.
        path (str): Đường dẫn để lưu mô hình.
    """
    if model is None:
        print("Mô hình là None. Không thể lưu.")
        return
    try:
        print(f"\nĐang lưu mô hình vào: {path}")
        model.write().overwrite().save(path) # overwrite() để ghi đè nếu đã tồn tại
        print("Lưu mô hình thành công.")
    except Exception as e:
        print(f"Lỗi khi lưu mô hình: {e}")

if __name__ == "__main__":
    # Khởi tạo SparkSession
    spark = get_spark_session(app_name="StockPrediction_Train")

    # --- Cấu hình đường dẫn ---
    prices_path = "../data/prices.csv" # Dữ liệu giá để huấn luyện
    articles_path = "../data/articles.csv" # Dữ liệu bài báo để huấn luyện
    # Đường dẫn để lưu mô hình đã huấn luyện
    # Thư mục này sẽ được tạo nếu chưa có
    model_output_path = "../models/stock_prediction_pipeline_model"

    # --- Tải dữ liệu ---
    prices_df = load_stock_prices(spark, prices_path)
    articles_df = load_news_articles(spark, articles_path)

    if prices_df and articles_df:
        # Kết hợp dữ liệu giá và bài báo
        # Hàm join_data từ data_loader.py sẽ tạo cột 'full_article_text'
        # và các cột cần thiết như 'open_price', 'close_price'
        raw_joined_df = join_data(prices_df, articles_df)

        if raw_joined_df:
            print("\nDữ liệu đã join để huấn luyện:")
            raw_joined_df.printSchema()
            raw_joined_df.show(5, truncate=True)

            # --- Tạo pipeline tiền xử lý ---
            # Các tham số này nên nhất quán với cách bạn muốn xử lý dữ liệu
            preprocessing_pipeline_obj = create_preprocessing_pipeline(
                text_input_col="full_article_text",
                numerical_input_cols=["open_price"],
                output_features_col="features",
                output_label_col="label" # Cột nhãn sẽ được tạo bởi SQLTransformer trong pipeline này
            )
            
            # Lấy danh sách các stages từ đối tượng pipeline tiền xử lý
            preprocessing_stages = preprocessing_pipeline_obj.getStages()


            # --- Huấn luyện mô hình ---
            # raw_joined_df chứa các cột cần thiết (open_price, close_price, full_article_text)
            # mà pipeline tiền xử lý (cụ thể là SQLTransformer) sẽ sử dụng.
            trained_pipeline_model = train_model(spark, raw_joined_df, preprocessing_stages)

            # --- Lưu mô hình ---
            if trained_pipeline_model:
                save_model(trained_pipeline_model, model_output_path)
            else:
                print("Huấn luyện mô hình thất bại. Không có mô hình để lưu.")
        else:
            print("Không thể join dữ liệu huấn luyện.")
    else:
        print("Không thể tải dữ liệu giá hoặc bài báo để huấn luyện.")

    # Dừng SparkSession
    spark.stop()
