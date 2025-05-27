# src/preprocessing.py

from pyspark.ml.feature import VectorAssembler, SQLTransformer, RegexTokenizer
from pyspark.ml import Pipeline
from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
# Đảm bảo import các class cần thiết từ pyspark.ml.param.shared
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param, Params, TypeConverters

from pyspark.ml.feature import StringIndexer, HashingTF, Tokenizer, StopWordsRemover, IDF
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import col, udf # udf được dùng nhiều lần
from pyspark.sql.types import StringType, ArrayType, DoubleType # Giữ lại các import này
import re
import hashlib
import numpy as np
# from pyspark.ml import Pipeline as SparkNlpPipeline # Alias không cần thiết nếu không có xung đột

# Import cấu hình
try:
    from . import config
except ImportError:
    print("Cảnh báo: Không thể import .config trong preprocessing.py, sử dụng giá trị mặc định nếu có fallback.")
    class FallbackConfig:
        TEXT_INPUT_COLUMN = "full_article_text"
        NUMERICAL_INPUT_COLUMNS = ["open_price"]
        FEATURES_OUTPUT_COLUMN = "features"
        REGRESSION_LABEL_OUTPUT_COLUMN = "percentage_change"
        HASHING_TF_NUM_FEATURES = 10000
        VIETNAMESE_STOPWORDS = ["và", "là", "có", "của", "trong", "cho", "đến", "khi", "thì", "mà", "ở", "tại"]
        ARTICLE_SEPARATOR = " --- "
    config = FallbackConfig()


class StockChunkExtractor(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable):
    stockCol = Param(Params._dummy(), "stockCol", "name of the stock symbol column", typeConverter=TypeConverters.toString)
    outputCol = Param(Params._dummy(), "outputCol", "output column name", typeConverter=TypeConverters.toString)
    inputCol = Param(Params._dummy(), "inputCol", "input column name", typeConverter=TypeConverters.toString)

    def __init__(self, inputCol="full_article_text", stockCol="symbol", outputCol="relevant_text_chunks", stockMap = None):
        super(StockChunkExtractor, self).__init__()
        self._setDefault(inputCol=inputCol, stockCol=stockCol, outputCol=outputCol)
        # Set các giá trị một cách tường minh để đảm bảo Param được cập nhật
        self._set(inputCol=inputCol)
        self._set(stockCol=stockCol) # Quan trọng: stockCol là "symbol"
        self._set(outputCol=outputCol)

        self.stock_map = stockMap or {
            "VCB": ["Vietcombank", "Ngân hàng Vietcombank", "VCB"],
            "CTG": ["VietinBank", "Ngân hàng Công Thương", "Ngân hàng VietinBank", "CTG"],
            "VPB": ["VPBank", "Ngân hàng VPBank", "VPB"],
            "BID": ["BIDV", "Ngân hàng BIDV", "BID"],
            "TCB": ["Techcombank", "Ngân hàng Techcombank", "TCB"],
            "HPG": ["Hòa Phát", "Tập đoàn Hòa Phát", "HPG"],
            "MWG": ["Thế Giới Di Động", "Công ty Thế Giới Di Động", "MWG"],
            "VNM": ["Vinamilk", "Công ty sữa Vinamilk", "VNM"],
            "VIC": ["Vingroup", "Tập đoàn Vingroup", "VIC"],
            "FPT": ["FPT", "Công ty FPT", "FPT Corporation"],
        }

    def getInputCol(self):
        return self.getOrDefault(self.inputCol)

    def getStockCol(self):
        return self.getOrDefault(self.stockCol)

    def getOutputCol(self):
        return self.getOrDefault(self.outputCol)
    
    def _extract_sentences(self, text):
        if text is None: return []
        text = str(text).replace('\n', ' ').replace('\r', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        abbreviations = [r'TP\.', r'Tp\.', r'TS\.', r'PGS\.', r'ThS\.', r'ĐH\.', 
                         r'ĐHQG\.', r'TT\.', r'P\.', r'Q\.', r'CT\.', r'CTCP\.', r'UBND\.']
        for abbr in abbreviations:
            text = re.sub(abbr, abbr.replace('.', '<period>'), text)
        pattern = r'([.!?])\s+(?=[A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ0-9])'
        processed_text = re.sub(pattern, r'\1\n', text)
        sentences = [s.strip() for s in processed_text.split('\n') if s.strip()]
        result = [s.replace('<period>', '.') for s in sentences]
        return result

    def _extract_stock_chunks_from_article(self, article_text, stock_code_value, context_window=2):
        results_for_current_symbol = []
        if not article_text or not isinstance(article_text, str) or not stock_code_value:
            return "<chunk>".join(results_for_current_symbol) 

        sentences = self._extract_sentences(article_text)
        if not sentences: return "<chunk>".join(results_for_current_symbol)

        current_symbol_aliases = self.stock_map.get(str(stock_code_value).upper(), [str(stock_code_value).upper()])

        stock_mention_indices = set()
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            for alias in current_symbol_aliases:
                if alias and isinstance(alias, str):
                    pattern = r'\b' + re.escape(alias.lower()) + r'\b'
                    if re.search(pattern, sentence_lower):
                        stock_mention_indices.add(i)
                        break
        
        if stock_mention_indices:
            mention_groups = []
            sorted_indices = sorted(list(stock_mention_indices))
            if not sorted_indices:
                return "<chunk>".join(results_for_current_symbol)

            current_group = [sorted_indices[0]]
            for i in range(1, len(sorted_indices)):
                idx = sorted_indices[i]
                if idx <= current_group[-1] + context_window + 1:
                    current_group.append(idx)
                else:
                    mention_groups.append(current_group)
                    current_group = [idx]
            if current_group:
                mention_groups.append(current_group)
            
            for group in mention_groups:
                if not group: continue
                start_idx = max(0, min(group) - context_window)
                end_idx = min(len(sentences), max(group) + context_window + 1)
                chunk = " ".join(sentences[start_idx:end_idx])
                results_for_current_symbol.append(chunk)
        
        return "<chunk>".join(results_for_current_symbol)

    def _transform(self, dataset):
        input_col_name = self.getInputCol()
        stock_col_name = self.getStockCol() 
        output_col_name = self.getOutputCol()

        extract_udf_func = udf(lambda article_text, current_symbol: \
                               self._extract_stock_chunks_from_article(article_text, current_symbol), StringType())
        
        return dataset.withColumn(output_col_name, extract_udf_func(dataset[input_col_name], dataset[stock_col_name]))


class SimpleTextEmbedder(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable):
    """Simple text embedder that creates random vectors based on text hash"""
    inputCol = Param(Params._dummy(), "inputCol", "input column name", typeConverter=TypeConverters.toString)
    outputCol = Param(Params._dummy(), "outputCol", "output column name", typeConverter=TypeConverters.toString)
    vectorSize = Param(Params._dummy(), "vectorSize", "size of the output vector", typeConverter=TypeConverters.toInt)

    def __init__(self, inputCol="text_feature", outputCol="text_embedding", vectorSize=128):
        super(SimpleTextEmbedder, self).__init__()
        self._setDefault(inputCol=inputCol, outputCol=outputCol, vectorSize=vectorSize)
        self._set(inputCol=inputCol)
        self._set(outputCol=outputCol)
        self._set(vectorSize=vectorSize)

    def getInputCol(self):
        return self.getOrDefault(self.inputCol)

    def getOutputCol(self):
        return self.getOrDefault(self.outputCol)

    def getVectorSize(self):
        return self.getOrDefault(self.vectorSize)

    def _transform(self, dataset):
        input_col_name = self.getInputCol()
        output_col_name = self.getOutputCol()
        vec_size = self.getVectorSize()

        def text_to_vector(text):
            if not text or not isinstance(text, str) or text.strip() == "":
                return Vectors.dense([0.0] * vec_size)
            else:
                text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
                seed = int(text_hash[:8], 16)
                np.random.seed(seed)
                vector = np.random.randn(vec_size)
                norm = np.linalg.norm(vector)
                if norm == 0: 
                    return Vectors.dense([0.0] * vec_size)
                vector = vector / norm
                return Vectors.dense(vector.tolist())
        
        text_to_vector_udf = udf(text_to_vector, VectorUDT())
        return dataset.withColumn(output_col_name, text_to_vector_udf(dataset[input_col_name]))

    
def create_preprocessing_pipeline(text_input_col=None,
                                  numerical_input_cols=None,
                                  symbol_col=None, # Thêm tham số symbol_col
                                  output_features_col=None,
                                  output_label_col=None
                                ):
    # Lấy giá trị từ config hoặc dùng giá trị mặc định
    text_input_col_resolved = text_input_col or getattr(config, 'TEXT_INPUT_COLUMN', 'full_article_text')
    numerical_input_cols_resolved = numerical_input_cols or getattr(config, 'NUMERICAL_INPUT_COLUMNS', ['open_price'])
    symbol_col_resolved = symbol_col or "symbol" # Mặc định là "symbol"
    output_features_col_resolved = output_features_col or getattr(config, 'FEATURES_OUTPUT_COLUMN', 'features')
    output_label_col_resolved = output_label_col or getattr(config, 'REGRESSION_LABEL_OUTPUT_COLUMN', 'percentage_change')

    print("Đang tạo pipeline tiền xử lý...")

    # Step 1: Generate label
    sql_transformer_label = SQLTransformer(
        statement=f"SELECT *, (close_price - open_price) AS {output_label_col_resolved} FROM __THIS__"
    )

    # Step 2: Extract text chunks per stock
    chunk_text_transformer = StockChunkExtractor(
        inputCol=text_input_col_resolved,
        stockCol=symbol_col_resolved, # Sử dụng symbol_col_resolved
        outputCol="text_feature_chunks" 
    )

    # Step 3: Encode stock symbol to index
    symbol_indexer = StringIndexer(
        inputCol=symbol_col_resolved, # Sử dụng symbol_col_resolved
        outputCol="symbol_index",
        handleInvalid="keep" 
    )

    # Step 4: Simple text embedding
    text_embedder = SimpleTextEmbedder(
        inputCol=chunk_text_transformer.getOutputCol(), 
        outputCol="text_embedding",
        vectorSize=128 
    )
    
    # Các bước xử lý text truyền thống (Tokenizer, HashingTF, IDF) không còn cần thiết
    # nếu SimpleTextEmbedder đã tạo ra vector "text_embedding" trực tiếp từ "text_feature_chunks".
    # Nếu bạn muốn xử lý "text_feature_chunks" bằng Tokenizer, HashingTF, IDF trước khi embedding
    # hoặc thay thế SimpleTextEmbedder, bạn cần điều chỉnh logic ở đây.
    # Hiện tại, giả định "text_embedding" là output vector từ SimpleTextEmbedder.

    # Step 5: Assemble all features into one vector
    assembler_input_cols = [text_embedder.getOutputCol(), symbol_indexer.getOutputCol()]
    
    # Chỉ thêm numerical_assembler nếu có cột số được cung cấp
    stages_to_add_before_final_assembler = []
    if numerical_input_cols_resolved:
        numerical_assembler = VectorAssembler(
            inputCols=numerical_input_cols_resolved,
            outputCol="numerical_vector_features",
            handleInvalid="skip" 
        )
        stages_to_add_before_final_assembler.append(numerical_assembler)
        assembler_input_cols.append(numerical_assembler.getOutputCol())


    if not assembler_input_cols:
         raise ValueError("Không có cột đặc trưng nào được chọn cho VectorAssembler cuối cùng.")

    vector_assembler = VectorAssembler(
        inputCols=assembler_input_cols,
        outputCol=output_features_col_resolved
    )

    # Final pipeline
    all_stages = [
        sql_transformer_label,
        chunk_text_transformer,
        symbol_indexer,
        text_embedder,
    ]
    all_stages.extend(stages_to_add_before_final_assembler) 
    all_stages.append(vector_assembler)
    
    preprocessing_pipeline = Pipeline(stages=all_stages)

    print("Pipeline tiền xử lý đã được tạo với các stages:")
    for i, stage in enumerate(all_stages):
        stage_info = f"  Stage {i}: {stage.__class__.__name__}"
        try: stage_info += f" | InputCol(s): {stage.getInputCol() if hasattr(stage, 'getInputCol') and callable(stage.getInputCol) else (stage.getInputCols() if hasattr(stage, 'getInputCols') and callable(stage.getInputCols) else 'N/A')}"
        except: pass 
        try: stage_info += f" | OutputCol(s): {stage.getOutputCol() if hasattr(stage, 'getOutputCol') and callable(stage.getOutputCol) else (stage.getOutputCols() if hasattr(stage, 'getOutputCols') and callable(stage.getOutputCols) else 'N/A')}"
        except: pass
        if isinstance(stage, StockChunkExtractor):
            try: stage_info += f" | StockCol: {stage.getStockCol()}"
            except: pass
        print(stage_info)
    return preprocessing_pipeline

if __name__ == "__main__":
    from pyspark.sql import SparkSession
    from pyspark.sql.types import StructType, StructField # Thêm import này
    
    spark = SparkSession.builder.appName("PreprocessingTest").master("local[*]").getOrCreate()

    sample_data_train = [
        (1.0, "2023-01-01", "FPT", "Giá FPT tăng mạnh. Vietcombank (VCB) cũng có tin tốt.", 150.0, 155.0),
        (2.0, "2023-01-01", "VCB", "Vietcombank công bố lợi nhuận quý. FPT thì không.", 250.0, 252.0),
        (3.0, "2023-01-02", "FPT", "Apple đối mặt khó khăn, nhưng FPT vẫn ổn định.", 154.0, 153.0),
        (4.0, "2023-01-02", "MWG", "Thế Giới Di Động khai trương cửa hàng mới.", 80.0, 81.0),
        (5.0, "2023-01-03", "FPT", "Không có tin gì về FPT hôm nay", 153.0, 153.0),
        (6.0, "2023-01-03", "NONAME", "Một công ty ABC nào đó tăng giá", 10.0, 11.0),
        (7.0, "2023-01-04", "VCB", None, 251.0, 253.0), # Text rỗng
    ]
    schema_train = StructType([
        StructField("id", DoubleType(), True),
        StructField("date", StringType(), True),
        StructField("symbol", StringType(), True),
        StructField("full_article_text", StringType(), True),
        StructField("open_price", DoubleType(), True),
        StructField("close_price", DoubleType(), True)
    ])
    data_df_train = spark.createDataFrame(sample_data_train, schema_train)
    print("Dữ liệu huấn luyện mẫu:")
    data_df_train.show(truncate=False)

    pipeline_obj = create_preprocessing_pipeline(
        text_input_col="full_article_text",
        numerical_input_cols=["open_price", "close_price"],
        symbol_col="symbol", # Truyền symbol_col
        output_features_col="features",
        output_label_col="label"
    )

    columns_to_check_null = ["full_article_text", "symbol", "open_price", "close_price"]
    cleaned_data_df = data_df_train.na.drop(subset=columns_to_check_null)

    if cleaned_data_df.count() == 0:
        print("Không có dữ liệu sau khi loại bỏ các hàng null. Không thể fit pipeline.")
    else:
        print(f"Số lượng mẫu sau khi làm sạch null: {cleaned_data_df.count()}")
        print("\nFitting preprocessing pipeline...")
        try:
            pipeline_model = pipeline_obj.fit(cleaned_data_df)
            print("\nTransforming data using the fitted pipeline...")
            processed_df = pipeline_model.transform(cleaned_data_df)
            print("\nDữ liệu sau khi qua pipeline tiền xử lý:")
            processed_df.printSchema()
            # Hiển thị các cột output của từng bước quan trọng
            processed_df.select(
                "date", "symbol", "label", 
                "text_feature_chunks", # Output của StockChunkExtractor
                "symbol_index",        # Output của StringIndexer
                "text_embedding",      # Output của SimpleTextEmbedder
                # "numerical_vector_features", # Output của numerical_assembler (nếu có)
                "features"             # Output cuối cùng
            ).show(truncate=30)

            if processed_df.count() > 0 and "features" in processed_df.columns:
                first_row_features = processed_df.select("features").first()
                if first_row_features and first_row_features[0] is not None:
                    num_features_in_vector = len(first_row_features[0])
                    print(f"\nSố lượng đặc trưng trong vector 'features': {num_features_in_vector}")
        except Exception as e:
            print(f"Lỗi trong quá trình test pipeline: {e}")
            import traceback
            traceback.print_exc()
    spark.stop()
