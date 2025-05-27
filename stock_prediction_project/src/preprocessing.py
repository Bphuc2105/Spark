from pyspark.ml.feature import VectorAssembler, SQLTransformer, RegexTokenizer
from pyspark.ml import Pipeline
from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml.feature import VectorAssembler, SQLTransformer, StringIndexer, HashingTF, Tokenizer
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import col, lag, sum as spark_sum, collect_list
from pyspark.sql.window import Window
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType, ArrayType, DoubleType
import re
import hashlib
import numpy as np
from pyspark.ml import Pipeline as SparkNlpPipeline
class StockChunkExtractor(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, inputCol="full_article_text", stockCol="symbol", outputCol="text_feature", stockMap = None):
        super().__init__()
        self.inputCol = inputCol
        self.stockCol = stockCol
        self.outputCol = outputCol
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

    def _transform(self, dataset):
        def extract_stock_chunks(article_text, stock_map, context_window=2):
            def extract_sentences(text):
                text = text.replace('\n', ' ').replace('\r', ' ')
                text = re.sub(r'\s+', ' ', text).strip()
                abbreviations = [r'TP\.', r'Tp\.', r'TS\.', r'PGS\.', r'ThS\.', r'ĐH\.', 
                                 r'ĐHQG\.', r'TT\.', r'P\.', r'Q\.', r'CT\.', r'CTCP\.', r'UBND\.']
                for abbr in abbreviations:
                    text = re.sub(abbr, abbr.replace('.', '<period>'), text)
                pattern = r'([.!?])\s+([A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ0-9])'
                text = re.sub(pattern, r'\1\n\2', text)
                sentences = [s.strip() for s in text.split('\n') if s.strip()]
                result = []
                for sentence in sentences:
                    sentence = sentence.replace('<period>', '.')
                    result.append(sentence)
                return result

            results = {}
            sentences = extract_sentences(article_text)
            for stock_code, aliases in stock_map.items():
                stock_mention_indices = set()
                for i, sentence in enumerate(sentences):
                    sentence_lower = sentence.lower()
                    for alias in aliases:
                        pattern = r'\b' + re.escape(alias.lower()) + r'\b'
                        if re.search(pattern, sentence_lower):
                            stock_mention_indices.add(i)
                            break
                if stock_mention_indices:
                    mention_groups = []
                    current_group = []
                    for idx in sorted(stock_mention_indices):
                        if not current_group or idx <= current_group[-1] + context_window + 1:
                            current_group.append(idx)
                        else:
                            mention_groups.append(current_group)
                            current_group = [idx]
                    if current_group:
                        mention_groups.append(current_group)
                    stock_chunks = []
                    for group in mention_groups:
                        start_idx = max(0, min(group) - context_window)
                        end_idx = min(len(sentences), max(group) + context_window + 1)
                        chunk = " ".join(sentences[start_idx:end_idx])
                        stock_chunks.append(chunk)
                    if stock_chunks:
                        results[stock_code] = stock_chunks
            return results

        def extract_text_feature(full_article_text, symbol, article_separator="<s>"):
            articles = full_article_text.split(article_separator) 
            selected_chunks = []
            for text in articles:
                chunks_article = extract_stock_chunks(text, self.stock_map)
                selected_chunks.extend(chunks_article.get(symbol, []))
            return "<chunk>".join(selected_chunks)
        
        extract_udf = udf(extract_text_feature, StringType())
        return dataset.withColumn(self.outputCol, extract_udf(dataset[self.inputCol], dataset[self.stockCol]))
    
class SimpleTextEmbedder(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, inputCol="text_feature", outputCol="text_embedding", vectorSize=128):
        super().__init__()
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.vectorSize = vectorSize

    def _transform(self, dataset):
        def text_to_vector(text):
            if not text or text.strip() == "":
                vector_array = [0.0] * self.vectorSize
            else:
                text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
                seed = int(text_hash[:8], 16)
                np.random.seed(seed)
                vector = np.random.randn(self.vectorSize)
                vector = vector / np.linalg.norm(vector)
                vector_array = vector.tolist()
            return Vectors.dense(vector_array)
        
        text_to_vector_udf = udf(text_to_vector, VectorUDT())
        return dataset.withColumn(self.outputCol, text_to_vector_udf(dataset[self.inputCol]))
    
def create_preprocessing_pipeline(text_input_col="full_article_text",
                                  numerical_input_cols=["open_price", "close_price"],
                                  symbol_col="symbol",
                                  output_features_col="features",
                                  output_label_col="label"
                                ):
    print("Đang tạo pipeline tiền xử lý...")
    sql_transformer_label = SQLTransformer(
        statement=f"SELECT *, (close_price - open_price) AS {output_label_col} FROM __THIS__"
    )
    chunk_text_transformer = StockChunkExtractor(
        inputCol=text_input_col,
        stockCol=symbol_col,
        outputCol="text_feature"
    )
    symbol_indexer = StringIndexer(
        inputCol=symbol_col,
        outputCol="symbol_index",
        handleInvalid="keep"
    )
    text_embedder = SimpleTextEmbedder(
        inputCol="text_feature",
        outputCol="text_embedding",
        vectorSize=128
    )
    assembler_input_cols = ["text_embedding", "symbol_index"] + numerical_input_cols
    vector_assembler = VectorAssembler(
        inputCols=assembler_input_cols,
        outputCol=output_features_col
    )
    preprocessing_pipeline = Pipeline(stages=[
        sql_transformer_label,
        chunk_text_transformer,
        symbol_indexer,
        text_embedder,
        vector_assembler
    ])
    print("Pipeline tiền xử lý đã được tạo.")
    return preprocessing_pipeline

if __name__ == "__main__":
    from pyspark.sql import SparkSession
    try:
        from data_loader_testing import get_spark_session, load_stock_prices, load_news_articles, join_data
    except ImportError:
        print("Không thể import data_loader. Vui lòng đảm bảo nó tồn tại và có thể truy cập.")
        exit()
    spark = get_spark_session("PreprocessingTest")
    prices_path = "data/prices.csv"
    articles_path =  "data/articles.csv"
    prices_df = load_stock_prices(spark, prices_path)
    articles_df = load_news_articles(spark, articles_path)
    if prices_df and articles_df:
        raw_data_df = join_data(prices_df, articles_df)
        if raw_data_df:
            print("\nDữ liệu thô sau khi join:")
            raw_data_df.show(5, truncate=True)
            raw_data_df.printSchema()
            pipeline = create_preprocessing_pipeline(
                    text_input_col="full_article_text",
                    numerical_input_cols=["open_price", "close_price"],
                    output_features_col="features",
                    output_label_col="label"
            )
            columns_to_check_null = ["date", "symbol", "open_price", "close_price", "full_article_text"]
            cleaned_data_df = raw_data_df.na.drop(subset=columns_to_check_null)
            if cleaned_data_df.count() == 0:
                print("Không có dữ liệu sau khi loại bỏ các hàng null. Không thể fit pipeline.")
            else:
                print(f"Số lượng mẫu sau khi làm sạch null: {cleaned_data_df.count()}")
                print("\nFitting preprocessing pipeline...")
                pipeline_model = pipeline.fit(cleaned_data_df)
                print("\nTransforming data using the fitted pipeline...")
                processed_df = pipeline_model.transform(cleaned_data_df)
                print("\nDữ liệu sau khi qua pipeline tiền xử lý:")
                processed_df.printSchema()
                processed_df.select("date", "symbol", "label", "features").show(10, truncate=50)
                if processed_df.count() > 0:
                    num_features_in_vector = len(processed_df.select("features").first()[0])
                    print(f"\nSố lượng đặc trưng trong vector 'features': {num_features_in_vector}")
        else:
            print("Không thể join dữ liệu.")
    else:
        print("Không thể tải dữ liệu giá hoặc bài báo.")
    spark.stop()