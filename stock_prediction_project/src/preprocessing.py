# src/preprocessing.py

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
# import sparknlp
# from sparknlp.base import DocumentAssembler, EmbeddingsFinisher
# from sparknlp.annotator import (
#     BertSentenceEmbeddings,
#     SentenceEmbeddings
# )
from pyspark.ml import Pipeline as SparkNlpPipeline

class StockChunkExtractor(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, inputCol="full_article_text", stockCol="stock_code", outputCol="text_feature", stockMap = None):
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
            """Find stock codes in an article and return continuous text chunks containing the mentions and their context.
            
            Args:
                article_text: The text of the article to analyze
                stock_map: Dictionary mapping stock codes to their aliases
                context_window: Number of sentences before and after to include as context
                
            Returns:
                Dictionary with stock codes as keys and lists of text chunks as values
            """
            def extract_sentences(text):
                """Split text into sentences with improved handling for Vietnamese text."""
                # Clean the text first
                text = text.replace('\n', ' ').replace('\r', ' ')
                text = re.sub(r'\s+', ' ', text).strip()
                
                # Common Vietnamese abbreviations to handle
                abbreviations = [r'TP\.', r'Tp\.', r'TS\.', r'PGS\.', r'ThS\.', r'ĐH\.', 
                                r'ĐHQG\.', r'TT\.', r'P\.', r'Q\.', r'CT\.', r'CTCP\.', r'UBND\.']
                
                # Replace periods in abbreviations temporarily to avoid splitting sentences there
                for abbr in abbreviations:
                    text = re.sub(abbr, abbr.replace('.', '<period>'), text)
                
                # Split text into sentences
                # Look for end punctuation followed by space and capital letter or digit
                pattern = r'([.!?])\s+([A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ0-9])'
                text = re.sub(pattern, r'\1\n\2', text)
                
                # Split by newline which now represents sentence boundaries
                sentences = [s.strip() for s in text.split('\n') if s.strip()]
                
                # Restore periods in abbreviations
                result = []
                for sentence in sentences:
                    sentence = sentence.replace('<period>', '.')
                    result.append(sentence)
                
                return result

            results = {}
            sentences = extract_sentences(article_text)
            
            # For each stock code, find mentions and extract surrounding chunks
            for stock_code, aliases in stock_map.items():
                stock_mention_indices = set()
                
                # Find sentences with direct mentions
                for i, sentence in enumerate(sentences):
                    sentence_lower = sentence.lower()
                    for alias in aliases:
                        pattern = r'\b' + re.escape(alias.lower()) + r'\b'
                        if re.search(pattern, sentence_lower):
                            stock_mention_indices.add(i)
                            break
                
                # If we found mentions, extract text chunks
                if stock_mention_indices:
                    # Group adjacent mentions together to avoid overlapping chunks
                    mention_groups = []
                    current_group = []
                    
                    for idx in sorted(stock_mention_indices):
                        if not current_group or idx <= current_group[-1] + context_window + 1:
                            # This mention is close to the previous one, add to current group
                            current_group.append(idx)
                        else:
                            # This mention is far from previous ones, start new group
                            mention_groups.append(current_group)
                            current_group = [idx]
                    
                    if current_group:  # Add the last group
                        mention_groups.append(current_group)
                    
                    # Extract text chunks based on mention groups
                    stock_chunks = []
                    for group in mention_groups:
                        # Determine the chunk boundaries with context
                        start_idx = max(0, min(group) - context_window)
                        end_idx = min(len(sentences), max(group) + context_window + 1)
                        
                        # Join the sentences to form a continuous chunk
                        chunk = " ".join(sentences[start_idx:end_idx])
                        stock_chunks.append(chunk)
                    
                    if stock_chunks:
                        results[stock_code] = stock_chunks
            
            return results

        def extract_text_feature(full_article_text, symbol, article_separator="<s>"):
            articles = full_article_text.split(article_separator) # <s> là seperator giữa các article
            selected_chunks = []
            for text in articles:
                chunks_article = extract_stock_chunks(text, self.stock_map)
                selected_chunks.extend(chunks_article.get(symbol, []))
            return "<chunk>".join(selected_chunks)
        
        extract_udf = udf(extract_text_feature, StringType())
        return dataset.withColumn(self.outputCol, extract_udf(dataset[self.inputCol], dataset[self.stockCol]))

class SimpleTextEmbedder(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    """Simple text embedder that creates random vectors based on text hash"""
    def __init__(self, inputCol="text_feature", outputCol="text_embedding", vectorSize=128):
        super().__init__()
        self.inputCol = inputCol
        self.outputCol = outputCol
        self.vectorSize = vectorSize

    def _transform(self, dataset):
        def text_to_vector(text):
            """Convert text to a simple numeric vector using hash-based approach"""
            if not text or text.strip() == "":
                # Return zero vector for empty text
                vector_array = [0.0] * self.vectorSize
            else:
                # Use text hash as seed for reproducible random vectors
                text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
                seed = int(text_hash[:8], 16)
                np.random.seed(seed)
                
                # Generate random vector normalized to unit length
                vector = np.random.randn(self.vectorSize)
                vector = vector / np.linalg.norm(vector)
                vector_array = vector.tolist()
            
            # Return as Spark ML Vector (dense vector)
            return Vectors.dense(vector_array)
        
        text_to_vector_udf = udf(text_to_vector, VectorUDT())
        return dataset.withColumn(self.outputCol, text_to_vector_udf(dataset[self.inputCol]))
    
def create_preprocessing_pipeline(text_input_col="full_article_text",
                                numerical_input_cols=["open_price", "close_price"],
                                symbol_col="symbol",
                                output_features_col="features",
                                output_label_col="label"
                            ):
    """
    Create Spark ML pipeline for processing stock prediction data.
    - Generates labels (close - open).
    - Extracts per-stock chunked text.
    - Embeds text using simple hash-based vectors (instead of BERT).
    - Encodes stock symbol as index.
    - Combines features into a single vector.

    Returns:
        pyspark.ml.Pipeline
    """
    print("Đang tạo pipeline tiền xử lý...")

    # Step 1: Generate label (close - open)
    sql_transformer_label = SQLTransformer(
        statement=f"SELECT *, (close_price - open_price) AS {output_label_col} FROM __THIS__"
    )

    # Step 2: Extract text chunks per stock
    chunk_text_transformer = StockChunkExtractor(
        inputCol=text_input_col,
        stockCol=symbol_col,
        outputCol="text_feature"
    )

    # Step 3: Encode stock symbol to index
    symbol_indexer = StringIndexer(
        inputCol=symbol_col,
        outputCol="symbol_index",
        handleInvalid="keep"
    )

    # Step 4: Simple text embedding (replaces Spark NLP)
    text_embedder = SimpleTextEmbedder(
        inputCol="text_feature",
        outputCol="text_embedding",
        vectorSize=128  # Smaller vector size for simplicity
    )
    # local_model_path = "models/embedding_model"
    # # Commented out Spark NLP embedding pipeline
    # def create_sparknlp_embedding_pipeline(inputCol="text_feature", outputCol="text_embedding"):
    #     document_assembler = DocumentAssembler() \
    #         .setInputCol(inputCol) \
    #         .setOutputCol("document")

    #     sentence_embeddings = BertSentenceEmbeddings.load(local_model_path) \
    #         .setInputCols(["document"]) \
    #         .setOutputCol("sentence_embeddings")

    #     embeddings_finisher = EmbeddingsFinisher() \
    #         .setInputCols(["sentence_embeddings"]) \
    #         .setOutputCols([outputCol]) \
    #         .setOutputAsVector(True)

    #     return SparkNlpPipeline(stages=[
    #         document_assembler,
    #         sentence_embeddings,
    #         embeddings_finisher
    #     ])

    # nlp_embedding_pipeline = create_sparknlp_embedding_pipeline()

    # Step 5: Assemble all features into one vector
    assembler_input_cols = ["text_embedding", "symbol_index"] + numerical_input_cols
    vector_assembler = VectorAssembler(
        inputCols=assembler_input_cols,
        outputCol=output_features_col
    )

    # Final pipeline (simplified without Spark NLP stages)
    preprocessing_pipeline = Pipeline(stages=[
        sql_transformer_label,
        chunk_text_transformer,
        symbol_indexer,
        text_embedder,
        # *nlp_embedding_pipeline.getStages(),  # Simple text embedder instead of Spark NLP
        vector_assembler
    ])

    print("Pipeline tiền xử lý đã được tạo.")
    return preprocessing_pipeline

    

if __name__ == "__main__":
    from pyspark.sql import SparkSession
    # Khi chạy trực tiếp, relative import có thể gây lỗi nếu không chạy bằng `python -m src.preprocessing`
    # Đoạn test này có thể cần điều chỉnh hoặc bỏ qua nếu chỉ tập trung vào việc module được import đúng
    try:
        # Modified import - removed nlp requirement
        from data_loader_testing import get_spark_session, load_stock_prices, load_news_articles, join_data
    except ImportError:
        print("Không thể import data_loader. Vui lòng đảm bảo nó tồn tại và có thể truy cập.")
        exit()

    # Modified to use regular Spark session instead of NLP-enabled one
    spark = get_spark_session("PreprocessingTest")

    # --- Cấu hình đường dẫn (giống như trong data_loader.py) ---
    prices_path = "data/prices.csv"
    articles_path =  "data/articles.csv"

    # --- Tải dữ liệu ---
    prices_df = load_stock_prices(spark, prices_path)
    articles_df = load_news_articles(spark, articles_path)

    if prices_df and articles_df:
        # Kết hợp dữ liệu
        # Hàm join_data từ data_loader.py sẽ tạo cột 'full_article_text'
        raw_data_df = join_data(prices_df, articles_df)

        if raw_data_df:
            print("\nDữ liệu thô sau khi join:")
            raw_data_df.show(5, truncate=True)
            raw_data_df.printSchema()
            
            # --- Tạo và kiểm thử pipeline tiền xử lý ---
            # Giả sử cột văn bản là 'full_article_text' và cột số là 'open_price'
            # Cột nhãn sẽ được pipeline tạo ra là 'label'
            # Cột đặc trưng cuối cùng sẽ là 'features'
            pipeline = create_preprocessing_pipeline(
                    text_input_col="full_article_text",
                    numerical_input_cols=["open_price", "close_price"],
                    output_features_col="features",
                    output_label_col="label"
            )
            # Huấn luyện pipeline tiền xử lý trên dữ liệu (chỉ các transformer không yêu cầu huấn luyện trước)
            # Đối với các Estimator như IDF, chúng cần được fit.
            # Loại bỏ các hàng có giá trị null trong các cột quan trọng trước khi fit
            # Ví dụ: cột 'full_article_text', 'open_price', 'close_price'
            # Cột 'close_price' cần thiết cho SQLTransformer để tạo nhãn
            columns_to_check_null = ["date", "symbol", "open_price", "close_price", "full_article_text"]
            cleaned_data_df = raw_data_df.na.drop(subset=columns_to_check_null)

            if cleaned_data_df.count() == 0:
                print("Không có dữ liệu sau khi loại bỏ các hàng null. Không thể fit pipeline.")
            else:
                print(f"Số lượng mẫu sau khi làm sạch null: {cleaned_data_df.count()}")
                print("\nFitting preprocessing pipeline...")
                # cleaned_data_df.select("full_article_text").show(1, truncate=False)
                pipeline_model = pipeline.fit(cleaned_data_df)

                # Áp dụng pipeline đã fit để biến đổi dữ liệu
                print("\nTransforming data using the fitted pipeline...")
                processed_df = pipeline_model.transform(cleaned_data_df)

                print("\nDữ liệu sau khi qua pipeline tiền xử lý:")
                processed_df.printSchema()
                # Hiển thị các cột quan trọng: nhãn và vector đặc trưng
                processed_df.select("date", "symbol", "label", "features").show(10, truncate=50)

                # Kiểm tra số lượng đặc trưng trong vector 'features'
                if processed_df.count() > 0:
                    num_features_in_vector = len(processed_df.select("features").first()[0])
                    print(f"\nSố lượng đặc trưng trong vector 'features': {num_features_in_vector}")
        else:
            print("Không thể join dữ liệu.")
    else:
        print("Không thể tải dữ liệu giá hoặc bài báo.")

    spark.stop()