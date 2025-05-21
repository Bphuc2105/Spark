# src/preprocessing.py

from pyspark.ml import Pipeline
from pyspark.ml import Transformer
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml.feature import VectorAssembler, SQLTransformer
from pyspark.sql.functions import col, lag, sum as spark_sum, collect_list
from pyspark.sql.window import Window
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType
import sparknlp
from sparknlp.base import DocumentAssembler, EmbeddingsFinisher
from sparknlp.annotator import (
    BertSentenceEmbeddings,
    SentenceEmbeddings
)
from pyspark.ml import Pipeline as SparkNlpPipeline
import re

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

        def extract_text_feature(full_article_text, symbol):
            chunks_article = extract_stock_chunks(full_article_text, self.stock_map)
            selected_chunks = chunks_article.get(symbol, [])
            return "<chunk>".join(selected_chunks)
        
        extract_udf = udf(extract_text_feature, StringType())
        return dataset.withColumn(self.outputCol, extract_udf(dataset[self.inputCol], dataset[self.stockCol]))
    
def create_preprocessing_pipeline(text_input_col="full_article_text",
                                numerical_input_cols=["open_price", "close_price"],
                                output_features_col="features",
                                output_label_col="label"):
    """
    Tạo một Spark ML Pipeline để tiền xử lý dữ liệu.
    Pipeline bao gồm:
    1. Tạo nhãn (sử dụng SQLTransformer cho tính linh hoạt trong pipeline).
    2. Chunking full_article_text
    3. Gán embedding
    4. VectorAssembler: Kết hợp embedding + number input + stock code index.

    Args:
        text_input_col (str): Tên cột chứa văn bản đầu vào (ví dụ: 'text_feature').
        numerical_input_cols (list): Danh sách tên các cột đặc trưng số (ví dụ: ['number_feature']).
        output_features_col (str): Tên cột chứa vector đặc trưng kết hợp cuối cùng.
        output_label_col (str): Tên cột nhãn.

    Returns:
        pyspark.ml.Pipeline: Đối tượng Pipeline đã được cấu hình.
    """
    print("Đang tạo pipeline tiền xử lý...")
    

    def create_sparknlp_embedding_pipeline(inputCol="text_feature", outputCol="text_embedding"):
        # Convert raw text to Document type
        document_assembler = DocumentAssembler() \
            .setInputCol(inputCol) \
            .setOutputCol("document")

        # Use pretrained BERT sentence embeddings
        sentence_embeddings = BertSentenceEmbeddings.pretrained("sent_small_bert_L2_128", "vi") \
            .setInputCols(["document"]) \
            .setOutputCol("sentence_embeddings")

        # Convert Spark NLP embeddings to Spark ML-compatible array<float>
        embeddings_finisher = EmbeddingsFinisher() \
            .setInputCols(["sentence_embeddings"]) \
            .setOutputCols([outputCol]) \
            .setOutputAsVector(True)

        return SparkNlpPipeline(stages=[
            document_assembler,
            sentence_embeddings,
            embeddings_finisher
        ])
    # Bước : Tạo nhãn (sử dụng SQLTransformer để tích hợp vào pipeline)
    sql_transformer_label = SQLTransformer(
        statement=f"SELECT *, (open_price - prev_close_price) AS {output_label_col} FROM __THIS__"
    )
    chunk_text_transformer = StockChunkExtractor(
        inputCol=text_input_col,
        stockCol="symbol",  # Assuming you have this column
        outputCol="text_feature"
    )
    # Step : Spark NLP embedding pipeline
    nlp_embedding_pipeline = create_sparknlp_embedding_pipeline(
        inputCol="text_feature",
        outputCol="text_embedding"
    )
    

    # Bước : VectorAssembler để kết hợp đặc trưng văn bản và đặc trưng số
    # Đầu vào là cột text_features (từ IDF) và các cột số được chỉ định
    assembler_input_cols = ["text_embedding"] + numerical_input_cols
    vector_assembler = VectorAssembler(inputCols=assembler_input_cols,
                                    outputCol=output_features_col)

    # Tạo Pipeline với tất cả các bước
    # Lưu ý thứ tự của các bước là quan trọng
    preprocessing_pipeline = Pipeline(stages=[
        sql_transformer_label, # Tạo nhãn trước
        chunk_text_transformer,
        *nlp_embedding_pipeline.getStages(),  # unpack NLP stages
        vector_assembler
    ])

    print("Pipeline tiền xử lý đã được tạo.")
    return preprocessing_pipeline

    

if __name__ == "__main__":
    # Phần này dùng để kiểm thử pipeline tiền xử lý
    # Bạn cần có data_loader.py để chạy phần này
    try:
        from data_loader_testing import get_spark_session_with_nlp, load_stock_prices, load_news_articles, join_data
    except ImportError:
        print("Không thể import data_loader. Vui lòng đảm bảo nó tồn tại và có thể truy cập.")
        exit()

    spark = get_spark_session_with_nlp("PreprocessingTest")

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
            print("\nFitting preprocessing pipeline...")
            # Loại bỏ các hàng có giá trị null trong các cột quan trọng trước khi fit
            # Ví dụ: cột 'full_article_text', 'open_price', 'close_price'
            # Cột 'close_price' cần thiết cho SQLTransformer để tạo nhãn
            columns_to_check_null = ["text_feature", "number_feature", "prev_open", "prev_close", "open", "close"]
            cleaned_data_df = raw_data_df.na.drop(subset=columns_to_check_null)

            if cleaned_data_df.count() == 0:
                print("Không có dữ liệu sau khi loại bỏ các hàng null. Không thể fit pipeline.")
            else:
                print(f"Số lượng mẫu sau khi làm sạch null: {cleaned_data_df.count()}")
                pipeline_model = pipeline.fit(cleaned_data_df)

                # Áp dụng pipeline đã fit để biến đổi dữ liệu
                print("\nTransforming data using the fitted pipeline...")
                processed_df = pipeline_model.transform(cleaned_data_df)

                print("\nDữ liệu sau khi qua pipeline tiền xử lý:")
                processed_df.printSchema()
                # Hiển thị các cột quan trọng: nhãn và vector đặc trưng
                processed_df.select("date", "symbol", "text_feature", "number_feature", "prev_open", "prev_close", "open", "close", "label", "features").show(5, truncate=True)

                # Kiểm tra số lượng đặc trưng trong vector 'features'
                if processed_df.count() > 0:
                    num_features_in_vector = len(processed_df.select("features").first()[0])
                    print(f"\nSố lượng đặc trưng trong vector 'features': {num_features_in_vector}")
        else:
            print("Không thể join dữ liệu.")
    else:
        print("Không thể tải dữ liệu giá hoặc bài báo.")

    spark.stop()
