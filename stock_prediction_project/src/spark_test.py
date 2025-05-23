from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, to_timestamp
from pyspark.sql import functions as F
import os
os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'

def get_spark_session_with_nlp(app_name="StockPredictionApp"):
    """
    Khởi tạo và trả về một SparkSession có cấu hình Spark NLP.
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.1.1") \
        .config("spark.driver.extraJavaOptions", "-Dfile.encoding=UTF-8 -Dsun.stdout.encoding=UTF-8 -Dsun.stderr.encoding=UTF-8") \
        .config("spark.executor.extraJavaOptions", "-Dfile.encoding=UTF-8 -Dsun.stdout.encoding=UTF-8 -Dsun.stderr.encoding=UTF-8") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("INFO")

    return spark

spark = get_spark_session_with_nlp("Test")