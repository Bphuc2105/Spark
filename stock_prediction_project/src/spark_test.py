from pyspark.sql import SparkSession
import os

spark = SparkSession.builder.appName("Test").getOrCreate()
print(spark.version)
spark.stop()
