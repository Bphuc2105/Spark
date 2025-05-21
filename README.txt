Chạy docker-compose.yml: Để tạo container Spark
Khi chạy chương trình: Ấn lệnh docker-compose up --build: Chạy chương trình thông qua Docker(Phải bật Docker Desktop trước)
Lệnh chạy predict: docker-compose run --rm spark-app spark-submit --master spark://spark-master:7077 --deploy-mode client --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1,org.elasticsearch:elasticsearch-spark-30_2.12:8.11.0 /app/main.py predict
Lệnh train: docker-compose run --rm spark-app spark-submit --master spark://spark-master:7077 --deploy-mode client --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1,org.elasticsearch:elasticsearch-spark-30_2.12:8.11.0 /app/main.py train        
Build lại Spark: docker-compose build --no-cache spark-app
Lệnh tạo lại Spark: docker-compose up -d --force-recreate   
