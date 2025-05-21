# data/kafka_producer.py

import time
import json
import csv
from kafka import KafkaProducer
import os
from datetime import datetime, timezone # Import datetime và timezone

# Cấu hình Kafka
# Sử dụng địa chỉ external port 9093 để chạy từ host machine
# Nếu chạy script này bên trong container, dùng internal address 'kafka:9092'
KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "localhost:9093") # Mặc định là localhost:9093 cho chạy từ host
NEWS_ARTICLES_TOPIC = os.environ.get("NEWS_ARTICLES_TOPIC", "news_articles")

# Đường dẫn đến file CSV mẫu (trong thư mục data)
# Giả sử bạn có một file CSV mới với cấu trúc tương tự articles.csv + open_price
# Ví dụ: new_data_sample.csv có các cột: id,date,link,title,text,symbol,open_price
SAMPLE_DATA_FILE = "data/new_data_sample.csv" # File CSV mẫu chứa dữ liệu mới

def read_csv_and_send_to_kafka(file_path, kafka_broker, topic):
    """
    Đọc dữ liệu từ file CSV và gửi từng dòng dưới dạng JSON tới Kafka.
    """
    print(f"Đang kết nối tới Kafka broker: {kafka_broker}")
    try:
        # Khởi tạo Kafka Producer
        producer = KafkaProducer(
            bootstrap_servers=[kafka_broker],
            value_serializer=lambda x: json.dumps(x).encode('utf-8'), # Serialize dict sang JSON bytes
            api_version=(0, 10, 2) # Tùy chọn, có thể cần thiết tùy phiên bản Kafka
        )
        print("Kết nối Kafka Producer thành công.")

        if not os.path.exists(file_path):
            print(f"Lỗi: Không tìm thấy file mẫu tại {file_path}")
            # Tạo file mẫu nếu không tồn tại
            print(f"Đang tạo file mẫu: {file_path}")
            sample_content = """id,date,link,title,text,symbol,open_price
pred_id1,2024-01-10T10:00:00+07:00,link1,title1,FPT news good,FPT,85.0
pred_id2,2024-01-10T11:00:00+07:00,link2,title2,VCB stock rally,VCB,60.0
pred_id3,2024-01-11T09:30:00+07:00,link3,title3,FPT shares up,FPT,86.5
pred_id4,2024-01-11T10:15:00+07:00,link4,title4,VCB announces dividend,VCB,61.2
"""
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                 f.write(sample_content)
            print(f"Đã tạo file mẫu: {file_path}. Vui lòng chạy lại script.")
            return # Dừng lại để người dùng chạy lại sau khi file mẫu được tạo


        print(f"Đang đọc dữ liệu từ file: {file_path}")
        with open(file_path, mode='r', encoding='utf-8') as csvfile:
            # Sử dụng DictReader để đọc các dòng dưới dạng dictionary
            reader = csv.DictReader(csvfile)
            row_count = 0
            for row in reader:
                # Mỗi 'row' là một dictionary, sẵn sàng để gửi dưới dạng JSON
                # Đảm bảo các cột cần thiết tồn tại và đúng kiểu dữ liệu (Spark sẽ parse lại)
                # Cần có 'date', 'symbol', 'text', 'open_price'
                if 'date' in row and 'symbol' in row and 'text' in row and 'open_price' in row:
                     # Chuyển đổi open_price sang float nếu nó là string
                     try:
                         row['open_price'] = float(row['open_price'])
                     except (ValueError, TypeError):
                         print(f"Cảnh báo: Không thể chuyển đổi open_price '{row.get('open_price')}' sang số tại dòng {row_count+1}. Bỏ qua dòng này.")
                         continue # Bỏ qua dòng nếu open_price không hợp lệ

                     # Thêm cột 'id' nếu không có (hoặc dùng cột id có sẵn)
                     if 'id' not in row or not row['id']:
                         row['id'] = f"auto_id_{int(time.time() * 1000)}_{row_count}" # Tạo ID đơn giản

                     # Gửi dữ liệu tới Kafka topic
                     # Key có thể là symbol để đảm bảo các bản ghi cùng mã cổ phiếu đi vào cùng partition
                     # Value là dictionary (sẽ được serialize thành JSON)
                     try:
                         future = producer.send(topic, key=row['symbol'].encode('utf-8'), value=row)
                         result = future.get(timeout=10) # Chờ gửi thành công (có timeout)
                         print(f"Đã gửi bản ghi cho {row.get('symbol')} (ID: {row.get('id')}) tới topic {topic}")
                         row_count += 1
                         # Có thể thêm độ trễ nhỏ để mô phỏng stream thời gian thực
                         # time.sleep(0.1)
                     except Exception as e:
                         print(f"Lỗi khi gửi bản ghi tới Kafka: {e}")
                         # Tiếp tục hoặc dừng tùy chiến lược xử lý lỗi

                else:
                    print(f"Cảnh báo: Dòng {row_count+1} thiếu các cột cần thiết (date, symbol, text, open_price). Bỏ qua.")
                    row_count += 1 # Vẫn tăng count để theo dõi dòng

        print(f"Đã gửi {row_count} bản ghi từ file {file_path} tới Kafka topic {topic}.")

    except Exception as e:
        print(f"Đã xảy ra lỗi trong Kafka Producer: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'producer' in locals() and producer:
            producer.flush() # Đảm bảo tất cả tin nhắn đang chờ được gửi đi
            producer.close()
            print("Kafka Producer đã đóng.")

if __name__ == "__main__":
    # Chạy hàm gửi dữ liệu
    # Đảm bảo bạn đang ở thư mục gốc của project khi chạy script này
    # hoặc điều chỉnh đường dẫn SAMPLE_DATA_FILE cho phù hợp.
    read_csv_and_send_to_kafka(SAMPLE_DATA_FILE, KAFKA_BROKER, NEWS_ARTICLES_TOPIC)
