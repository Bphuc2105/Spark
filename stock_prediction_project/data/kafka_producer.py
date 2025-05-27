import time
import json
import pandas as pd # Sử dụng pandas
import os
from datetime import datetime, timezone
from kafka import KafkaProducer

# Cấu hình Kafka
KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "localhost:9093")
NEWS_ARTICLES_TOPIC = os.environ.get("NEWS_ARTICLES_TOPIC", "news_articles")

# --- THAY ĐỔI CÁCH XÁC ĐỊNH ĐƯỜNG DẪN FILE ---
# Lấy đường dẫn tuyệt đối của thư mục chứa script này (tức là thư mục 'data')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Các file CSV mẫu được mong đợi nằm cùng thư mục với script này
ARTICLES_DATA_FILE = os.path.join(SCRIPT_DIR, "new_articles.csv")
PRICES_DATA_FILE = os.path.join(SCRIPT_DIR, "new_prices.csv")
# ---------------------------------------------

def load_and_join_data(articles_file_path, prices_file_path):
    """
    Đọc dữ liệu từ file articles và prices, sau đó join chúng lại.
    """
    if not os.path.exists(articles_file_path):
        print(f"Lỗi: Không tìm thấy file bài báo tại {articles_file_path}")
        print(f"Đang tạo file mẫu: {articles_file_path}")
        sample_articles_content = """id,date,link,title,text,symbol
a1,2024-05-01,http://example.com/news/fpt1,"FPT công bố kế hoạch tăng trưởng mới","Tập đoàn FPT hôm nay đã đưa ra chiến lược phát triển cho giai đoạn tới, tập trung vào công nghệ AI và Cloud. Giá cổ phiếu FPT phản ứng tích cực.",FPT
a2,2024-05-01,http://example.com/news/vcb1,"VCB chia cổ tức cao kỷ lục","Ngân hàng Vietcombank thông báo mức chia cổ tức cao nhất từ trước đến nay, gây bất ngờ cho thị trường. Cổ phiếu VCB tăng trần.",VCB
a3,2024-05-02,http://example.com/news/fpt2,"FPT ký hợp đồng lớn với đối tác Nhật Bản","FPT Software vừa công bố hợp đồng trị giá hàng chục triệu USD với một tập đoàn lớn của Nhật. Điều này củng cố vị thế của FPT.",FPT
a4,2024-05-02,http://example.com/news/vcb2,"VCB mở rộng mạng lưới chi nhánh","Vietcombank tiếp tục mở thêm các chi nhánh mới tại các tỉnh thành trọng điểm, nhằm nâng cao dịch vụ khách hàng.",VCB
"""