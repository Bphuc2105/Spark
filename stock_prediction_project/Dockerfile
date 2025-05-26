# Sử dụng cùng base image với các service Spark khác
FROM bitnami/spark:3.4.1

# Thiết lập thư mục làm việc bên trong container
WORKDIR /app

# Cập nhật package list và cài đặt zip utility
# Chạy với quyền root (mặc định trong Dockerfile)
USER root
RUN apt-get update && \
    apt-get install -y zip && \
    rm -rf /var/lib/apt/lists/*

# Sao chép file __init__.py vào thư mục làm việc /app
# Điều này giúp Python nhận diện /app là một package root khi PYTHONPATH được thiết lập
COPY __init__.py .

# Sao chép file requirements.txt và cài đặt các thư viện Python
# Nên làm bước này trước khi copy code để tận dụng cache của Docker layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn ứng dụng
# Đảm bảo các lệnh COPY này nằm sau khi bạn đã sửa đổi file trên máy host
COPY main.py .
COPY src ./src

# Thêm /app vào PYTHONPATH để các import như "from src import ..." hoạt động đúng cách
# cho các script chạy từ /app (ví dụ: main.py)
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Bạn có thể tùy chọn copy thêm thư mục data hoặc models nếu muốn chúng có sẵn trong image
# Tuy nhiên, việc sử dụng volumes trong docker-compose là cách tốt hơn để quản lý dữ liệu/mô hình
# COPY data ./data
# COPY models ./models

# Có thể thiết lập Entrypoint hoặc Command mặc định (sẽ bị ghi đè bởi docker-compose command)
# ENTRYPOINT ["spark-submit"]
# CMD ["/app/main.py", "predict"] # Hoặc "train"