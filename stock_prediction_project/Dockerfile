# Sử dụng cùng base image với các service Spark khác
FROM bitnami/spark:3.4.1

# Thiết lập thư mục làm việc bên trong container
WORKDIR /app

# Sao chép file requirements.txt và cài đặt các thư viện Python
# Nên làm bước này trước khi copy code để tận dụng cache của Docker layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn ứng dụng
# Đảm bảo các lệnh COPY này nằm sau khi bạn đã sửa đổi file trên máy host
COPY main.py .
COPY src ./src  

# Bạn có thể tùy chọn copy thêm thư mục data hoặc models nếu muốn chúng có sẵn trong image
# Tuy nhiên, việc sử dụng volumes trong docker-compose là cách tốt hơn để quản lý dữ liệu/mô hình
# COPY data ./data
# COPY models ./models

# Có thể thiết lập Entrypoint hoặc Command mặc định (sẽ bị ghi đè bởi docker-compose command)
# ENTRYPOINT ["spark-submit"]
# CMD ["/app/main.py", "predict"] # Hoặc "train"

