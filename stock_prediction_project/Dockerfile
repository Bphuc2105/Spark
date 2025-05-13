# Dockerfile

# Sử dụng một base image Python chính thức. Chọn phiên bản Python phù hợp với project của bạn.
FROM python:3.11-slim

# Ngăn chặn các câu hỏi tương tác trong quá trình cài đặt gói
ENV DEBIAN_FRONTEND=noninteractive

# Các biến môi trường cho Spark và Hadoop
ENV SPARK_VERSION=3.4.1 
ENV HADOOP_VERSION=3    
ENV SPARK_HOME=/opt/spark
ENV PATH="${SPARK_HOME}/bin:${SPARK_HOME}/sbin:${PATH}"
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 

# Cài đặt các gói cần thiết:
# Cập nhật danh sách gói, sau đó cài đặt Java, wget, gnupg, tini, và procps.
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless \
    wget \
    gnupg \
    tini \
    procps && \
# Dọn dẹp cache của apt để giảm kích thước image
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Tải và cài đặt Apache Spark
# Lấy Spark từ archive của Apache.
RUN wget -qO- "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" | tar xz -C /opt/ && \
    mv "/opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}" "${SPARK_HOME}" && \
    chown -R root:root "${SPARK_HOME}" # Đảm bảo quyền sở hữu đúng

# Tạo thư mục làm việc cho ứng dụng bên trong container
WORKDIR /app

# Sao chép tệp requirements.txt trước để tận dụng Docker layer caching.
# Nếu tệp này không đổi, Docker sẽ không cần chạy lại bước pip install.
COPY requirements.txt .

# Cài đặt các thư viện Python từ requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn của project (bao gồm main.py, thư mục src/, etc.)
# vào thư mục làm việc /app trong container.
COPY . .

# (Tùy chọn) Thiết lập entrypoint để sử dụng tini làm init process.
# Tini giúp quản lý các process con và xử lý tín hiệu đúng cách.
# ENTRYPOINT ["/usr/bin/tini", "--"]

# (Tùy chọn) Thiết lập CMD mặc định.
# Lệnh này sẽ bị ghi đè bởi `command:` trong docker-compose.yml cho service spark-app.
# CMD ["python", "main.py", "train"]
