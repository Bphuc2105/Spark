# Cần cài đặt các thư viện: pip install supabase pandas
import os
from supabase import create_client, Client
import pandas as pd

def supabase_table_to_csv(supabase_url: str, supabase_key: str, table_name: str, output_csv_path: str):
    """
    Lấy dữ liệu từ một bảng trong Supabase và lưu vào tệp CSV.

    Args:
        supabase_url (str): URL của Supabase project của bạn.
        supabase_key (str): Service key (hoặc anon key nếu chỉ đọc dữ liệu public) của Supabase project.
                            Nên sử dụng service key nếu cần truy cập dữ liệu được bảo vệ.
        table_name (str): Tên của bảng bạn muốn lấy dữ liệu.
        output_csv_path (str): Đường dẫn để lưu tệp CSV đầu ra.
    """
    try:
        # 1. Kết nối tới Supabase
        print(f"Đang kết nối tới Supabase tại URL: {supabase_url[:30]}...") # Chỉ log một phần URL để bảo mật
        supabase: Client = create_client(supabase_url, supabase_key)
        print("Kết nối Supabase thành công.")

        # 2. Lấy dữ liệu từ bảng
        print(f"Đang lấy dữ liệu từ bảng '{table_name}'...")
        all_data = []
        offset = 0
        limit = 1000 # Số lượng dòng lấy mỗi lần, có thể điều chỉnh
        
        while True:
            response = supabase.table(table_name).select("*").range(offset, offset + limit - 1).execute()
            if response.data:
                all_data.extend(response.data)
                if len(response.data) < limit:
                    break # Đã lấy hết dữ liệu
                offset += limit
                print(f"Đã lấy {len(all_data)} dòng...")
            else:
                # Kiểm tra lỗi nếu có
                if hasattr(response, 'error') and response.error:
                    print(f"Lỗi khi lấy dữ liệu: {response.error}")
                    return
                break # Không còn dữ liệu hoặc có lỗi không xác định

        if not all_data:
            print(f"Không tìm thấy dữ liệu nào trong bảng '{table_name}'.")
            return

        print(f"Lấy dữ liệu thành công. Tổng số {len(all_data)} dòng.")

        # 3. Chuyển đổi dữ liệu sang Pandas DataFrame
        print("Đang chuyển đổi dữ liệu sang DataFrame...")
        df = pd.DataFrame(all_data)
        print("Chuyển đổi DataFrame thành công.")
        print("5 dòng dữ liệu đầu tiên:")
        print(df.head())

        # 4. Lưu DataFrame thành tệp CSV
        print(f"Đang lưu DataFrame vào tệp CSV: {output_csv_path}")
        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig') # index=False để không ghi chỉ số của DataFrame vào CSV
                                                                    # encoding='utf-8-sig' để hỗ trợ tiếng Việt tốt hơn khi mở bằng Excel
        print(f"Dữ liệu đã được lưu thành công vào '{output_csv_path}'.")

    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")

if __name__ == "__main__":
    # --- Cấu hình cần thay đổi ---
    SUPABASE_URL = os.environ.get("") 
    SUPABASE_KEY = os.environ.get("") 

    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Vui lòng thiết lập biến môi trường SUPABASE_URL và SUPABASE_KEY,")
        print("hoặc nhập trực tiếp vào trong code (không khuyến khích cho production).")
        # Ví dụ:
        SUPABASE_URL = "https://fmjjbdnghgdamhcoliun.supabase.co"
        SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZtampiZG5naGdkYW1oY29saXVuIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0MTc0ODAyNiwiZXhwIjoyMDU3MzI0MDI2fQ.Boqfsg63piR-BE-ilSSqK505C3Cf6QuF21tMH8YhBTk" # Hoặc anon key
        # if not SUPABASE_URL or not SUPABASE_KEY:
        #     exit("Thông tin Supabase URL và Key là bắt buộc.")

    TABLE_TO_EXPORT = "article_data"  # Thay bằng tên bảng bạn muốn xuất
    # Thay đổi tên tệp đầu ra thành 'articles.csv'
    OUTPUT_FILE_PATH = "articles.csv" 

    if SUPABASE_URL and SUPABASE_KEY and TABLE_TO_EXPORT:
        print(f"Sẽ xuất dữ liệu từ bảng '{TABLE_TO_EXPORT}' ra tệp '{OUTPUT_FILE_PATH}'.")
        supabase_table_to_csv(SUPABASE_URL, SUPABASE_KEY, TABLE_TO_EXPORT, OUTPUT_FILE_PATH)
    else:
        print("Thiếu thông tin SUPABASE_URL, SUPABASE_KEY hoặc TABLE_TO_EXPORT.")
        print("Vui lòng cung cấp đầy đủ thông tin và thử lại.")

