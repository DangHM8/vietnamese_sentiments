import streamlit as st
import torch
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from underthesea import word_tokenize

# --- 1. CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="centered")

st.title("📊 Phân loại Sắc thái Cảm xúc")
st.markdown("---")

# --- 2. HÀM TẢI MÔ HÌNH (Sử dụng Cache) ---
@st.cache_resource
def load_all_models():
    # 1. Tải PhoBERT từ Hugging Face
    pb_tokenizer = AutoTokenizer.from_pretrained("danghm/vietnamese_sentiments")
    pb_model = AutoModelForSequenceClassification.from_pretrained("danghm/vietnamese_sentiments")
    
    # 2. Tải Logistic Regression và SỬA LỖI VERSION
    with open("tfidf_logistic_model.pkl", "rb") as f:
        log_data = pickle.load(f)
        # THÊM DÒNG NÀY ĐỂ SỬA LỖI AttributeError
        if not hasattr(log_data['classifier'], 'multi_class'):
            log_data['classifier'].multi_class = 'auto' 
        
    # 3. Tải Linear SVM và SỬA LỖI TƯƠNG TỰ (nếu có)
    with open("svm_sentiment_model.pkl", "rb") as f:
        svm_data = pickle.load(f)
        if not hasattr(svm_data['classifier'], 'multi_class'):
            svm_data['classifier'].multi_class = 'auto'
            
    return pb_tokenizer, pb_model, log_data, svm_data

# Thực hiện tải mô hình
try:
    pb_tokenizer, pb_model, log_data, svm_data = load_all_models()
    label_map = {0: "Tiêu cực 😡", 1: "Trung tính 😐", 2: "Tích cực 😍"}
except Exception as e:
    st.error(f"⚠️ Lỗi nạp file: {e}. Đảm bảo các file .pkl nằm cùng cấp với app.py và có kết nối internet để tải PhoBERT.")
    st.stop()

# --- 3. PHẦN LỰA CHỌN MÔ HÌNH (Hiển thị ngay tại màn hình chính) ---
st.subheader("1. Cài đặt cấu hình")
model_choice = st.selectbox(
    "Chọn thuật toán bạn muốn sử dụng để dự đoán:",
    ("PhoBERT (Deep Learning)", "Logistic Regression (TF-IDF)", "Linear SVM (TF-IDF)")
)

# Hiển thị ghi chú nhanh về hiệu năng thực tế của mô hình đã chọn
if model_choice == "PhoBERT (Deep Learning)":
    st.success("✨ **PhoBERT:** Hiểu ngữ cảnh tốt nhất. F1-Macro đạt **0.6663** tại Epoch 2.")
elif model_choice == "Logistic Regression (TF-IDF)":
    st.info("📈 **Logistic:** Cân bằng tốt. Accuracy **78.18%** và F1-Macro **0.64**.")
else:
    st.warning("⚖️ **SVM:** Accuracy cao nhất (**78.63%**) nhưng nhận diện lớp Trung tính kém (Recall **0.23**).")

st.markdown("---")

# --- 4. KHU VỰC NHẬP DỮ LIỆU & DỰ ĐOÁN ---
st.subheader("2. Nhập nội dung")
user_input = st.text_area("Nhập bình luận khách hàng tại đây:", height=100, placeholder="Ví dụ: Shop phục vụ rất tốt, mình sẽ ủng hộ tiếp...")

if st.button("🔍 Phân tích cảm xúc"):
    if not user_input.strip():
        st.error("Vui lòng không để trống ô nhập liệu!")
    else:
        # Tiền xử lý chung (Tách từ tiếng Việt)
        text_segmented = word_tokenize(user_input, format="text")
        
        # Biến trung gian để hiển thị kết quả
        final_label = ""
        final_conf = None

        # Logic dự đoán theo từng mô hình
        if model_choice == "PhoBERT (Deep Learning)":
            inputs = pb_tokenizer(text_segmented, return_tensors="pt", padding=True, truncation=True, max_length=128)
            with torch.no_grad():
                outputs = pb_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]
                idx = np.argmax(probs)
                final_label = label_map[idx]
                final_conf = probs[idx]

        elif model_choice == "Logistic Regression (TF-IDF)":
            tfidf_vec = log_data['vectorizer']
            clf = log_data['classifier']
            X_tfidf = tfidf_vec.transform([text_segmented])
            idx = clf.predict(X_tfidf)[0]
            final_label = label_map[idx]
            final_conf = clf.predict_proba(X_tfidf).max()

        else: # Linear SVM
            tfidf_vec = svm_data['vectorizer']
            clf = svm_data['classifier']
            X_tfidf = tfidf_vec.transform([text_segmented])
            idx = clf.predict(X_tfidf)[0]
            final_label = label_map[idx]
            # SVM không hỗ trợ độ tin cậy mặc định

        # --- 5. HIỂN THỊ KẾT QUẢ ---
        st.markdown("### Kết quả phân tích:")
        st.write(f"Mô hình đang dùng: **{model_choice}**")
        
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("Nhãn dự đoán", final_label)
        if final_conf:
            res_col2.metric("Độ tin cậy", f"{final_conf:.2%}")
        else:
            res_col2.write("**Độ tin cậy:** (Không hỗ trợ trên Linear SVM)")

# --- 6. PHẦN THỐNG KÊ (Dưới cùng) ---
with st.expander("📊 Xem bảng đối chiếu thông số thực tế từ quá trình huấn luyện"):
    st.write("Bảng số liệu dựa trên kết quả kiểm thử trên tập Validation:")
    st.table({
        "Tiêu chí": ["Accuracy (Độ chính xác)", "F1-Macro (Độ cân bằng)", "Ưu điểm"],
        "PhoBERT": ["78.36%", "0.6663", "Hiểu ngữ cảnh sâu"],
        "Logistic": ["78.18%", "0.6400", "Ổn định, tốc độ nhanh"],
        "Linear SVM": ["78.63%", "0.6200", "Accuracy tổng thể cao"]
    })