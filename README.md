# 📊 Vietnamese Sentiment Analysis Dashboard

This project is a comprehensive toolkit for **Vietnamese Sentiment Analysis**, featuring a web-based dashboard built with Streamlit. It compares three different machine learning approaches to classify text into **Negative**, **Neutral**, or **Positive** sentiments.

## 🚀 Features

- **Multi-Model Support**: Compare predictions from three different architectures:
  - **PhoBERT (Deep Learning)**: State-of-the-art transformer model for Vietnamese.
  - **Logistic Regression**: Reliable baseline using TF-IDF vectorization.
  - **Linear SVM**: High-accuracy traditional machine learning model.
- **Real-time Analysis**: Interactive web interface to test your own Vietnamese comments.
- **Performance Evaluation**: Built-in comparison table showing Accuracy and F1-Macro scores.
- **Vietnamese NLP Preprocessing**: Uses `underthesea` for word segmentation and custom regex for cleaning (teencode, special characters).

## 🛠️ Tech Stack

- **Framework**: [Streamlit](https://streamlit.io/)
- **Model Provider**: [Hugging Face Transformers](https://huggingface.co/)
- **Language Model**: [PhoBERT](https://huggingface.co/vinai/phobert-base) (fine-tuned: [danghm/vietnamese_sentiments](https://huggingface.co/danghm/vietnamese_sentiments))
- **Machine Learning**: Scikit-learn
- **NLP Library**: [Underthesea](https://github.com/undertheseanlp/underthesea)
- **Deep Learning**: PyTorch

## 📊 Model Performance

| Metric       | PhoBERT               | Logistic Regression | Linear SVM       |
| :----------- | :-------------------- | :------------------ | :--------------- |
| **Accuracy** | 78.36%                | 78.18%              | **78.63%**       |
| **F1-Macro** | **0.6663**            | 0.6400              | 0.6200           |
| **Strength** | Context understanding | Balanced & Fast     | Overall Accuracy |

## 📁 Repository Structure

- `app.py`: The main Streamlit application.
- `vietnamese-sentiments.ipynb`: Training notebook and experimentation.
- `tfidf_logistic_model.pkl`: Pre-trained Logistic Regression model metadata.
- `svm_sentiment_model.pkl`: Pre-trained Linear SVM model metadata.
- `sentiment_model_v1/`: (Optional) Local storage for fine-tuned PhoBERT.
- `requirements.txt`: List of Python dependencies.

## ⚙️ Installation & Usage

1. **Clone the repository**:

   ```bash
   git clone <https://github.com/DangHM8/vietnamese_sentiments>
   cd btl
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Dashboard**:
   ```bash
   streamlit run app.py
   ```

## 📝 Dataset

The models were trained on the [anotherpolarbear/vietnamese-sentiment-analysis](https://huggingface.co/datasets/anotherpolarbear/vietnamese-sentiment-analysis) dataset, which consists of over 10,000 customer reviews labeled from 1 to 5 stars, mapped into 3 sentiment classes.

---

_Developed as part of the LLM course at HUST._
