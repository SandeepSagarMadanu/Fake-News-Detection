# 📰 Fake News Detection using Machine Learning

This project focuses on detecting fake news articles using Natural Language Processing (NLP) and supervised machine learning algorithms. The core goal is to classify news content as **real** or **fake** based on text data.

## 🧠 About the Project

In the age of digital media, the spread of misinformation has become a critical issue. This project utilizes a machine learning-based approach to detect fake news using a labeled dataset. By vectorizing the text using **TF-IDF** and training models like **Multinomial Naive Bayes** and **Passive Aggressive Classifier**, the model learns to differentiate between real and fake news articles effectively.

The solution is implemented in Python using Scikit-learn, with exploratory data analysis and performance evaluation.

## 📁 Dataset

The dataset (`data.csv`) consists of news articles labeled as "REAL" or "FAKE". It includes:

- `title`: Title of the news article
- `text`: The body content of the article
- `label`: Target label - REAL or FAKE

> **Note**: You must include the `data.csv` file in your working directory.

## ⚙️ Technologies Used

- Python
- Jupyter Notebook
- Pandas, NumPy
- Scikit-learn
- Matplotlib
- TF-IDF Vectorizer

## 🏗️ Machine Learning Models Used

- **TF-IDF Vectorizer** for converting text into numerical form.
- **Multinomial Naive Bayes** classifier
- **Passive Aggressive Classifier**

## 📊 Evaluation Metrics

- Accuracy
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/fake-news-detection.git
   cd fake-news-detection
   ```

2. Install required libraries (optional: use a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook Fake_news_detection.ipynb
   ```

4. Make sure `data.csv` is in the same directory as the notebook.

## ✅ Results

- Both classifiers demonstrate high accuracy in distinguishing fake from real news.
- Passive Aggressive Classifier showed slightly better performance in some runs.

## 📌 Future Improvements

- Deploy model as a web app using Flask or Streamlit
- Add LSTM/Transformer-based models for improved accuracy
- Handle out-of-vocabulary and multilingual news detection

## 👨‍💻 Author

- M. Sandeep Sagar
- https://github.com/SandeepSagarMadanu
