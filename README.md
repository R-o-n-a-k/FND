# Fake News Detection
<p align="center">
  <img src="https://github.com/R-o-n-a-k/R-o-n-a-k/blob/cc804e68b237228c7a54aeb0268759ff28998680/assets/FND.gif?raw=true" alt="Fake News Detection" width="100%" />
</p>

## 🚀 About Project

**Fake News Detection App** is a machine learning-powered web application that determines whether a news article is real or fake. Built using Python Flask for the backend and a UI layer, it leverages both **PassiveAggressiveClassifier** and **Naive Bayes** algorithms for classification. The model is trained using a Kaggle-sourced dataset and integrated into a minimal, responsive interface for ease of use and rapid testing.

## 🌐 Technologies Used

-  **ML Models:** Passive Aggressive Classifier, Naive Bayes
-  **Backend:** Python, Flask
-  **ML Tools:** Scikit-learn, Pandas, NumPy
-  **Dataset:** [Kaggle Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
-  **Frontend/UI:** HTML, CSS (basic Flask templates)
-  **Deployment:** Render

## ✨ Features

- 🧠 ML-based fake vs real news classification
- 🔎 Real-time prediction on custom input
- 📊 Trained on large dataset from Kaggle
- 💡 Model with highest accuracy gets connected (Naive Bayes / Passive Aggressive)
- ⚙️ Flask-powered backend with Python model integration
- 📱 Minimal UI for testing predictions

##  ⚙️ Setup Instructions

To run this project locally:

1. **Clone the repository:**
```
git clone https://github.com/R-o-n-a-k/FakeNewsDetector.git

cd FakeNewsDetector
```

2. **Install dependencies:**

```
pip install -r requirements.txt
```

3. **Run the flask app:**

```
python app.py
```
