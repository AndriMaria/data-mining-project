# 🏃 Human Activity Recognition — Data Mining
Python project applying machine learning and deep learning techniques to classify human physical activities from accelerometer sensor data, developed as part of the **Data Mining** course at the **University of Patras**.


## 📌 About
The goal of this project is to recognize human activities (e.g. walking, sitting, standing) using accelerometer data collected from sensors placed on the **back** and **thigh** of participants. Multiple ML models were trained and compared to find the best classifier for this task.
---

## 📁 Notebooks

| Notebook | Description |
|----------|-------------|
| `Neural_Networks.ipynb` | Deep Neural Network with TensorFlow (4 Dense layers, ~93% accuracy) |
| `3Classifiers.ipynb` | Comparison of Random Forest, MLP, and Naive Bayes classifiers |
| `K-means_Cluster.ipynb` | K-Means clustering of activity data |
| `DBSCAN_Cluster.ipynb` | DBSCAN density-based clustering |
| `projectDataMining.ipynb` | Main project notebook |
| `Object2-12.ipynb` | Per-subject analysis notebooks |


## 🛠️ Tech Stack

| | |
|---|---|
| Language | Python 3 |
| ML Library | scikit-learn |
| Deep Learning | TensorFlow / Keras |
| Data | pandas, numpy |
| Visualization | matplotlib |

---

## 🔬 Models & Results

### Classifiers (`3Classifiers.ipynb`)
Trained on accelerometer data with a **sliding window** approach (window size = 10):

| Model | Accuracy |
|-------|----------|
| Random Forest | **97%** |
| Neural Network (MLP) | 96% |
| Naive Bayes | 72% |

### Deep Learning (`Neural_Networks.ipynb`)
A 4-layer Dense Neural Network trained with TensorFlow:
- 20 epochs, batch size 32
- **Test Accuracy: ~93%**

### Clustering
- **K-Means** — grouping activities by movement patterns
- **DBSCAN** — density-based clustering to detect activity regions

---

## 📊 Dataset

The dataset contains accelerometer readings from sensors on the **back** and **thigh**:

| Feature | Description |
|---------|-------------|
| `back_x/y/z` | Back sensor acceleration (3 axes) |
| `thigh_x/y/z` | Thigh sensor acceleration (3 axes) |
| `label` | Activity class (1–8) |

Data was **normalized** via standardization and reduced using a **sliding window** technique before training.

---

## 🎓 What I Learned

- Preprocessing time-series sensor data (normalization, sliding window).
- Training and comparing multiple ML classifiers with scikit-learn.
- Building a deep learning model with TensorFlow/Keras.
- Evaluating models with confusion matrices and classification reports.
- Applying unsupervised learning (K-Means, DBSCAN) to activity data.

## ⚠️ Notes

- The DBSCAN notebook encountered memory limitations on large datasets.
- The dataset CSV files are not included in this repo.

---

## 📄 License

This project was created for academic purposes.
