# 📈 Stock Chart Pattern Recognition using Deep Learning

This project focuses on identifying classical technical analysis patterns from stock candlestick chart images to study market behavior and explore the feasibility of pattern-based prediction using deep learning.
Instead of using pre-labeled datasets, the entire pipeline — from data collection to image generation, labeling, and model training — is built from scratch using real market data.

# 🛠️ Tools & Libraries Used

yfinance – downloading historical stock market data
mplfinance – generating candlestick chart images
NumPy, Pandas – data processing
TensorFlow / Keras – deep learning & transfer learning
scikit-learn – evaluation metrics and class weighting

# 📊 Patterns Considered

Head and Shoulders

Double Top

Double Bottom

Ascending Triangle

Descending Triangle

No Pattern

# 📉 Stocks Used

The following Indian stocks were downloaded using yfinance:

RELIANCE.NS
HDFCBANK.NS
ICICIBANK.NS
ITC.NS
INFY.NS
TCS.NS
SBIN.NS
MARUTI.NS
HINDUNILVR.NS


All raw OHLC data was stored in the data/raw directory.

# 🧱 Dataset Generation Pipeline

Historical OHLC data for each stock was downloaded and stored as CSV files.

Each stock’s OHLC data was converted into 60-day candlestick chart images, with a sliding window step of 10 days.

This resulted in 6,468 chart images, all initially placed in the no_pattern folder under the training directory.

Rule-based logic using price peaks and troughs was implemented to detect classical technical patterns.

Based on these rules, images were classified into their respective pattern folders.

# 📂 Data Splitting

A 75/25 split was applied between training and validation datasets.

Due to real market behavior, the dataset exhibited significant class imbalance, with patterns like Head and Shoulders having far fewer samples.

(Insert image/table showing per-pattern image counts here)

# 🧠 Model Architecture

Transfer learning was used with MobileNetV2 as the backbone:

Pretrained on ImageNet for efficient feature extraction

Base layers were frozen initially to prevent overfitting

Global Average Pooling applied to reduce spatial dimensions

Batch Normalization for training stability

Dense layer with 128 neurons, ReLU activation

Dropout of 0.4 for regularization

Final output layer using sigmoid activation

(Insert model summary image here)

# ⚖️ Handling Class Imbalance

Since some patterns had significantly fewer samples than others, class weights were applied during training.
This ensured that minority classes contributed more strongly to the loss function and improved recall for underrepresented patterns.

# 📈 Training & Evaluation

The model was trained using class-weighted loss and evaluated using multiple metrics:

Training vs Validation Accuracy

Training vs Validation Loss

Confusion Matrix

Per-class Recall (preferred over raw accuracy due to imbalance)

(Insert final epoch metrics here)
(Insert accuracy & loss plots here)
(Insert confusion matrix here)

# 🔍 Key Observations & Learnings

One of the most important takeaways from this project is that real-world stock pattern prediction is extremely challenging:

Technical patterns are often ambiguous and overlapping

Class imbalance significantly impacts model generalization

High training accuracy does not necessarily translate to real-world performance

Achieving consistently high (>85%) validation accuracy on real market data is rare and often unrealistic

This project highlights the practical limitations of deep learning in financial pattern recognition, while emphasizing the importance of proper evaluation beyond simple accuracy metrics.

# 📌 Final Note

This project is intended as an exploratory and educational study rather than a trading system.
All scripts required to reproduce the dataset and experiments are provided, while large generated datasets are excluded due to size constraints.

