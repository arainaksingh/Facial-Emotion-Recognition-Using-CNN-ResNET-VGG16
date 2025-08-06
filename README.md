Facial Emotion Recognition (FER) using CNN
This project implements a Facial Emotion Recognition system using Convolutional Neural Networks (CNN) to classify human emotions from facial images.
The model is trained on a dataset of 7 emotion classes:

😡 Angry
🤢 Disgust
😨 Fear
😀 Happy
😐 Neutral
😢 Sad
😲 Surprise

🚀 Features
Loads and preprocesses grayscale facial images (48x48)

CNN model with Conv2D, MaxPooling, and Dropout layers

Training and Validation Accuracy Visualization

Random Test Predictions with Original vs Predicted Labels

Supports Transfer Learning (can be extended to ResNet or VGG16)

📊 Model Performance
Training Accuracy: ~80%

📈 Sample Graphs
Accuracy Graph (Training vs Validation)
Loss Graph (Training vs Validation)

📦 Dependencies
Python 3.8+
TensorFlow / Keras
NumPy
Pandas
Matplotlib
Seaborn
scikit-learn

🔮 Future Improvements
Implement Transfer Learning with ResNet50 or VGG16

Compute Precision, Recall, and F1-score for each emotion

Deploy as a Web App using Flask or Streamlit
