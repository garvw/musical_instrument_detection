# 🎵 Musical Instrument Detection using Deep Learning

This project uses a Convolutional Neural Network (CNN) to analyze audio files and detect which musical instruments are playing over time. It processes audio into Mel-spectrograms and uses deep learning to classify the instruments in 5-second block summaries.

## 🚀 Features
* Converts audio waveforms into 2D Mel-spectrograms using `librosa`.
* Trained on the IRMAS dataset to recognize multiple instruments (Piano, Guitar, Voice, Drums, Saxophone, etc.).
* Excludes specific background noise to focus on primary instruments.
* Outputs a clean, timestamped summary of which instrument is dominating every 5 seconds.

## 🛠️ Tech Stack
* **Python**
* **TensorFlow / Keras** (CNN Architecture)
* **Librosa** (Audio Processing)
* **NumPy & Scikit-learn** (Data Manipulation & Label Encoding)

## 📂 How to Run
Since the model is already trained and saved, you do not need to retrain it to test a new song!

1. **Clone the repository** and ensure your `.keras` and `.npy` files are in the main folder.
2. **Install dependencies:**
   `pip install tensorflow librosa numpy scikit-learn`
3. **Run the Notebook:**
   * Run the setup cell to initialize the spectrogram function.
   * Run the Load Model cell to load `instrument_model.keras` into memory.
   * Run the testing cell on your `.wav` file to see the timestamped output!
