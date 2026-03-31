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
Since the model is already trained, you **do not** need to retrain it to test a new song!

1. **Download the Model:** Click here to download the pre-trained `instrument_model.keras` file from [Google Drive]((https://drive.google.com/drive/folders/1gFpMn0pdYeu9Eq2Tuv72Bxqb0T5voENF)). 
2. **Move the file:** Place that downloaded `.keras` file in the exact same folder as the Jupyter Notebook on your computer.
3. **Install dependencies:** Open your terminal and run:
   `pip install tensorflow librosa numpy scikit-learn`
4. **Run the Notebook:**
   * Run the first setup cell.
   * Run the "Load Model" cell to load the `.keras` and `.npy` files.
   * Run the final testing cell on your `.wav` file to see the timestamped output!
