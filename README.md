
# 🎶 Melody Extraction with Melodic Segnet 🎶

Welcome to the enhanced fork of "A Streamlined Encoder/Decoder Architecture for Melody Extraction"! Here, we've added some fantastic features to this already robust source code. Check out the original research [here](https://arxiv.org/abs/1810.12947).

## 🚀 Getting Started

These instructions will guide you to set up the project and run it on your machine for development, testing, and melody extraction fun!

### 📋 Prerequisites

Make sure you have the following packages installed:

- Python 3.8
- PyTorch 0.4.1
- NumPy
- SciPy
- PySoundFile
- Pandas

### 🎤 Melody Extraction

#### Predict on Audio

Want to extract melody from an audio file? It's super easy! The output will be a `.txt` file containing time (in seconds) and frequency (in Hz).

```bash
# Extract melody from an audio file
python predict_on_audio.py [-fp FILEPATH] [-t MODEL_TYPE] [-gpu GPU_INDEX] [-o OUTPUT_DIR] [-e EVALUATE] [-m MODE]
```

#### Evaluate Performance

Curious about how well it performs? Run the evaluation script on various datasets like ADC2004, MIREX05, MedleyDB. The results will be in a `.csv` file with evaluation metrics.

```bash
# Evaluate the model
python evaluate.py [-dd DATA_DIR] [-t MODEL_TYPE] [-gpu GPU_INDEX] [-o OUTPUT_DIR] [-ds DATASET]
```

#### Data Arrangement

Before training, let's arrange our data properly.

```bash
# Arrange data for training
python data_arrangement.py [-df DATA_FOLDER] [-t MODEL_TYPE] [-o OUTPUT_FOLDER]
```

#### Training Time! 🏋️

Ready to train the model? Make sure you've prepared the h5py file with `data_arrangement.py`.

```bash
# Start training
python training.py [-fp FILEPATH] [-t MODEL_TYPE] [-gpu GPU_INDEX] [-o OUTPUT_DIR] [-ep EPOCH_NUM] [-lr LEARN_RATE] [-bs BATCH_SIZE]
```

### 🆕 What's New in this Fork? 🌟

- **`data_arrangement_ground_truth.py`**: Now you can generate F0 pitch using the Crepe Pitch Detector! 🎵

To generate X and Y values for your dataset:

```bash
# Generate dataset
python data_arrangement.py -df MIR-1K/LyricsWav -t vocal -o data
```

To kickstart your model training with a specific batch size and number of epochs:

```bash
# Begin the training journey
python training.py -bs 4 -ep 10
```

### 🤝 Contributing

Got ideas to make this even better? Contributions are welcome!

### 📄 License


### 💡 Acknowledgments

---

Feel free to customize this template further to match the specific details and style of your project. Enjoy enhancing the world of melody extraction! 🎉🎼