# Real-Time Gender Recognition from Audio Using Convolutional Neural Networks in MATLAB

## Overview

This repository contains a MATLAB-based implementation of a real-time gender recognition system that uses spectrograms of audio data and Convolutional Neural Networks (CNN). It is an interesting application of deep learning techniques on audio data.

<img width="566" alt="confusionMatrix" src="https://github.com/farukaydogan/Gender-Detection-On-Realtime-Voice/assets/57232389/15cb6512-0421-4a06-b503-877d1351c4b5">


## Prerequisites

- MATLAB
- MATLAB's Deep Learning Toolbox

## Implementation Details

The system operates by first converting real-time audio data into a spectrogram. The spectrogram is then processed by the CNN, which has been trained to recognize gender-based characteristics within the spectrogram, thereby identifying the gender of the speaker in real-time.

The CNN model was trained on a large dataset of labeled audio data. Different genders have distinct characteristics that are reflected in their spectrograms, and our CNN model has been taught to identify these with a high degree of accuracy.

<img width="1797" alt="Accurracy" src="https://github.com/farukaydogan/Gender-Detection-On-Realtime-Voice/assets/57232389/978a7788-4dcf-4971-990d-180d10e051a0">
## Getting Started

### Installation

1. Clone the repository: git clone https://github.com/farukaydogan/Gender-Detection-On-Realtime-Voice
2. Navigate to the repository: cd Gender-Detection-On-Realtime-Voice

3. Open MATLAB and run the main file.


### [Dataset](https://drive.google.com/)
### Usage

After running the main file, the system will start processing real-time audio data from your computer's default microphone. The gender of the speaker will be outputted in real-time.

## Results and Future Work

The system has shown impressive accuracy rates during preliminary tests. However, as with all machine learning projects, there is always room for improvement. Future efforts will be directed towards refining the model to increase its accuracy and exploring its potential in other applications.

## Contributing

Contributions are always welcome! Please read the [contribution guidelines](CONTRIBUTING.md) first.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any queries or discussions, feel free to open an issue.

