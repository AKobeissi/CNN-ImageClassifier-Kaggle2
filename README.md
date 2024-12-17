# Kaggle competition 2 

Team: Amine & Danil 

Students: Amine Kobeissi, Danil Garmaev

This repository contains a collection of machine learning models designed for classifying images using Convolutional Neural Networks (CNN) and other models. It includes code for training, testing, and evaluating image classifiers using several techniques, including Logistic Regression, Support Vector Machine (SVM), Random Forest, ResNet and U-Net. The custom ResNet provided the best solution, resulting in 3rd place in the Kaggle Competition.


## Overview

### Models:

- **ResNet (FinalSubmission.py)**: This model was used for the final Kaggle submission and performed well in the private leaderboard. It leverages the ResNet architecture for deep learning-based image classification.

- **Random Forest (RandomForest.py)**: This model used a Random Forest algorithm and outperformed the baseline model, providing a solid alternative to more complex deep learning models.

- **U-Net (UNET.ipynb)**: A Jupyter Notebook for performing image classification using the U-Net architecture. This notebook is meant to be run in Kaggle’s notebook environment.

- **Logistic Regression (logregAndSVM.py)**: A basic logistic regression classifier, used as a baseline for comparison with more advanced models. **Support Vector Machine **: Implements an SVM classifier, useful for linear and non-linear image classification tasks.

## Dataset

The dataset for this project is available on Kaggle and consists of retinal disease identification data. The files used for training and testing are:

- `train_data.pkl`: The training dataset.
- `test_data.pkl`: The test dataset.

The dataset is available under the path `/kaggle/input/ift3395-ift6390-identification-maladies-retine`.

## Dependencies

To run the models and notebooks, the following Python packages are required:

- numpy
- pandas
- scikit-learn
- tensorflow
- keras
- matplotlib
- seaborn
- opencv-python
- pillow
- jupyter

##  Instruction for Running the Models

1. Clone the repository into your Kaggle notebook.

2. Run the models by executing the relevant Python scripts or Jupyter notebook for segmentation tasks:

- To train the ResNet model:
 !python FinalSubmission.py or upload the file to Kaggle notebook and run with GPU.

- To train the Random Forest model:
 !python RandomForest.py or upload the file to Kaggle notebook and run.

- To train U-Net: 
Upload UNet.ipynb to Kaggle notebook and run with GPU.

- To train logistic regression and SVM:  
 !python logregAndSVM.py upload the file to Kaggle notebook and run.


3. Dataset Path: The dataset should be available in the Kaggle environment under the path /kaggle/input/ift3395-ift6390-identification-maladies-retine


## Results

The models are evaluated based on their performance on the test dataset. The final Kaggle submission uses the ResNet model as the private leaderboard model, while the Random Forest model has shown to beat the baseline performance.


