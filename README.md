# Skin Cancer Classifier (HAM10000)

This project uses a CNN trained on the HAM10000 dataset to classify skin lesion images into 7 categories, including melanoma and other cancerous types.  
It demonstrates data preprocessing, model training, evaluation, and prediction visualization. The trained model can assist in detecting potential skin cancer from dermoscopic images.  

## Usage
1. Download the dataset via [KaggleHub](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000).
2. Run the script to train the CNN.
3. The trained model is saved as `HAMCancer.keras`.

## Requirements
- TensorFlow / Keras  
- Pandas, NumPy, Matplotlib, Seaborn  
- scikit-learn  
- kagglehub  

## Output
- Accuracy and loss plots  
- Test accuracy evaluation  
- Example cancer prediction visualization  
