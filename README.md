 CIFAR-10 Image Classification using CNN

This project demonstrates how to build and train a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset using TensorFlow and Keras. The goal is to accurately predict one of 10 classes for each image in the dataset. The model is trained on labeled data and evaluated for accuracy using test data.



1.Dataset: CIFAR-10

The CIFAR-10 dataset is a widely used benchmark dataset in computer vision. It consists of:

- 60,000 color images (32x32 pixels)
- 10 classes:  
  `airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`
- Split:  
  - 50,000 images for training  
  - 10,000 images for testing

Images are evenly distributed across the 10 classes.


 2.Technologies Used

This project is built with the following tools and libraries:

- Python – Programming Language  
- TensorFlow/Keras – Deep Learning Framework  
- NumPy – Numerical Operations  
- Matplotlib – Data Visualization  
- Seaborn – Heatmap for Confusion Matrix  
- Google Colab – Cloud environment for training models efficiently

---

3.Model Architecture

The CNN architecture used in this project is a sequential model consisting of the following layers:

1. Conv2D Layer – 32 filters of size (3x3), activation: ReLU  
2. MaxPooling2D – pool size (2x2)  
3. Dropout – 0.2 (to prevent overfitting)  
4. Conv2D Layer – 64 filters of size (3x3), activation: ReLU  
5. MaxPooling2D – pool size (2x2)  
6. Dropout – 0.3  
7. Conv2D Layer – 64 filters, (3x3), activation: ReLU  
8. Flatten Layer – Converts 2D feature maps to 1D  
9. Dense Layer – 64 units, activation: ReLU  
10. Dropout – 0.4  
11. Output Layer – Dense(10) with Softmax activation (for multiclass classification)

4.Training Settings
- Loss Function: Sparse Categorical Crossentropy  
- Optimizer: Adam  
- Metrics: Accuracy  
- Epochs: Customizable (usually between 20-50)  
- Batch Size: 64 (can vary)



5.Results

- Final Test Accuracy: ~70%  
- Visualizations:  
  - Training vs Validation Accuracy  
  - Confusion Matrix  
  - Sample Predictions with Actual Labels

These results show the model can generalize reasonably well to unseen data in the CIFAR-10 dataset.



6.How to Run This Project

1. Open the notebook in [Google Colab](https://colab.research.google.com/)  
2. Upload the notebook if working locally  
3. Run all cells:
   - Dataset loading
   - Normalization
   - Model building
   - Training and evaluation
4. Observe the metrics and visual outputs



7.Key Learnings

- Understanding how CNNs work on image data
- How to load and preprocess image datasets
- Building and fine-tuning deep learning models
- Preventing overfitting using dropout layers
- Evaluating model performance using confusion matrices and predictions



8.License

This project is open-source and available under the **MIT License**.


 
9.Acknowledgments

- CIFAR-10 dataset by University of Toronto 
- Keras Documentation for reference  
- Google Colab for providing free GPU resources  
- TensorFlow for model building and training tools

---

## 🔗 Let's Connect

If you're interested in deep learning, feel free to fork this repo, raise issues, or connect for collaboration!
