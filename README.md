**Task 05 - Food Item Recognition and Calorie Estimation Model Using CNN**

**Dataset:** [[Food-101 Dataset](https://www.kaggle.com/datasets/dansbecker/food-101)](https://www.kaggle.com/datasets/dansbecker/food-101)

The Python files `train.py`, `cnnmodel.py`, and `calorie.py` perform the tasks involved in building a food recognition model using Convolutional Neural Networks (CNN) and estimating the calorie content of recognized food items.

- Project Overview
This project focuses on developing a model that can accurately recognize food items from images and estimate their calorie content. The system allows users to track their dietary intake and make informed food choices.

- Key Steps in the Project

1. Data Loading:
   The dataset, containing images of 101 types of food items, was loaded from the provided Kaggle link. For this project, I focused on recognizing 7 food items for simplicity, with each item having up to 100 images.

2. Data Preprocessing:  
   The images were resized to 400x400 pixels for uniformity, and data was split into training and testing sets. Labels were assigned to each food category, and the data was shuffled to avoid bias during training.

3. Model Architecture:
   A CNN model was built using TFLearn with the following layers:
   - **Convolutional layers**: For feature extraction from the images.
   - **MaxPooling layers**: For downsampling and reducing dimensionality.
   - **Dropout layers**: For regularization to prevent overfitting.
   - **Fully Connected layers**: For final classification into one of the food categories.
   The network was trained using the Adam optimizer and categorical cross-entropy as the loss function.

4. Model Training: 
   The model was trained for 10 epochs using a subset of the dataset with a learning rate of 0.001. During training, a validation set was used to track the model's performance. The final model was saved for future use.

5. Calorie Estimation:  
   Using the `calorie.py` script, after recognizing the food item, the system estimates its calorie content based on pre-defined density and calorie values for each food type. The calorie calculation is performed by estimating the food item's volume and mass from the image and applying the corresponding calorie density.

6. Prediction Functionality: 
   After training, the model can take an input image of a food item and return the predicted class (type of food). The `calorie.py` script then calculates the estimated calorie content of the recognized food item.

- Model Performance 
The model's performance was evaluated on a test set, and accuracy metrics were calculated. Further improvements can be made by increasing the dataset size or tuning the model architecture.

- Saving the Model
The trained model was saved in a format that allows for reuse in future projects. The saved model can be loaded to make predictions on new images of food items.

- Visualization  
You can visualize the model's performance by plotting metrics like accuracy and loss over epochs during training. A confusion matrix can also be generated to observe which food items were misclassified. 

