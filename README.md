# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
![1](https://github.com/user-attachments/assets/a010e694-7771-45be-aacc-27d113f9ad2b)




## DESIGN STEPS

## Import Libraries:
Import necessary libraries like pandas, numpy, etc.

## Load and Explore Data:
Load the dataset using appropriate functions. Explore the data to understand its structure, content, and missing values.

## Preprocess and Clean Data:
Handle missing values (e.g., imputation, deletion). Deal with outliers and inconsistencies. Perform feature scaling or normalization if necessary.

## Feature Engineering:
Encode categorical features (e.g., one-hot encoding, label encoding). Create new features from existing ones if relevant.

## Exploratory Data Analysis (EDA):
Visualize data distribution and relationships using various plots (e.g., histograms, scatter plots). Gain insights into data patterns and trends.

## Split Data into Training and Testing Sets:
Split the data into training and testing sets for model development and evaluation.

## Build Deep Learning Model:
Design the model architecture with appropriate layers (e.g., dense, convolutional) and activation functions. Compile the model with an optimizer and loss function.

## Train the Model:
Train the model on the training set for a specified number of epochs. Monitor training progress and adjust hyperparameters (e.g., learning rate, batch size) if needed.

## Evaluate Model Performance:
Evaluate the model's performance on the testing set using various metrics (e.g., accuracy, precision, recall). Analyze the results to assess the model's effectiveness and identify potential areas for improvement.

## Visualize Training and Validation Performance:
Plot learning curves to visualize the model's training and validation loss and accuracy over time. Gain insights into model convergence and potential overfitting/underfitting issues.

## Save the Model:
Save the trained model using serialization techniques like pickle for future use or deployment.

## Make Predictions:


## PROGRAM

### Name: Thirugnanamoorthi G
### Register Number:212221230117

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pylab as plt

customer_df = pd.read_csv('customers.csv')


customer_df_cleaned = customer_df.dropna(axis=0)

categories_list = [['Male', 'Female'],
                   ['No', 'Yes'],
                   ['No', 'Yes'],
                   ['Healthcare', 'Engineer', 'Lawyer', 'Artist', 'Doctor',
                    'Homemaker', 'Entertainment', 'Marketing', 'Executive'],
                   ['Low', 'Average', 'High']]

enc = OrdinalEncoder(categories=categories_list)
customers_1 = customer_df_cleaned.copy()
customers_1[['Gender',
             'Ever_Married',
             'Graduated', 'Profession',
             'Spending_Score']] = enc.fit_transform(customers_1[['Gender',
                                                                   'Ever_Married',
                                                                   'Graduated', 'Profession',
                                                                   'Spending_Score']])
le = LabelEncoder()
customers_1['Segmentation'] = le.fit_transform(customers_1['Segmentation'])

customers_1 = customers_1.drop('ID', axis=1)
customers_1 = customers_1.drop('Var_1', axis=1)

# Splitting the dataset into features and target variable
X = customers_1[['Gender', 'Ever_Married', 'Age', 'Graduated', 'Profession', 'Work_Experience', 'Spending_Score', 'Family_Size']].values
y1 = customers_1[['Segmentation']].values


one_hot_enc = OneHotEncoder()
one_hot_enc.fit(y1)
y = one_hot_enc.transform(y1).toarray()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=50)


scaler_age = MinMaxScaler()
scaler_age.fit(X_train[:, 2].reshape(-1, 1))
X_train_scaled = np.copy(X_train)
X_test_scaled = np.copy(X_test)
X_train_scaled[:, 2] = scaler_age.transform(X_train[:, 2].reshape(-1, 1)).reshape(-1)
X_test_scaled[:, 2] = scaler_age.transform(X_test[:, 2].reshape(-1, 1)).reshape(-1)


updated_model = Sequential([
    Dense(10, input_shape=(8,), activation='relu'),
    Dense(16, activation='relu'),
    Dense(24, activation='relu'),
    Dense(4, activation='softmax')
])


updated_model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])


history = updated_model.fit(x=X_train_scaled, y=y_train,
                            epochs=200, batch_size=256,
                            validation_data=(X_test_scaled, y_test))


plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss vs Validation Loss')
plt.legend()
plt.show()

y_pred = updated_model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)


print(confusion_matrix(y_true, y_pred_classes))
print(classification_report(y_true, y_pred_classes))


updated_model.save('customer_classification_model_updated.h5')
with open('customer_data_updated.pickle', 'wb') as fh:
    pickle.dump([X_train_scaled, y_train, X_test_scaled, y_test, customers_1, customer_df_cleaned, scaler_age, enc, one_hot_enc, le], fh)


updated_model_loaded = load_model('customer_classification_model_updated.h5')
with open('customer_data_updated.pickle', 'rb') as fh:
    [X_train_scaled, y_train, X_test_scaled, y_test, customers_1, customer_df_cleaned, scaler_age, enc, one_hot_enc, le] = pickle.load(fh)

x_single_prediction = np.argmax(updated_model_loaded.predict(X_test_scaled[1:2, :]), axis=1)
print(x_single_prediction)
print(le.inverse_transform(x_single_prediction))

```
## Dataset Information


![2](https://github.com/user-attachments/assets/0a3db15a-a40d-4e45-995f-02aca2800008)


## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
![3](https://github.com/user-attachments/assets/d2c6e68e-44ac-4dff-b915-e1e8d7f757dc)



### Classification Report
![4](https://github.com/user-attachments/assets/d2650bf6-4649-412d-b6fa-d5b6b2a35c6c)


### Confusion Matrix
![5](https://github.com/user-attachments/assets/786b4f87-bea8-406b-8ce4-0439da953a53)


### New Sample Data Prediction
![6](https://github.com/user-attachments/assets/c8eb8c3d-57c5-41e6-a903-47864bde5f60)



## RESULT
A neural network classification model is developed for the given dataset.
