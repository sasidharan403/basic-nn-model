# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model
![WhatsApp Image 2023-08-16 at 10 05 10 PM](https://github.com/sasidharan403/basic-nn-model/assets/94154712/d861eb49-d756-4bf8-bc6b-418db7054714)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM

~~~
Developed by: A.Sasi Dharan
Reg no : 212221240049
~~~
~~~
### To Read CSV file from Google Drive :

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

### Authenticate User:

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

### Open the Google Sheet and convert into DataFrame :

worksheet = gc.open('data 1').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns = rows[0])
df = df.astype({'input':'int','output':'int'})
### Import the packages :
df.head()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


X = df[['input']].values
Y = df[['output']].values
X

### Split Training and testing set :

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.33,random_state = 42)

### Pre-processing the data :

Scaler = MinMaxScaler()
Scaler.fit(X_train)
Scaler.fit(X_test)

X_train1 = Scaler.transform(X_train)
X_test1 = Scaler.transform(X_test)
X_train1

### Model :

ai_brain = Sequential([
    Dense(5,activation = 'relu'),
    Dense(7,activation = 'relu'),
    Dense(1)])

ai_brain.compile(
    optimizer = 'rmsprop',
    loss = 'mse'
)

ai_brain.fit(X_train1,Y_train,epochs = 4000)

### Loss plot :

loss_df = pd.DataFrame(ai_brain.history.history)

loss_df.plot()
### Testing with the test data and predicting the output :

ai_brain.evaluate(X_test1,Y_test)

X_n1 = [[38]]

X_n1_1 = Scaler.transform(X_n1)

ai_brain.predict(X_n1_1)
~~~

## Dataset Information

![data1](https://github.com/sasidharan403/basic-nn-model/assets/94154712/a96a2f87-0a5c-4602-b02b-0b5e4fdc9769)


## OUTPUT

### Training Loss Vs Iteration Plot

![deep 1](https://github.com/sasidharan403/basic-nn-model/assets/94154712/4e2e7838-79d1-483c-940b-57a833ee3656)


### Test Data Root Mean Squared Error

![deep 1](https://github.com/sasidharan403/basic-nn-model/assets/94154712/38b31e6b-addf-4785-ab5f-9e292d8db0c3)

### New Sample Data Prediction

![deep](https://github.com/sasidharan403/basic-nn-model/assets/94154712/c81f9481-cc58-44ce-9014-01ee5ed65ad9)


## RESULT
Thus a neural network model for regression using the given dataset is written and executed successfully.
