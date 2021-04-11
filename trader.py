import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from statistics import mean
from numpy import newaxis
import csv
import argparse


# Arguments
epochs = 100
batch_size = 32
past_day = 5
future_day = 1


# 以收盤價為train, 以開盤價為target label
def split_dataset(df, past_day, future_day):
  X, Y = [], []
  for i in range(len(df) - future_day - past_day):
    X.append(np.array(df[i:i+past_day, 0]))
    Y.append(np.array(df[i+past_day:i+past_day+future_day, 0]))
  return np.array(X), np.array(Y)


def build_model(shape):
  model = Sequential()
  model.add(LSTM(64, input_shape=(shape[1], shape[2]), return_sequences=True))  
  model.add(Dropout(0.2))
  model.add(LSTM(64, return_sequences=True))
  model.add(Dropout(0.2))
  model.add(Dense(1))
  return model 

def plotting(input1, input2, title, legend, x_label=None, y_label=None, grid=True, figsize=(20, 8)):
    plt.figure(figsize=figsize)
    plt.plot(input1)
    plt.plot(input2)
    plt.title(title)
    plt.legend(legend)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(grid)
    plt.show()

def calculate_revenue(size, nu):
    #  1: buy
    #  0: hold
    # -1: short/sell
    
    # size = x_test.shape[0] - 1
    status = 0
    flag = 0
    revenue = 0
    with open('output.csv', mode='w') as csv_file:
        writer = csv.writer(csv_file)
        for i in range(size):
            if (status == 1):
                if (nu[i+1]<nu[i]):
                    writer.writerow(['-1'])
                    status = 0
                    revenue = revenue+nu[i]
                else:
                    writer.writerow(['0'])
            elif (status == 0):
                if (nu[i+1]>nu[i]):
                    writer.writerow(['1'])
                    status = 1
                    revenue = revenue-nu[i]
                elif (nu[i+1]<nu[i]):
                    writer.writerow(['-1'])
                    status = -1
                    revenue = revenue+nu[i]
                else: 
                    writer.writerow(['0'])
            else :
                if (nu[i+1]>nu[i]):
                    writer.writerow(['1'])
                    status = 0
                    revenue = revenue-nu[i]
                else:
                    writer.writerow(['0'])
    if (status==1) :
        revenue = revenue + nu[size]
    elif (status==-1) :
        revenue = revenue - nu[size] 

    return revenue

if __name__ == "__main__":    
    
    # Main Arguments
    main_path = '';

    parser = argparse.ArgumentParser()
    parser.add_argument("--training", default="dataset/training.csv", help="input training data file name")
    parser.add_argument("--testing", default="dataset/testing.csv", help="input testing data file name")
    parser.add_argument("--output", default="output.csv", help="output file name")
    args = parser.parse_args()

    train_df = pd.read_csv(args.training, header=None)
    test_df = pd.read_csv(args.testing, header=None)

    train_df.drop([1,2,3], inplace=True, axis=1)
    test_df.drop([1,2,3], inplace=True, axis=1)

    test_df = pd.DataFrame(np.insert(test_df.to_numpy(), 0, train_df.to_numpy()[-(past_day+1):], axis=0))
    train_df = pd.DataFrame(train_df.to_numpy()[:-(past_day+1)])

    # Scaling
    sc = MinMaxScaler(feature_range=(-1, 1))
    scaled_train_df = sc.fit_transform(train_df)
    scaled_test_df  = sc.transform(test_df)

    # Generate training data and label
    x_train, y_train = split_dataset(scaled_train_df, past_day, future_day)
    x_test, y_test = split_dataset(scaled_test_df, past_day, future_day)

    # Plotting Original Open and Close Price of testing.csv    
    plotting(x_test[:,-1], y_test, 'Price', ['close price','open price'])

    # Reshape the data into (Samples, Timestep, Features)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    # Build model
    model = build_model(x_train.shape)
    model.summary()

    # Compile and Fit
    reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
    early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1, mode='auto')
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,validation_data=(x_test, y_test), shuffle=False, callbacks=[reduce_lr, early_stopping])
    model.save('model.h5')

    # Plotting Model Loss
    plotting(input1=history.history['loss'], input2=history.history['val_loss'], title='Model Loss', legend=['Train','Valid'], x_label='Epochs', y_label='Loss')    

    # Start Predicting
    predicted = model.predict(x_test)
    predict = sc.inverse_transform(predicted.reshape(predicted.shape[0], predicted.shape[1]))

    # 將預測data前移三天，並利用預測結果預測剩餘最後三天的股票
    last = np.array([x_test[-1, 1:], x_test[-1, 2:], x_test[-1, 3:]], dtype=object)
    # print(last)
    # print(np.array(predicted[newaxis, -2, -1]))
    last[0] = np.array(np.concatenate((np.array(last[0]), np.array(predicted[newaxis, -3, -1]))))
    last[1] = np.concatenate((last[1], predicted[newaxis, -3, -1]))
    last[1] = np.array(np.concatenate((last[1], predicted[newaxis, -2, -1])))
    last[2] = np.concatenate((last[2], predicted[newaxis, -3, -1]))
    last[2] = np.concatenate((last[2], predicted[newaxis, -2, -1]))
    last[2] = np.array(np.concatenate((last[2], predicted[newaxis, -1, -1])))

    last[0] = pd.DataFrame(last[0])
    last[1] = pd.DataFrame(last[1])
    last[2] = pd.DataFrame(last[2])


    X = []
    X.append(np.array(last[0]))
    X.append(np.array(last[1]))
    X.append(np.array(last[2]))
    X = np.array(X)
    # print(X)

    predicted_last = model.predict(X)
    predicted_last = sc.inverse_transform(predicted_last.reshape(predicted_last.shape[0], predicted_last.shape[1]))

    nu = []
    for i in range(20):
        nu.append(predict[i, -1])
    nu = nu[3:]
    nu.append(predicted_last[0, -1])
    nu.append(predicted_last[1, -1])
    nu.append(predicted_last[2, -1])
    print(nu)

    ground_truth = sc.inverse_transform(y_test.reshape(-1,1))

    plotting(input1=ground_truth, input2=nu, title='Open Price', legend=['y_test','predict'])

    revenue = calculate_revenue(x_test.shape[0]-1, nu)
    print('===========================')
    print('Revenue is :', revenue)
    print('===========================')
    
