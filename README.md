# audio_ml

## Voice Recognizion
## Inspired by: Deep Learning for Audio by [Valerio Velardo](https://www.youtube.com/watch?v=fMqL5vckiU0&list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf)


```
    # Generates a RNN-LSTM Model
    input_shape = (X_train.shape[1], X_train.shape[2]) # 130, 13    130 slices of mfcc's     13 mfcc's per slice => 130*13 features.
    model = keras.Sequential()  

    # 2 LSTM layers
    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))    # 64 neurons  130X13   True passes the prediciton to th e next layer
    model.add(keras.layers.LSTM(64))                                                    # 64 neurons

    # dense layer
    model.add(keras.layers.Dense(64, activation='relu'))    # 64 neurons
    model.add(keras.layers.Dropout(0.3))


    # output layer
    model.add(keras.layers.Dense(5, activation='softmax'))  # five classification . 10 neurons
```
