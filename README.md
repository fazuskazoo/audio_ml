# audio_ml

## Voice Recognizion
## Inspired by: Deep Learning for Audio by Valerio
```
# input layer
Each track have multiple MFCC vectors ... keras.layers.Flattten(
            input.shape[1] (the number of mfcc vectors, 
            input.shape[2] the number of mfccs ( 13)
            
keral.layers.MFCC(64, activation=relu)  # 64 Neurons
```

```
    model = keras.Sequential()  

    # 2 LSTM layers
    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(64))

    # dense layer
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
     # dense layer
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))
```
