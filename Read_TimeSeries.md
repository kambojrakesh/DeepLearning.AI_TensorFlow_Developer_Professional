https://colab.research.google.com/github/https-deeplearning-ai/tensorflow-1-public/blob/main/C4/W2/ungraded_labs/C4_W2_Lab_3_deep_NN.ipynb#scrollTo=mIi2fTzPTJxe
https://en.wikipedia.org/wiki/Huber_loss


----------------------------------------------------------------------------------------------------
#Time Series applications - stock prices, weather forecasts, historical trends, such as Moore's law.
#set of random values producing what's typically called white noise


--------------------------------------------------
windowed dataset?
return two data first being batch size and number of timestep

but with RNN we expect three dimensions batch size, the number of timestamps, and the series dimensionality

# Generate a TF Dataset from the series values
dataset = tf.data.Dataset.from_tensor_slices(series)

# Window the data but only take those with the specified size
dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)

# Flatten the windows by putting its elements in a single batch
dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

# Create tuples with features and labels 
dataset = dataset.map(lambda window: (window[:-1], window[-1]))

# Shuffle the windows
dataset = dataset.shuffle(shuffle_buffer)

# Create batches of windows
dataset = dataset.batch(batch_size).prefetch(1)

# Compute the metrics
print(tf.keras.metrics.mean_squared_error(x_valid, results).numpy())
print(tf.keras.metrics.mean_absolute_error(x_valid, results).numpy())

-----------------------------------------------------------------------------------------------------------

# Set the learning rate scheduler
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))

history = model_tune.fit(dataset, epochs=100, callbacks=[lr_schedule])

draw graph between the learning rate and loss

we will pick a learning rate by running the tuning code 


-----------------------------------------------------------------------------------------------------------------

return_sequence = return the last output in the output sequence, or the full sequence
return_sequence = true means sequence to sequence 
return_sequence = False sequence to vector


----------------------------------------------------

lambda :  arbitrary operation of tensorflow

In lambda : input_length  => [none] - model can expect sequence of any length 

  tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                      input_shape=[None]),
   tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),

----------------------------------------------

Huber less sensitive to outliers in data than the squared error loss.

loss=tf.keras.losses.Huber()

----------------------------------------------------

# Generate data windows of the validation set
val_set = windowed_dataset(x_valid, window_size, batch_size, shuffle_buffer_size)

----------------------------------------------------------

tf.keras.backend.clear_session() --> clear all previously used memory


---------------------

class myCallback(tf.keras.callbacks.Callback) can also check the validation across mae and mse








