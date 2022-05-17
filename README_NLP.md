https://projector.tensorflow.org/

----------------------------------------------------------------------------------------------------

from tensorflow.keras.preprocessing.text import Tokenizer


TensorFlow Data Services or TFDS for short, and that contains many data sets and lots of different categories

---------------------------\\
Five important steps of tokenizer:-

Training steps:-

tokenizer = Tokenizer()

tokenizer.fit_on_texts(corpus) --> corpus is also list

texts_to_sequences( --> always pass list not str

max_sequence_len = max([len(x) for x in input_sequences])

input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))


	
	
-------------------

Few basic testing steps must to remember:-

# Convert the text into sequences
token_list = tokenizer.texts_to_sequences([seed_text])[0]
# Pad the sequences
token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
# Get the probabilities of predicting a word
predicted = model.predict(token_list, verbose=0)
# Choose the next word based on the maximum probability
predicted = np.argmax(predicted, axis=-1).item()
# Get the actual word from the word index
output_word = tokenizer.index_word[predicted]




