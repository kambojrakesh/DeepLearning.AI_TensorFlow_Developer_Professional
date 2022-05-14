https://www.linkedin.com/learning/speaking-confidently-and-effectively/great-speaking-skills-are-a-must-have?autoplay=true

# E-learning-AI1-git-work-DeepLearning.AI_TensorFlow_Developer_Professional
tensorflow==2.7.0
scikit-learn==1.0.1
pandas==1.1.5
matplotlib==3.2.2
seaborn==0.11.2

# Callback - 3steps approach
# ImageDataGenerator class -  4steps approach
# Transfer Learning 
# Create Test set from Validation Set
------------------------------------

https://faroit.com/keras-docs/0.2.0/optimizers/
https://keras.io/api/metrics/
https://keras.io/api/losses/
https://www.tensorflow.org/guide/keras/save_and_serialize
https://www.tensorflow.org/guide/keras/sequential_model
https://medium.com/ai%C2%B3-theory-practice-business/tensorflow-1-0-vs-2-0-part-3-tf-keras-ea403bd752c0

https://www.tensorflow.org/api_docs/python/tf


revist- https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C1/W4/ungraded_labs/C1_W4_Lab_1_image_generator_no_validation.ipynb

transfer_learning - https://www.tensorflow.org/tutorials/images/transfer_learning#data_preprocessing
-----------------------------------------------------------------------------------------------
#Callback


Callback class - tf.keras.callbacks.Callback
on_epoch_end(self, epoch, logs={}) 

------------------------------------------------------------------------------------------------

loss='sparse_categorical_crossentropy',  - for multiple ouput with softmax, 10

------------------------------------------------------------------------------------------------
#ImageDataGenerator class: two steps craete objects of Image Generator and then pass appropriate arguments to flow_from_directory method

# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(parameters)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir)

--------------------------------------------------------------------------------------------------

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
np.vstack((a,b))

add extra axis in and merge both numpy

--------------------------------------------------------------------------------------------------

x = np.array([3, 1, 2])
np.argsort(x) --- sort the argument and return index of sorted array

----------------------------------------------------------------------------------------------------
# Transfer Learning steps:- 4 steps approach downnload, initialize, load, and Freeze layer

from tensorflow.keras.applications.inception_v3 import InceptionV3

# Set the weights file you downloaded into a variable
local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

# Initialize the base model.
# Set the input shape and remove the dense layers.
pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 
                                include_top = False, 
                                weights = None)

# Load the pre-trained weights you downloaded.
pre_trained_model.load_weights(local_weights_file)

# Freeze the weights of the layers.(disabled retraining of all old layers)
for layer in pre_trained_model.layers:
  layer.trainable = False
  
----------------------------------------------------------------------------------------------------
# cardinality : - Create Test set from Validation Set
 
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

