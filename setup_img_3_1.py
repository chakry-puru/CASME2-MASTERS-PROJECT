#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os, cv2, matplotlib as plt, numpy as np, pandas as pd,csv,glob
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import array
import tensorflow as tf
import copy
from tensorflow import keras

from tensorflow.keras import layers
import PIL
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from IPython.display import Image
from tensorflow.keras.callbacks import TensorBoard


def runTraining(X_train, X_test, y_train, y_test, final_results):

	print("STARTING TRAINING...")



	
	X_train.shape

	y_test.shape

	data_train = tf.data.Dataset.from_tensor_slices(X_train)
	data_test = tf.data.Dataset.from_tensor_slices(X_test)

	def process_img(frame_path):
	    print("process_img acivated...")
	    img = tf.io.read_file(frame_path)
	    img = tf.image.decode_png(img, channels=4)
	    img = img[...,0:2]
	    
	    #convert unit8 tensor to floats in the [0,1]range
	    img = tf.image.convert_image_dtype(img, tf.float32) 
	    lenimage = tf.math.reduce_euclidean_norm(img, axis=-1, keepdims=True)
	    img = tf.concat([img, lenimage], -1)
	    #resize 
	    return tf.image.resize(img, [224, 224])

	process_img(X_train[0]).shape
	X_train[0]


	data_train = data_train.map(process_img, num_parallel_calls=5)

	data_test=data_test.map(process_img,num_parallel_calls=5)

	data_train.element_spec.shape
	data_test.element_spec.shape

	label_train=tf.data.Dataset.from_tensor_slices(y_train)
	label_test=tf.data.Dataset.from_tensor_slices(y_test)


	# In[ ]:


	train_ds = tf.data.Dataset.zip((data_train, label_train))
	test_ds = tf.data.Dataset.zip((data_test, label_test))


	# In[ ]:


	train_ds.element_spec


	# In[ ]:


	train_ds=train_ds.batch(4).prefetch(tf.data.experimental.AUTOTUNE).cache()
	test_ds=test_ds.batch(4).prefetch(tf.data.experimental.AUTOTUNE).cache()


	# In[ ]:

	# In[ ]:


	data_augmentation = tf.keras.Sequential([
	  layers.experimental.preprocessing.RandomFlip("horizontal"),
	  #layers.experimental.preprocessing.RandomRotation(0.2),
	])


	# In[ ]:


	base_model = tf.keras.applications.ResNet50(input_shape=(224,224,3),
	                                               include_top=False,
	                                               weights='imagenet')


	# In[ ]:


	NAME = "apex_Resnet50_3_Ff"

	tboard_log_dir = os.path.join("logsNew",NAME)
	tensorboard = TensorBoard(log_dir = tboard_log_dir)


	# In[ ]:


	train_ds


	# In[ ]:


	for images, labels in train_ds.take(1):
	    plt.figure(figsize=(10, 10))
	    first_image = images[0]
	    for i in range(9):
	        #ax = plt.subplot(3, 3, i + 1)
	        augmented_image = data_augmentation(
	            tf.expand_dims(first_image, 0), training=True
	        )
	        ##plt.imshow(augmented_image[0].numpy().astype("int32"))
	        #plt.title(int(labels[i]))
	        #plt.axis("off")


	# In[ ]:


	base_model.trainable = True


	# In[ ]:


	inputs = keras.Input(shape=(224, 224,3))
	x = data_augmentation(inputs)


	# In[ ]:




	x = base_model(x)
	x = keras.layers.MaxPooling2D(2)(x)
	x = keras.layers.GlobalAveragePooling2D()(x)
	x = keras.layers.Flatten()(x)
	x= keras.layers.Dense(128,activation='relu')(x)
	#x = keras.layers.GlobalAveragePooling2D(2,2)(x)
	x = keras.layers.Dropout(0.2)(x)
	x= keras.layers.Dense(64,activation='relu')(x)
	x = keras.layers.Dropout(0.2)(x)

	outputs = keras.layers.Dense(5,activation='softmax')(x)

	model = keras.Model(inputs, outputs)

	model.summary()


	# In[ ]:


	#from tensorflow.keras.utils import plot_model
	#plot_model(model, to_file='model_resnet.png',show_shapes=True, show_dtype=False,show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96)


	# In[ ]:


	model.compile(optimizer=keras.optimizers.Adam(),
	              loss=keras.losses.CategoricalCrossentropy(),
	              metrics=keras.metrics.CategoricalAccuracy())


	# In[ ]:



	model.fit(train_ds,validation_data=test_ds, epochs=100) # 100)#validation_data=test_ds,callbacks=tensorboard)


	# In[ ]:


	#base_model = tf.keras.applications.resnet50(input_shape=(224,224,2),include_top=False,weights=None)


	# In[ ]:

	train_pred = model.predict(train_ds,verbose=1)
	y_train_ground = y_train.argmax(1)
	y_train_pred = train_pred.argmax(1)


	# In[ ]:

	###
	#from sklearn.metrics import accuracy_score,confusion_matrix
	#train_cmat = confusion_matrix(y_train_ground, y_train_pred)
	#print(train_cmat)


	# In[ ]:


	#import seaborn as sns
	#sns.heatmap(train_cmat, annot=True)


	# In[ ]:


	#accuracy_score(y_train_ground, y_train_pred)


	# In[ ]:





	# In[ ]:


	test_pred = model.predict(test_ds,verbose=1)
	y_test_ground = y_test.argmax(1)
	y_test_pred = test_pred.argmax(1)


	# In[ ]:


	#test_cmat = confusion_matrix(y_test_ground, y_test_pred)
	#print(test_cmat)


	# In[ ]:


	#accuracy_score(y_test_ground, y_test_pred)


	# In[ ]:


	#df_cm = pd.DataFrame(test_cmat, columns=np.unique(l), index = np.unique(l))


	# In[ ]:


	#plt.figure(figsize = (10,7))
	#sns.set(font_scale=1.4)#for label size
	#sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})

	final_results["train"] = y_train_pred
	final_results["test"] = y_test_pred
	final_results["train_ground"]=y_train_ground
	final_results["test_ground"]=y_test_ground
	print(final_results)


# In[ ]:




