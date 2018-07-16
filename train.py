from keras.preprocessing.image import ImageDataGenerator
from classifier import ImageClassifier
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os, sys

def parse_arguments():
	# Command-line flags are defined here.
	parser = argparse.ArgumentParser(\
		description = "Argument Parser for Product Classifier")
	parser.add_argument("-d", "--dataset", required=True, \
		help="path to input dataset (i.e., directory of images)")
	parser.add_argument("-m", "--model", required=True, \
		help="path to output model")
	parser.add_argument("-l", "--labelbin", required=True, \
		help="path to output label binarizer")
	parser.add_argument("-p", "--plot", type=str, default="plot.png", \
		help="path to output accuracy/loss plot")

	return parser.parse_args()

def train(args):

	args = parse_arguments()
	EPOCHS = 100
	INIT_LR = 1e-3
	BS = 32
	IMAGE_DIMS = (96, 96, 3)

	data = []
	labels = []

	print("[INFO] loading images...")
	imagePaths = sorted(list(paths.list_images(args.dataset)))
	random.seed(42)
	random.shuffle(imagePaths)
	# loop over the input images
	for imagePath in imagePaths:
		# load the image, pre-process it, and store it in the data list
		try:
			image = cv2.imread(imagePath)
			image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
			image = img_to_array(image)
			data.append(image)
		 
			# extract the class label from the image path and update the
			# labels list
			label = imagePath.split(os.path.sep)[-2]
			labels.append(label)
		except:
			pass

	data = np.array(data, dtype="float") / 255.0
	labels = np.array(labels)

	# binarize the labels
	lb = LabelBinarizer()
	labels = lb.fit_transform(labels)
	 
	# partition the data into training and testing splits using 80% of
	# the data for training and the remaining 20% for testing
	(trainX, testX, trainY, testY) = train_test_split(data,
		labels, test_size=0.2, random_state=42)

	aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

	# initialize the model
	print("[INFO] compiling model...")
	model = ImageClassifier(IMAGE_DIMS[1], IMAGE_DIMS[0],\
		channels=IMAGE_DIMS[2],classes=len(lb.classes_)).build()
	opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
	model.compile(loss="categorical_crossentropy", optimizer=opt,
		metrics=["accuracy"])

	# train the network
	print("[INFO] training network...")
	H = model.fit_generator(
		aug.flow(trainX, trainY, batch_size=BS),
		validation_data=(testX, testY),
		steps_per_epoch=len(trainX) // BS,
		epochs=EPOCHS, verbose=1)

	# save the model to disk
	print("[INFO] serializing network...")
	model.save(args.model)
	 
	# save the label binarizer to disk
	print("[INFO] serializing label binarizer...")
	f = open(args.labelbin, "wb")
	f.write(pickle.dumps(lb))
	f.close()

	# plot the training loss and accuracy
	plt.style.use("ggplot")
	plt.figure()
	N = EPOCHS
	plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
	plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
	plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="upper left")
	plt.savefig(args["plot"])

if __name__ == "__main__":
	train(sys.argv)