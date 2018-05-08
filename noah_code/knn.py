from sklearn.neighbors import NearestNeighbors
import numpy as np
import face_recognition as fr
import sqlite3
import pickle
import os
import csv
from shogun import MulticlassLabels, RealFeatures
from shogun import KNN, EuclideanDistance
from shogun import LMNN

N_NEIGHBORS = 3
MAX_LMNN_ITERS = 2000

db_file = '../eric_code/labed_encodings.db'
val_data_dir = '../images-val-pub'
output_file = 'submission.csv'

#FR is abbreviation for face_recognition library
def saveTrainingDataFR():
	toSave1 = []
	toSave2 = []

	conn = sqlite3.connect(db_file)
	c = conn.cursor()
	for row in c.execute('SELECT * FROM labed_encodings ORDER BY label'):
		label, encodings = load_encoding(row)
		for encoding in encodings:
			toSave1.append(label)
			toSave2.append(encoding)
			if encoding == 'Unknown':
				continue
		print("Add training label/encoding")

	with open('tr_labels_encodings_fr.pickle', 'wb') as f:
		pickle.dump((toSave1, toSave2), f, protocol=2)

def load_encoding(row):
	label, encoding_str = row
	encoding = map(np.fromstring, eval(encoding_str))
	return label, encoding

def saveValDataFR():
	toSave1 = []
	toSave2 = []

	files = sorted(os.listdir(val_data_dir))
	for f in files:
		img_file = '{}/{}'.format(val_data_dir, f)
		num_jitters=100

		img = fr.api.load_image_file(img_file)

		try:
			unknown_encoding = fr.api.face_encodings(img)
			if len(unknown_encoding) == 0:
				unknown_encoding = fr.api.face_encodings(img, num_jitters)
			unknown_encoding = unknown_encoding[0]
			toSave1.append(f)
			toSave2.append(unknown_encoding)
			print("Add validation encoding")
		except Exception as e:
			print 'Failed to get face encoding: %s' % str(e)

	with open('val_imgNames_encodings_fr.pickle', 'wb') as f:
		pickle.dump((toSave1, toSave2), f, protocol=2)

def knnFR(useLMNN = False):
	with open("val_imgNames_encodings_fr.pickle", "rb") as f:
		tmp = pickle.load(f)
		valImgNames = tmp[0]
		valEncodings = tmp[1]
	with open("tr_labels_encodings_fr.pickle", "rb") as f:
		tmp = pickle.load(f)
		trLabels = tmp[0]
		trEncodings = tmp[1]

	predictions = knn(trLabels, trEncodings, valImgNames, valEncodings, useLMNN)

	return valImgNames, predictions

# Input: For any point i, features[i] = that point's feature array and strLabels[i] = that point's string label
# Output: Array of size len(points) predicting labels for each point in unknownPoints
def knn(strLabels, featuresArr, unknownFeatureNames, unknownFeaturesArr, useLMNN = False):
	labelsToInt = {}
	labelsIntToStr = {}
	ret = {}
	intLabels = []

	# convert string labels to int labels (requirement for shogun)
	counter = 0
	for strLabel in strLabels:
		if strLabel not in labelsToInt:
			labelsToInt[strLabel] = counter
			labelsIntToStr[counter] = strLabel
			counter = counter + 1
	for strLabel in strLabels:
		intLabels.append(labelsToInt[strLabel])

	features = RealFeatures(np.array(featuresArr).T)
	unknownFeatures = RealFeatures(np.array(unknownFeaturesArr).T)
	labels = MulticlassLabels(np.array(intLabels).astype(np.float64))
	knn = KNN(N_NEIGHBORS, EuclideanDistance(features, features), labels)

	if useLMNN:
		lmnn = LMNN(features, labels, N_NEIGHBORS)

		# set an initial transform as a start point of the optimization (if not set, principal component analysis is used to obtain this value)
		# init_transform = numpy.eye(2)
		# lmnn.train(init_transform)

		lmnn.train()

		# maximum number of iterations
		lmnn.set_maxiter(MAX_LMNN_ITERS)
		
		# lmnn is used as a distance measure for knn
		knn.set_distance(lmnn.get_distance())

	knn.train()
	predictLabelInts = knn.apply_multiclass(unknownFeatures).get_int_labels()
	for i in range(len(predictLabelInts)):
		predictLabel = labelsIntToStr[predictLabelInts[i]]
		ret[unknownFeatureNames[i]] = predictLabel

	return ret

# Input: For any point i, points[i] = that point's feature array and labels[i] = that point's label
# Output: Array of size len(points) predicting labels for each point in unknownPoints
def old_knn(labels, points, unknownPointNames, unknownPoints):
	nbrs = NearestNeighbors(n_neighbors = N_NEIGHBORS, algorithm = 'ball_tree').fit(np.array(points))
	distances, indices = nbrs.kneighbors(unknownPoints)
	predLabels = {}

	for i in range(len(unknownPoints)):
		predLabel = 'Unknown'
		knn_i = indices[i] # k nearest neighbors to point with indexed i in unknownPoints
		labelFreq = {}
		for neighbor in knn_i:
			if neighbor in labelFreq:
				labelFreq[neighbor] = labelFreq[neighbor] + 1
			else:
				labelFreq[neighbor] = 1
		sortedLabels = sorted(labelFreq, key = labelFreq.get, reverse = True)
		if len(sortedLabels) > 0:
			predLabel = labels[sortedLabels[0]]

		predLabels[unknownPointNames[i]] = predLabel

	return predLabels

def writeToFile(imageNames, predictions):
	with open(output_file, 'wb') as f:
		writer = csv.writer(f, delimiter=',')
		writer.writerow(['image_label', 'celebrity_name'])

		for i in range(len(imageNames)):
			writer.writerow([imageNames[i], predictions[imageNames[i]]])

def main():
	# Data only needs to be pickled once
	# saveTrainingDataFR()
	# saveValDataFR()

	imageNames, predictions = knnFR(True)
	writeToFile(imageNames, predictions)

if __name__ == '__main__':
	main()