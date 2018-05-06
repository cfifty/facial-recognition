from sklearn.neighbors import NearestNeighbors
import numpy as np
import face_recognition as fr
import sqlite3

N_NEIGHBORS = 3

#FR is abbreviation for face_recognition library
def saveTrainingDataFR():
	toSave1, toSave2 = [], []

	conn = sqlite3.connect(db_file)
    c = conn.cursor()
    for row in c.execute('SELECT * FROM labed_encodings ORDER BY label'):
        label, encodings = load_encoding(row)
        toSave1.append(label)
        toSave2.append(encodings)

    with open('tr_labels_encodings_fr.pickle', 'wb') as f:
        pickle.dump((toSave1, toSave2), f, protocol=2)

def saveValDataFR():
	toSave = []

	files = sorted(os.listdir(val_data_dir))
    for f in files:
        img_file = '{}/{}'.format(val_data_dir, f), num_jitters=100

        img = fr.api.load_image_file(img_file)
	    try:
	        unknown_encoding = fr.api.face_encodings(img)
	        if len(unknown_encoding) == 0:
	            unknown_encoding = fr.api.face_encodings(img, num_jitters)
	        unknown_encoding = unknown_encoding[0]
	        toSave.append(unknown_encoding)
	    except Exception as e:
	        print 'Failed to get face encoding: %s' % str(e)
	        toSave.append('Unknown')

	with open('val_encodings_fr.pickle', 'wb') as f:
        pickle.dump(toSave, f, protocol=2)

def knnFR(labels, unknownLabels):
	with open("val_encodings_fr.pickle", "rb") as f:
		valEncodings = pickle.load(f)
	with open("tr_labels_encodings_fr.pickle", "rb") as f:
		(trLabels, trEncodings) = pickle.load(f)

	return knn(trLabels, trEncodings, valEncodings)

# Input: For any point i, points[i] = that point's feature array and labels[i] = that point's label
# Output: Array of size len(points) predicting labels for each point in unknownPoints
def knn(labels, points, unknownPoints):
	nbrs = NearestNeighbors(n_neighbors = N_NEIGHBORS, algorithm = 'ball_tree').fit(np.array(points))
	distances, indices = nbrs.kneighbors(valEncodings)
	predLabels = []

	for i in range(len(valEncodings)):
		predLabel = 'Unknown'
		knn_i = indices[i] # k nearest neighbors to point with indexed i in valEncodings
		labelFreq = []
		for neighbor in knn_i:
			if neighbor in labelFreq:
				labelFreq[neighbor] = labelFreq[neighbor] + 1
			else:
				labelFreq[neighbor] = 1
		sortedLabels = sorted(labelFreq, key=my_dict.get, reverse=True)
		if len(sortedLabel) > 0:
			predLabel = sortedLabels[0]

		predLabels.append(predLabel)

	return predLabels