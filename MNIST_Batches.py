#Tommy Fang
#dkfang7@gmail.com
#MNIST classifier using custom backprop and batch training

import numpy as np
import matplotlib
import pickle 
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
#nItr = 3000, eta = 0.44, accuracy = 4676/5000
#nItr = 10000m eta: 0.3, accuracy = 4769/5000
#with relu, nItr = 20000, eta = 0.006, accuracy = 4404
#relu - eta = 0.002, a = 4456.0
#0.00200195 - 4550
numIterations = 75000
batchSize = 50
hiddenNodes = 100
k = 10
#toggles matplot lib drawing of  weights
displayPlot = False
#learning rate
eta = tf.constant(0.00299)
#toggle using numpy implementation or tensorflow
#write to file?
writeFile = True

def main():
	print("Loading training data")
	trainX = load_data("./p2/train_data")
	trainY = read_labels("./p2/labels/train_label.txt")
	M, N = np.shape(trainX)
	data = np.arange(0, M)
	batch_shuffle = tf.train.shuffle_batch([data], enqueue_many = True, batch_size=50, capacity=M, min_after_dequeue=10, allow_smaller_final_batch=True)
	#print(batch_shuffle)
	x = tf.placeholder(tf.float32, [None, N])
	y = tf.placeholder(tf.float32, [None, k])
	#derivative = tf.placeholder(tf.float32,[None, k])


	w1 = tf.Variable(tf.random_normal([N, hiddenNodes])) # 784 x 100
	w2 = tf.Variable(tf.random_normal([hiddenNodes, hiddenNodes])) # 100 x 100
	w3 = tf.Variable(tf.random_normal([hiddenNodes, k])) # 100 x 10
	w = {'1':w1, '2': w2, '3': w3 }

	b1 = tf.Variable(tf.zeros([1, hiddenNodes])) # 100 x 1 tensor
	b2 = tf.Variable(tf.zeros([1, hiddenNodes])) # 100 x 1 tensor
	b3 = tf.Variable(tf.zeros([1, k])) # 10 x 1
	b = {'1':b1, '2': b2, '3':b3}

	#forward propagation
	layer1 = (tf.add(tf.matmul(x, w['1']), b['1']))
	a1 = tf.nn.relu(layer1)
	layer2 = tf.add(tf.matmul(a1, w['2']), b['2'])
	a2 = tf.nn.relu(layer2)
	layer3 = tf.add(tf.matmul(a2, w['3']), b['3'])
	a3 = sigmoid(layer3)

 	#loss
	loss = tf.subtract(a3, y)
	cost = tf.reduce_mean(tf.square(loss))
	#total_loss = tf.reduce_mean(loss)
	#cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=a3)
#tf.losses.sparse_softmax_cross_entropy(labels=y, logits=a3)#tf.reduce_mean(-tf.reduce_sum(y * tf.log(a3), reduction_indices=[1]))
	#avg_loss = tf.reduce_mean(diff)
	#cost = tf.reduce_mean(tf.divide(diff, 2*M))
	#Backpropagation	
	#output layer to middle


	d_z3 = tf.multiply(loss, sigmoidPrime(a3))
	d_b3 = d_z3
	d_w3 = tf.matmul(tf.transpose(a2), d_z3)

	#2nd middle layer to 1st
	d_a2 = tf.matmul(d_z3, tf.transpose(w['3']))
	d_z2 = tf.multiply(d_a2, ReLUPrime(layer2) )
	d_b2 = d_z2
	d_w2 = tf.matmul(tf.transpose(a1), d_z2)

	#1st hidden layer to input
	d_a1 = tf.matmul(d_z2, tf.transpose(w['2']))
	d_z1 = tf.multiply(d_a1, ReLUPrime(layer1) )
	d_b1 = d_z1
	d_w1 = tf.matmul(tf.transpose(x), d_z1)

	#Training step function
	step = [
    tf.assign(w['1'],
            tf.subtract(w['1'], tf.multiply(eta, d_w1)))
  , tf.assign(b['1'],
            tf.subtract(b['1'], tf.multiply(eta,
                               tf.reduce_mean(d_b1, axis=[0]))))
  , tf.assign(w['2'],
            tf.subtract(w['2'], tf.multiply(eta, d_w2)))
  , tf.assign(b['2'],
            tf.subtract(b['2'], tf.multiply(eta,
                               tf.reduce_mean(d_b2, axis=[0]))))
	, tf.assign(w['3'],
            tf.subtract(w['3'], tf.multiply(eta, d_w3)))
  , tf.assign(b['3'],
            tf.subtract(b['3'], tf.multiply(eta,
                               tf.reduce_mean(d_b3, axis=[0]))))
]
	# Accuracy check
	acct_mat = tf.equal(tf.argmax(a3, 1), tf.argmax(y, 1))
	acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32))

	init = tf.global_variables_initializer()
	testX = load_data("./p2/test_data")
	testY = read_labels("./p2/labels/test_label.txt")

	#Clear data file
	f = open("cost.txt", "w")
	f.write("")
	f.close()

	f = open("cost.txt", "a")

	print("Beginning tf session")
	with tf.Session() as sess:
		sess.run(init)
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		# Begin training
		total_error = 0
		for itr in range(numIterations):
			#Select batch of 50 random samples
			bxs, bys = nextBatch(trainX, trainY, sess.run([batch_shuffle]))
			c = sess.run(cost, feed_dict = {x: bxs, y: bys})
			#c = sess.run(total_loss,feed_dict = {x: bxs, y: bys} )
			if (itr%10000 ==0):
				print(itr, c)

			training_error = sess.run(acct_res, feed_dict = {x: bxs, y : bys})
			total_error += training_error
			#Save important data
			
			f.write(str(itr) + "," + str(c) + "," + str(total_error/(itr+1)) + '\n')
			#Run the training step on the batch
			sess.run(step, feed_dict = {x: bxs, y: bys})

		#Check model precision through test data.
		testX = load_data("./p2/test_data")
		testY = read_labels("./p2/labels/test_label.txt")
		res = sess.run(acct_res, feed_dict =
               {x: testX,
                y : testY})
		print(res)

		saveWeights([w['1'].eval(), b['1'].eval(), w['2'].eval(), b['2'].eval(), w['3'].eval(), b['3'].eval()])

		#function for computing classification rate for each number
		evaluate(sess,a3,x,testX,testY, res)
	coord.request_stop()
	coord.join(threads)
	sess.close()
	f.close()
	return
def evaluate(sess, a3, x, testX, testY, res):
	f = open("classification.txt", "w")
	predictions = sess.run(a3, feed_dict={x:testX})
	predictions = sess.run(tf.argmax(predictions, axis=1))
	truth = sess.run(tf.argmax(testY, axis=1))
	seen = []
	correct = []
	for ks in range(k):
		seen.append(0)
		correct.append(0)
	for idx in range(0, len(predictions)):
		seen[truth[idx]] += 1
		if (predictions[idx] == truth[idx]):
			correct[predictions[idx]] += 1
	out = str(res) + "\n"
	for p in range(k):
		out += str(p) + "," + str(correct[p]/seen[p]) + "\n"
	f.write(out)
	f.close()

def nextBatch(data, labels, batch):
	x_shuffle = np.array([data[i] for i in batch][0])
	y_shuffle = np.array([labels[i] for i in batch][0])
	return x_shuffle, y_shuffle
def saveWeights(W):
	f = open("multiclass_parameters.txt", "wb")
	pickle.dump(W, f)
	f.close()
def read_labels(labelPath):
	y = []
	f = open(labelPath, "r")
	for i in f:
		yMat = np.zeros(k)
		idx = int(i.strip())
		yMat[idx] = 1
		y.append(yMat)
	f.close()
	return np.array(y)
def sigmoidPrime(z):
	return tf.multiply(sigmoid(z), tf.subtract(tf.constant(1.0), sigmoid(z)))
def sigmoid(z):
	return tf.div(tf.constant(1.0), tf.add(tf.constant(1.0), tf.exp(tf.negative(z))))
def softmax(X):
    exps = tf.exp(X)
    return exps / np.sum(exps)
def ReLU(x):
	return tf.argmax(x,0)
def ReLUPrime(x):
	return tf.cast(tf.greater(x, 0), tf.float32)
def softmaxPrime(z):
	return tf.multiply(softmax(z), tf.subtract(tf.constant(1.0), softmax(z)))
def load_data(dataDir):
	out_data = []
	for f in os.listdir(dataDir):
		fname = os.path.join(dataDir, f)
		if os.path.isfile(fname):
			#data = fname#out_data.append(fname)
			image = matplotlib.image.imread(fname)
			imageData = np.array(image).reshape(784)/255
			out_data.append(imageData)
			#image = matplotlib.image.imread(os.path.join(dataDir, f))
	#normalized = normalize_data(out_data)
	return np.array(out_data)

if __name__ == "__main__":
	main()