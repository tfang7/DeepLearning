#MNIST classifier for numbers 0-5 using numpy/tensorflow
import numpy as np
import matplotlib
import pickle
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
numIterations = 2000
tolerance = 0.0001
#toggles matplot lib drawing of  weights
displayPlot = False
#learning rate
alpha = 0.002666
#toggle using numpy implementation or tensorflow
usingTF = False
#write to file?
writeFile = True

#I implemented a numpy version first and then implemented the tensorflow version.
def main():
	print("Loading training data")
	trainX = load_data("./train_data")
	trainY = read_labels("./labels/train_label.txt")
	M, N = np.shape(trainX)
	k = 5
	if (usingTF):
		W = tf.Variable(tf.random_normal([N,k]))#
		X = tf.convert_to_tensor(trainX, np.float32, name = "images")
		Y = tf.convert_to_tensor(trainY, np.float32, name = "labels")
		b = tf.Variable(tf.zeros([k]), name='biases')
		x = tf.placeholder("float", [None,784])
		y = tf.placeholder("float", [None,5])
		h = tf.add(tf.matmul(x,W),b)
		pred = tf.nn.softmax(h)
		cost = tf.reduce_sum(tf.pow(pred-y, 2))/(2*M)
		W_grad = -tf.matmul(tf.transpose(X), Y - h)
		b_grad = -tf.reduce_mean(tf.matmul(tf.transpose(X), Y - h), reduction_indices=0)
		train(trainX, x, trainY, y, h, M, W, b, cost, W_grad, b_grad)
	else:
		#Numpy version
		W=np.random.rand(N,k)*0.01
		bias = np.zeros(k)
		print("Beginning gradient descent")
		gradient_descent(trainX, trainY, W, bias, alpha, M)
	return
def train(X, x, Y, y, h, M, W, b, c, wg, bg):
	init = tf.global_variables_initializer()
	
	gradW, gradB = tf.gradients(xs=[W, b], ys=c)
	new_W = W.assign(W - alpha * wg)
	new_b = b.assign(b - alpha * bg)
	
	with tf.Session() as sess:
		sess.run(init)
		print("Running sess")
		for itr in range(numIterations):
			#sess.run([W, b, c], feed_dict={x:X,y:Y})
			curr_W, curr_b, curr_loss  = sess.run([new_W, new_b, c], feed_dict={x:X,y:Y})
			print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
		testX = load_data("./test_data")
		testY = read_labels("./labels/test_label.txt")

		correct_prediction = tf.equal(tf.argmax(testY,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		#print(sess.run(accuracy, feed_dict={x: testX, y: testY}))

def gradient_descent(x, y, W, bias, alpha, M):
	xT = np.transpose(x)
	#print(str(np.shape(y)))
	thetaList = []
	#clear this file before training

	f = open("loss_data.txt", "w")
	f.write("")
	f.close()
	# reopen file
	f = open("loss_data.txt", "a")
	#Train our weight layer by taking all the sample data and forming a hypthosis
	for i in range(0, numIterations):
		#form the hypothesis
		h = np.dot(x, W)
		#check the difference between the hypothesis and actual y values
		loss = (h - y) + bias
		sumLoss = np.sum(loss)
		#compute the mean squared error
		cost = np.sum( (loss) ** 2) / (2 * M)
		#write loss to file to plot later
		if (writeFile):
			f.write(str(i) + "," + str(sumLoss) + "," + str(cost) + "\n")

		#compute the gradient over xT
		w_gradient = np.dot(xT, loss) / M
		Wnext = W - alpha * w_gradient
		deltaW = np.sum(Wnext - W)
		bias += deltaW

		if (i % 10 == 0):
			print("Iteration %d | Cost: %f " % (i, cost) )
			#print("Bias: " + str(bias))
			#print("Loss: " + str(loss))
			#print("DeltaW: " + str(np.sum(abs(Wnext - W))))
		if (np.sum(abs(Wnext - W)) < tolerance):
			print("Convergence")
			thetaList.append(Wnext)
			break
		W = Wnext
		if i == numIterations - 1:
			print("Finished training iterations")
			thetaList.append(Wnext)
	Wtest = thetaList[0]
	
	f.close()
	testWeights(Wtest)
	return Wtest, bias

def testWeights(W):
	print("Loading test data")
	tX = load_data("./test_data")
	tY = read_labels("./labels/test_label.txt")
	Mt, Nt = np.shape(tX)
	Wt = np.transpose(W)
	saveWeights(Wt)
	print("Testing weights")
	if (displayPlot):
		for wk in Wt:
			img = wk.reshape(28,28)
			plt.imshow(img)
			plt.colorbar()
			plt.show()
	correct = 0
	correctCounts = [0 for i in range(5)]
	seenCount = [0 for i in range(5)]
	for m in range(Mt):
		t = sigmoid(np.dot(tX[m], W))
		actual = np.argmax(tY[m])
		predicted = np.argmax(t)
		if (actual == predicted):
			correctCounts[actual] += 1
			correct += 1
		seenCount[actual] += 1

	print("Correct : " + str(correct) + " / " + str(Mt) + " = " + str(correct/(Mt) ))
	countString = ""
	if (writeFile):
		f = open("test_data.txt", "w")
		toFile = ""
		for i in range(len(correctCounts)):
			countString +=str(i+1) + ": " + str(correctCounts[i]) + " / " + str(seenCount[i]) + " = " + str(correctCounts[i]/seenCount[i]) + "\n"
			toFile += str(correctCounts[i]/seenCount[i]) + "\n"
		f.write(toFile + "\n" + str(correct/(Mt)))
		f.close()
		print(countString)

def saveWeights(W):
	f = open("multiclass_parameters.txt", "wb")
	pickle.dump(W, f)
	f.close()


def read_labels(labelPath):
	y = []
	f = open(labelPath, "r")
	for i in f:
		yMat = np.zeros(5)
		k = int(i.strip())
		yMat[k-1] = 1
		y.append(yMat)
	f.close()
	return np.array(y)

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

def load_data(dataDir):
	out_data = []
	for f in os.listdir(dataDir):
		fname = os.path.join(dataDir, f)
		if os.path.isfile(fname):
			out_data.append(fname)
			#image = matplotlib.image.imread(os.path.join(dataDir, f))
	normalized = normalize_data(out_data)
	return normalized

#input: dataset -- a list of image files
#output: normalized vector [784x1] 
def normalize_data(dataset):
	normalized = []
	#loop through images in dataset
	for data in dataset:
		image = matplotlib.image.imread(data)
		imageData = np.array(image).reshape(784) / 255
		#img = np.reshape(np.array(image),(1,784))
		normalized.append(imageData)
	return np.array(normalized)

if __name__ == "__main__":
	main()