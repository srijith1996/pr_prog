import numpy as np
import tensorflow as tf
np.set_printoptions(threshold='nan')
# open file for reading training data
try:
	file = open("/home/ashwin/pr_project/m", 'r')
except:
	print "Error opening file for reading."
num_train = 6124
num_test = 2000
num_attr = 21

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

# initialize training attributes and classes
train_data = np.empty([num_train, num_attr])
test_data = np.empty([num_test, num_attr])
train_classes = np.empty(num_train)
test_classes = np.empty(num_test)

# iteration variable
i = 0
for x in file:

	attrs = x.split(" ")

	# fill up the training example labels
	if i < num_train:
		train_classes[i] = attrs[0]
	else:
		test_classes[i-num_train] = attrs[0]

	# strip trailing new lines
	attrs = attrs[1:-1]
	attrs_new = []

	# strip trailing ":1"
	for attr in attrs:
		attr = attr[:-2]
		attrs_new.append(attr)

	# add data to new list
	attrs_new = np.asarray(attrs_new)

	if i < num_train:
		# fill up the training example attributes
		train_data[i] = attrs_new
	else:
		test_data[i-num_train] = attrs_new

	i += 1

# normalize the attribute values
mins = np.empty(21)
mins = np.amin(train_data, axis=0)
train_data -= mins
train_classes -= 1	

# print train_data
train_classes = train_classes.reshape(train_classes.shape[0],-1)
test_classes = test_classes.reshape(test_classes.shape[0],-1)
# print np.shape(train_classes)
# print np.shape(test_classes)
# print np.shape(train_data)
# print np.shape(test_data)
# # print isinstance(train_data, np.ndarray)

no_data_samples = num_train
no_of_batches = 10
no_of_features = 21
no_of_labels = 2

#network initialisations
no_of_inputs = 21
no_weights_first_layer  = 21
no_bias_first_layer  = 21
no_of_outputs = 1
no_of_epochs = 100
learning_rate = 0.9

x = tf.placeholder(tf.float32, [None,no_of_inputs])

W_1 = weight_variable([no_of_inputs,no_weights_first_layer])
b_1 = bias_variable([no_bias_first_layer])
y_1 = tf.nn.sigmoid(tf.matmul(x, W_1) + b_1)

W_2 = weight_variable([no_weights_first_layer,no_of_outputs])
b_2 =  bias_variable([no_of_outputs])
y = tf.nn.sigmoid(tf.matmul(y_1,W_2) + b_2)

y_ = tf.placeholder(tf.float32, [None,1])

mean_square_error = -tf.reduce_mean(tf.mul(tf.log(y),y_)+tf.mul((1-y_),tf.log(1-y)))

# Optimiser to minimize cost function
train_step = tf.train.MomentumOptimizer(learning_rate,momentum=0.5).minimize(mean_square_error)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
tf.train.start_queue_runners(sess=sess)

saver = tf.train.Saver()

for m in range(no_of_epochs):          

    i_train_data_tensor , o_train_data_tensor = train_data,train_classes
    _,error,o = sess.run([train_step,mean_square_error,y], feed_dict={x: i_train_data_tensor, 
        y_: o_train_data_tensor})

    print'batch: ' , i , 'epoch: ', m+1 , 'error is:' , error# 'output',o

saver.save(sess, "/home/ashwin/mushroom.ckpt") 

correct_prediction = tf.equal(tf.round(y), y_)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

i_test_data_tensor , o_test_data_tensor = test_data,test_classes
acc,o = sess.run([accuracy,y], feed_dict={x: i_test_data_tensor, 
		y_: o_test_data_tensor})



print acc, np.sum(np.round(o)), W_1