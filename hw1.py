import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,  confusion_matrix
import time

#seed random to get the same weights every run
# np.random.seed(9)
'''
PLR - w = w + ada(y-t)w - perceptron learning rule

1. Choose small random initial weights, 𝑤! ∈ [−.05, .05]. Compute the accuracy on the training
and test sets for this initial set of weights, to include in your plot. (Call this “epoch 0”.)
Recall that the bias unit is always set to 1, and the bias weight is treated like any other weight.

2. Repeat: cycle through the training data, changing the weights (according to the perceptron
learning rule) after processing each training example xk , as follows:

• Compute 𝒘 ∙ 𝒙(i) at each output unit.

• The unit with the highest value of 𝒘 ∙ 𝒙(i) is the prediction for this training example.

• If this is the correct prediction, don’t change the weights and go on to the next
training example.

• Otherwise, update all weights in the perceptron:
𝑤i ⟵𝑤i +𝜂 𝑡(i) −𝑦(i) 𝑥 i(i) ,
where
𝑡(i) =
and
1 if the ouput unit is the correct one for this training example
0 otherwise
𝑦(i) =
1 if 𝒘 ∙ 𝒙(i) >0 0
otherwise
Thus, 𝑡(i) −𝑦(i) can be 1, 0, or −1.

'''

#gather all the variables and setup
train = np.loadtxt("CS445/data/mnist_train.csv", delimiter = ",", skiprows = 1) #first value is the header *from kaggle
test = np.loadtxt("CS445/data/mnist_test.csv", delimiter = ",", skiprows = 1)

#all columns but the first label
x = train[:, 1:] #shape (60000,784)
z = test[:, 1:] #test grab

#grab the labels shape(60000, ) w/ index 0 
label = train[:, 0]
z_label = test[:, 0]

#preprocess the data to keep weights down
x = x/255 
z = z/255
print(z.shape)

#create the bias column with 1's
#add it to x using np.hstack
bias = np.ones([x.shape[0], 1]) 
z_bias = np.ones([z.shape[0], 1])
x = np.hstack([x, bias]) #new shape (60000,785)
z = np.hstack([z, z_bias])

#compute w * x so lets make a weights array for 10 perceptrons
w = np.random.uniform(-0.5, 0.5, size = (785,10))

print("Data loaded...\n")

#3 learning rates and epochs
l_rate1 = 0.001
l_rate2 = 0.01
l_rate3 = 0.1
epochs = 70

#preprocessing the data and setup complete
train_acc = []
test_acc = []
confusion = np.zeros([10,10])


ep0_train_pre = np.argmax(np.dot(x,w), axis = 1)
ep0_test_acc = accuracy_score(ep0_train_pre, label) * 100

print("Training accuracy: ", ep0_train_pre)
print(f"Test Accuracy based on training: {ep0_test_acc:.2f}")
train_acc.append(ep0_test_acc) #add to the array to keep
test_acc.append(ep0_test_acc) #keep for later

#now we loop over ptrons and data for epochs

start = time.time()
for epoch in range(1, epochs + 1): #move past the 0
    true = 0
    start = time.time()
    for i in range(x.shape[0]): #range of the data 
        #grab current values
        data = x[i]
        truth = int(label[i])

        result = np.dot(data, w)
        pred = np.argmax(result)

        #if the pred is = to the truth, then we were correct
        #no need to update anything
        if pred == truth:
            true += 1 

        #otherwise update the weights
        else:
            y = (result > 0).astype(int)

            #encode the "one hot" in the correct location
            t_vec = np.zeros(10)
            t_vec[truth] = 1

            #update with plr
            for j in range(10): #ie all perceptrons
                w[:, j] = w[:, j] + l_rate1 * (t_vec[j] - y[j]) * data

        
    #end of each epoch compare to the test set
    #and training set
    
    t_result = np.dot(z, w)
    t_pred = np.argmax(t_result, axis = 1)
    t_acc = accuracy_score(t_pred, z_label) * 100

    pred = np.argmax(np.dot(x,w), axis = 1)
    acc = accuracy_score(pred, label) * 100


    #shuffle the data
    index = np.arange(x.shape[0])
    np.random.shuffle(index)
    x = x[index]
    label = label[index] #first try did not realign the labels and the success tanked

    train_acc.append(acc)
    test_acc.append(t_acc)

    end = time.time()
    print(f"Epoch{epoch}:  Test Accuracy: {t_acc:.2f} - Accuracy: {acc:.2f} Time: {end - start:.2f}") 

conf_pred = []
conf_truth = []
raw_con = np.zeros([10,10], dtype = int)

for i in range(z.shape[0]):
    result = np.dot(z[i],w)
    pred = np.argmax(result)
    conf_pred.append(pred)
    conf_truth.append(z_label[i])
    raw_con[int(z_label[i])][pred] += 1

print(raw_con)





plt.plot(train_acc, label = 'Training')
plt.plot(test_acc, label = 'Test')
plt.title('LR = 0.1')
plt.legend()
plt.ylabel('Accuracy (%)')
plt.xlabel('Epoch')
plt.ylim(80,100)
plt.show()






