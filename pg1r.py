import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

'''
Task: Each output unit corresponds to one of the 10 classes (‘0’ to ‘9’). Set the target
value tk for output unit k to 0.9 if the input class is the kth class, 0.1 otherwise.

Network classification: An example x is propagated forward from the input to the output.
The class predicted by the network is the one corresponding to the most highly activated
output unit. The activation function for each hidden and output unit is the sigmoid
function.
'''



# Download training data from open datasets.
#from pytorch tutorial
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

#data loaded in
train_load = DataLoader(training_data, batch_size=1, shuffle=True)
test_load = DataLoader(test_data, batch_size=1)
print('Data Loaded...')

#var for selecting device
device = torch.device("cuda")

#vars for epochs, layer size, accuracy lists
epochs = 50
h1 = 20
h2 = 50
h3 = 100
train_acc = []
test_acc = []
lr = .1
momentum = 0.9

#this is specifically for Exp3 creating percentage data splits for testing use TTS
split_labels = training_data.targets.numpy()
split_index = np.arange(len(training_data))

#making the smaller size data
#t_t_s returns a touple that needs ignoring otherwise the dataloading throws an error
quarter, _ = train_test_split(split_index, train_size=.25, shuffle=True, stratify=split_labels)
half, _ = train_test_split(split_index, train_size=.5, shuffle=True, stratify=split_labels)

#store them for iterations
sets = [(half)]

def sigmoid(x):
    #sigmoid equation as denoted in lecture
    moid = 1 / (1 + torch.exp(-x))
    return moid

def loss(pred, truth):
    #we defined loss as 1/2(t - y)^2
    loss = .5 * ((truth - pred)**2) #"squared loss"
    return loss




class NeuralNetwork():
    def __init__(self, h_size): #take the variable hidden size as an arg

        #all the setup variables for size
        self.input_size = 784 #the inputs for the 28x28
        self.hidden_size = h_size #will be variable for some experiments
        self.output_size = 10 #standard 0-9 outputs like hw1

        #actual layers, pytorch.org states that bias defaults to true, so no need to include
        #hand defined weight and bias for the two layers
        self.hl_w = torch.empty(self.hidden_size, self.input_size).to(device) # [20 x 784]
        self.hl_b = torch.empty(1, self.hidden_size).to(device)
        self.ol_w = torch.empty(self.output_size, self.hidden_size).to(device) #{20 x 784}
        self.ol_b = torch.empty(1, self.output_size).to(device)

        #doing initializations for the starting values
        torch.nn.init.uniform_(self.hl_w, -0.05, 0.05)  
        torch.nn.init.ones_(self.hl_b)
        torch.nn.init.uniform_(self.ol_w, -0.05, 0.05)       
        torch.nn.init.ones_(self.ol_b)
        
        #needed to create var to store the previous momentum updates
        #like lets me copy the structure
        self.ol_w_m = torch.zeros_like(self.ol_w).to(device)
        self.ol_b_m = torch.zeros_like(self.ol_b).to(device)
        self.hl_w_m = torch.zeros_like(self.hl_w).to(device)
        self.hl_b_m = torch.zeros_like(self.hl_b).to(device)

    def forward(self, x, pred): #UPDATED

        #x is our data point of choice for this epoch
        x = torch.flatten(x, start_dim=1)

        #from lecture
        #pass inputs through the layers, then the activation function
        #forward propogate that output to the next layer, and activation then we decide the "outcome"

        #layer 1 mult the matrix + bias [1,SIZE]
        h_out = torch.mm(x, self.hl_w.T) + self.hl_b 
        h_sig = sigmoid(h_out)

        #the next layer gets the "squashed" output from the previous layer
        o_out = torch.mm(h_sig, self.ol_w.T) + self.ol_b
        o_sig = sigmoid(o_out)

        #or all are sent to backprop algo
        if pred: return o_sig

        #or all are sent to backprop algo
        return x, h_out, h_sig, o_out, o_sig

    def backprop(self, x, h_out, h_sig, o_out, o_sig, truth):
        #take the forward outputs and calc the error
        #first on the outputs, then hidden later
        #then update weights
        #er(k) = [o(1-o)] -> der of sig * (o-t)
        '''
        Backwards phase:
        · compute the error at the output using:
            δo(κ) = (yκ − tκ) yκ(1 − yκ) (4.8)
        · compute the error in the hidden layer(s) using:
            δh(ζ) = aζ (1 − aζ ) N∑ k=1 wζ δo(k) (4.9)
        · update the output layer weights using:
            wζκ ← wζκ − ηδo(κ)ahidden ζ (4.10)
        · update the hidden layer weights using:
            vι ← vι − ηδh(κ)x
        '''
        o_error = (o_sig - truth) * (o_sig * ( 1 - o_sig))

        #now we need output grad
        grad_out = torch.mm(o_error.T, h_sig) 
        grad_out_b = o_error

        #hidden grad
        #thinking backwards, the hidden error is the ending error
        #and how much the middle layer is connected to the output layer in this case
        #so the error is the mm of the end error and the synaptic weight of the two layers
        error_hid = torch.mm(o_error, self.ol_w) 
        h_error = error_hid * (h_sig * (1 - h_sig))

        grad_hid = torch.mm(h_error.T, x)
        grad_hid_b = h_error

        #now we should have what we need to update the weights/bias
        #w = w - delta(w) 

        #but with momentum, we need the previous update
        #(lr * grad) + (momentum * previous term) 
        self.ol_w_m = (lr * grad_out) + (momentum * self.ol_w_m)  
        self.ol_b_m = (lr * grad_out_b) + (momentum * self.ol_b_m)
        self.hl_w_m = (lr * grad_hid) + (momentum * self.hl_w_m)
        self.hl_b_m = (lr * grad_hid_b) + (momentum * self.hl_b_m)

        #update the weights with momentum term
        self.ol_w -= self.ol_w_m
        self.ol_b -= self.ol_b_m
        self.hl_w -= self.hl_w_m
        self.hl_b -= self.hl_b_m


'''
Task: Each output unit corresponds to one of the 10 classes (‘0’ to ‘9’). Set the target
value tk for output unit k to 0.9 if the input class is the kth class, 0.1 otherwise.

The task part is confusing to me , target is the set of labels, the "one hot encoded"
from hw1, so we need a vector of [0.1,0.1,0.9.0.1...] where the correct target is 2 in
this example case

Network training: Use back-propagation with stochastic gradient descent to train the
network. Include the momentum term in the weight updates, as described in the lectures.
Set the learning rate to 0.1 and the momentum to 0.9.
'''

def target_vector(label):
    '''
    previously achieved this was
    #encode the "one hot" in the correct location
            t_vec = np.zeros(10)
            t_vec[truth] = 1
    
    now we need this -> [0.1,0.1,0.9.0.1...]
    into a tensor...

    torch.full(size, fill_value, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
    like np.zeros tensors have a .full() function that fills with specified
    ours is a 
    '''
    
    one_hot = torch.full((1,10), 0.1, dtype=torch.float).to(device) #setting all locations to 0.1 by default
    #[0.1,0.1,0.1.0.1...]
    #then we need to place the 0.9 in the "true" answer index
    one_hot[0, label.item()] = 0.9
    return one_hot

#function used to run the exp3 in sequence
def exp3_setup():

    for subs in sets:
        #using subset from the torch utils
        #each "subs" has the varying data size 
        train = Subset(training_data, subs)
        train_loaded = DataLoader(train, batch_size=1, shuffle=True)

        #this is needed to compare against training set of the same size
        eval_loaded = DataLoader(train,batch_size=1)

        #create the new model, with fresh functions
        model = NeuralNetwork(h3)
    
        #train the models
        train_model(model, train_loaded, epochs, test_load, eval_loaded)

#trains the model for set epochs, prints stats and graphs etc
def train_model(model, train_load, epochs, test_load, eval_loaded):

    #train over the epochs same looping like hw1
    for epoch in range(epochs):
        start = time.time()
        #dataloaders return tuples
        for i, (data, label) in enumerate(train_load):
            data, label = data.to(device), label.to(device)
            #get truth vector
            truth = target_vector(label)

            #forward pass
            x, h_out, h_sig, o_out, o_sig = model.forward(data, False)

            #grab the prediction
            result = o_sig

            #lr and momentum are access in this from globals
            model.backprop(x, h_out, h_sig, o_out, o_sig, truth)

            #calc loss
            calc_loss = loss(result, truth) 

            
            #that seems to be the basic loop
            #now we check accuracy at the end of the epoch

        
        #"After each epoch, calculate the network's accuracy on the training set and the test set for your plot."
        #this really should be a function of its own
        train_time = time.time()

        #local lists for accuracy_score usage
        #using a float gets you an error turns out
        train_pred = []
        train_true = []
        test_pred = []
        test_true = []

        #evaluate
        #after submission move this into a function that gets called to save redundancy
        
        
        for i, (data, label) in enumerate(eval_loaded):
                data, label = data.to(device), label.to(device)
                #as in hw1, get a prediction and find the max
                result = model.forward(data, True)
                _, pred = torch.max(result,1)

                #accuracy_score wants lists to get accuracy so use local lists
                train_pred.append(pred.item())
                train_true.append(label.item())

            #test accuracy and store it for graphing
        acc = accuracy_score(train_pred, train_true) * 100  
        train_acc.append(acc)

        for j, (data, label) in enumerate(test_load):
                data, label = data.to(device), label.to(device)
                #as in hw1, get a prediction and find the max
                t_result = model.forward(data, True)
                _, t_pred = torch.max(t_result,1)
                
                test_pred.append(t_pred.item())
                test_true.append(label.item())
                
        t_acc = accuracy_score(test_pred, test_true) * 100
        test_acc.append(t_acc)
        ################################################################################

        #set train again and record some times, print out some stats
        end = time.time()
        eval_time = end - train_time
        print(f"Epoch{epoch + 1}:  Test Accuracy: {t_acc:.2f} - Accuracy: {acc:.2f} Total Time: {end - start:.2f} Eval Time: {eval_time:.2f} Train Time: {train_time - start:.2f}")

    #prints the confusion matrix and a graph of the model performance over the epochs
    confusion = confusion_matrix(test_true, test_pred)
    print(confusion)

    plt.plot(train_acc, label = 'Training')
    plt.plot(test_acc, label = 'Test')
    plt.title(f'Example Size - 0.5')
    plt.legend()
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epoch')
    plt.ylim(80,100)
    plt.show()



#used in exp 1,2
#after these, train was modified to include exp3 extra data set
#that can be removed and these can be run again if needed

#create a model object
#model = NeuralNetwork(h3)
#train_model(model, train_load, epochs, test_load)

#exp3 runs sequentially
exp3_setup()

