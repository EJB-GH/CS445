import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


'''
1. Create training and test set:
Split the data into a training and test set. Each of these should have about 2,300 instances,
and each should have about 40% spam, 60% not-spam, to reflect the statistics of the full
data set. Since you are assuming each feature is independent of all others, here it is not
necessary to standardize the features.
'''

#globals for setting
data_size = 2300
minimal = 0.0001

#setting up the data sets
# fetch dataset 
spambase = fetch_ucirepo(id=94) 
  
# data (as pandas dataframes) 
X = spambase.data.features 
Y = spambase.data.targets 

x = X.to_numpy()
y = Y.to_numpy().flatten() #a flat vector of [0,1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=data_size, stratify=y)
spam = y_train.mean()

#size/input checking
print(f"Training Data Info\n Size: {len(x_train)} Spam: {spam}")
print(x_train.shape) #x is a (2301,57)  inputs x features
print(x_test.shape)
#y is a (2301, ) of [0,1,0,1.....]
print(y_train)

'''
2. Create probabilistic model. (Write your own code to do this.)

• Compute the prior probability for each class, 1 (spam) and 0 (not-spam) in
the training data. As described in part 1, P(1) should be about 0.4.

• For each of the 57 features, compute the mean and standard deviation in the
training set of the values given each class. If any of the features has zero standard
deviation, assign it a “minimal” standard deviation (e.g., 0.0001) to avoid a divide-by-
zero error in Gaussian Naïve Bayes.
'''

class prob_model():
    def __init__(self):
        #using dicts lets me keep the labels separate
        #which makes the training loop much simpler for the prior prob computing
        self.prior_prob = {}
        self.mean = {}
        self.std = {}
        self.cls = np.unique(y_train)


    '''
    Compute the prior probability for each class, 1 (spam) and 0 (not-spam) in
    the training data. As described in part 1, P(1) should be about 0.4.
    '''
    def train(self, x_train, y_train):
        num_data, num_features = x_train.shape #returns touple explored above (2301, 57)

        #the prior probs of 1 or 0 
        #sum the points where x/y
        for label in self.cls: #checking for 0 then 1

            #we can sum the points where y is equal to 0/1 / the total inputs
            #we get a float of the total entries of that value
            self.prior_prob[label] = np.sum(y_train == label) / float(num_data) #type hint to ensure

            #extract the matching class values from x
            #gives us the flattened vector of points where it matches t/f with the equality
            x_label = x_train[y_train == label] 

            #then we need mean and std for these labels
            self.mean[label] = np.mean(x_label, axis=0)
            temp_std = np.std(x_label, axis=0)

            #check for the 0 deviation where the temp is = 0
            temp_std[temp_std == 0] = minimal
            self.std[label] = temp_std # we assign after making sure theres no 0's


    '''
    Because a product of 58 probabilities will be very small, we will instead use the
    log of the product. 
    the gaussian naive bayes algo
    '''
    def gnb(self, data, mean, std):
        '''
        argmaxf(data) = argmaxlog(data)
        argmax(log(class) + log(data1 | class)...etc up to n)
        logP(class) + sumlogP(data | class)
        p(data|class) = N(data, mean, std)
        recall from log stuff that log/log = log - log, log 1 is also 0
        log*log = log + log
        '''
        base = (-0.5 *  np.log(2 * np.pi)) #part 1 of the denom
        base2 = -(np.log(std)) #part 2 denom
        exponent =  -1 * ((data - mean)**2 / (2 * (std)**2)) #the exponent portion
        return base + base2 + exponent


    def prediction(self, data):
        #after we train, we get a prediction for evaluation
        #this one will go over all of training data more like 

        all_probs = {}
        total_pred = []

        for d in data:
            for label in self.cls:
                #logP(class)
                logp = np.log(self.prior_prob[label])

                #+ sumlogP(data | class) from above
                sum_log = 0.0
                for x_i, x_d in enumerate(d):
                    # p(data|class) = N(data, mean, std)
                    x_mean = self.mean[label][x_i]
                    x_std = self.std[label][x_i]

                    #gnb function to get the N value
                    sum_log += self.gnb(x_d, x_mean, x_std)

                #then append
                all_probs[label] = logp + sum_log

            #find the max as we always do
            true_pred = max(all_probs, key=all_probs.get)
            #keep it for accuracy check
            total_pred.append(true_pred)

        #make sure to return an np, as y_test is an np for the library calls to follow
        return np.array(total_pred)

        
#create model
bayes = prob_model()

#train
bayes.train(x_train, y_train)

#test
predictions = bayes.prediction(x_test)

#matrix creation
confusion = confusion_matrix(predictions, y_test)
print(confusion)

print(classification_report(y_test, predictions, labels=[0,1]))

#print the accuracy
print(accuracy_score(y_test, predictions) * 100)
