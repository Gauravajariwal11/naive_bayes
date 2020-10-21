#Name- Gaurav ajariwal
#UTA id- 1001396273

import sys
import statistics as stat
import math
import numpy as np


class Classifier :
    def __init__(self, id) :
        self.classID = id
        self.prob = float()
        self.attributes = []

class Attribute :
    def __init__(self, attID) :
        self.attributeID = attID
        self.mean = float()
        self.stdev = float()
        self.prob = float()
        self.values = []

class testing_data :
    def __init__(self, id, trueClass) :
        self.id = id
        self.predict = int()
        self.prob = float()
        self.actual_class = trueClass
        self.accuracy = float()
        self.p_x_c = []
        self.p_x = 0
    
classes = []
test_set = []

def processfile(filename) :
    return [line.rstrip('\n') for line in open(filename)]

def std_dev(obs, avg):
    square_diff = []            #need to find square difference
    # stdev = stat.stdev(obs)    could have just used this
    # print(std)
    for i in obs:
        square_diff.append((i-avg)**2)
    var = ((1/(len(square_diff)-1)) * sum(square_diff))
    if(var<0.0001):
        var = 0.0001
    stdev = math.sqrt(var)
    return stdev

def gaussians_vals(x, mean, stdev) :
    return (1./(np.sqrt(2.*np.pi)*stdev)*np.exp(-np.power((x - mean)/stdev, 2.)/2))


def accuracy(prob_of_class, actualClass):               #evaluating function
    acc = []
    for i, prob in enumerate(prob_of_class):
        if(prob == max(prob_of_class)):
            acc.append(i+1)

    if(len(acc) == 1):
        if(acc[0] == actualClass):
            return 1
        else:
            return 0
    else:
        if(actualClass in acc):
            return 1/len(acc)
        else:
            return 0


def naive_bayes(training_file, test_file) :

    train_data=processfile(training_file)
    num_of_classes = []

    for i in train_data :
        temp_data = i.split()
        temp = [float(x) for x in temp_data]
        
        if (not(temp[-1] in num_of_classes)):
            num_of_classes.append(temp[-1])                        #total number of classes in the gven order

    for i in range(0, len(num_of_classes)) :
        classes.append(Classifier(i+1))
        
        for j in range( 0, len(train_data[0].split())-1):
            classes[i].attributes.append(Attribute(j+1))              #creating objects of the class
    
    for i in train_data :
        temp_data = i.split()
        temp = [float(x) for x in temp_data]
        class_num = int(temp[-1])

        for j in classes :
            if (j.classID == class_num) :
                for index, k in enumerate(temp[:-1]):
                    j.attributes[index].values.append(k)                #values of the attributes

    for k in classes :
        k.prob = len(k.attributes[0].values)/len(train_data)              #Calculate p(C)

    for i in classes :
        for j in i.attributes :
            if( not (len(j.values) == 0) ) :                              #Calulate mean and standard deviation for each attribute
                j.mean = stat.mean(j.values)                              #use stat for finding mean, and std_dev()
            j.stdev = std_dev(j.values, j.mean)

    for i in classes :
        for j in i.attributes :
            print("Class %d, attribute %d, mean = %.2f, std = %.2f" % (i.classID, j.attributeID, j.mean, j.stdev))          #output for training stage
        print()
    
    test_data = processfile(test_file)                                      #testing stage

    for index, i in enumerate(test_data) :
        temp_data = i.split()
        temp2 = [float(x) for x in temp_data]
        temp = testing_data(index+1, temp2[-1])
        
        # Calculate gaussians for each class on the test object
        for j in range( 0, len(classes)) :
            p_x_c = 1

            for x, k in enumerate(temp2[:-1]) :
                temp_test = classes[j].attributes[x]
                p_x_c *= gaussians_vals(k, temp_test.mean, temp_test.stdev)

            temp.p_x_c.append(p_x_c)                                            #first p_x_c comes from the class definition. It means prob of x given c

        for j in range(0, len(classes)) :
            temp.p_x += (temp.p_x_c[j]*classes[j].prob)                           # Calculate p(x) with sum rule

        class_prob = []

        for j in range(0, len(classes)) :
            class_prob.append((temp.p_x_c[j]*classes[j].prob)/temp.p_x)            #Bayes Rule to calculate P(C|x)

        temp.prob = max(class_prob)
        temp.predict = class_prob.index(max(class_prob))+1
        temp.accuracy = accuracy(class_prob, temp.actual_class)
        print("ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f" % (temp.id, temp.predict, temp.prob, temp.actual_class, temp.accuracy))
        test_set.append(temp)

    total_acc = 0
    for i in test_set :
        total_acc += i.accuracy
    print("classification accuracy = %6.4f" % (total_acc/len(test_set)))



train, test = sys.argv[1], sys.argv[2]
naive_bayes(train, test)