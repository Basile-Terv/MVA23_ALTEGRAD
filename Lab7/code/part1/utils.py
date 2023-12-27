"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2023
"""

import numpy as np


def create_train_dataset():
    n_train = 100000
    max_train_card = 10
    max_digit=10

    ############## Task 1
    
    ##################
    # your code here #
    X_train= np.zeros((n_train,max_train_card))
    y_train=np.zeros(n_train)
    for i in range(n_train):
        card=np.random.randint(1,max_train_card+1)
        X_train[i,-card:]=np.random.randint(1,max_digit+1,size=card)
        y_train[i]=np.sum(X_train[i,:])

    print('the lst training sample is ',X_train[0,:])
    print('the lst training label is ',y_train[0])
    ##################

    return X_train, y_train


def create_test_dataset():
    
    ############## Task 2
    
    ##################
    # your code here #
    n_test = 200000
    min_test_card = 5
    max_test_card = 100
    step_test_card = 5
    max_digit=10

    cards=range(min_test_card,max_test_card+1,step_test_card)
    print('cards=',cards)
    n_samples_per_card=n_test//len(cards)
    print('n_samples_per_card=',n_samples_per_card)

    X_test=list()
    y_test=list()

    for card in cards:
        X=np.random.randint(1,max_digit+1,size=(n_samples_per_card,card))
        y=np.sum(X,axis=1)

        X_test.append(X)
        y_test.append(y)
    ##################
    return X_test,y_test
    #return np.array(X_test), np.array(y_test)