import pickle
import numpy as np
import copy

def Accuracy(y_predict,y_true):
    return np.mean(y_predict==y_true)

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classifier=weak_classifier
        self.n_weakers_limit=n_weakers_limit
        self.alphalist=[]
        self.hlist=[]        

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self,X,y,X_test,y_test):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        self.alphalist=[]
        self.hlist=[]
        Accuracytrainlist=[]
        Accuracytestlist=[]
        omega = np.ones((X.shape[0],1)) * (1/X.shape[0])   #omega.shape  (200,1)
        for m in range(self.n_weakers_limit):
            clf = copy.deepcopy(self.weak_classifier)
            clf.fit(X, y,sample_weight=omega.reshape(-1))
            epsilon=np.sum(omega*(clf.predict(X).reshape(-1,1) != y))
            if epsilon> 0.5 or epsilon == 0 :
                break
            alpha = (1/2)*np.log((1-epsilon)/epsilon)
            self.alphalist.append(alpha)
            self.hlist.append(clf)
            print(' %d / %d  weak_classifier : Train_dataset Accuracy  %f'%(m+1,self.n_weakers_limit,Accuracy(self.predict(X).reshape(-1,1),y)))
            Accuracytrainlist.append(Accuracy(self.predict(X).reshape(-1,1),y))
            Accuracytestlist.append(Accuracy(self.predict(X_test).reshape(-1,1),y_test))         
            z=np.sum(omega*np.exp(-alpha*y*(clf.predict(X).reshape(-1,1))))
            omega=omega*(np.exp(-alpha*y*(clf.predict(X).reshape(-1,1))))/z            
        return  Accuracytrainlist,Accuracytestlist

    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        scores = sum(clf.predict(X).reshape(-1,1) * alpha for clf, alpha in zip(self.hlist , self.alphalist))
        return scores

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        scores = sum(clf.predict(X).reshape(-1,1) * alpha for clf, alpha in zip(self.hlist , self.alphalist))
        return np.sign(scores)

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
