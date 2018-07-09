from sklearn.metrics import f1_score

from sklearn.base import BaseEstimator

from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer


import scipy.sparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),"data"))
import bigrams
import numpy as np
from sklearn.base import BaseEstimator
    
from sklearn.model_selection import KFold,StratifiedKFold

class MetaCV(BaseEstimator):
    def __init__(self, C=1.0, splits=5, tf=0):
        self.C=C
        self.splits=5
        self.n_features=20000
        self.tf=tf

    def fit(self, sents_train, labels_nr_train):
        self.meta=LinearSVC(C=self.C, random_state=42)
        k_fold=KFold(n_splits=self.splits)

        train_m=[]
        labels_m=[]
        index=np.array(range(len(labels_nr_train)))
        for train_di,test_di in k_fold.split(index):
            train_d=[sents_train[tk] for tk in train_di]
            labels_trd = [labels_nr_train[tk] for tk in train_di]
            test_d=[sents_train[tk] for tk in test_di]
            labels_tsd = [labels_nr_train[tk] for tk in test_di]
            tf_vectorizer = CountVectorizer( min_df=2,
                                        max_features=self.n_features)

            tf_train = tf_vectorizer.fit_transform(train_d)
            tf_train = normalize(tf_train)



            tfidf_vectorizer = TfidfVectorizer( min_df=2,
                                         norm="l2",
                                           max_features=self.n_features)
            tfidf_train = tfidf_vectorizer.fit_transform(train_d)
            X_train_ngrams, X_ngrams  = bigrams.transform_features_obj(train_d, n_grams=7)
            X_ngrams=X_ngrams


            tfidf_vectorizer_ngram = TfidfVectorizer( min_df=2,
                                         norm="l2",
                                           max_features=self.n_features, ngram_range=(1, 7))
            tfidf_train_ngram = tfidf_vectorizer_ngram.fit_transform(train_d)



            tf_vectorizer_char_ngram = CountVectorizer( min_df=2,
                                                       analyzer="char",
                                           max_features=self.n_features, ngram_range=(1, 7))
            tf_train_char_ngram = tf_vectorizer_char_ngram.fit_transform(train_d)


            tf_train_char_ngram = normalize(tf_train_char_ngram)
            tfidf_vectorizer_char_ngram = TfidfVectorizer( min_df=2,
                                         norm="l2",
                                                       analyzer="char",
                                           max_features=self.n_features, ngram_range=(1, 7))
            tfidf_train_char_ngram = tfidf_vectorizer_char_ngram.fit_transform(train_d)


            svms=[]
            if self.tf==0:
                arr=[tfidf_train, tfidf_train_ngram,X_train_ngrams,tfidf_train_char_ngram,tf_train_char_ngram]
            else:
                arr=[tf_train,tfidf_train, tfidf_train_ngram,X_train_ngrams,tfidf_train_char_ngram,tf_train_char_ngram]
            for tk in arr:
                svms.append(LinearSVC(random_state=42, C=1))
                svms[-1].fit(tk,labels_trd)




            tf_dev = tf_vectorizer.transform(test_d)
            tf_dev =  normalize(tf_dev)
            tfidf_dev = tfidf_vectorizer.transform(test_d)
            tf_dev_char_ngram = tf_vectorizer_char_ngram.transform(test_d)
            tfidf_dev_ngram = tfidf_vectorizer_ngram.transform(test_d)

            tf_dev_char_ngram = normalize(tf_dev_char_ngram)

            tfidf_dev_char_ngram = tfidf_vectorizer_char_ngram.transform(test_d)

            X_test_ngrams = X_ngrams[0].transform(test_d)

            svms_preds=[]
            if self.tf==0:
                arr=[tfidf_dev, tfidf_dev_ngram,X_test_ngrams,tfidf_dev_char_ngram,tf_dev_char_ngram]
            else:
                arr=[tf_dev,tfidf_dev, tfidf_dev_ngram,X_test_ngrams,tfidf_dev_char_ngram,tf_dev_char_ngram]
                
            for i1,tk in enumerate(arr):
                svms_preds.append(svms[i1].decision_function(tk))
            #sumpres=sum([tk for tk in svms_preds])
            if len(svms_preds[0].shape)==1: #(0,1) class problem
                sumpres=np.vstack((svms_preds)).transpose()
            else:
                sumpres=sum([tk for tk in svms_preds])
            #sumpres=np.hstack((svms_preds))
            train_m.append(sumpres)
            labels_m.append(labels_tsd)
            #print("SVM TF-IDF separate svm for features Bigrams   char 7 ngram",f1_score(np.argmax(sumpres,1),labels_nr_dev, average="weighted"))



        self.meta.fit(np.vstack((train_m)),np.hstack((labels_m)))


        self.tf_vectorizer = CountVectorizer( min_df=2,
                                    max_features=self.n_features)

        tf_train = self.tf_vectorizer.fit_transform(sents_train)
        tf_train = normalize(tf_train)



        self.tfidf_vectorizer = TfidfVectorizer( min_df=2,
                                     norm="l2",
                                       max_features=self.n_features)
        tfidf_train = self.tfidf_vectorizer.fit_transform(sents_train)
        X_train_ngrams, X_ngrams  = bigrams.transform_features_obj(sents_train, n_grams=7)
        self.X_ngrams=X_ngrams


        self.tfidf_vectorizer_ngram = TfidfVectorizer( min_df=2,
                                     norm="l2",
                                       max_features=self.n_features, ngram_range=(1, 7))
        tfidf_train_ngram = self.tfidf_vectorizer_ngram.fit_transform(sents_train)



        self.tf_vectorizer_char_ngram = CountVectorizer( min_df=2,
                                                   analyzer="char",
                                       max_features=self.n_features, ngram_range=(1, 7))
        tf_train_char_ngram = self.tf_vectorizer_char_ngram.fit_transform(sents_train)


        tf_train_char_ngram = normalize(tf_train_char_ngram)
        self.tfidf_vectorizer_char_ngram = TfidfVectorizer( min_df=2,
                                     norm="l2",
                                                   analyzer="char",
                                       max_features=self.n_features, ngram_range=(1, 7))
        tfidf_train_char_ngram = self.tfidf_vectorizer_char_ngram.fit_transform(sents_train)


        self.svms=[]
        if self.tf==0:
            arr=[tfidf_train, tfidf_train_ngram,X_train_ngrams,tfidf_train_char_ngram,tf_train_char_ngram]
        else:
            arr=[tf_train,tfidf_train, tfidf_train_ngram,X_train_ngrams,tfidf_train_char_ngram,tf_train_char_ngram]
        for tk in arr:
            self.svms.append(LinearSVC(random_state=42, C=1))
            self.svms[-1].fit(tk,labels_nr_train)


    def predict(self,sents_dev):


        tf_dev = self.tf_vectorizer.transform(sents_dev)
        tf_dev = normalize(tf_dev)
        tfidf_dev = self.tfidf_vectorizer.transform(sents_dev)
        tf_dev_char_ngram = self.tf_vectorizer_char_ngram.transform(sents_dev)
        tfidf_dev_ngram = self.tfidf_vectorizer_ngram.transform(sents_dev)

        tf_dev_char_ngram = normalize(tf_dev_char_ngram)

        tfidf_dev_char_ngram = self.tfidf_vectorizer_char_ngram.transform(sents_dev)

        X_test_ngrams = self.X_ngrams[0].transform(sents_dev)

        svms_preds=[]
        if self.tf==0:
            arr=[ tfidf_dev, tfidf_dev_ngram,X_test_ngrams,tfidf_dev_char_ngram,tf_dev_char_ngram]
        else:
            arr=[tf_dev, tfidf_dev, tfidf_dev_ngram,X_test_ngrams,tfidf_dev_char_ngram,tf_dev_char_ngram]
        for i1,tk in enumerate(arr):
            svms_preds.append(self.svms[i1].decision_function(tk))
        #sumpres=sum([tk for tk in svms_preds])
        if len(svms_preds[0].shape)==1: #(0,1) class problem
            sumpres=np.vstack((svms_preds)).transpose()
        else:
            sumpres=sum([tk for tk in svms_preds])
        #sumpres=np.hstack((svms_preds))

        preds=self.meta.predict(sumpres)
        return preds

    def decision_function(self,sents_dev):


        tf_dev = self.tf_vectorizer.transform(sents_dev)
        tf_dev = normalize(tf_dev)
        tfidf_dev = self.tfidf_vectorizer.transform(sents_dev)
        tf_dev_char_ngram = self.tf_vectorizer_char_ngram.transform(sents_dev)
        tfidf_dev_ngram = self.tfidf_vectorizer_ngram.transform(sents_dev)

        tf_dev_char_ngram = normalize(tf_dev_char_ngram)

        tfidf_dev_char_ngram = self.tfidf_vectorizer_char_ngram.transform(sents_dev)

        X_test_ngrams = self.X_ngrams[0].transform(sents_dev)

        svms_preds=[]
        if self.tf==0:
            arr=[ tfidf_dev, tfidf_dev_ngram,X_test_ngrams,tfidf_dev_char_ngram,tf_dev_char_ngram]
        else:
            arr=[tf_dev, tfidf_dev, tfidf_dev_ngram,X_test_ngrams,tfidf_dev_char_ngram,tf_dev_char_ngram]
        for i1,tk in enumerate(arr):
            svms_preds.append(self.svms[i1].decision_function(tk))
        #sumpres=sum([tk for tk in svms_preds])
        if len(svms_preds[0].shape)==1: #(0,1) class problem
            sumpres=np.vstack((svms_preds)).transpose()
        else:
            sumpres=sum([tk for tk in svms_preds])
        #sumpres=np.hstack((svms_preds))
        preds=self.meta.decision_function(sumpres)
        return preds


def perform(uall,comp,tf):
    meta=MetaCV(splits=5, tf=tf)
    print ("uall",uall,"comp",comp,"tf",tf)
    #if "labels_nr_train" not in globals():
    if uall==0:
            labels_train, labels_nr_train, labels_dict_train,sents_train_raw, words_train, chars_train, char_counts_train = get_data()
            labels_dev_dev, labels_nr_dev, labels_dict_dev,sents_dev_raw, words_dev, chars_dev, char_counts_dev = get_dev_data()
            sents_train= [" ".join(tk).lower() for tk in sents_train_raw]
            sents_dev= [" ".join(tk).lower() for tk in sents_dev_raw]
            labels_dev_dev, labels_nr_dev, labels_dict_dev,sents_dev_raw, words_dev, chars_dev, char_counts_dev = get_dev_data()
    else:
            labels_train, labels_nr_train, labels_dict_train,sents_train_raw, words_train, chars_train, char_counts_train = preprocessing.get_data_all()
            labels_dev_dev, labels_nr_dev, labels_dict_dev,sents_dev_raw, words_dev, chars_dev, char_counts_dev = preprocessing.get_dev_data_all()
            sents_train= [" ".join(tk).lower() for tk in sents_train_raw]
            sents_dev= [" ".join(tk).lower() for tk in sents_dev_raw]
            labels_dev_dev, labels_nr_dev, labels_dict_dev,sents_dev_raw, words_dev, chars_dev, char_counts_dev = preprocessing.get_dev_data_all()

    if "sents_test_raw" not in globals():
        #sents_test_raw, words_test, chars_test, char_counts_test = preprocessing.get_test()
        labels_test, labels_nr_test, labels_dict_test,sents_test_raw, words_test, chars_test, char_counts_test = preprocessing.get_data(fname=os.path.join(os.path.dirname(os.path.realpath(__file__)),"./data/gold/gold.txt"))
    if comp==0:
        meta.fit(sents_train,labels_nr_train)
        preds=meta.predict(sents_dev)
        print("SVM TF",f1_score(preds,labels_nr_dev, average="weighted"))
        print("SVM TF macro",f1_score(preds,labels_nr_dev, average="macro"))
    else:
        rev_labels_dict_train = dict([(tv,tk) for tk,tv in labels_dict_train.items()])
        sents_train_comp= [" ".join(tk).lower() for tj in [sents_train_raw,sents_dev_raw] for tk in tj]
        labels_nr_train_comp = [ tk for tj in [labels_nr_train,labels_nr_dev] for tk in tj]
        meta.fit(sents_train_comp,labels_nr_train_comp) 
        sents_test_comp= [" ".join(tk).lower() for tk in sents_test_raw]
        preds=meta.predict(sents_test_comp)
        labels_preds=[rev_labels_dict_train[np.argmax(tk)] for tk in preds]
        with open("prediction.labels","w") as f1:
            for label in labels_preds:
                f1.write(label+"\n")

        from sklearn.model_selection import cross_val_score


        #scores_a = cross_val_score(meta, sents_train_a, labels_nr_train_a, scoring="f1_macro", cv=5, n_jobs=-1)
        #scores = cross_val_score(meta, sents_train, labels_nr_train, scoring="f1_macro", cv=5, n_jobs=-1)

        preds_score=meta.decision_function(sents_test_comp)

        trf=np.argmax(preds_score,1)

        #trf[np.where(preds_score.max(1)<-0.3)[0]]=-1


        rev_labels_dict_train=dict([(tv,tk) for tk,tv in labels_dict_train.items()])
        rev_labels_dict_train[-1]="XY"

        tr_labs=[rev_labels_dict_train[trf[tk]] for tk in range(trf.shape[0])] 

        with open("predictions_c5_metacv_multiclass_threshold.labels","w") as f1:
            for tr1 in tr_labs:
                f1.write(tr1+"\n")

        gdi4=np.isin(labels_nr_test,[labels_dict_test[tk] for tk in ["ZH","LU","BE","BS"]])
        index=np.where(gdi4)[0]

        print("SVM TF",f1_score([labels_dict_test[rev_labels_dict_train[tk]] for tk in np.array(trf)[index]],np.array(labels_nr_test)[index], average="weighted"))
        print("SVM TF macro",f1_score([labels_dict_test[rev_labels_dict_train[tk]] for tk in np.array(trf)[index]],np.array(labels_nr_test)[index], average="macro"))
        #print("SVM
        #TF",f1_score(preds_score[index],labels_nr_test[index],
        #average="weighted"))
        return index, gdi4,trf, labels_dict_test,rev_labels_dict_train,labels_nr_test, preds_score

if __name__=="__main__":

    import preprocessing
    experiment_all=0
    if experiment_all==1:
        for tcomp in range(2):
            comp=tcomp
            for tuall in range(2):
                uall=tuall
                for tft in range(2):
                    tf=tft
                    meta=MetaCV(splits=5, tf=tf)
                    print ("uall",uall,"comp",comp,"tf",tf)
                    #
                    perform(uall,comp,tf)
    else:
        uall=0
        comp=1
        tf=0
        index,gdi4,trf,labels_dict_test,rev_labels_dict_train,labels_nr_test,preds_score =perform(uall,comp,tf)
        print("SVM TF",f1_score([labels_dict_test[rev_labels_dict_train[tk]] for tk in np.array(trf)[index]],np.array(labels_nr_test)[index], average="macro"))
        trf5=trf.copy()
        trf5[np.where(preds_score.max(1)<-0.3)[0]]=-1
        results=[]
        for tk in np.linspace(-0.3,0,20):
             trf5=trf.copy()
             trf5[np.where(preds_score.max(1)<tk)[0]]=-1
             results.append(f1_score([labels_dict_test[rev_labels_dict_train[tk]] for tk in np.array(trf5)],np.array(labels_nr_test), average="macro"))


        y=results

        x=np.linspace(-0.3,0,20)
        import matplotlib.pyplot as plt
        plt.plot(x,y)


        plt.xlabel("Threshold range")


        plt.ylabel("F-1 Macro")


        plt.title("F-1 Macro versus Threshold")
        plt.savefig("threshold_dependency.png")
        #plt.show()
        lab=range(4)
        threshold=[0 for i in lab]
        tresults=[]
        uthresholds=[]
        maxits=16
        minv=-0.4
        maxv=0.1
        for tk0 in np.linspace(minv,maxv,maxits):
            threshold[0]=tk0
            for tk1 in np.linspace(minv,maxv,maxits):
                threshold[1]=tk1
                for tk2 in np.linspace(minv,maxv,maxits):
                    threshold[2]=tk2
                    for tk3 in np.linspace(minv,maxv,maxits):
                        threshold[3]=tk3
                        trf5=trf.copy()
                        for i in lab:
                            trf5[(preds_score.max(1)<threshold[i]) &( preds_score.argmax(1)==i)]=-1
                        uthresholds.append([tk for tk in threshold])
                        tresults.append(f1_score([labels_dict_test[rev_labels_dict_train[tk] ] for tk in np.array(trf5)],np.array(labels_nr_test), average="macro"))

        print(np.max(tresults),np.argmax(tresults),uthresholds[np.argmax(tresults)])
  

        
