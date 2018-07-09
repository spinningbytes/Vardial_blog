def gather_bigrams(data):
    res = set()
    for n1 in data:
        res.update(bigrams(n1))
    return list(res)

def bigrams(word):
    chars = [c for c in word]
    bigrams = [c1 + c2 for c1, c2 in zip(chars, chars[1:])]
    features = chars + bigrams
    return features

def transform_features(data_train, data_test, n_grams=1):

    from sklearn.feature_extraction.text import CountVectorizer
    bigrams_list = gather_bigrams([tj for tk in data_train for tj in tk.split()])
    print(n_grams)
    cv = CountVectorizer(
            analyzer=bigrams,
       # analyzer="char",
            preprocessor=lambda x : x,
            vocabulary=bigrams_list,
        ngram_range=(1, n_grams))

    
    X_train = cv.fit_transform(data_train)

    X_test = cv.transform(data_test)
    return X_train, X_test

def transform_features_obj(data_train, n_grams=1):

    from sklearn.feature_extraction.text import CountVectorizer
    bigrams_list = gather_bigrams([tj for tk in data_train for tj in tk.split()])
    
    cv = CountVectorizer(
            analyzer=bigrams,
       # analyzer="char",
            preprocessor=lambda x : x,
            vocabulary=bigrams_list,
        ngram_range=(1, n_grams))

    
    X_train = cv.fit_transform(data_train)

    return X_train, (cv,bigrams_list)
