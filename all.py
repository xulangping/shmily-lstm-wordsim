import matplotlib.pyplot as plt
import time
time.sleep(2)
def func():
    from sklearn import metrics
    from lib import data_split, features_word2vec, model_lstm, model_randomforest
    import pandas as pd
    import os

    # Read data
    # Use the kaggle Bag of words vs Bag of popcorn data:
    # The data is downloaded from:
    # https://www.kaggle.com/c/word2vec-nlp-tutorial/data
    data = pd.read_csv("./data/labeledTrainData.tsv", header=0,
                       delimiter="\t", quoting=3, encoding="utf-8")

    print("The labeled training set dimension is:\n")
    print(data.shape)

    data2 = pd.read_csv("./data/unlabeledTrainData.tsv", header=0,
                        delimiter="\t", quoting=3, encoding="utf-8")

    print("The unlabeled training set dimension is:\n")
    print(data.shape)

    # Labeled data(data) and Unlabeled data(data2)
    # are combined to train the word2vec model
    data2.append(data)
    print(data2.shape)

    model_path = "./model/300features_40minwords_10context"

    # If we have a pre-trained model we'd like to use, it can be loaded here directly.
    # Otherwise we will use the existing data to train it from scratch
    if not os.path.isfile(model_path):
        model = features_word2vec.get_word2vec_model(data2, "review", num_features=300, downsampling=1e-3,
                                                     model_name=model_path)
    else:
        # After model is created, we can load it as an existing file
        model = features_word2vec.load_word2vec_model(model_name=model_path)
    embedding_weights = features_word2vec.create_embedding_weights(model)
    print(embedding_weights.shape)

    # We also need to prepare the word2vec features, so that they are
    # each word is now mapped to an index, consistents with the training embedding
    # Currently, we are limiting each review article to 500 words.
    # By default, we pad the LHS of each vector with zeros.
    # e.g [ 0, 0, 0 .... 0.27, 0.89, 0.35]
    features = features_word2vec.get_indices_word2vec(data, "review", model, maxLength=500,
                                                      writeIndexFileName="./model/imdb_indices.pickle", padLeft=True)

    print(embedding_weights.shape)

    # Now we separate data for training and validation
    y = data["sentiment"]
    X_train, y_train, X_test, y_test = data_split.train_test_split_shuffle(y, features, test_size=0.1)

    model_lstm.classif_imdb(X_train, y_train, X_test, y_test, embedding_weights=embedding_weights,
                            dense_dim=256, nb_epoch=3)

    model_lstm.classif_imdb(X_train, y_train, X_test, y_test, embedding_weights=embedding_weights, dense_dim=256,
                            nb_epoch=3, include_cnn=True)

    model_lstm.classif_imdb(X_train, y_train, X_test, y_test, embedding_weights=None, dense_dim=256, nb_epoch=3)

    features_avg_word2vec = features_word2vec.get_avgfeatures_word2vec(data, "review", model)
    X_train, y_train, X_test, y_test = data_split.train_test_split_shuffle(y, features_avg_word2vec, test_size=0.1)
    model_randomforest.classif(X_train, y_train, X_test, y_test)

x = ['0', 'Nearest 10', 'Nearest 20', 'Nearest 30', 'Nearest 40', 'Nearest 50', 'Nearest 60', 'Nearest 70', 'Nearest 80']
y = [0.69, 0.72, 0.75, 0.75, 0.72, 0.78, 0.71, 0.75, 0.75]
plt.figure(figsize=(12, 8))
plt.plot(x, y)
plt.title('Accuracy')
plt.grid()
plt.show()



