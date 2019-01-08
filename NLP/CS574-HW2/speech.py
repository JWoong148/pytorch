import os
import codecs

def read_files(data_loc):
    """Read the data files.
    The returned object contains various fields that store the data.
    The data is preprocessed for use with scikit-learn.
    
    Description of each fileds of returned object are as follows.
    
    - count_vec: CountVectorizer used to process the data (for reapplication on new data)
    - trainX,devX,testX,unlabeledX: Array of vectors representing Bags of Words, i.e. documents processed through the vectorizer
    - le: LabelEncoder, i.e. a mapper from string labels to ints (stored for reapplication)
    - target_labels: List of labels (same order as used in le)
    - trainy,devy: Array of int labels, one for each document
    - test_fnames: List of test file names
    """

    class Data: pass
    speech = Data()

    print("-- train data")
    speech.train_data, speech.train_fnames, speech.train_labels = read_tsv(data_loc, "train.tsv")
    print(len(speech.train_data))

    print("-- dev data")
    speech.dev_data, speech.dev_fnames, speech.dev_labels = read_tsv(data_loc, "dev.tsv")
    print(len(speech.dev_data))

    print ("-- test data")
    test_data, test_fnames = read_unlabeled(data_loc,'test')
    print (len(test_fnames))

    print("-- unlabeled data")
    unlabeled_data, unlabeled_fnames = read_unlabeled(data_loc, 'unlabeled')
    print(len(unlabeled_fnames))

    print("-- transforming data and labels")
    from sklearn.feature_extraction.text import CountVectorizer
    speech.count_vect = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)
    speech.trainX = speech.count_vect.fit_transform(speech.train_data)

    speech.devX = speech.count_vect.transform(speech.dev_data)
    speech.testX = speech.count_vect.transform(test_data)
    speech.test_fnames = test_fnames
    speech.unlabeledX = speech.count_vect.transform(unlabeled_data)

    from sklearn import preprocessing
    speech.le = preprocessing.LabelEncoder()
    speech.le.fit(speech.train_labels)
    speech.target_labels = speech.le.classes_
    speech.trainy = speech.le.transform(speech.train_labels)
    speech.devy = speech.le.transform(speech.dev_labels)
    return speech

def read_tsv(data_loc, fname):
    """Reads the labeled data described in tsv file.
    The returned object contains three fields that represent the unlabeled data.
    data: documents, each document is represented as list of words
    fnames: list of filenames, one for each document
    labels: list of labels for each document (List of string)
    """
    tf = codecs.open(data_loc + fname, 'r', encoding='utf-8')
    data = []
    labels = []
    fnames = []
    for line in tf:
        (ifname,label) = line.strip().split("\t")
        content = read_instance(data_loc, ifname)
        labels.append(label)
        fnames.append(ifname)
        data.append(content)
    tf.close()
    return data, fnames, labels

def read_unlabeled(data_loc, dname):
    """Reads the unlabeled data.
    The returned object contains two fields that represent the unlabeled data.
    data: documents, each document is represented as list of words
    fnames: list of filenames, one for each document
    """
    data = []
    fnames = []
    raw_fnames = os.listdir(data_loc+dname)
    for raw_fname in raw_fnames:
        fname = dname+'/'+ raw_fname
        content = read_instance(data_loc, fname)
        data.append(content)
        fnames.append(fname)
    return data, fnames

def read_instance(data_loc, ifname):
    """Reads the document file.
    Each document file has a string represents sequence of words, 
    and each words are seperated by semicolon.
    This function convert this string into list of words and return it.
    """
    inst = data_loc + ifname
    ifile = codecs.open(inst, 'r', encoding='utf-8')
    content = ifile.read().strip()
    content = content.split(';')
    return content

def write_pred_kaggle_file(cls, outfname, speech):
    """Writes the predictions in Kaggle format.

    Given the classifier, output filename, and the speech object,
    this function write the predictions of the classifier on the test data and
    writes it to the outputfilename. 
    """
    yp = cls.predict(speech.testX)
    labels = speech.le.inverse_transform(yp)
    f = codecs.open(outfname, 'w')
    f.write("FileIndex,Category\n")
    for i in range(len(speech.test_fnames)):
        fname = speech.test_fnames[i]
        f.write(fname + ',' + labels[i] + '\n')
    f.close()

if __name__ == "__main__":
    print("Reading data")
    data_loc = "data/"
    speech = read_files(data_loc)

    print("Training classifier")
    import classify
    cls = classify.train_classifier(speech.trainX, speech.trainy)

    print("Evaluating")
    classify.evaluate(speech.trainX, speech.trainy, cls)
    classify.evaluate(speech.devX, speech.devy, cls)

    print("Writing Kaggle pred file")
    write_pred_kaggle_file(cls, "data/speech-pred.csv", speech)

