package camel.classifier_dictionary;

import classifiers.NaiveBayesClassifier;
import classifiers.RandomForestClassifier;
import classifiers.SVMClassifier;
import utils.MyStopwordsHandler;
import weka.core.Debug;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.core.stemmers.SnowballStemmer;
import weka.core.tokenizers.NGramTokenizer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;

import java.io.File;
import java.util.logging.Level;
import java.util.logging.Logger;

public class CamelClassifierDictionary {

    public static void main(String[] args) throws Exception {

        Instances data = loadDataset("./dataset/camel.arff");

        NGramTokenizer tokenizer = new NGramTokenizer();	// get a tokenizer
        SnowballStemmer stemmer = new SnowballStemmer();	// get a word-stemmer
        MyStopwordsHandler stopword = new MyStopwordsHandler();

        FixedDictionaryStringToWordVector filter = new FixedDictionaryStringToWordVector();

        filter.setDictionaryFile(new File("./dataset/security_keywords.txt"));
        filter.setInputFormat(data);
        filter.setTokenizer(tokenizer);
        filter.setLowerCaseTokens(true);
        filter.setIDFTransform(false);
        filter.setTFTransform(true);
        filter.setOutputWordCounts(true);
        //filter.setStemmer(stemmer);
        filter.setStopwordsHandler(stopword);

        Instances dataFiltered = Filter.useFilter(data, filter);

        Instances sample =	dataFiltered;
        int numFolds = 2; 	// 50/50 split

        // setting up train- and test-set
        sample.randomize(new Debug.Random(42));
        sample.stratify(numFolds);

        Instances trainingSet = sample.trainCV(numFolds, 0);
        Instances testSet = sample.testCV(numFolds, 0);

        RandomForestClassifier.classify(trainingSet, testSet, "camel_dictionary");

        NaiveBayesClassifier.classify(trainingSet, testSet, "camel_dictionary");

        SVMClassifier.classify(trainingSet, testSet, "camel_dictionary");

    }

    public static Instances loadDataset(String path) {
        Instances dataset = null;
        try {
            dataset = ConverterUtils.DataSource.read(path);
            if (dataset.classIndex() == -1) {
                dataset.setClassIndex(dataset.numAttributes() - 1);
            }
        } catch (Exception ex) {
            Logger.getLogger(CamelClassifierDictionary.class.getName()).log(Level.SEVERE, null, ex);
        }

        return dataset;
    }
}
