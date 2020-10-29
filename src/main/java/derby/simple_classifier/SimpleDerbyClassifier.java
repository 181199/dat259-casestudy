package derby.simple_classifier;

import classifiers.NaiveBayesClassifier;
import classifiers.RandomForestClassifier;
import classifiers.SVMClassifier;
import utils.MyStopwordsHandler;
import utils.MyStringToWordVector;
import weka.core.Debug;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.core.stemmers.SnowballStemmer;
import weka.core.tokenizers.NGramTokenizer;
import weka.filters.Filter;

import java.util.logging.Level;
import java.util.logging.Logger;

public class SimpleDerbyClassifier {

    public static void main(String[] args) throws Exception {

        Instances data = loadDataset("./dataset/derby.arff");

        NGramTokenizer tokenizer = new NGramTokenizer();	// get a tokenizer
        SnowballStemmer stemmer = new SnowballStemmer();	// get a word-stemmer
        MyStopwordsHandler stopword = new MyStopwordsHandler();

        MyStringToWordVector filter = new MyStringToWordVector();

        filter.setTokenizer(tokenizer);
        filter.setInputFormat(data); 		// pass the schema of the data to the filter; throws exception
        filter.setWordsToKeep(1000000);
        filter.setDoNotOperateOnPerClassBasis(true);
        filter.setLowerCaseTokens(true);
        filter.setIDFTransform(true);
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

        RandomForestClassifier.classify(trainingSet, testSet, "derby_simple");

        NaiveBayesClassifier.classify(trainingSet, testSet, "derby_simple");

        SVMClassifier.classify(trainingSet, testSet, "derby_simple");

    }

    public static Instances loadDataset(String path) {
        Instances dataset = null;
        try {
            dataset = ConverterUtils.DataSource.read(path);
            if (dataset.classIndex() == -1) {
                dataset.setClassIndex(dataset.numAttributes() - 1);
            }
        } catch (Exception ex) {
            Logger.getLogger(SimpleDerbyClassifier.class.getName()).log(Level.SEVERE, null, ex);
        }

        return dataset;
    }
}
