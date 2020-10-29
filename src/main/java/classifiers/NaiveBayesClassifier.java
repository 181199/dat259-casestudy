package classifiers;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;

public class NaiveBayesClassifier {

    public NaiveBayesClassifier(){
    }

    public static void classify(Instances train, Instances test, String dataset) throws Exception {
        // Naive bayes classifier
        NaiveBayes naiveBayes = new NaiveBayes();

        // set up FilteredClassifier
        FilteredClassifier classifier = new FilteredClassifier();
        classifier.setClassifier(naiveBayes);

        classifier.buildClassifier(train);

        // create new Evaluation object and pass the schema of the dataset
        Evaluation eval = new Evaluation(train);

        // evaluate classifier on test-set
        eval.evaluateModel(classifier, test);

        System.out.println("Naive bayes:");
        // print some stats about the result:
        System.out.println(eval.toSummaryString());
        // more details:
        System.out.println(eval.toClassDetailsString());
        // print confusion matrix
        System.out.println(eval.toMatrixString());

        weka.core.SerializationHelper.write("./models/" + dataset + "_naivebayes.model", classifier);
    }
}
