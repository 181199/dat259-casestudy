package classifiers;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.SelectedTag;

public class SVMClassifier {

    public SVMClassifier() {
    }

    public static void classify(Instances train, Instances test, String dataset) throws Exception {
        // configure SVM
        LibSVM svm = new LibSVM();
        svm.setKernelType(new SelectedTag(LibSVM.KERNELTYPE_RBF,LibSVM.TAGS_KERNELTYPE)); // type: RBF
        svm.setCost(100.0);		// the C parameter
        svm.setProbabilityEstimates(false);
        svm.setDoNotReplaceMissingValues(true);

        // set up FilteredClassifier
        FilteredClassifier classifier = new FilteredClassifier();
        classifier.setClassifier(svm);

        classifier.buildClassifier(train);

        // create new Evaluation object and pass the schema of the dataset
        Evaluation eval = new Evaluation(train);

        // evaluate classifier on test-set
        eval.evaluateModel(classifier, test);

        System.out.println("SVM:");
        // print some stats about the result:
        System.out.println(eval.toSummaryString());
        // more details:
        System.out.println(eval.toClassDetailsString());
        // print confusion matrix
        System.out.println(eval.toMatrixString());

        weka.core.SerializationHelper.write("./models/" + dataset + "_svm.model", classifier);
    }
}
