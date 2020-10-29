package derby.classifier;

import classifiers.NaiveBayesClassifier;
import classifiers.RandomForestClassifier;
import classifiers.SVMClassifier;
import utils.MyStopwordsHandler;
import utils.MyStringToWordVector;
import weka.core.Attribute;
import weka.core.Debug;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.core.tokenizers.NGramTokenizer;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveWithValues;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

public class DerbyClassifier {

    public static void main(String[] args) throws Exception {

        Instances data = loadDataset("./dataset/derby.arff");
        data.setClassIndex(2);

        // Make a new dataset with only SBRs
        RemoveWithValues removeWithValues = new RemoveWithValues();
        removeWithValues.setAttributeIndex("3");
        removeWithValues.setNominalIndices("first");
        removeWithValues.setInputFormat(data);
        Instances SBRs = Filter.useFilter(data, removeWithValues);

        // Make a new dataset with only NSBRs
        RemoveWithValues removeWithValues2 = new RemoveWithValues();
        removeWithValues2.setAttributeIndex("3");
        removeWithValues2.setNominalIndices("last");
        removeWithValues2.setInputFormat(data);
        Instances NSBRs = Filter.useFilter(data, removeWithValues2);

        System.out.println("SBRs="+SBRs.size()+" | NSBRs="+NSBRs.size());
        NGramTokenizer tokenizer = new NGramTokenizer();	// get a tokenizer
        //SnowballStemmer stemmer = new SnowballStemmer();	// get a word-stemmer
        MyStopwordsHandler stopword = new MyStopwordsHandler(); // get stopword handler

        // Saving the 100 (or so) most frequent words in the SBRs dataset to a new file
        MyStringToWordVector filter = new MyStringToWordVector();
        filter.setTokenizer(tokenizer);
        filter.setInputFormat(SBRs); 		// pass the schema of the data to the filter; throws exception
        filter.setWordsToKeep(100);
        filter.setDoNotOperateOnPerClassBasis(true);
        filter.setLowerCaseTokens(true);
        filter.setIDFTransform(true);
        filter.setTFTransform(true);
        filter.setOutputWordCounts(true);
        filter.setDictionaryFileToSaveTo(new File("./dataset/features_derby.csv"));
        //filter.setStemmer(stemmer);
        filter.setStopwordsHandler(stopword);

        Instances keywords = Filter.useFilter(SBRs, filter);

        Map<String, Double> scored_words = scoreWords(SBRs, NSBRs, "./dataset/features_derby.csv");

        Double[] final_scores = scoreReport(NSBRs, scored_words);

        Instances newNSBRs = filterOutNSBRs(NSBRs, final_scores);
        newNSBRs.setClassIndex(2);
        SBRs.setClassIndex(2);

        Instances newDataset = merge(SBRs, newNSBRs);

        MyStringToWordVector filter2 = new MyStringToWordVector();
        filter2.setTokenizer(tokenizer);
        filter2.setInputFormat(newDataset); 		// pass the schema of the data to the filter; throws exception
        filter2.setWordsToKeep(1000000);
        filter2.setDoNotOperateOnPerClassBasis(true);
        filter2.setLowerCaseTokens(true);
        filter2.setIDFTransform(true);
        filter2.setTFTransform(true);
        filter2.setOutputWordCounts(true);
        //filter.setStemmer(stemmer);
        filter2.setStopwordsHandler(stopword);

        Instances dataFiltered = Filter.useFilter(newDataset, filter2);

        Instances sample = dataFiltered;
        int numFolds = 2; 	// 50/50

        // setting up train- and test-set
        sample.randomize(new Debug.Random(42));
        sample.stratify(numFolds);

        Instances trainingSet = sample.trainCV(numFolds, 0);
        Instances testSet= sample.testCV(numFolds, 0);

        RandomForestClassifier.classify(trainingSet, testSet, "derby");

        NaiveBayesClassifier.classify(trainingSet, testSet, "derby");

        SVMClassifier.classify(trainingSet, testSet, "derby");
    }

    public static Instances filterOutNSBRs(Instances NSBRs, Double[] score){

        int j = 0;
        for(int i = 0; i < NSBRs.numInstances(); i++){
            if(score[i] > 0.1){
                j++;
                NSBRs.delete(i);
            }
        }
        System.out.println("Number of NSBRs removed: " + j);

        return NSBRs;

    }

    public static Double[] scoreReport(Instances dataset, Map<String, Double> scored_words){

        Double[] score = new Double[dataset.numInstances()];


        for(int i = 0; i < dataset.numInstances(); i++){
            score[i] = 0.0;
            List<Double> M = new ArrayList<>(); 		// need to collect the scorei per word
            List<Double> _M = new ArrayList<>();		// collect the inverse (1-scorei)
            for (Map.Entry<String, Double> entry : scored_words.entrySet()) {
                //System.out.println(entry.getKey() + " = " + entry.getValue());

                if (dataset.instance(i).toString().contains(entry.getKey())) {
                    M.add(entry.getValue());
                    _M.add(1-entry.getValue());

                    /*if(score[i] == 0.0){
                        score[i] = entry.getValue();
                    } else {
                        score[i] *= entry.getValue();
                    }*/
                }
            }
            double pm = prod(M);					// do the product
            double p_m = prod(_M);					// do the product
            score[i] = pm / (pm + p_m);				// final score
            /*if(score[i] > 0.0) {
                score[i] = score[i] / (score[i] + (1 - score[i]));
            }*/
            System.out.println(i+" | "+score[i]);
        }

        return score;
    }

    private static double prod(List<Double> score) {
        double p = 1.0;
        for(double e : score)
            p *= e;

        return p;
    }

    public static Map<String, Double> scoreWords(Instances SBRs, Instances NSBRs, String keywordsFile){
        Map<String, Double> scoredWords = new HashMap<String, Double>();

        String[] keywords = new String[100];

        BufferedReader br = null;
        String line = "";
        int i = 0;

        try {
            br = new BufferedReader(new FileReader(keywordsFile));
            while ((line = br.readLine()) != null) {

                String[] toks = line.split(",");
                keywords[i] = toks[0];
                i++;
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        Double score = 0.0;
        Double probabilitySBRs = 0.0;
        Double probabilityNSBRs = 0.0;

        for(String w : keywords){

            probabilitySBRs = Math.min(1, probabilitySBRs(SBRs, w));
            probabilityNSBRs = Math.min(1, probabilityNSBRs(NSBRs, w));

            // Using the support function of squaring the numerator
            score = Math.max(0.01, Math.min(0.99, (Math.pow(probabilitySBRs, 2)/(probabilitySBRs + probabilityNSBRs))));
            scoredWords.put(w, score);
            //System.out.println(w+" = "+score);
        }

        return scoredWords;
    }

    //
    public static Double probabilitySBRs(Instances dataset, String word){
        Double p = 0.0;

        for(int i = 0; i < dataset.numInstances(); i ++){
            if(dataset.instance(i).toString().contains(word)){
                //System.out.println(dataset.instance(i).toString()+" : "+word);
                p++;
            }
        }

        /*if(p == 0.0){
            return 0.01;
        }*/

        double pr = Math.min(1, ((double)p)/((double)dataset.size()));

        return pr; //p/dataset.numInstances();
    }

    public static Double probabilityNSBRs(Instances dataset, String word){
        Double p = 0.0;

        for(int i = 0; i < dataset.numInstances(); i ++){
            if(dataset.instance(i).toString().contains(word)){
                p++;
            }
        }

        /*if(p == 0.0){
            return 0.99;
        }*/

        double pr = Math.min(1, ((double)p)/((double)dataset.size()));
        return pr; //p/dataset.numInstances();
    }

    public static Instances loadDataset(String path) {
        Instances dataset = null;
        try {
            dataset = ConverterUtils.DataSource.read(path);
            if (dataset.classIndex() == -1) {
                dataset.setClassIndex(dataset.numAttributes() - 1);
            }
        } catch (Exception ex) {
            Logger.getLogger(DerbyClassifier.class.getName()).log(Level.SEVERE, null, ex);
        }

        return dataset;
    }

    public static Instances merge(Instances data1, Instances data2)
            throws Exception
    {
        // Check where are the string attributes
        int asize = data1.numAttributes();
        boolean strings_pos[] = new boolean[asize];
        for(int i=0; i<asize; i++)
        {
            Attribute att = data1.attribute(i);
            strings_pos[i] = ((att.type() == Attribute.STRING) ||
                    (att.type() == Attribute.NOMINAL));
        }

        // Create a new dataset
        Instances dest = new Instances(data1);
        dest.setRelationName(data1.relationName() + "+" + data2.relationName());

        ConverterUtils.DataSource source = new ConverterUtils.DataSource(data2);
        Instances instances = source.getStructure();
        Instance instance = null;
        while (source.hasMoreElements(instances)) {
            instance = source.nextElement(instances);
            dest.add(instance);

            // Copy string attributes
            for(int i=0; i<asize; i++) {
                if(strings_pos[i]) {
                    dest.instance(dest.numInstances()-1)
                            .setValue(i,instance.stringValue(i));
                }
            }
        }

        return dest;
    }
}