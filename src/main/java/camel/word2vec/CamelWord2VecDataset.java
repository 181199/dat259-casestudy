package camel.word2vec; /**
 * 
 */

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.stopwords.StopWords;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.text.NumberFormat;
import java.util.Arrays;
import java.util.Collection;
import java.util.Locale;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * @author tdoy
 *
 */
public class CamelWord2VecDataset {

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {

        File gModel = new File("/Users/anja/Desktop/DAT259/GoogleNews-vectors-negative300.bin.gz");
        Word2Vec vec = WordVectorSerializer.readWord2VecModel(gModel);
        writeWord2VectorsGoogle("./dataset/camel_mod.csv", "./dataset/camel_wordvec.csv", vec);
	}

    public static void writeWord2VectorsGoogle(String infile, String outfile, Word2Vec word2vec) {

        try(BufferedWriter bw = new BufferedWriter(new FileWriter(outfile))){

            try(BufferedReader br = new BufferedReader(new FileReader(infile))){
                String line = "";
                int index = 0;
                while((line=br.readLine())!=null) {

                    String[] toks = line.split(";");
                    Collection<String> tokens = normalizeText(toks[2]);
                    INDArray invector = getVectorGoogle(tokens, word2vec);

                    String rowvecpluslabel = getWordVectorsAndLabel(invector, index);
                    rowvecpluslabel = rowvecpluslabel+toks[0]; 		// append the label at the end
                    bw.write(rowvecpluslabel+"\n");
                    index++;
                }

            }catch(Exception e) {
                e.printStackTrace();
            }

        }catch(Exception e) {
            //
        }
    }

   private static String getWordVectorsAndLabel(INDArray vec, int index) {
    	
	   String rowvecs = "";
    	for(int i=0; i<vec.columns()-1; i++) {
    		NumberFormat nf = NumberFormat.getInstance(new Locale("en", "US"));
    		nf.setMaximumFractionDigits(6);
    		String val = nf.format(vec.getRow(0).getDouble(i));
    		rowvecs += val+";";
    	}
    	
    	return rowvecs;
    	
    }

    public static Collection<String> normalizeText(String text){
        Pattern charsPunctuationPattern = Pattern.compile("[\\d:,\"\'\\`\\_\\|?!\n\r@;]+");
        String input_text = charsPunctuationPattern.matcher(text.trim().toLowerCase()).replaceAll("");
        input_text = input_text.replaceAll("\\{.*?\\}", "");
        input_text = input_text.replaceAll("\\[.*?\\]", "");
        input_text = input_text.replaceAll("\\(.*?\\)", "");
        input_text = input_text.replaceAll("[^A-Za-z0-9(),!?@\'\\`\"\\_\n]", " ");
        input_text = input_text.replaceAll("[/]"," ");
        input_text = input_text.replaceAll(";"," ");
        Collection<String> labels = Arrays.asList(input_text.split(" ", 299)).parallelStream().filter(label->label.length()>0).collect(Collectors.toList());
        labels = labels.parallelStream().filter(label ->  !StopWords.getStopWords().contains(label.trim())).collect(Collectors.toList());
        return labels;
    }

    public static INDArray getVectorGoogle(Collection<String> labels, Word2Vec word2Vec){
        double sentence_vector[] = new double[300];
        // initializing the vector
        Arrays.fill(sentence_vector,0d);
        int i=0;
        for(String word: labels)
        {
            // make vector embeddings for a one word at a time
            double[] word_vec = word2Vec.getWordVector(word);
            // array sum of vectors
            sentence_vector = (i==0) ? word_vec : double_array_sum(word_vec,sentence_vector);
            ++i;
        }

        // average of array
        return Nd4j.create(double_array_avg(sentence_vector, labels.size()));
    }

    public static double[] double_array_sum(double[] word_vec, double[] sentence_vector){
        double[] sum = new double[word_vec.length];

        for (int i = 0; i < word_vec.length; i++) {
            sum[i] = word_vec[i] + sentence_vector[i];
        }
        return sum;
    }

    public static double[] double_array_avg(double[] sentence_vector, int size){
        double[] avg = new double[sentence_vector.length];
        for (int i=0; i < size; i++) {
            avg[i] = sentence_vector[i]/size;
        }

        return avg;
    }

}
