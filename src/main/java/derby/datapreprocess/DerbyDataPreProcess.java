package derby.datapreprocess; /**
 * 
 */

import weka.core.Instances;
import weka.core.converters.CSVSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.*;

/**
 * @author tdoy
 *
 */
public class DerbyDataPreProcess {

	/**
	 * 
	 */
	public DerbyDataPreProcess() {
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		
		loadFileToARFF();
		// Get data
		Instances data = new Instances(new FileReader("./dataset/derby.arff"));
		data.setClassIndex(2);
		
		// apply the StringToWordVector
		StringToWordVector filter = new StringToWordVector();
	    filter.setInputFormat(data);
	    filter.setWordsToKeep(10);							// just testing
	    filter.setLowerCaseTokens(true);
	    filter.setTFTransform(false);
	    filter.setIDFTransform(true);
	    
	    Instances dataFiltered = Filter.useFilter(data, filter);
	    //System.out.println("\n\nFiltered data:\n\n" + dataFiltered);
	    
	    instancesToCsv(dataFiltered, "./dataset/derby_tokenized.csv");

	}
	
	private static void instancesToCsv(Instances data, String path) throws IOException {
		
		CSVSaver saver = new CSVSaver();
		saver.setFieldSeparator(";");
	    saver.setInstances(data);
	    saver.setFile(new File(path));
	    saver.writeBatch();
	}
	
	private static void loadFileToARFF() {
		// create a your own arff file from the polarity data. This must contain both the neg and the pos dataset
		File out = new File("./dataset/derby.arff");
		File fin = new File("./dataset/derby_mod.csv");
		
		String header = "@RELATION 'derby-security'\n" +
				"\n" + 
				"@ATTRIBUTE summary string\n" +
                "@ATTRIBUTE description string\n" +
                "@ATTRIBUTE class-att {0,1} nominal\n" +
				"\n" + 
				"@data\n";
		
		writeHeader(header, out.getAbsolutePath());
		readWriteFile(fin.getAbsolutePath(),out.getAbsolutePath());
	}
	
	private static void writeHeader(String header, String outfile) {
		
		try(BufferedWriter bw = new BufferedWriter(new FileWriter(outfile))){
			
			bw.write(header);
		}catch(Exception e) {
			//
		}
	}
	
	private static void readWriteFile(String path, String outfile) {
		
		try(BufferedWriter bw = new BufferedWriter(new FileWriter(outfile, true))){
			
			try(BufferedReader br = new BufferedReader(new FileReader(path))){
				String line = "";
				
				int i=0;
				while((line=br.readLine())!=null) {
					if(i++==0) continue;									// ignore header in file
					line = line.replace("\"", "");
					line = line.replace("\\", "");
					String[] cols = line.split(";");					// this is separated by semicolon
					bw.write("\""+cols[1]+"\","+"\""+cols[2]+"\","+cols[0]+"\n");			// cols[4]=bugreport: cols[2]=type-of-bug (sec/no-sec)
				}
				
			}catch(Exception e) {
				//
			}
		}catch(Exception e) {
			//
		}		
	}

}
