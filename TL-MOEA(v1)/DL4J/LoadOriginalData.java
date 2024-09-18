package mylesson2;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.security.PublicKey;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.bytedeco.javacpp.cuda.double1;
import org.bytedeco.javacpp.cuda.int1;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import TL4DMOEA.Tool;
 




public class LoadOriginalData {
	int nSample = 0;
	static final int seed = 12345;
    int numInput = 0;
    int numOutput = 0;
    double[][] feature = null;
    double[][] labels = null;
    String featureName = null;
	String labelName = null;
   
	public static void main(String[] args) throws IOException {
		
		
		//loadData(nSample,numInput,numOutput,feature,labels);
	
	}
	
	
	//构造函数
	public LoadOriginalData( ) {
		featureName = new String("G:\\eclipseWorkspace\\2.tldmoea4wrokflowScheduling2\\dl4jtransferLearning\\datann\\originalFeature_"+Tool.workflowIndex + ".txt");
		labelName = new String("G:\\eclipseWorkspace\\2.tldmoea4wrokflowScheduling2\\dl4jtransferLearning\\datann\\originalLabel_"+Tool.workflowIndex + ".txt");
		nSample = Tool.nSample;
	    feature = new double[nSample][numInput];
	    labels = new double[nSample][numOutput];
	    numInput = Tool.TaskNum;
	    numOutput = Tool.TaskNum;
	}
	
	public static void produceData (int nSamples,int numInput,int numOutput) throws IOException {
		
	}
	
	
	//加载数据
	public void loadData (int nSamples,int numInput,int numOutputs,double feature[][] ,double labels[][] ) throws IOException {
		
		
		File fileFeature = new File(featureName);
		File fileLabel = new File(labelName);
		

		
		BufferedReader brf = new BufferedReader(new FileReader(fileFeature));
		BufferedReader brl = new BufferedReader(new FileReader(fileLabel));
		
		String st;
		int i=0;
		int j;
		int z = 0;
		while ((st = brf.readLine()) != null) {
			z = z + 1;
		}
		System.out.println("z :" + z);
		System.out.println("featureLen:" + feature.length);
		System.out.println("featureName: " + featureName);
		
		while ((st = brf.readLine()) != null) {
			j = 0;
			for (String retval: st.split(" ")){
				feature[i][j] = Double.valueOf(retval.toString());
				j = j + 1;
				
			}
			i = i + 1;
		}	
		
		//label
		i=0;
		
		while ((st = brl.readLine()) != null) {
			j = 0;
			for (String retval: st.split(" ")){
				labels[i][j] = Double.valueOf(retval.toString());
				j = j + 1;
				
			}
			i = i + 1;
			
		}	
		
//		System.out.println(feature[12][1]);
//		System.out.println(labels[12][1]);
	}


	    //加载数据  加载上一个环境的数据用于输入
		public static void loadTestFeatureData (double feature[][] ,String featureName ) throws IOException {
			
			
			File fileFeature = new File(featureName);
			
			
			BufferedReader brf = new BufferedReader(new FileReader(fileFeature));
			
			
			String st;
			int i=0;
			int j;
			while ((st = brf.readLine()) != null) {
				j = 0;
				for (String retval: st.split(" ")){
					feature[i][j] = Double.valueOf(retval.toString());
					j = j + 1;
					
				}
				i = i + 1;
				
				
			}	

//			System.out.println(feature[12][1]);
//			System.out.println(labels[12][1]);
		}

	
	//加载数据
			public void loadDataX (int nSamples,int numInput,int numOutputs,double feature[][] ,double labels[][] ) throws IOException {
				
				try {
					produceData (nSamples,numInput,numOutputs);
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				File fileFeature = new File(featureName);
				File fileLabel = new File(labelName);
				
				BufferedReader brf = new BufferedReader(new FileReader(fileFeature));
				BufferedReader brl = new BufferedReader(new FileReader(fileLabel));
				
				String st;
				int i=0;
				int j;
				while ((st = brf.readLine()) != null) {
					j = 0;
					for (String retval: st.split(" ")){
						feature[i][j] = Double.valueOf(retval.toString());
						j = j + 1;
						
					}
					i = i + 1;
				}	
				
				//label
				i=0;
				
				while ((st = brl.readLine()) != null) {
					j = 0;
					for (String retval: st.split(" ")){
						labels[i][j] = Double.valueOf(retval.toString());
						j = j + 1;
						
					}
					i = i + 1;
					
				}	
				
//				System.out.println(feature[12][1]);
//				System.out.println(labels[12][1]);
			}



	
	
}







