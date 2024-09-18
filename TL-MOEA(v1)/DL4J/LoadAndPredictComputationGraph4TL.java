package mylesson2;

import org.bytedeco.javacpp.cuda.double1;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import TL4DMOEA.Tool;

import java.io.File;
import java.io.IOException;

/**
 * A very simple example for saving and loading a ComputationGraph
 *
 * @author Alex Black
 */
public class LoadAndPredictComputationGraph4TL {

	public ComputationGraph myCG = null;
	public DataSet testData = null;
	public static double proportionTraiAndTest = 0.00000001;//表示训练的比例为0
	public int nSamples = 0;
	public int numInput = 0;
	public int numOutputs = 0;
	public double[][] originalFeature = null;
    public static void main(String[] args) throws Exception {
//        //Define a simple ComputationGraph:
//        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
//            .weightInit(WeightInit.XAVIER)
//                .updater(new Nesterovs(0.01, 0.9))
//            .graphBuilder()
//            .addInputs("in")
//            .addLayer("layer0", new DenseLayer.Builder().nIn(4).nOut(3).activation(Activation.TANH).build(), "in")
//            .addLayer("layer1", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX).nIn(3).nOut(3).build(), "layer0")
//            .setOutputs("layer1")
//            .backprop(true).pretrain(false).build();
//
//        ComputationGraph net = new ComputationGraph(conf);
//        net.init();
//
//
//        //Save the model
//        File locationToSave = new File("model/MyComputationGraph.zip");       //Where to save the network. Note: the file is in .zip format - can be opened externally
//        boolean saveUpdater = true;                                             //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
//        ModelSerializer.writeModel(net, locationToSave, saveUpdater);
//
//        //Load the model
//        ComputationGraph restored = ModelSerializer.restoreComputationGraph(locationToSave);
//
//
//        System.out.println("Saved and loaded parameters are equal:      " + net.params().equals(restored.params()));
//        System.out.println("Saved and loaded configurations are equal:  " + net.getConfiguration().equals(restored.getConfiguration()));
    }
    
    //构造
    public LoadAndPredictComputationGraph4TL() {
    	myCG = null;
    	testData = new DataSet();
    	nSamples = Tool.netInputs.size();
        numInput = Tool.TaskNum; //相当于维度
        numOutputs = Tool.TaskNum;//相当于维度
    	originalFeature = new double[nSamples][numInput];
    }
    
    //加载计算图
    public void loadCompuGraph(String fileName) throws IOException {
    	File locationToSave = new File(fileName); 
    	myCG = ModelSerializer.restoreComputationGraph(locationToSave);
    }

    
    //通过模型预测，Elite-led Transfer Learning Strategy的一部分，还包括train net, update memory, add noise
    public double[][] predictData(String fileName) throws IOException {
    	
    	//加载上一个环境的时刻
    	LoadOriginalData.loadTestFeatureData(originalFeature,fileName);
        splitData(1);//1表示归一化
        INDArray fe = testData.getFeatures();

        
    	INDArray out[] = myCG.output(fe);//留意这里的out[0]就是一个矩阵 out里只有一个矩阵
        double douMatrix[][] = new double[out[0].rows()][out[0].columns()];
        for(int i=0;i<out[0].rows();i++) {
        	for(int j=0; j<out[0].columns();j++) {
        		douMatrix[i][j] = out[0].getDouble(i, j);
        	}
        }
    	return douMatrix;
    }
    
    //划分数据集
    public void splitData (int isNorma) {
    	DataSet dataSet = null;
    	if (isNorma == 1) {
    		dataSet = dataNormalization();
    	} else {
    		dataSet = noDataNormalization();
		}
        
		SplitTestAndTrain testAndTrain = dataSet.splitTestAndTrain(proportionTraiAndTest);
        
        testData = testAndTrain.getTest();
        
    }
    
    //数据归一化
    public DataSet dataNormalization() {
        INDArray featureNdarray = Nd4j.create(originalFeature);
        
        //两个凑一起，但不用特征
        DataSet dataSet = new DataSet(featureNdarray, featureNdarray);
        
        //归一化
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(dataSet);
        normalizer.transform((dataSet));        
        return dataSet;
    }
    
    //数据不归一化
    public DataSet noDataNormalization() {
    	 //两个凑一起，但不用特征
        INDArray featureNdarray = Nd4j.create(originalFeature);
        INDArray labelsNdarray = Nd4j.create(originalFeature);
        DataSet dataSet = new DataSet(featureNdarray, labelsNdarray);
        return dataSet;
    }
}
