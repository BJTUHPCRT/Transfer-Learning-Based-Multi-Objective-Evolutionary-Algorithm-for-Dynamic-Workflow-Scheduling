package mylesson2;

import org.bytedeco.javacpp.cuda.double1;
import org.bytedeco.javacpp.cuda.int1;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.ExponentialSchedule;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

//import baidudianshi.DataSetAccuracyLossCalculator;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.deeplearning4j.util.ModelSerializer;


import java.io.File;
import java.io.IOException;

/**
 * Created by Joe on 2018/5/20.
 * 多元到多元回归
 */
import TL4DMOEA.Tool;

//不同于MultiToMultiRegression.java，这里使用的是ComputationGraphConfiguration设计网络，主要是用于后面迁移学习，可保存计算图。
//MultiToMultiRegression.java可用 MultiLayerNetwork 设计网络，可用于保存模型，但是在迁移学习时不可修改网络，例如冻结什么的。
public class MultiToMultiRegressionUsingComGraph4TL {
	private static final int seed = 12345;
    //对于每个miniBatch的迭代次数
//    private static final int iterations = 10;
    //epoch数量(全部数据的训练次数)
    private int nEpochs = Tool.nEpochs;

    int numHiddenNodes = 0;
    //一共生成多少样本点
    int nSamples = 0;
    //Batch size: i.e., each epoch has nSamples/batchSize parameter updates
    int batchSize = 0;
    //网络模型学习率
    private static final double learningRate = 0.01;//0.01
    //动量的设置
    private static final double momentum = 0.9;
    //训练集和测试集的比例
    private static double proportionTraiAndTest = 0.99999999;

    int numInput = 0; //相当于维度
    int numOutputs = 0;//相当于维度
    
    double[][] originalFeature = null;
    double[][] originalLabels = null;
    double[][] norFeature = null;
    double[][] norLabels = null;
    
    DataSet trainingData = null;
    DataSet testData = null;
    
    private static final Random rng = new Random(seed);
    
    
	public MultiToMultiRegressionUsingComGraph4TL() {
		
		    batchSize = Tool.batchSize;
	    	nSamples = Tool.netInputs.size();
	        numInput = Tool.TaskNum;
	        numOutputs = Tool.TaskNum;
	        originalFeature = new double[nSamples][numInput];
	        originalLabels = new double[nSamples][numOutputs];
	        
	        norFeature = new double[nSamples][numInput];
	        norLabels = new double[nSamples][numOutputs];
	        numHiddenNodes = Tool.TaskNum*2;
	        
	        trainingData = new DataSet();
	        testData = new DataSet();
	    }

   
    public static void main(String[] args) throws IOException {
       
    }

    
    //训练神经网络
    public  void trainingNeuralNetwork() {
    	DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);

        Nd4j.setDataType(DataBuffer.Type.DOUBLE);

        System.out.println(Nd4j.dataType());

        //加载数据
        try {
			loadData();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        splitData(1);//1表示归一化

        
        //这里统一设置下参数
        Map<Integer, Double> lrSchedule = new HashMap<>();
        if (Tool.TaskNum <= 100 ){
            lrSchedule.put(0, 0.1);
//            lrSchedule.put(150, 0.01);
            nEpochs = 200;
        } else {
            lrSchedule.put(0, 0.1);
//            lrSchedule.put(350, 0.01);
            nEpochs = 400;
        }
        
        
      //Define a simple ComputationGraph:
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
        	.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs((ISchedule) new MapSchedule(ScheduleType.EPOCH, lrSchedule),momentum))
            .graphBuilder()
            .addInputs("in")
            .addLayer("layer0", new DenseLayer.Builder().nIn(numInput).nOut(numHiddenNodes).activation(Activation.RELU).build(), "in")
            .addLayer("layer1", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY).nIn(numHiddenNodes).nOut(numOutputs).build(), "layer0")
            .setOutputs("layer1")
            .backprop(true).pretrain(false).build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();
        //设置监听器
        net.setListeners(new ScoreIterationListener(1));

        //训练
        DataSetIterator iterator = getTrainingData(batchSize);
        for( int i=0; i<nEpochs; i++ ){
        	iterator.reset();
            net.fit(getTrainingData(batchSize));
        }

        //输出网络配置
        Logger log = LoggerFactory.getLogger(ComputationGraphConfiguration.class);
        log.info(net.summary());
        
        //输出当前模型的参数
//        System.out.println("当前网络参数：");
//        Map<String, INDArray> params = net.paramTable();
//        params.forEach((key, value) -> System.out.println("key:" + key +", value = " + value));
//        
        
        //保存计算图
        File locationToSave = new File("G:\\eclipseWorkspace\\2.tldmoea4wrokflowScheduling2\\dl4jtransferLearning\\modeltl\\MyMultiLayerNetwork_"+Tool.workflowIndex + "_" + Tool.env + ".zip");       //Where to save the network. Note: the file is in .zip format - can be opened externally
        boolean saveUpdater = true;                                             //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
        try {
			ModelSerializer.writeModel(net, locationToSave, saveUpdater);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
         
    }
    
    
    //加载数据
    public void loadData() throws IOException {
    	LoadOriginalData4TL loadTata4TL = new LoadOriginalData4TL();
    	loadTata4TL.loadData(nSamples, numInput, numOutputs, originalFeature,originalLabels);
	
    }


    
    //数据不归一化
    public DataSet noDataNormalization() {
        INDArray featureNdarray = Nd4j.create(originalFeature);
        INDArray labelsNdarray = Nd4j.create(originalLabels);
        DataSet dataSet = new DataSet(featureNdarray, labelsNdarray);
        return dataSet;
    }
    
    
    //数据归一化
    public  DataSet dataNormalization() {
        INDArray featureNdarray = Nd4j.create(originalFeature);
        INDArray labelsNdarray = Nd4j.create(originalLabels);
        DataSet dataSet = new DataSet(featureNdarray, labelsNdarray);
        
        //归一化
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(dataSet);
        normalizer.transform((dataSet));
        
        
        //放回nor数组中 #OK
        INDArray featureNdarray1 = dataSet.getFeatures();
        INDArray labelsNdarray1 = dataSet.getLabels();
        NdIndexIterator iter1 = new NdIndexIterator(nSamples, numInput);
        int index1 = 0;
        while (iter1.hasNext()) {
            long[] nextIndex = iter1.next();
            double nextVal = featureNdarray1.getDouble(nextIndex);
            norFeature[index1/numInput][index1%numInput] = nextVal;
            index1 = index1 + 1;
        }
        
        NdIndexIterator iter2 = new NdIndexIterator(nSamples, numOutputs);
        int index2 = 0;
        while (iter2.hasNext()) {
            long[] nextIndex = iter2.next();
            double nextVal = labelsNdarray1.getDouble(nextIndex);
            norLabels[index2/numOutputs][index2%numOutputs] = nextVal;
            index2 = index2 + 1;
        }
        
        return dataSet;
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
        trainingData = testAndTrain.getTrain();
        testData = testAndTrain.getTest();
    }
    
    
    //得到训练数据
    public DataSetIterator getTrainingData(int batchSize) {
        List<DataSet> allData = trainingData.asList();
        return new ListDataSetIterator(allData,batchSize);
    }
    
    //得到测试集
    public DataSetIterator getTestData() {
        List<DataSet> allData = testData.asList();
        return new ListDataSetIterator(allData);
    }



    
    
}
