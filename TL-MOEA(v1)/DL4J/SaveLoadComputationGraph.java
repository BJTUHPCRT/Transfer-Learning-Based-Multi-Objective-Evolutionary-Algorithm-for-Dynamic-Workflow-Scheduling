package mylesson2;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;

/**
 * A very simple example for saving and loading a ComputationGraph
 *
 * @author Alex Black
 */
public class SaveLoadComputationGraph {

    public static void main(String[] args) throws Exception {
        //Define a simple ComputationGraph:
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
            .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(0.01, 0.9))
            .graphBuilder()
            .addInputs("in")
            .addLayer("layer0", new DenseLayer.Builder().nIn(4).nOut(3).activation(Activation.TANH).build(), "in")
            .addLayer("layer1", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX).nIn(3).nOut(3).build(), "layer0")
            .setOutputs("layer1")
            .backprop(true).pretrain(false).build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();


        //Save the model
        File locationToSave = new File("model/MyComputationGraph.zip");       //Where to save the network. Note: the file is in .zip format - can be opened externally
        boolean saveUpdater = true;                                             //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
        ModelSerializer.writeModel(net, locationToSave, saveUpdater);

        //Load the model
        ComputationGraph restored = ModelSerializer.restoreComputationGraph(locationToSave);


        System.out.println("Saved and loaded parameters are equal:      " + net.params().equals(restored.params()));
        System.out.println("Saved and loaded configurations are equal:  " + net.getConfiguration().equals(restored.getConfiguration()));
    }

}
