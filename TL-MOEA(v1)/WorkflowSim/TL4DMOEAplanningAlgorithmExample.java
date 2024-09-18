package TL4DMOEA;


import java.io.File;


import java.io.FileWriter;
import java.io.IOException;
import java.security.PublicKey;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import org.uma.jmetal.algorithm.multiobjective.tl4dmoea.FastNonDominatedSortRanking2;
import org.uma.jmetal.algorithm.multiobjective.tl4dmoea.Individual;
import org.uma.jmetal.algorithm.multiobjective.tl4dmoea.TL4DMOEA;
import org.uma.jmetal.algorithm.multiobjective.tl4dmoea.MyUtils;
import org.cloudbus.cloudsim.CloudletSchedulerSpaceShared;
import org.cloudbus.cloudsim.Log;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.lists.VmList;
import org.workflowsim.CondorVM;
import org.workflowsim.Job;
import org.workflowsim.Task;
import org.workflowsim.WorkflowDatacenter;
import org.workflowsim.WorkflowEngine;
import org.workflowsim.WorkflowParser;
import org.workflowsim.WorkflowPlanner;
import org.workflowsim.examples.WorkflowSimBasicExample1;
import org.workflowsim.utils.ClusteringParameters;
import org.workflowsim.utils.OverheadParameters;
import org.workflowsim.utils.Parameters;
import org.workflowsim.utils.ReplicaCatalog;


import mylesson2.MultiToMultiRegressionTransferLearningUsingComGraph;
import mylesson2.MultiToMultiRegressionUsingComGraph4TL;


public class TL4DMOEAplanningAlgorithmExample extends WorkflowSimBasicExample1{


	public static void main1(String[] args) {
		
	}

    public static void main(String[] args) {

    	
    	//一些实验设置
    	int experimentNum = 1;
    	int workflowNum = 20;
    	
		/**************先进行一个初始化的参数，用于tl4dmoea配置相应的参数，**************************/
		try {
			Tool.configureParameters(Tool.workflowIndex);
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		InitCloudParameters();
		
    	for(int workflowIndex = 0; workflowIndex < workflowNum; workflowIndex++) {
    		Tool.workflowIndex = workflowIndex;
    		//判断是否存在
    		
    		boolean exit = false;
    		for (int i = 0; i < Tool.subWorkflows.length; i++) {
    			if (workflowIndex == Tool.subWorkflows[i]) {
    				exit = true;
    			}
    		}
    		if ( exit == false) {
    			continue;
    		}
    		for(int experIndex = 0; experIndex < experimentNum; experIndex++) {
    			Tool.experIndex = experIndex;
    			
    			Tool.memory = new ArrayList<>();
    			try {
    				Tool.configureParameters(Tool.workflowIndex);
    			} catch (IOException e1) {
    				// TODO Auto-generated catch block
    				e1.printStackTrace();          
    			}
    			InitCloudParameters();
    			
    			long startTime = System.currentTimeMillis();
    			List<Integer> LastPoolList = new  ArrayList<>();
    			for(int env = 0; env < Tool.envNum; env++) {
    				Tool.isDiversityLearning = 0;
//    				Tool.batchSize = (Tool.netInputs.size())/(5);
    				Tool.batchSize = 32;
               		TL4DMOEA tl4dmoea = new TL4DMOEA();
               		tl4dmoea.generatePoints();
               		
               		Tool.points = tl4dmoea.obtainPoints();//参考点一直不变，所以不做复制了
    				Tool.env = env;
    	
    				System.out.println("第 " + env + " 次环境");
    				
    				
        	    	try {
        	    		//配置和初始化相应参数
        	    		Tool.configureParameters(Tool.workflowIndex);
        	    		
        	    		
        				//生成资源池 基本的资源池
        				List<Integer> pooList = new  ArrayList<>();
        				if (Tool.dynamicType[env] == 0) {
        					for (int po = 0; po < Tool.VmNum; po++) {
            					pooList.add(new Integer(po));
            				}
        					Tool.resourcePool = pooList;
        				} else  {
        					//这一部分包括了remove 和 add 相对于上一个环境
        					//相当于完全从DynVMInfoList读取数据，相较于上一个环境的VMList，相当于有remove也有add，
        					//记住startPos----> endPos这里面的数据是要保存下来的，而不是删除。
        					//每次的资源池数目是不固定的。
        					//VMInfoList是保存每个VM最基本的数据，用于计算能耗
        					//resourcePool是存储一些当前的VM序号
        					int startPos = -1;
        					if (Tool.VmNum == 18) {
        						startPos = env * Tool.VmNum + Tool.severity[env];
        					} else if(Tool.VmNum == 60) {
        						startPos = env * Tool.VmNum + Tool.severity2[env];
        					} else {
        						//
        					}
        					
        					int endPos = (env+1) * Tool.VmNum;
        					for(int s = startPos; s < endPos; s++) {
        						pooList.add(Tool.DynVMInfoList.get(0).dynamicVMs.get(s));
        					}
        					LastPoolList.clear();
        					LastPoolList.addAll(pooList);
        					
        					//update
        					int Index = 0;
        					int updateLen = 0;
        					if (env == 0 || env == 1) {
        						Index = 0;
        					} else if (env == 2 || env == 3) {
        						Index = 1;
        					} else if (env == 4 || env == 5) {
        						Index = 2;
        					} else if (env == 6 || env == 7) {
        						Index = 9;
        					} else {
        						Index = 12;
        					}
        					
        					double everyvCPUPower = 10.42; //1000.0/96
        					if (Tool.VmNum == 18) {
//        						updateLen = Tool.severity[env];
        						if (Tool.severity[env] == 3) {
        							updateLen = 1;
        						} else if (Tool.severity[env] == 6 ) {
        							updateLen = 3;
        						} else {
        							//
        						}
        					} else if ( Tool.VmNum == 60) {
        						updateLen = Tool.severity2[env]; 
        						
        					}
        					
        					//选择一些vm进行更新 
        					VMInfo bestVm  = new VMInfo();
        					bestVm = Tool.VMInfoList.get(Index);	
        					for(int up = 0; up < updateLen; up++) {

        						Tool.VMInfoList.get(LastPoolList.get(up).intValue()).vCPU = bestVm.vCPU;
        						Tool.VMInfoList.get(LastPoolList.get(up).intValue()).memory = bestVm.memory;
        						Tool.VMInfoList.get(LastPoolList.get(up).intValue()).bandWidth = bestVm.bandWidth;
        						Tool.VMInfoList.get(LastPoolList.get(up).intValue()).bandwidthCost = bestVm.bandwidthCost;
        						Tool.VMInfoList.get(LastPoolList.get(up).intValue()).mips = bestVm.mips;
        						Tool.VMInfoList.get(LastPoolList.get(up).intValue()).storage = bestVm.storage;
        						Tool.VMInfoList.get(LastPoolList.get(up).intValue()).storageCost = bestVm.storageCost;
        						Tool.VMInfoList.get(LastPoolList.get(up).intValue()).vmCost = bestVm.vmCost;
        					}
        					
        					//更新功耗  所有
        					int vCPUArray[] = new int[Tool.hostNum];
        					for(int h = 0; h < Tool.hostNum; h++) {
        						vCPUArray[h] = 0;
        					}
        					for (int v = 0; v < Tool.VmNum; v++) {
        						vCPUArray[Tool.VMInfoList.get(v).hostId - 1] = vCPUArray[Tool.VMInfoList.get(v).hostId - 1] + Tool.VMInfoList.get(v).vCPU;
        					}
        					for(int h = 0; h < Tool.hostNum; h++) {
        						Tool.HostInfoList.get(h).powerMax = vCPUArray[h]*everyvCPUPower;
        					}
        					
        					Tool.resourcePool = LastPoolList;
        					
        				}
        				InitCloudParameters();
        	    	
        	    		
	       	    		
	       	    		FileWriter fw1 = new FileWriter("G:\\eclipseWorkspace\\2.tldmoea4wrokflowScheduling2\\file\\tl4dmoea\\tl4dmoea_initFitness_"+Tool.workflowIndex+"_" + Tool.experIndex + "_"+ Tool.env + ".txt");
	       	    		FileWriter fw2 = new FileWriter("G:\\eclipseWorkspace\\2.tldmoea4wrokflowScheduling2\\file\\tl4dmoea\\tl4dmoea_initPop_"+Tool.workflowIndex+"_" + Tool.experIndex + "_"+ Tool.env + ".txt");
	               		FileWriter fw3 = new FileWriter("G:\\eclipseWorkspace\\2.tldmoea4wrokflowScheduling2\\file\\tl4dmoea\\tl4dmoea_pf_"+Tool.workflowIndex+"_"+ Tool.experIndex + "_" + Tool.env + ".txt");
	               		FileWriter fw4 = new FileWriter("G:\\eclipseWorkspace\\2.tldmoea4wrokflowScheduling2\\file\\tl4dmoea\\tl4dmoea_ps_"+Tool.workflowIndex+"_" + Tool.experIndex + "_"+ Tool.env + ".txt");
	               			               		              		
	               		//保存样本
	               		FileWriter fw7 = new FileWriter("G:\\eclipseWorkspace\\2.tldmoea4wrokflowScheduling2\\dl4jtransferLearning\\datatl\\originalFeature_"+Tool.workflowIndex  + "_" + env + ".txt");
	               		FileWriter fw8 = new FileWriter("G:\\eclipseWorkspace\\2.tldmoea4wrokflowScheduling2\\dl4jtransferLearning\\datatl\\originalLabel_"+Tool.workflowIndex + "_" +  env + ".txt");
//
//	               		
//	               		
        	    		if(env != 0) {    
        	    			tl4dmoea.obtainMemory();
        	    			tl4dmoea.reInitPop();//重新初始化，符合条件
        	    			tl4dmoea.evaluatePop2(tl4dmoea.population);
        	    			
        	    			
            	          //保存环境变化后第一次迭代的数据
            				for(int i = 0; i < tl4dmoea.population.size(); i++) {
            					for(int j = 0; j < Tool.nobj; j++) {
            						fw1.write(tl4dmoea.population.get(i).getObjectives(j)+" ");
            					}	
            					fw1.write("\n");
            					for(int k = 0; k < Tool.TaskNum; k++) {
            						fw2.write(tl4dmoea.population.get(i).getVariables(k)+" ");
            					}
            					fw2.write("\n");
            				}
            	            fw1.close();
            	            fw2.close();
        	    		} else {
        	    			tl4dmoea.obtainMemory();
        	    			tl4dmoea.initPop();			
        	    			tl4dmoea.evaluatePop2(tl4dmoea.population);
        	    			
        	    			
        	    			Tool.lastEnvirPop.clear();
        	    			Tool.lastEnvirPop.addAll(tl4dmoea.population);
      	    			
        	    			//保存最初始数据
            				for(int i = 0; i < tl4dmoea.population.size(); i++) {
            					for(int j = 0;j < tl4dmoea.nobj; j++) {
            						fw1.write(tl4dmoea.population.get(i).getObjectives(j)+" ");
            					}	
            					fw1.write("\n");
            					for(int k = 0; k < tl4dmoea.dimension; k++) {
            						fw2.write(tl4dmoea.population.get(i).getVariables(k)+" ");
            					}
            					fw2.write("\n");
            				}
            	            fw1.close();
            	            fw2.close();
        	    		}
        	    		        		         
        		        //迭代
        	    		tl4dmoea.maxIter = Tool.ChangeFrequency[env];
        				for(int i = 0; i < tl4dmoea.maxIter; i++) {	
        					System.out.println("iteration: "+i);
        					if(i >= tl4dmoea.maxIter * 0.9 && i%5 == 0) {
        						Tool.isDiversityLearning = 1;
        					}
        					tl4dmoea.offspringPopulation = tl4dmoea.reproduction(tl4dmoea.population, tl4dmoea.crossoverProbability, tl4dmoea.mutationProbability);
        					tl4dmoea.population = tl4dmoea.preferenceBasedSelectionStrategy(tl4dmoea.population, tl4dmoea.offspringPopulation, Math.min(((double)i/(tl4dmoea.maxIter*0.1))*tl4dmoea.angleBase, tl4dmoea.angleBase));		
        					Tool.isDiversityLearning = 0;
        					
        				}
        				tl4dmoea.updateMemory();//更新了Tool中
        				

        	        
     
        	            /****处理数据包括 过滤不满足条件的，移除重复的，再获得第一层**********/
        	            
        	            
        	            //用于辅助计算变化程度
        	            Tool.lastPop.clear();
        	            Tool.lastPop.addAll(tl4dmoea.population);
           	            //先清理，后添加
//  
        	            
        	            tl4dmoea.population = filterNotMeetConditions(tl4dmoea.population, Tool.g);
        	            //特殊处理，解决样本不够的问题
        	            if(tl4dmoea.population.size() < 10) {
        	            	tl4dmoea.population.clear();
        	            	tl4dmoea.population.addAll(Tool.lastPop);
        	            }
        	            tl4dmoea.population = removeDuplication(tl4dmoea.population);    
        	            tl4dmoea.population = obtainFirstLevelIndividuals(tl4dmoea.population);        	            
        	            //特殊处理，解决样本不够的问题
        	            if(tl4dmoea.population.size() < 10) {
        	            	tl4dmoea.population.clear();
        	            	tl4dmoea.population.addAll(Tool.lastPop);
        	            }
        	            //保存最终数据
        				for(int i = 0; i < tl4dmoea.population.size(); i++) {
        					for(int j = 0; j < tl4dmoea.nobj; j++) {
        						fw3.write(tl4dmoea.population.get(i).getObjectives(j)+" ");
        					}	
        					fw3.write("\n");
        					for(int k = 0; k < tl4dmoea.dimension; k++) {
        						fw4.write(tl4dmoea.population.get(i).getVariables(k)+" ");
        					}
        					fw4.write("\n");
        				}
        				//保存数据

        	            fw3.close();
        	            fw4.close();
        	            
        	           
        	            
        	            Tool.netInputs.clear();
        	            Tool.netInputs.addAll(Tool.lastEnvirPop);  
        	            Tool.netoutputs.clear();
        	            Tool.netoutputs.addAll(tl4dmoea.population);
//    	            	tl4dmoeaSampleProcess(Tool.netInputs, Tool.netoutputs, Tool.points);
    	            	nnnsga2SampleProcess(Tool.netInputs, Tool.netoutputs);

        	            Tool.lastEnvirPop.clear();//清理元素
        	            Tool.lastEnvirPop.addAll(tl4dmoea.population);
        	            
        	            //处理样本数理 写入文件& 训练网络    
        				for(int i=0;i<Tool.netInputs.size();i++) {
        					for(int j=0; j<Tool.TaskNum; j++) {
        						fw7.write(Tool.netInputs.get(i).getVariables(j)+" ");
        						fw8.write(Tool.netoutputs.get(i).getVariables(j)+" ");
        					}
        					fw7.write("\n");
        					fw8.write("\n");
        				}
        				fw7.close();
        				fw8.close();
        				
    	            	if (env == 0) {
    	            		//
    	            	} else if(env == 1){
    	            		MultiToMultiRegressionUsingComGraph4TL mlp = new MultiToMultiRegressionUsingComGraph4TL(); 
    	            		mlp.trainingNeuralNetwork();
    	            		
    	            	} else {
    	            		//迁移学习
    	            		System.out.println("主 " + Tool.netInputs.size());
    	            		MultiToMultiRegressionTransferLearningUsingComGraph tl = new MultiToMultiRegressionTransferLearningUsingComGraph();
    	            		tl.tlTrainNet();
    	            	}
            				
        	            
        	    	}catch (IOException e) {
        	            // TODO Auto-generated catch block
        	            e.printStackTrace();
        	        }
        		}
    			
    			
	            //最后一次仿真系统自带
	            //lastSimulation();
	            long endTime = System.currentTimeMillis();
    	    	System.out.println("程序运行时间：" + (endTime - startTime) + "ms");    //输出程序运行时间                   
	            System.out.println("finish~");
    	   }		
	    }
    	
        
    }

    
    //nn-nsga2数据处理 Paper: Neural network based multi-objective evolutionary algorithm for dynamic workflow scheduling in cloud computing
    public static void nnnsga2SampleProcess(List<Individual> netInputs, List<Individual> netOutputs) {

    	
    	//这里的inputs和output对应的样本数据是一样的，都是种群的大小
    	double eucliDisMatrix[][] = new double[netInputs.size()][netOutputs.size()];
    	int NetInputOouputMatch[][] = new int[netInputs.size()][2];//第一列为input的序号，从0开始
        List<Individual> netoutputsTemp = new ArrayList<>();
    	for(int i=0;i<netInputs.size();i++) {
    		for(int j=0;j<netOutputs.size();j++) {
    			eucliDisMatrix[i][j] = MyUtils.euclideanDistance(netInputs.get(i).getAllObjectives(), netOutputs.get(j).getAllObjectives());
    		}
    	}

		
		int outputIndex = -1;
		
		for (int i = 0; i < netInputs.size(); i++ ) {
			outputIndex = MyUtils.findMinIndex(eucliDisMatrix[i]);
			NetInputOouputMatch[i][0] = i;
			NetInputOouputMatch[i][1] = outputIndex;
		}

    	//根据匹配池数据匹配int  out
    	for(int i = 0; i < netInputs.size(); i++) {
    		netoutputsTemp.add(netOutputs.get(NetInputOouputMatch[i][1]));
    	}
    	netOutputs.clear();
    	netOutputs.addAll(netoutputsTemp);
    	
    }
    
    
    
 
    public static void tl4dmoeaSampleProcess(List<Individual> netInputs, List<Individual> netOutputs,double points[][]) {
        
    	//1.先inputs与points匹配，找到与之对应的点
    	double angleMatrix[][] = new double[netInputs.size()][points.length];
    	double maxValue = Double.POSITIVE_INFINITY;
    	double basePoint[] = new double[Tool.nobj]; 
    	int NetInputPointsMatch[][] = new int[netInputs.size()][2];//第一列放netinput序号，第二列放points序号
    	for(int i=0; i<Tool.nobj; i++) {
    		basePoint[i] = 0;
    	}

    	for(int i=0;i<netInputs.size();i++) {
    		for(int j=0;j<points.length;j++) {
    			angleMatrix[i][j] = MyUtils.getAngle(netInputs.get(i).getAllObjectives(), points[j], basePoint, basePoint);		
    		}
    	}
    	
    	int count = 0;
    	while(count != netInputs.size()) {
    		double minValue = maxValue;
    		int inputIndex = -1;//标记序号
    		int pointIndex = -1;
    		//全局找最小值
    		for(int i=0;i<netInputs.size();i++) {
    			for(int j=0; j<points.length; j++) {
    				if(angleMatrix[i][j] < minValue) {
    					minValue = angleMatrix[i][j];
    					inputIndex = i;
    					pointIndex = j;
    				}
    			}	
    		}
    		//加入到匹配池中
    		NetInputPointsMatch[inputIndex][0] = inputIndex;
    		NetInputPointsMatch[inputIndex][1] = pointIndex;
    		//对应的两个点设置为最大值
    		for(int i=0;i<points.length;i++) {//行
    			angleMatrix[inputIndex][i] = maxValue;
    		}
    		for(int i=0;i<netInputs.size();i++) {//列
    			angleMatrix[i][pointIndex] = maxValue;
    		}
    		count = count + 1;
    		
    	}
    	
//    	//找到角度最小的序号
//    	for(int i=0;i<netInputs.size();i++) {
//    		int order[] = MyUtils.sortIndex(angleMatrix[i]);
//    		NetInputPointsMatch[i][0] = i;
//    		NetInputPointsMatch[i][1] = order[0];
//    		//剩下 所有的对应序号全部设置为最大值，避免重复被选
//    		for(int j=0; j<netInputs.size(); j++) {
//    			angleMatrix[j][order[0]] = maxValue;
//    		}
////    		System.out.print(order[0] + " ");
//    	}
    	
    	//2.找到netoput  与 现在所匹配的向量之间的最小值
    	int samplesNum = netInputs.size() > netOutputs.size() ? netOutputs.size() : netInputs.size();
    	int otherNum = netInputs.size() < netOutputs.size() ? netOutputs.size() : netInputs.size();
    	int NetInputOouputMatch[][] = new int[otherNum][2];
    	double angleMatrix2[][] = new double[netInputs.size()][netOutputs.size()];

    	//计算角度 intput与points相当于已经绑定了
		for(int i=0;i<netInputs.size();i++) {
    		for(int j=0;j<netOutputs.size();j++) {
    			angleMatrix2[i][j] = MyUtils.getAngle(netInputs.get(i).getAllObjectives(), netOutputs.get(j).getAllObjectives(), points[NetInputPointsMatch[i][1]], points[NetInputPointsMatch[i][1]]);		
    		}
    	}
    	
		int count1 = 0;
		int samNetInput[] = new int[samplesNum];//保存netInt被选择的序号
    	while(count1 != samplesNum) {
    		double minValue = maxValue;
    		int inputIndex = -1;//标记序号
    		int outputIndex = -1;
    		//全局找最小值
    		for(int i=0;i<netInputs.size();i++) {
    			for(int j=0; j<netOutputs.size(); j++) {
    				if(angleMatrix2[i][j] < minValue) {
    					minValue = angleMatrix2[i][j];
    					inputIndex = i;
    					outputIndex = j;
    				}
    			}	
    		}
    		//加入到匹配池中
    		NetInputOouputMatch[inputIndex][0] = inputIndex;
    		samNetInput[count1] = inputIndex;
    		NetInputOouputMatch[inputIndex][1] = outputIndex;
    		//对应的两个点设置为最大值
    		for(int i=0;i<netOutputs.size();i++) {//行
    			angleMatrix2[inputIndex][i] = maxValue;
    		}
    		for(int i=0;i<netInputs.size();i++) {//列
    			angleMatrix2[i][outputIndex] = maxValue;
    		}
    		count1 = count1 + 1;
    		
    	}
    	
    	List<Individual> netInputsTemp = new ArrayList<>();
    	List<Individual> netoutputsTemp = new ArrayList<>();
    	//根据匹配池数据匹配in  out
    	for(int i=0;i<samNetInput.length;i++) {
    		netInputsTemp.add(netInputs.get(samNetInput[i]));
    		netoutputsTemp.add(netOutputs.get(NetInputOouputMatch[samNetInput[i]][1]));
    	}
    	
    	netInputs.clear();
    	netInputs.addAll(netInputsTemp);
    	
    	netOutputs.clear();
    	netOutputs.addAll(netoutputsTemp);

    }
    
 
    //输入和输出并不是唯一的匹配，符合论文当中的说法
 public static void tl4dmoeaSampleProcess2(List<Individual> netInputs, List<Individual> netOutputs,double points[][]) {
        
    	//1.先inputs与points匹配，找到与之对应的点
    	double angleMatrix[][] = new double[netInputs.size()][points.length];
    	double maxValue = Double.POSITIVE_INFINITY;
    	double basePoint[] = new double[Tool.nobj]; 
    	int NetInputPointsMatch[][] = new int[netInputs.size()][2];//第一列放netinput序号，第二列放points序号
    	for(int i=0; i<Tool.nobj; i++) {
    		basePoint[i] = 0;
    	}

    	for(int i=0;i<netInputs.size();i++) {
    		for(int j=0;j<points.length;j++) {
    			angleMatrix[i][j] = MyUtils.getAngle(netInputs.get(i).getAllObjectives(), points[j], basePoint, basePoint);		
    		}
    	}
    	
    	
		//全局找最小值
		for(int i=0;i<netInputs.size();i++) {
    		double minValue = maxValue;
    		int inputIndex = i;
    		int pointIndex = -1;
			for(int j=0; j<points.length; j++) {
				if(angleMatrix[i][j] < minValue) {
					minValue = angleMatrix[i][j];
					pointIndex = j;
				}
			}
    		//加入到匹配池中
    		NetInputPointsMatch[inputIndex][0] = inputIndex;
    		NetInputPointsMatch[inputIndex][1] = pointIndex;
		}

    		
    		
    
    	
//    	//找到角度最小的序号
//    	for(int i=0;i<netInputs.size();i++) {
//    		int order[] = MyUtils.sortIndex(angleMatrix[i]);
//    		NetInputPointsMatch[i][0] = i;
//    		NetInputPointsMatch[i][1] = order[0];
//    		//剩下 所有的对应序号全部设置为最大值，避免重复被选
//    		for(int j=0; j<netInputs.size(); j++) {
//    			angleMatrix[j][order[0]] = maxValue;
//    		}
////    		System.out.print(order[0] + " ");
//    	}
    	
    	//2.找到netoput  与 现在所匹配的向量之间的最小值
    	int samplesNum = netInputs.size() > netOutputs.size() ? netOutputs.size() : netInputs.size();
    	int otherNum = netInputs.size() < netOutputs.size() ? netOutputs.size() : netInputs.size();
    	int NetInputOouputMatch[][] = new int[otherNum][2];
    	double angleMatrix2[][] = new double[netInputs.size()][netOutputs.size()];

    	//计算角度 intput与points相当于已经绑定了
		for(int i=0;i<netInputs.size();i++) {
    		for(int j=0;j<netOutputs.size();j++) {
    			angleMatrix2[i][j] = MyUtils.getAngle(netInputs.get(i).getAllObjectives(), netOutputs.get(j).getAllObjectives(), points[NetInputPointsMatch[i][1]], points[NetInputPointsMatch[i][1]]);		
    		}
    	}
    	
		int count1 = 0;
		int samNetInput[] = new int[samplesNum];//保存netInt被选择的序号
		//全局找最小值
		for(int i=0;i<netInputs.size();i++) {
			double minValue = maxValue;
    		int inputIndex = i;//标记序号
    		int outputIndex = -1;
			for(int j=0; j<netOutputs.size(); j++) {
				if(angleMatrix2[i][j] < minValue) {
					minValue = angleMatrix2[i][j];
					outputIndex = j;
				}
			}
    		NetInputOouputMatch[inputIndex][0] = inputIndex;
    		NetInputOouputMatch[inputIndex][1] = outputIndex;
		}
		//加入到匹配池中


    	

    	List<Individual> netoutputsTemp = new ArrayList<>();
    	//根据匹配池数据匹配in  out
    	for(int i=0;i<netInputs.size();i++) {
    		netoutputsTemp.add(netOutputs.get(NetInputOouputMatch[i][1]));
    	}
    	
    	netOutputs.clear();
    	netOutputs.addAll(netoutputsTemp);

    }
    
    //得到第一层数据
    public static List<Individual> obtainFirstLevelIndividuals(List<Individual> population) {
		FastNonDominatedSortRanking2  fndsr = new FastNonDominatedSortRanking2();
		List<ArrayList<Individual>> ranking = fndsr.computeRanking(population);
//		List<Individual> pop = new ArrayList<>();
		List<Individual> first = new ArrayList<>();
		first = fndsr.getSubFront(0);
		return first;
    }

    //去重
    public static List<Individual> removeDuplication(List<Individual> population) {
    	double fitnessSum[] = new double[population.size()];
    	for(int i=0;i<population.size();i++) {
    		double temp = 0.0;
    		for(int j=0;j<Tool.nobj;j++) {
    			temp = temp + population.get(i).getObjectives(j);
    		}
    		fitnessSum[i] = temp;
    	}
    	List<Individual> pop = new ArrayList<>();
    	pop.add(population.get(0));
    	for(int i=0;i<population.size();i++) {
    		int j;
    		for(j=0;j<pop.size();j++) {
    			if (fitnessSum[i] == fitnessSum[j]) {
    				break;
    			} else {
    				
    			}
    		}
    		if(j==pop.size()) {
    			pop.add(population.get(i));
    		}
    	}
    	return pop;
    	
    }
    
    
    //过滤不符合条件的解
    public static List<Individual>  filterNotMeetConditions(List<Individual> population,double g[]) {
    	List<Individual> pop = new ArrayList<>();
    	double temp1;
		for (int i=0;i<population.size();i++) {
			temp1 = 0;
			for(int j=0;j<Tool.nobj;j++) {
				if(g[j] >= population.get(i).getObjectives(j)) {
					temp1 = temp1 + 1;
				}else{
//					
				}
				
			}
			//对统计的的值分配值
			if (temp1 == Tool.nobj) {
				pop.add(population.get(i));
			}
		}
    	return pop;
    }
    
    
    
    //初始化
    public static void InitCloudParameters() {
    	

    	try {
	    	// First step: Initialize the WorkflowSim package. 
	
	        /**
	         * However, the exact number of vms may not necessarily be vmNum If
	         * the data center or the host doesn't have sufficient resources the
	         * exact vmNum would be smaller than that. Take care.
	         */
	        int vmNum = Tool.VmNum;//number of vms;
	        
	        /**
	         * Should change this based on real physical path
	         */
	        String daxPath = "G:\\eclipseWorkspace\\2.tldmoea4wrokflowScheduling2\\WorkflowSim\\WorkflowSim\\WorkflowSim-1.0-master\\config\\dax\\" + String.valueOf(Tool.workflowName)+"_"+ Integer.toString(Tool.fileTaskNum)+"_"+Integer.toString(Tool.env)+".xml";
	        
	        
	        File daxFile = new File(daxPath);
	        if (!daxFile.exists()) {
	            Log.printLine("Warning: Please replace daxPath with the physical path in your working environment!");
	            return;
	        }
	
	        /**
	         * Since we are using HEFT planning algorithm, the scheduling
	         * algorithm should be static such that the scheduler would not
	         * override the result of the planner
	         */
	        
	    	
	        Parameters.SchedulingAlgorithm sch_method = Parameters.SchedulingAlgorithm.STATIC;
	        Parameters.PlanningAlgorithm pln_method = Parameters.PlanningAlgorithm.TL4DMOEA;
	        ReplicaCatalog.FileSystem file_system = ReplicaCatalog.FileSystem.LOCAL;
	
	        /**
	         * No overheads
	         */
	        OverheadParameters op = new OverheadParameters(0, null, null, null, null, 0);
	
	        /**
	         * No Clustering
	         */
	        ClusteringParameters.ClusteringMethod method = ClusteringParameters.ClusteringMethod.NONE;
	        ClusteringParameters cp = new ClusteringParameters(0, 0, method, null);
	
	        /**
	         * Initialize static parameters
	         */
	        Parameters.init(vmNum, daxPath, null,
	                null, op, cp, sch_method, pln_method,
	                null, 0);
	        ReplicaCatalog.init(file_system);
	
	        // before creating any entities.
	        int num_user = 1;   // number of grid users
	        Calendar calendar = Calendar.getInstance();
	        boolean trace_flag = false;  // mean trace events
	
	        // Initialize the CloudSim library
	        CloudSim.init(num_user, calendar, trace_flag);
	        
	        
	        //得到taskList
	        WorkflowParser wfp = new WorkflowParser(0);
	        wfp.parse();
	       
	        //更新实际的task数目
	        Tool.TaskNum = wfp.getTaskList().size();//少的task都是末尾连续的
	        
	        //如果需要对 task预处理形成pipeline 则需要这句
//	        Tool.pipelineTaskList = genePipelinePairTaskList(wfp.getTaskList());
	        
	        
	        //对task进行了分层处理
	        Tool.setTaskList(taskLayerProcess(wfp.getTaskList()));
	        
	        WorkflowDatacenter datacenter0 = createDatacenter("Datacenter_0");

            /**
             * Create a WorkflowPlanner with one schedulers.
             */
            WorkflowPlanner wfPlanner = new WorkflowPlanner("planner_0", 1);
            /**
             * Create a WorkflowEngine.
             */
            WorkflowEngine wfEngine = wfPlanner.getWorkflowEngine();
            /**
             * Create a list of VMs.The userId of a vm is basically the id of
             * the scheduler that controls this vm.
             */
            List<CondorVM> vmlist0 = createVM(wfEngine.getSchedulerId(0), Parameters.getVmNum());
            
            Tool.setVmList(vmlist0);
            /**
             * Submits this list of vms to this WorkflowEngine.
             */
            wfEngine.submitVmList(vmlist0, 0);

            /**
             * Binds the data centers with the scheduler.
             */
            wfEngine.bindSchedulerDatacenter(datacenter0.getId(), 0);
            
      

    	}catch (Exception e) {
            Log.printLine("The simulation has been terminated due to an unexpected error");
        }
    }
    
    
    //得到pipelinePairTasks  input:wfp.getTaskList()
    public static List<ArrayList<Integer>> genePipelinePairTaskList(List<Task> taskList1) {
    	List<ArrayList<Integer>> pipelineTaskList = new ArrayList<>();
    	int pipelineTaskLen = 0;//保存有多少对这样的task
    	for(int i=0;i<taskList1.size();i++) {
    		Task task = taskList1.get(i);
    		if (task.getChildList().size() == 1 ) {
    			Task childTask = task.getChildList().get(0);
    			if (childTask.getParentList().size() == 1) {
    				//满足两个条件，添加进来
    				ArrayList<Integer> pipeList = new ArrayList<>();
    				pipeList.add(Integer.valueOf(task.getCloudletId()));
    				pipeList.add(Integer.valueOf(childTask.getCloudletId()));
    				pipelineTaskList.add(new ArrayList<Integer>());
    				pipelineTaskList.set(pipelineTaskLen, pipeList);
    				pipelineTaskLen = pipelineTaskLen + 1; //加1 下标
    				
    				//System.out.println(task.getCloudletId() + " " + childTask.getCloudletId()); //输出pipepair
    			}
    		} else {
    			
    		}
    	}
    	
    	return pipelineTaskList;
    	
    }
    
    
    
    //task分层处理  把根据task1-N的序号换成根据分层来重新安排序号
    public static List<Task> taskLayerProcess(List<Task> taskList1) {
    	List<Task> taskList = new ArrayList<Task>();

    	List<ArrayList<Task>> layerTaskList = new ArrayList<>();
		Task ta;
		//找到最大层
		int maxLayer = -1;
		for(int i=0;i<taskList1.size();i++) {
			ta = taskList1.get(i);
			if(ta.getDepth()>maxLayer) {
				maxLayer = ta.getDepth();
			}
		}
		
		
		//为list分层分配好空间
		for(int i=0;i<maxLayer;i++) {
			layerTaskList.add(new ArrayList<Task>());
		}
    	//1.分层
    	for(int i=0;i<taskList1.size();i++) {
    		ta = taskList1.get(i);
    		ArrayList<Task> taskList2  = layerTaskList.get(ta.getDepth()-1);
    		taskList2.add(ta);
    		layerTaskList.set(ta.getDepth()-1,  taskList2);
    	}
    	
    	//2.根据层重新组合成list
    	for(int i=0;i<layerTaskList.size();i++) {
    		ArrayList<Task> taskList2  = layerTaskList.get(i);
    		for(int j=0;j<taskList2.size();j++) {
    			ta = taskList2.get(j);
    			taskList.add(ta);
    		}
    		
    	}
    	
    	
    	return taskList;
    }
    
    //接着仿真最后一个仿真
    public static void lastSimulation() {
    	try {
	    	// First step: Initialize the WorkflowSim package. 
	
	        /**
	         * However, the exact number of vms may not necessarily be vmNum If
	         * the data center or the host doesn't have sufficient resources the
	         * exact vmNum would be smaller than that. Take care.
	         */
	        int vmNum = Tool.VmNum;//number of vms;
	        /**
	         * Should change this based on real physical path
	         */
	        String daxPath = "G:\\eclipseWorkspace\\2.tldmoea4wrokflowScheduling2\\WorkflowSim\\WorkflowSim\\WorkflowSim-1.0-master\\config\\dax\\" + String.valueOf(Tool.workflowName)+"_"+Integer.toString(Tool.fileTaskNum)+".xml";
	        
	        
	        File daxFile = new File(daxPath);
	        if (!daxFile.exists()) {
	            Log.printLine("Warning: Please replace daxPath with the physical path in your working environment!");
	            return;
	        }
	
	        /**
	         * Since we are using HEFT planning algorithm, the scheduling
	         * algorithm should be static such that the scheduler would not
	         * override the result of the planner
	         */
	        Parameters.SchedulingAlgorithm sch_method = Parameters.SchedulingAlgorithm.STATIC;
	        Parameters.PlanningAlgorithm pln_method = Parameters.PlanningAlgorithm.TL4DMOEA;
	        ReplicaCatalog.FileSystem file_system = ReplicaCatalog.FileSystem.LOCAL;
	
	        /**
	         * No overheads
	         */
	        OverheadParameters op = new OverheadParameters(0, null, null, null, null, 0);
	
	        /**
	         * No Clustering
	         */
	        ClusteringParameters.ClusteringMethod method = ClusteringParameters.ClusteringMethod.NONE;
	        ClusteringParameters cp = new ClusteringParameters(0, 0, method, null);
	
	        /**
	         * Initialize static parameters
	         */
	        Parameters.init(vmNum, daxPath, null,
	                null, op, cp, sch_method, pln_method,
	                null, 0);
	        ReplicaCatalog.init(file_system);
	
	        // before creating any entities.
	        int num_user = 1;   // number of grid users
	        Calendar calendar = Calendar.getInstance();
	        boolean trace_flag = false;  // mean trace events
	
	        // Initialize the CloudSim library
	        CloudSim.init(num_user, calendar, trace_flag);
	
	        WorkflowDatacenter datacenter0 = createDatacenter("Datacenter_0");
	
	        /**
	         * Create a WorkflowPlanner with one schedulers.
	         */
	        WorkflowPlanner wfPlanner = new WorkflowPlanner("planner_0", 1);
	        /**
	         * Create a WorkflowEngine.
	         */
	        WorkflowEngine wfEngine = wfPlanner.getWorkflowEngine();
	        /**
	         * Create a list of VMs.The userId of a vm is basically the id of
	         * the scheduler that controls this vm.
	         */
	        List<CondorVM> vmlist0 = createVM(wfEngine.getSchedulerId(0), vmNum);
	    	
	        /**
	         * Submits this list of vms to this WorkflowEngine.
	         */
	        wfEngine.submitVmList(vmlist0, 0);
	
	
	        /**
	         * Binds the data centers with the scheduler.
	         */
	        wfEngine.bindSchedulerDatacenter(datacenter0.getId(), 0);
	
	        CloudSim.startSimulation();
	        List<Job> outputList0 = wfEngine.getJobsReceivedList();
	        
            CloudSim.stopSimulation();
            printJobList(outputList0);
    	}catch (Exception e) {
            Log.printLine("The simulation has been terminated due to an unexpected error");
        }
    }
    
    

    ////////////////////////// STATIC METHODS ///////////////////////
    protected static List<CondorVM> createVM(int userId, int vms) {

    	
    	
    	
    	
        //Creates a container to store VMs. This list is passed to the broker later
    	//根据task数目产生vm数目


        LinkedList<CondorVM> list = new LinkedList<CondorVM>();

        //VM Parameters
        long size = 10000; //image size (MB)
        int ram = 512; //vm memory (MB)
        double mips = 0;
        long bw = 0;
        int pesNumber = 1; //number of cpus
        String vmm = "Xen"; //VMM name
        double cost=0.0;//CPU cost
        double costPerMem=0.0;
        double costPerStorage = 0.0;
        double costPerBW = 0.0;
        
        
        //create VMs
        CondorVM[] vm = new CondorVM[vms];

        //Random bwRandom = new Random(System.currentTimeMillis());
        
        for (int i = 0; i < vms; i++) {
        	//加载数据
        	//i 是 vmid
        	mips = Tool.VMInfoList.get(i).mips;
        	ram = Tool.VMInfoList.get(i).memory;
        	bw = (long)Tool.VMInfoList.get(i).bandWidth;
        	size = (long)Tool.VMInfoList.get(i).storage;
        	vmm = String.valueOf(Tool.VMInfoList.get(i).vmName);
        	cost = Tool.VMInfoList.get(i).vmCost;
        	costPerStorage = Tool.VMInfoList.get(i).storageCost;
        	costPerBW = Tool.VMInfoList.get(i).bandwidthCost;
        	
            //double ratio = bwRandom.nextDouble();
            //double mipsValue = Tool.getVmlist().get(i);
            
            vm[i] = new CondorVM(i, userId,mips, pesNumber, ram, bw, size, vmm, cost,costPerMem,costPerStorage,costPerBW,new CloudletSchedulerSpaceShared());
            list.add(vm[i]);
        }
        
        Tool.setVmList(list);
        
        //Collections.copy(descList, srcList)
        return list;
    }

    /**
     * Creates main() to run this example This example has only one datacenter
     * and one storage
     * @throws InterruptedException 
     */
    
//  
    
    public static double calMakespan() {


        try {
            // First step: Initialize the WorkflowSim package.

            /**
             * However, the exact number of vms may not necessarily be vmNum If
             * the data center or the host doesn't have sufficient resources the
             * exact vmNum would be smaller than that. Take care.
             */
            int vmNum = Tool.VmNum;//number of vms;
            /**
             * Should change this based on real physical path
             */
            String daxPath = "G:\\eclipseWorkspace\\2.tldmoea4wrokflowScheduling2\\WorkflowSim\\WorkflowSim\\WorkflowSim-1.0-master\\config\\dax\\" + String.valueOf(Tool.workflowName)+"_"+Integer.toString(Tool.fileTaskNum)+".xml";
            
            File daxFile = new File(daxPath);
            if(!daxFile.exists()){
                Log.printLine("Warning: Please replace daxPath with the physical path in your working environment!");
                return -1;
            }

            /**
             * Since we are using HEFT planning algorithm, the scheduling algorithm should be static
             * such that the scheduler would not override the result of the planner
             */
            Parameters.SchedulingAlgorithm sch_method = Parameters.SchedulingAlgorithm.STATIC;
            Parameters.PlanningAlgorithm pln_method = Parameters.PlanningAlgorithm.TL4DMOEA;
            ReplicaCatalog.FileSystem file_system = ReplicaCatalog.FileSystem.LOCAL;

            /**
             * No overheads
             */
            OverheadParameters op = new OverheadParameters(0, null, null, null, null, 0);;

            /**
             * No Clustering
             */
            ClusteringParameters.ClusteringMethod method = ClusteringParameters.ClusteringMethod.NONE;
            ClusteringParameters cp = new ClusteringParameters(0, 0, method, null);

            /**
             * Initialize static parameters
             */
            Parameters.init(vmNum, daxPath, null,
                    null, op, cp, sch_method, pln_method,
                    null, 0);
            ReplicaCatalog.init(file_system);

            // before creating any entities.
            int num_user = 1;   // number of grid users
            Calendar calendar = Calendar.getInstance();
            boolean trace_flag = false;  // mean trace events

            // Initialize the CloudSim library
            CloudSim.init(num_user, calendar, trace_flag);

            WorkflowDatacenter datacenter0 = createDatacenter("Datacenter_0");

            /**
             * Create a WorkflowPlanner with one schedulers.
             */
            WorkflowPlanner wfPlanner = new WorkflowPlanner("planner_0", 1);
            /**
             * Create a WorkflowEngine.
             */
            WorkflowEngine wfEngine = wfPlanner.getWorkflowEngine();
            /**
             * Create a list of VMs.The userId of a vm is basically the id of
             * the scheduler that controls this vm.
             */
            List<CondorVM> vmlist0 = createVM(wfEngine.getSchedulerId(0), Parameters.getVmNum());

            /**
             * Submits this list of vms to this WorkflowEngine.
             */
            wfEngine.submitVmList(vmlist0, 0);

            /**
             * Binds the data centers with the scheduler.
             */
            wfEngine.bindSchedulerDatacenter(datacenter0.getId(), 0);



            CloudSim.startSimulation();


            List<Job> outputList0 = wfEngine.getJobsReceivedList();
             
            

            CloudSim.stopSimulation();
//            printJobList(outputList0);

            
            double[] fTime = new double[Tool.TaskNum+1];

            DecimalFormat dft = new DecimalFormat("###.##");
            String tab = "\t";
            Log.printLine("========== OUTPUT ==========");
            Log.printLine("TaskID" + tab + "vmID" + tab + "RunTime" + tab + "StartTime" + tab + "FinishTime" + tab + "Depth"+tab+"STATUS");

            for (int i = 0; i < outputList0.size(); i++) {
                Job oneJob = outputList0.get(i);
                Log.printLine(oneJob.getCloudletId() + tab
                        + oneJob.getVmId() + tab
                        + dft.format(oneJob.getActualCPUTime()) + tab
                        + dft.format(oneJob.getExecStartTime()) + tab+tab
                        + dft.format(oneJob.getFinishTime()) + tab +tab
                        + oneJob.getDepth()+ tab +oneJob.getCloudletStatusString() );

                fTime[oneJob.getCloudletId()] = oneJob.getFinishTime();
            }

            double makespan = outputList0.get((outputList0.size()-1)).getFinishTime()-outputList0.get(0).getFinishTime();
//            Thread.sleep(100000000);
//            System.out.print(makespan);
            return makespan;


        } catch (Exception e) {
            Log.printLine("The simulation has been terminated due to an unexpected error");
            return -1;
        }
    }
    
    
    
    public static void initWorkflow()
    {

        /**
         * However, the exact number of vms may not necessarily be vmNum If
         * the data center or the host doesn't have sufficient resources the
         * exact vmNum would be smaller than that. Take care.
         */
        int vmNum = Tool.VmNum;//number of vms;
        /**
         * Should change this based on real physical path
         */
        String daxPath = "G:\\eclipseWorkspace\\2.tldmoea4wrokflowScheduling2\\WorkflowSim\\WorkflowSim\\WorkflowSim-1.0-master\\config\\dax\\" + String.valueOf(Tool.workflowName)+"_"+Integer.toString(Tool.fileTaskNum)+"_"+Integer.toString(Tool.env)+".xml";
        
        File daxFile = new File(daxPath);
        if(!daxFile.exists()){
            Log.printLine("Warning: Please replace daxPath with the physical path in your working environment!");
            return;
        }

        /**
         * Since we are using HEFT planning algorithm, the scheduling algorithm should be static
         * such that the scheduler would not override the result of the planner
         */
        Parameters.SchedulingAlgorithm sch_method = Parameters.SchedulingAlgorithm.STATIC;
        Parameters.PlanningAlgorithm pln_method = Parameters.PlanningAlgorithm.TL4DMOEA;
        ReplicaCatalog.FileSystem file_system = ReplicaCatalog.FileSystem.LOCAL;

        /**
         * No overheads
         */
        OverheadParameters op = new OverheadParameters(0, null, null, null, null, 0);;

        /**
         * No Clustering
         */
        ClusteringParameters.ClusteringMethod method = ClusteringParameters.ClusteringMethod.NONE;
        ClusteringParameters cp = new ClusteringParameters(0, 0, method, null);

        /**
         * Initialize static parameters
         */
        Parameters.init(vmNum, daxPath, null,
                null, op, cp, sch_method, pln_method,
                null, 0);
        ReplicaCatalog.init(file_system);

        // before creating any entities.
        int num_user = 1;   // number of grid users
        Calendar calendar = Calendar.getInstance();
        boolean trace_flag = false;  // mean trace events

        // Initialize the CloudSim library
        CloudSim.init(num_user, calendar, trace_flag);

        WorkflowParser wfp = new WorkflowParser(0);
        wfp.parse();

        Tool.setTaskList(wfp.getTaskList());

        
    }
    
    
    //用于产生makespan,cost,randEnergy，以便获得约束的上下界。
    public static void main4GA(String[] args) {

    	
    	//一些实验设置
    	int experimentNum = 10;
    	int workflowNum = 20;
    	
		/**************先进行一个初始化的参数，用于tl4dmoea配置相应的参数，**************************/
		try {
			Tool.configureParameters(Tool.workflowIndex);
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		InitCloudParameters();
    	for(int workflowIndex = 0; workflowIndex < workflowNum; workflowIndex++) {
    		Tool.workflowIndex = workflowIndex;
    		//判断是否存在
    		
    		boolean exit = false;
    		for (int i = 0; i < Tool.subWorkflows.length; i++) {
    			if (workflowIndex == Tool.subWorkflows[i]) {
    				exit = true;
    			}
    		}
    		if ( exit == false) {
    			continue;
    		}
    		for(int experIndex = 0;experIndex<experimentNum;experIndex++) {
    			long startTime = System.currentTimeMillis();
    	    	try {
    	    		//配置和初始化相应参数
    	    		Tool.configureParameters(Tool.workflowIndex);
    	    		//清理原有内容,即末尾不需要加true
 
            		FileWriter fw = new FileWriter("G:\\eclipseWorkspace\\2.tldmoea4wrokflowScheduling2\\constrain\\" + "energy_"+Tool.workflowIndex+ "_" + String.valueOf(experIndex) +".txt");
//            		FileWriter fw2 = new FileWriter("G:\\eclipseWorksapce\\2.tldmoea4wrokflowScheduling\\constrain\\" + (experIndex+1) + "\\cost_"+Tool.workflowIndex+".txt");
    		        InitCloudParameters();

    		        TL4DMOEA tl4dmoea = new TL4DMOEA();
    		        tl4dmoea.initBestIndividual();
    		        tl4dmoea.initPop();
    		        tl4dmoea.evaluatePop4GA(tl4dmoea.population);
    		        
    		        //迭代
    				for(int i=0; i<tl4dmoea.maxIter;i++) {	
    					System.out.println("iteration: "+i);
    					
    					tl4dmoea.offspringPopulation = tl4dmoea.reproduction4GA(tl4dmoea.population,tl4dmoea.crossoverProbability,tl4dmoea.mutationProbability);
    					tl4dmoea.population = tl4dmoea.replacement4GA(tl4dmoea.population, tl4dmoea.offspringPopulation);
    					
//    					fw.write(tl4dmoea.calMeanFitness()+ " ");
//    					fw.write(tl4dmoea.bestIndividual.getObjectives(Tool.objectiveNow)+ " ");
//    					fw.write("\n");
    				}
    				
    				//保存数据
//					for(int j=0;j<tl4dmoea.nobj;j++) {
//						fw.write(tl4dmoea.bestIndividual.getObjectives(j)+" ");
//					}
    				fw.write(tl4dmoea.randEnergy(tl4dmoea.population)+" ");
					fw.write("\n");
    	            fw.close();

    	            //最后一次仿真系统自带
    	            //lastSimulation();
    	            long endTime = System.currentTimeMillis();
        	    	System.out.println("程序运行时间：" + (endTime - startTime) + "ms");    //输出程序运行时间                   
    	            System.out.println("finish~");
    	    	}catch (IOException e) {
    	            // TODO Auto-generated catch block
    	            e.printStackTrace();
    	        }
    	    	
    	   }		
	    }
    	
        
    }



}
