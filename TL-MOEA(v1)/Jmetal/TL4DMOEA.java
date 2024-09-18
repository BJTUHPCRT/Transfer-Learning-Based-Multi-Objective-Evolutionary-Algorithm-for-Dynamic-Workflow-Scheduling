package org.uma.jmetal.algorithm.multiobjective.tl4dmoea;
import TL4DMOEA.Tool;
import mylesson2.LoadAndPredictComputationGraph4TL;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;


import TL4DMOEA.MyFitnessFunction;



//新的导入
import java.util.Random;
import java.util.Set;
import java.util.Vector;

import org.netlib.util.doubleW;



 













public class TL4DMOEA {

	public int getMaxIter() {
		return maxIter;
	}

	public void setMaxIter(int maxIter) {
		this.maxIter = maxIter;
	}

	public int getPopSize() {
		return popSize;
	}
	

	public void setPopSize(int popSize) {
		this.popSize = popSize;
	}

	public int getDimension() {
		return dimension;
	}

	public void setDimension(int dimension) {
		this.dimension = dimension;
	}

	public int getNobj() {
		return nobj;
	}

	public void setNobj(int nobj) {
		this.nobj = nobj;
	}

	//构造方法
	public TL4DMOEA() {

	}
	
	
	
	//设置相应的参数
	public int maxIter = Tool.IterationNum;
	public int popSize = Tool.populationSize;
	public int dimension = Tool.TaskNum;
	public static int nobj = Tool.nobj;
	public int memorySize = Tool.memorySize;
	public int archiveSize = Tool.archiveSize;

	public static double g[] = Tool.g;
	public double distributionIndex = 20;
	public double crossoverProbability = 0.9;//0.9
	public double mutationProbability = 1.0/dimension;
	public int numberOfDivisions = Tool.divisions;
	public double angleBase = Tool.angleBase; 
	
	

	
	public List<ReferencePoint2<Individual>> referencePoints = new Vector<>() ;
	public static double meanPoint[] = new double[Tool.nobj];
	public static double sigma = Double.MAX_VALUE;
	public static double iter = 0;
	
	/**********新增的*****************/
	public Individual bestIndividual = new Individual();
	
	//(2,99)=100 point, (3,13)=105 point   position对应点的值
	public void generatePoints() {
		(new ReferencePoint2<Individual>()).generateReferencePoints(referencePoints,nobj, numberOfDivisions);	
	}
	
	
	//获取参考点，并用于样本的匹配
	public double[][] obtainPoints() {
		
		double points[][] = new double[popSize][nobj]; //popsize是和参考点一样的大小
   		for(int i=0;i<referencePoints.size();i++) {
			List<Double> tempDoubles = 	referencePoints.get(i).position;
			for(int k=0;k< tempDoubles.size(); k++) {
				points[i][k] = tempDoubles.get(k).doubleValue();
			}
   		}
   		return points;
	}
	
	public List<Individual> population = new ArrayList<>();
	public List<Individual> offspringPopulation = new ArrayList<>();
	public List<Individual> noPrePopulation = new ArrayList<>();
	public List<Individual> tempPopulation = new ArrayList<>();
	public List<Individual> archive = new ArrayList<>();//用于多样性
	public List<Individual> memory = new ArrayList<>();//用于记忆存储
	
	
	//初始化种群
	public void initPop() {
		Random random = new Random();
		Bounds bo = new Bounds();
		
        for (int i = 0; i < popSize; i++) {
        	
        	Individual individual  = new Individual();
        	double val = 0;
        	double taskOrder = 0;
        	for(int j=0; j<dimension; j++) {
        		val = bo.getLowerBound(j) +(bo.getUpperBound(j) -  bo.getLowerBound(j)) * random.nextDouble();
        		taskOrder = bo.getLowerBound2(j) + (bo.getUpperBound2(j) -  bo.getLowerBound2(j)) * random.nextDouble();
        		individual.setVariables(j, val);
        		individual.setVariables2(j, taskOrder);
        	}
        	population.add(individual);
        }
	}
	
	
	//计算上下两次环境的变化程度
	public double degreeOfChange(List<Individual> oldPop) {
		List<Individual> newPop = new ArrayList<>();
		for(int i = 0; i < oldPop.size(); i++) {
			Individual individual = new Individual();
			individual = oldPop.get(i).copy();
			newPop.add(individual);
		}
		evaluatePop2(newPop);//新环境下种群的评估
		
		
		//计算变化程度
		double newPopFitness[][] = new double[newPop.size()][nobj];
		double oldPopFitness[][] = new double[oldPop.size()][nobj];
		double oldPopFitnessNorm[][] = new double[oldPop.size()][nobj];
		double newPopFitnessNorm[][] = new double[newPop.size()][nobj];
		double degree = 0.0;
		for(int i = 0; i < newPop.size(); i++) {
			newPopFitness[i] = newPop.get(i).getAllObjectives().clone();
			oldPopFitness[i] = oldPop.get(i).getAllObjectives().clone();
		}
		double oldMatrix[][]  = MyUtils.transposeMatrix(oldPopFitness);
		double newMatrix[][]  = MyUtils.transposeMatrix(newPopFitness);
		
		for(int i = 0; i < nobj; i++) {
			oldMatrix[i] = MyUtils.normalization(oldMatrix[i]);
			newMatrix[i] = MyUtils.normalization(newMatrix[i]);
		}
		//
		oldPopFitnessNorm  = MyUtils.transposeMatrix(oldMatrix);
		newPopFitnessNorm  = MyUtils.transposeMatrix(newMatrix);
		
		for (int i = 0; i < newPop.size(); i++ ) {
			for (int j = 0; j < nobj; j++) {
				degree = degree + Math.abs(newPopFitnessNorm[i][j] - oldPopFitnessNorm[i][j]);
			}
		}
		return degree / (newPop.size()*nobj);
	}
	


	//用于动态重新初始化种群
	public void reInitPop() throws IOException {
		Tool.degreeChange = degreeOfChange(Tool.lastPop);
		Random random = new Random();
		Bounds bo = new Bounds();
		population.addAll(Tool.memory);
		if (Tool.env == 1) {//这个时候没法用预测技术,因为还没有训练网络
			
			int deleteIndiviNum = (int) Math.ceil(population.size()*Tool.zeta);
			int indiviOrder[] =  MyUtils.generateUqiInt(population.size());
			for(int i = 0; i < deleteIndiviNum; i++) {
				population.remove(indiviOrder[i]);
				
				Individual individual  = new Individual();
	        	double val2 = 0;
	        	double taskOrder2 = 0;
	        	for(int j = 0; j < dimension; j++) {
	        		 
	        		val2 = bo.getLowerBound(j) + random.nextDouble()*(bo.getUpperBound(j) - bo.getLowerBound(j));
	        		taskOrder2 = bo.getLowerBound2(j) + random.nextDouble()*(bo.getUpperBound2(j) - bo.getLowerBound2(j));
	        		individual.setVariables(j, val2);
	        		individual.setVariables2(j, taskOrder2);
	        	}
	        	population.add(indiviOrder[i], individual);
			}
			
			addNoiseIndivis();
		} else {
			//用神经网络预测 加载上一时刻的数据 ,所以这里输入的是上一时刻的标签
			//Elite-led Transfer Learning Strategy的一部分，还包括train net, update memory, add noise
			System.out.println("通过网络预测ps");
			String fileName1 = new String("G:\\eclipseWorkspace\\2.tldmoea4wrokflowScheduling2\\dl4jtransferLearning\\modeltl\\MyMultiLayerNetwork_"+Tool.workflowIndex + "_" + (Tool.env-1) + ".zip");
			String fileName2 = new String("G:\\eclipseWorkspace\\2.tldmoea4wrokflowScheduling2\\dl4jtransferLearning\\datatl\\originalLabel_"+ Tool.workflowIndex + "_" + (Tool.env-1) + ".txt");
			LoadAndPredictComputationGraph4TL loadCG4TL = new LoadAndPredictComputationGraph4TL();
			loadCG4TL.loadCompuGraph(fileName1);
			double douMatrix[][];
			douMatrix = loadCG4TL.predictData(fileName2);
			repairPredictValueTL(douMatrix);
			
			//添加一些噪声
			addNoiseIndivis();
			
		}
		
		evaluatePop2(population);
		List<Individual> emptyList = new ArrayList<>();
		population =  replacement4(population, emptyList);
	}

	
	
	public void addNoiseIndivis() {
		Random random = new Random();
		Bounds bo = new Bounds();
		int currPopSize = population.size();
		
		//产生10%左右的噪声
		while(population.size() < popSize) {
			int randIndex =  random.nextInt(currPopSize);
			double val = 0;
			Individual individual  = new Individual();
        	for(int j = 0; j < dimension; j++) {
        		
        		if (random.nextDouble() < 0.1) {
        			val = bo.getLowerBound(j) + random.nextDouble()*(bo.getUpperBound(j) -bo.getLowerBound(j));
        		} else {
        			val = population.get(randIndex).getVariables(j);
        		}
        		
        		individual.setVariables(j, val);
        		
        	}
        	population.add(individual);
			
		}
	}
	
	
	//修复神经网络预测的值
		public void repairPredictValueTL(double douMatrix[][]) {
			Bounds bo = new Bounds();
			Random random = new Random(); 
			for(int i = 0; i < douMatrix.length; i++) {
				Individual individual  = new Individual();
				double temp = 0.0;
				for(int j = 0; j < douMatrix[0].length; j++) {
					//超范围随机生成
					temp = bo.getLowerBound(j) + random.nextDouble() * (bo.getUpperBound(j) - bo.getLowerBound(j));
					if (douMatrix[i][j] < bo.getLowerBound(j)) {
						individual.setVariables(j, temp);
					}else if(douMatrix[i][j]  >= bo.getUpperBound(j)) {
						individual.setVariables(j, temp);
					} else {
						//
					}

				}

				population.add(individual);	
			}
		}
	
	//初始化archive
	public void initArchive() {
		//已经初始化过了
	}
	
	public void obtainMemory() {
		//
		memory = Tool.memory;
	}
    
    
	
	
	
	
	//种群评估，资源池可能变小了
	public void evaluatePop2(List<Individual> population) {
			
			double MDD = 0;//makespan /  deadline
			double CDB = 0;
			double EDR = 0;
			int taskOrder[] = new int[Tool.TaskNum];
			double tempOrder[];
			int sortRank[] = new int[Tool.TaskNum];
			int tempSo = 0;
			
		
			for(int i=0;i<population.size();i++) {
				double temp[] = new double[Tool.TaskNum];
				int assignment[] = new int[Tool.TaskNum];
				temp = population.get(i).getAllVariables().clone();
				for(int j = 0; j < temp.length; j++) {
					tempSo	= (int)(0 + temp[j] *(Tool.resourcePool.size()));
					assignment[j] = Tool.resourcePool.get(tempSo).intValue();		
				}
				Tool.allot = assignment;
				
				tempOrder = population.get(i).getAllVariables2().clone();
				sortRank = MyUtils.sortIndex(tempOrder);
				for(int k = 0; k < sortRank.length; k++) {
					taskOrder[sortRank[k]] = k;
				}
				Tool.taskOrder = taskOrder;
				
				MyFitnessFunction myFitness = new MyFitnessFunction(); 
				myFitness.scheduleSimulation(assignment,taskOrder);

				

				MDD = myFitness.makespanDivideDeadline();
				CDB = myFitness.costDivideBudget();
				EDR = myFitness.energyDivideRandEnergy();
				
				population.get(i).setObjectives(0, MDD);
				population.get(i).setObjectives(1, CDB);				
				population.get(i).setObjectives(2, EDR);
				
			    
			}
			
		}
	
	
	
	
  //模拟二进制交叉
  	public List<Individual> Intercrossover( double probability, Individual parent1, Individual parent2) {
  		/** EPS defines the minimum difference allowed between real values */
  		
  		double EPS = 1.0e-14;
  	    
  	    
  	    List<Individual> offspring = new ArrayList<Individual>(2);

  	    
  	    //把两个父代复制到offspring中
  	    offspring.add(parent1.copy());
  	    offspring.add(parent2.copy()) ;
  	    
  	     
  	    int i;
  	    double rand;
  	    double y1, y2, yL, yu;
  	    double c1, c2;
  	    double alpha, beta, betaq;
  	    double valueX1, valueX2;
  	    Random random = new Random();
  	    if (random.nextDouble() <= probability) {
  	      for (i = 0; i < dimension ; i++) {
  	    	valueX1 = parent1.getVariables(i);
  	    	valueX2 = parent2.getVariables(i);

  	        if (random.nextDouble() <= 0.5) {
  	          if (Math.abs(valueX1 - valueX2) > EPS) {

  	            if (valueX1 < valueX2) {
  	              y1 = valueX1;
  	              y2 = valueX2;
  	            } else {
  	              y1 = valueX2;
  	              y2 = valueX1;
  	            }
  	            
  	            Bounds bound = new Bounds();
  	           
  	            yL = bound.getLowerBound(i);
  	            yu = bound.getUpperBound(i);
  	            rand = random.nextDouble();
  	            beta = 1.0 + (2.0 * (y1 - yL) / (y2 - y1));
  	            alpha = 2.0 - Math.pow(beta, -(distributionIndex + 1.0));

  	            if (rand <= (1.0 / alpha)) {
  	              betaq = Math.pow((rand * alpha), (1.0 / (distributionIndex + 1.0)));
  	            } else {
  	              betaq = Math.pow(1.0 / (2.0 - rand * alpha), 1.0 / (distributionIndex + 1.0));
  	            }

  	            c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1));
  	            beta = 1.0 + (2.0 * (yu - y2) / (y2 - y1));
  	            alpha = 2.0 - Math.pow(beta, -(distributionIndex + 1.0));

  	            if (rand <= (1.0 / alpha)) {
  	              betaq = Math.pow((rand * alpha), (1.0 / (distributionIndex + 1.0)));
  	            } else {
  	              betaq = Math.pow(1.0 / (2.0 - rand * alpha), 1.0 / (distributionIndex + 1.0));
  	            }

  	            c2 = 0.5 * (y1 + y2 + betaq * (y2 - y1));
  	            
  	            if (c1 < yL) {
  	              c1 = yL;
  	            
  	            }

  	            if (c2 < yL) {
  	              c2 = yL;
  	            }

  	            if (c1 > yu) {
  	              c1 = yu;
  	            
  	            }

  	            if (c2 > yu) {
  	              c2 = yu;
  	              
  	            }

  	            if (random.nextDouble() <= 0.5) {
  	            	offspring.get(0).setVariables(i, c2);
  	            	offspring.get(1).setVariables(i, c1);
  	      
  	            } else {
  	            	offspring.get(0).setVariables(i, c1);
  	            	offspring.get(1).setVariables(i, c2);
  	         
  	            }
  	          } else {
  	            	offspring.get(0).setVariables(i, valueX1);
  	            	offspring.get(1).setVariables(i, valueX2);

  	          }
  	        } else {
              	offspring.get(0).setVariables(i, valueX2);
              	offspring.get(1).setVariables(i, valueX1);

  	        }
  	      }
  	    }

  		
  		return offspring;
  	}
	
  	
	
  	
 
  //模拟二进制交叉
  	public List<Individual> IntercrossoverOrder( double probability, Individual parent1, Individual parent2) {
  		/** EPS defines the minimum difference allowed between real values */
  		
  		double EPS = 1.0e-14;
  	    
  	    List<Individual> offspring = new ArrayList<Individual>(2);

  	    
  	    //把两个父代复制到offspring中
  	    offspring.add(parent1.copy());
  	    offspring.add(parent2.copy()) ;
  	    
  	      
  	    int i;
  	    double rand;
  	    double y1, y2, yL, yu;
  	    double c1, c2;
  	    double alpha, beta, betaq;
  	    double valueX1, valueX2;
  	    Random random = new Random();
  	    if (random.nextDouble() <= probability) {
  	      for (i = 0; i < dimension ; i++) {
  	    	valueX1 = parent1.getVariables2(i);
  	    	valueX2 = parent2.getVariables2(i);

  	        if (random.nextDouble() <= 0.5) {
  	          if (Math.abs(valueX1 - valueX2) > EPS) {

  	            if (valueX1 < valueX2) {
  	              y1 = valueX1;
  	              y2 = valueX2;
  	            } else {
  	              y1 = valueX2;
  	              y2 = valueX1;
  	            }
  	            
  	            Bounds bound = new Bounds();
  	           
  	            yL = bound.getLowerBound2(i);
  	            yu = bound.getUpperBound2(i);
  	            rand = random.nextDouble();
  	            beta = 1.0 + (2.0 * (y1 - yL) / (y2 - y1));
  	            alpha = 2.0 - Math.pow(beta, -(distributionIndex + 1.0));

  	            if (rand <= (1.0 / alpha)) {
  	              betaq = Math.pow((rand * alpha), (1.0 / (distributionIndex + 1.0)));
  	            } else {
  	              betaq = Math.pow(1.0 / (2.0 - rand * alpha), 1.0 / (distributionIndex + 1.0));
  	            }

  	            c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1));
  	            beta = 1.0 + (2.0 * (yu - y2) / (y2 - y1));
  	            alpha = 2.0 - Math.pow(beta, -(distributionIndex + 1.0));

  	            if (rand <= (1.0 / alpha)) {
  	              betaq = Math.pow((rand * alpha), (1.0 / (distributionIndex + 1.0)));
  	            } else {
  	              betaq = Math.pow(1.0 / (2.0 - rand * alpha), 1.0 / (distributionIndex + 1.0));
  	            }

  	            c2 = 0.5 * (y1 + y2 + betaq * (y2 - y1));
  	            
  	           
  	            if (c1 < yL) {
  	              c1 = yL;
  	            
  	            }

  	            if (c2 < yL) {
  	              c2 = yL;
  	            }

  	            if (c1 > yu) {
  	              c1 = yu;
  	            
  	            }

  	            if (c2 > yu) {
  	              c2 = yu;
  	              
  	            }

  	            if (random.nextDouble() <= 0.5) {
  	            	offspring.get(0).setVariables2(i, c2);
  	            	offspring.get(1).setVariables2(i, c1);
  	      
  	            } else {
  	            	offspring.get(0).setVariables2(i, c1);
  	            	offspring.get(1).setVariables2(i, c2);
  	         
  	            }
  	          } else {
  	            	offspring.get(0).setVariables2(i, valueX1);
  	            	offspring.get(1).setVariables2(i, valueX2);

  	          }
  	        } else {
              	offspring.get(0).setVariables2(i, valueX2);
              	offspring.get(1).setVariables2(i, valueX1);

  	        }
  	      }
  	    }

  		
  		return offspring;
  	}
  	
  	
  	
	//多项式变异
	public void Intermutation(double probability, Individual individual) {
		double rnd, delta1, delta2, mutPow, deltaq;
	    double y, yl, yu, val, xy;
	    
	    Random random = new Random();
	    Bounds bound = new Bounds();
	    for (int i = 0; i < individual.varLen; i++) {
	      if (random.nextDouble() <= probability) {
	    	
	        y = (double)individual.getVariables(i);
	        
	        yl = (double)bound.getLowerBound(i);
	        yu = (double)bound.getUpperBound(i);
	        if (yl == yu) {
	          y = yl ;
	        } else {
	          delta1 = (y - yl) / (yu - yl);
	          delta2 = (yu - y) / (yu - yl);
	          rnd = random.nextDouble();
	          mutPow = 1.0 / (distributionIndex + 1.0);
	          if (rnd <= 0.5) {
	            xy = 1.0 - delta1;
	            val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (Math.pow(xy, distributionIndex + 1.0));
	            deltaq = Math.pow(val, mutPow) - 1.0;
	          } else {
	            xy = 1.0 - delta2;
	            val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (Math.pow(xy, distributionIndex + 1.0));
	            deltaq = 1.0 - Math.pow(val, mutPow);
	          }
	          y = y + deltaq * (yu - yl);
	          y = MyUtils.randValue(y, yl, yu);
	        }
	        individual.setVariables(i,  y);
	      }
	    }
	    

  }
 
	
	
		
			
	//针对order的变异
	public void IntermutationOrder(double probability, Individual individual) {
		double rnd, delta1, delta2, mutPow, deltaq;
	    double y, yl, yu, val, xy;
	    
	    Random random = new Random();
	    Bounds bound = new Bounds();
	    for (int i = 0; i < individual.varLen; i++) {
	      if (random.nextDouble() <= probability) {
	    	
	        y = (double)individual.getVariables2(i);
	        
	        yl = (double)bound.getLowerBound2(i);
	        yu = (double)bound.getUpperBound2(i);
	        if (yl == yu) {
	          y = yl ;
	        } else {
	          delta1 = (y - yl) / (yu - yl);
	          delta2 = (yu - y) / (yu - yl);
	          rnd = random.nextDouble();
	          mutPow = 1.0 / (distributionIndex + 1.0);
	          if (rnd <= 0.5) {
	            xy = 1.0 - delta1;
	            val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (Math.pow(xy, distributionIndex + 1.0));
	            deltaq = Math.pow(val, mutPow) - 1.0;
	          } else {
	            xy = 1.0 - delta2;
	            val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (Math.pow(xy, distributionIndex + 1.0));
	            deltaq = 1.0 - Math.pow(val, mutPow);
	          }
	          y = y + deltaq * (yu - yl);
	          y = MyUtils.randValue(y, yl, yu);
	        }
	        individual.setVariables2(i, y);
	      }
	    }
	    
  }
	

	
	//进行二进制锦标赛选择 
	public List<Individual> TournamentSelection(List<Individual> population) {
		
		List<Individual> cpopulation = new ArrayList<>(getPopSize());
		Random rand = new Random();
		int induIndex1 = 0;
		int induIndex2 = 0;
		int dominanceFlag = 0;
		int selectIndex = 0;
		FastNonDominatedSortRanking2 fndsr = new FastNonDominatedSortRanking2 ();
    	for (int i=0;i<getPopSize();i++) {
    		//选俩
    		induIndex1 = rand.nextInt(getPopSize());
    		induIndex2 = rand.nextInt(getPopSize());
    		
//    		dominanceFlag = fndsr.compare(population.get(induIndex1), population.get(induIndex2));
    		if (induIndex1 < induIndex2) {
    			selectIndex = induIndex1;
    		} else {
    			selectIndex = induIndex2;
    		}
    		
    		cpopulation.add(population.get(selectIndex));
        		
    	}
    	
    	population = cpopulation;
    	return population;
    }
	
	
	
	    //产生子代
		public List<Individual> reproduction(List<Individual> population,double crossoverProbability,double mutationProbability) {
		  List<Individual> offspringPopulation = new ArrayList<>(getPopSize());
		  
		  //选择交配
		  population = TournamentSelection(population);


		  for (int i = 0; i < getPopSize(); i+=2) {
		    List<Individual> parents = new ArrayList<>(2);
		    
		    parents.add(population.get(i));
		    parents.add(population.get(Math.min(i + 1, getPopSize()-1)));
		    
		    //交叉变异
		    List<Individual> offspring = Intercrossover(crossoverProbability, parents.get(0), parents.get(1));
		    //
		    offspring = IntercrossoverOrder(crossoverProbability, offspring.get(0), offspring.get(1));

		    Intermutation(mutationProbability, offspring.get(0));
		    //
		    IntermutationOrder(mutationProbability, offspring.get(0));
		    
		    Intermutation(mutationProbability, offspring.get(1));
		    //
		    IntermutationOrder(mutationProbability, offspring.get(1));
		    
		    
		    Random random = new Random();
		    //先多样性学习
		    if (archive.size() > 0 && random.nextDouble() < 0.1 && Tool.isDiversityLearning == 1) {
		    	multiSpaceDiversityLearningStrategy(offspring.get(0), archive);
		    	multiSpaceDiversityLearningStrategy(offspring.get(1), archive); 
		    }

		    
		    //再计算下fitness
		    evaluatePop2(offspring);
		    
		    offspringPopulation.add(offspring.get(0));
		    offspringPopulation.add(offspring.get(1));
		  }
		 
		  return offspringPopulation ;
		}
	
	
    
		  
     //参考点的复制
	  private List<ReferencePoint2<Individual>> getReferencePointsCopy() {
		  List<ReferencePoint2<Individual>> copy = new ArrayList<>();
		  for (ReferencePoint2<Individual> r : this.referencePoints) {
			  copy.add(new ReferencePoint2<>(r));
		  }
		  return copy;
	  }
		  
		  
 
	
	//
		//preferenceRegionSort
		public  List<List<Integer>>  preferenceRegionSortingStrategy2(List<Individual> jointPopulation) {
			//preferenceRegionSort
			int preFlagArray[]  = getPreferredFlag2(jointPopulation);
			List<Integer> A1 = new ArrayList<>();
			List<Integer> A2 = new ArrayList<>();
			
			//保存2个区的个体序号
			for(int i=0;i<preFlagArray.length;i++) {
				if(preFlagArray[i] == 1) {
					A1.add(Integer.valueOf(i));
				}else {
					A2.add(Integer.valueOf(i));
				}
			}
			
			//
			List<List<Integer>> FrontLayer = new ArrayList<>();
			FrontLayer.add(A1);
			FrontLayer.add(A2);		
			return FrontLayer;
		}
		
		
		public static int[] getFlagArray(List<Individual> jointPopulation) {
			
			int popLen = jointPopulation.size();
			int flagArray[] = new int[popLen];
			int temp1 = 0;
			int temp2 = 0;
			int temp3 = 0;
			//初始化
			for(int i=0;i<popLen;i++) {
				flagArray[i] = 0;
			}
			
			//统计判断属于哪个区域
			for (int i=0;i<popLen;i++) {
				temp1 = 0;
				temp2 = 0;
				temp3 = 0;
				for(int j=0;j<nobj;j++) {
					if(g[j]<=jointPopulation.get(i).getObjectives(j)) {
						temp1 = temp1 + 1;
					}else if (g[j]>=jointPopulation.get(i).getObjectives(j)) {
						temp2 = temp2 + 1;
					} else {
//						temp3 = temp3 + 1;
					}
					
				}
				//对统计的的值分配值
				if (temp1 == nobj) {
					flagArray[i] = 1;
				}
				if (temp2 == nobj) {
					flagArray[i] = 1;
				}
			}
			
			return flagArray;
		}

		
		//NSGA2的环境选择
		public List<Individual> replacement4(List<Individual> population, List<Individual> offspringPopulation) {
			
			List<Individual> jointPopulation = new ArrayList<>();
			
			
			//将父代和子代种群加入到混合种群中
			for(int i=0;i<population.size();i++) {
				jointPopulation.add(population.get(i)) ;
			}
			for(int i=0;i<offspringPopulation.size();i++) {
				jointPopulation.add(offspringPopulation.get(i)) ;
			}

			FastNonDominatedSortRanking2  fndsr = new FastNonDominatedSortRanking2();

			List<ArrayList<Individual>> ranking = fndsr.computeRanking(jointPopulation);
			

			
			
			//NSGA2中的步骤
			List<Individual> last = new ArrayList<>();
			List<Individual> pop = new ArrayList<>();
			List<List<Individual>> fronts = new ArrayList<>();
			

			int rankingIndex = 0;
			int candidateSolutions = 0;
			
			
			//一直添加，直到刚好满足或者超过了popSize的大小
			while (candidateSolutions < getPopSize()) {				
			  last = fndsr.getSubFront(rankingIndex);		

			  fronts.add(last);
			  candidateSolutions += last.size();
			  if ((pop.size() + last.size()) <= getPopSize()) {
				  for(int j=0;j<last.size();j++) {
					  pop.add(last.get(j));
				  }
				 
			  }			    
			  rankingIndex++;
			}
			if (pop.size() == this.getPopSize()) {
				return pop;
			}
			pop = crowdDistanceSelect(getPopSize() - pop.size(), pop,last);
			
			
			 
			return pop;
		}
		
		
		//根据拥挤距离来选择
		public List<Individual>  crowdDistanceSelect(int need,List<Individual> pop,List<Individual> last) {
			
			double fitnessArray[][] = new double[last.size()][nobj];
			double fitnessArrayNew[][] = new double[last.size()][nobj];
			double eps = 1.0e-14;
			double maxAndMinMatrix[][] = new double[2][nobj];//%第一行表示最大值，第二行最小值，第一列是第
			double maxValue = Double.MAX_VALUE;
			double minValue = Double.MIN_VALUE;
			//初始化
			for(int i=0;i<nobj;i++) {
				maxAndMinMatrix[0][i] = minValue;//这里是反着来的
				maxAndMinMatrix[1][i] = maxValue;
			}
		
			//先存入数组
			for(int i=0;i<last.size();i++) {
				for(int j=0;j<nobj;j++) {
					fitnessArray[i][j] = last.get(i).getObjectives(j);
				}
				
			}
			
			double oneObjec[] = new double[last.size()];
			for(int i=0;i<last.size();i++) {
				oneObjec[i] = fitnessArray[i][0];
			
			}
			
			int oriIndex[] = MyUtils.sortIndex(oneObjec);  //原有的顺序
//				
			//跟着变化
			for(int i=0;i<last.size();i++) {
				for (int j=0;j<nobj;j++) {
					fitnessArrayNew[i][j] = fitnessArray[oriIndex[i]][j];
				}
			}
			
			//计算拥挤距离
			double crowdD[] = new double[last.size()];
			//初始化
			for(int i=0;i<last.size();i++) {
				crowdD[i] = 0.0;
			}
			

			//计算最小值和最大值
			for(int i=0;i<last.size();i++) {
				for(int j=0;j<nobj;j++) {
					
					//比较最大值
					if (fitnessArray[i][j]>maxAndMinMatrix[0][j]) {
						maxAndMinMatrix[0][j] = fitnessArray[i][j];
					}
					//比较最小值
					if (fitnessArray[i][j]<maxAndMinMatrix[1][j]) {
						maxAndMinMatrix[1][j] = fitnessArray[i][j];
					} 


				}
			
		   }
			
			//拥挤距离计算
			for(int i=0;i<last.size();i++) {
				if(i==0 || i+1 ==last.size()) {
					crowdD[i] = maxValue;
				} else {
					for(int k=0;k<nobj;k++) {
						crowdD[i] = crowdD[i] +  Math.abs(fitnessArrayNew[i-1][k] - fitnessArrayNew[i+1][k]+eps)/(maxAndMinMatrix[0][k]-maxAndMinMatrix[1][k]+eps);
					}
					
				}
			}
			

			
			int oriIndex2[] = MyUtils.sortIndex(crowdD); 
			
			//加入到我所需要的队列中,从拥挤距离大那里开始挑
			int needArray[] = new int[need];
			int needIndex = 0;
			for (int i=last.size()-1;i>=0;i--) {
				needArray[needIndex] = oriIndex[oriIndex2[i]];
				
				needIndex = needIndex + 1;
				if(needIndex == need) {
					break;
				}
			}
			
			//添加到pop中
			for(int i=0;i<need;i++) {
				pop.add(last.get(needArray[i]));
			}
			return pop;
		}
		
		
	
		//偏好的环境选择，replacement3
		public List<Individual> preferenceBasedSelectionStrategy(List<Individual> population, List<Individual> offspringPopulation, double angle) {
		
			List<Individual> jointPopulation = new ArrayList<>();
			List<Individual> choosen = new ArrayList<>();
			
			//将父代和子代种群加入到混合种群中
			for(int i=0;i<population.size();i++) {
				jointPopulation.add(population.get(i)) ;
			}
			for(int i=0;i<offspringPopulation.size();i++) {
				jointPopulation.add(offspringPopulation.get(i)) ;
			}

			if (Tool.isDiversityLearning == 1) {
				estimatedSimilarity(jointPopulation);
			}
			
			//preferenceRegionSort 得到两个区域
			List<List<Integer>> FrontLayer =  preferenceRegionSortingStrategy2(jointPopulation);
	
			//
			List<Integer> canInPopIndivi = new ArrayList<>();
			List<Integer> lastRegionIndivi = new ArrayList<>();
			int FrontLayerFlag = 0;//用于标记当前处于第几个区域
			for(int i = 0; i < FrontLayer.size(); i++) {
				if (canInPopIndivi.size() + FrontLayer.get(i).size() >= getPopSize()) {
					lastRegionIndivi.addAll(FrontLayer.get(i));
					FrontLayerFlag = i;
					break;
				} else {
					canInPopIndivi.addAll(FrontLayer.get(i));
					FrontLayerFlag = i;
				}
			}
			
		    //new 的pop 
			List<Individual> pop = new ArrayList<>();
			if (FrontLayerFlag == 0) {
				
				if(lastRegionIndivi.size() == getPopSize()) {//如果刚刚好
					canInPopIndivi.addAll(lastRegionIndivi);
					for(int i = 0; i < canInPopIndivi.size(); i++) {
						pop.add(jointPopulation.get(canInPopIndivi.get(i).intValue()));
					}
					return pop;
				} else {
					//对第一个区域进行非支配排序
					List<Individual> lastRegionIndiviPopulation = new ArrayList<>();
					for(int i = 0; i < lastRegionIndivi.size(); i++) {
						lastRegionIndiviPopulation.add(jointPopulation.get(lastRegionIndivi.get(i).intValue()));
					}
					
					FastNonDominatedSortRanking2  fndsr = new FastNonDominatedSortRanking2();
					List<ArrayList<Individual>> ranking = fndsr.computeRanking(lastRegionIndiviPopulation);

					List<Individual> last = new ArrayList<>();
					
					List<List<Individual>> fronts = new ArrayList<>();
					int rankingIndex = 0;
					int candidateSolutions = 0;//canInPopIndivi.size()
					
					//再添加从 lastRegion中选择的 一直添加，直到刚好满足或者超过了popSize的大小
					while (candidateSolutions < getPopSize()) {				
					  last = fndsr.getSubFront(rankingIndex);	
					  fronts.add(last);
					  candidateSolutions += last.size();
					  if ((pop.size() + last.size()) <= getPopSize()) {
						  for(int j = 0; j < last.size(); j++) {
							  pop.add(last.get(j));
						  }
						 
					  }			    
					  rankingIndex++;
					}
					if (pop.size() == this.getPopSize()) {
						return pop;
					}
					
					EnvironmentalSelection2 selection =
					        new EnvironmentalSelection2(fronts,getPopSize() - pop.size(),getReferencePointsCopy(),
					               nobj);	
					
					
					choosen = selection.execute(last);
					for(int j = 0; j < choosen.size(); j++) {
						pop.add(choosen.get(j));
					}

					
				} 
			} else {//
				//对最上层的一个区域计算角度
				List<Individual> lastRegionIndiviPopulation = new ArrayList<>();
				double lastRegionPopulationFitness[][] = new double[nobj][lastRegionIndivi.size()];
				for(int i = 0; i < lastRegionIndivi.size(); i++) {
					lastRegionIndiviPopulation.add(jointPopulation.get(lastRegionIndivi.get(i).intValue()));
				}
				
				//得到上一层的fitness
				for(int k = 0; k < lastRegionIndiviPopulation.size(); k++) {
					for(int z = 0; z < nobj; z++) {
						lastRegionPopulationFitness[z][k] = lastRegionIndiviPopulation.get(k).getObjectives(z);
					}
					
				}
				//
				double angleArray[] = new double[lastRegionIndiviPopulation.size()];
				List<Integer> P = new ArrayList<>();
				List<Integer> Q = new ArrayList<>();
				double point[] = new double[g.length];
				for(int j = 0; j < lastRegionIndiviPopulation.size(); j++) {
					for(int k = 0; k < point.length; k++) {
						point[k] = lastRegionPopulationFitness[k][j];
					}
					
					angleArray[j] = getAngle(point, g);
					
					if (angleArray[j] < angle) {
						P.add(Integer.valueOf(lastRegionIndivi.get(j).intValue()));//直接换成以前联合种群的序号
					} else {
						Q.add(Integer.valueOf(lastRegionIndivi.get(j).intValue())) ;//直接换成以前联合种群的序号
					}
					
				}
				
				//再添加
				if (canInPopIndivi.size() + P.size() <= getPopSize()) {
					//把P换成原来的序号，已经换好了
					canInPopIndivi.addAll(P);
					if (canInPopIndivi.size() == getPopSize()) {
						for(int k = 0; k < canInPopIndivi.size(); k++) {
							pop.add(jointPopulation.get(canInPopIndivi.get(k).intValue()));
						}
						return pop;
					}else {
						//从Q中选 补充进去
						double QAllfitness[][] = new double[nobj][Q.size()];
						for(int k = 0; k < Q.size(); k++) {
							for(int z = 0; z < nobj; z++) {
								QAllfitness[z][k] = jointPopulation.get(Q.get(k).intValue()).getObjectives(z);
							}
						}
						double disArray1[] = getDPJ(QAllfitness,g);
						int sortIndex[] = MyUtils.sortIndex(disArray1);
						int currentCanInPopIndiviSize = canInPopIndivi.size();
						for (int k = 0; k < (getPopSize()-currentCanInPopIndiviSize); k++) {
							canInPopIndivi.add(Q.get(sortIndex[k]).intValue());
						}
						//添加到pop中
						for(int k = 0; k < canInPopIndivi.size(); k++) {
							pop.add(jointPopulation.get(canInPopIndivi.get(k).intValue()));
						}
						
					}
				} else {//对P非支配排序
					 
					List<Individual> PIndiviPopulation = new ArrayList<>();
					for(int i = 0; i < P.size(); i++) {
						PIndiviPopulation.add(jointPopulation.get(P.get(i).intValue()));
					}
					
					FastNonDominatedSortRanking2  fndsr = new FastNonDominatedSortRanking2();
					List<ArrayList<Individual>> ranking = fndsr.computeRanking(PIndiviPopulation);

					List<Individual> last = new ArrayList<>();
					List<List<Individual>> fronts = new ArrayList<>();
					int rankingIndex = 0;
					int candidateSolutions = canInPopIndivi.size();
					//先把之前的保存到pop
					for(int i = 0; i < canInPopIndivi.size(); i++) {
						pop.add(jointPopulation.get(canInPopIndivi.get(i).intValue()));
					}
					
					//再添加从 lastRegion中选择的 一直添加，直到刚好满足或者超过了popSize的大小
					while (candidateSolutions < getPopSize()) {				
					  last = fndsr.getSubFront(rankingIndex);			  
					  fronts.add(last);
					  candidateSolutions += last.size();
					  if ((pop.size() + last.size()) <= getPopSize()) {
						  for(int j=0;j<last.size();j++) {
							  pop.add(last.get(j));
						  }
						 
					  }			    
					  rankingIndex++;
					}
					if (pop.size() == this.getPopSize()) {
						return pop;
					}
					
//					
					EnvironmentalSelection2 selection =
					        new EnvironmentalSelection2(fronts,getPopSize() - pop.size(),getReferencePointsCopy(),
					               nobj);	
					
					choosen = selection.execute(last);
					for(int j = 0; j < choosen.size(); j++) {
						pop.add(choosen.get(j));
					}
				}				
			}
			return pop;
		}


			//得到偏好flag  共两个区域
			public static int[] getPreferredFlag2(List<Individual> jointPopulation) { 
				int popLen = jointPopulation.size();
				int preferredFlagArray[] = new int[popLen];
				int temp1 = 0;

				//初始化
				for(int i=0;i<popLen;i++) {
					preferredFlagArray[i] = 2;//默认属于第三个区域
				}
				
				//统计判断属于哪个区域
				for (int i=0;i<popLen;i++) {
					temp1 = 0;
					for(int j=0;j<nobj;j++) {
						if(g[j]>=jointPopulation.get(i).getObjectives(j)) {
							temp1 = temp1 + 1;
						}else{
//							temp3 = temp3 + 1;
						}
						
					}
					//对统计的的值分配值
					if (temp1 == nobj) {
						preferredFlagArray[i] = 1;
					}
				}
				
				return preferredFlagArray;
			
			}
			
			
			//得到每个点与射线的垂直距离
			public static double[] getDPJ(double lastRegionPopulationFitness[][], double R[]) {
				int popLen = lastRegionPopulationFitness[0].length;
				double dpij[] = new double[popLen];
				double OR[] = R.clone();
				double E[] = R.clone();
				double Oi[] = R.clone();
				double OB[] = R.clone();
				double OBOi[] = R.clone();
				double temp = 0.0;
				for (int i=0;i<R.length;i++) {
					E[i] = OR[i]/norm(R);
				}
				//
				for(int i=0;i<popLen;i++) {
					for(int j=0;j<Oi.length;j++) {
						Oi[j] = lastRegionPopulationFitness[j][i];
					}
					double OiE = 0.0;
					for(int k=0;k<Oi.length;k++) {
						OiE = OiE + Oi[k]*E[k];
					}
					temp = OiE/(norm(Oi)*norm(E));
					double sita = Math.acos(temp);
					for(int k=0;k<E.length;k++) {
						OB[k] = Math.cos(sita)*(norm(Oi)*E[k]);
					}
					for(int k=0;k<E.length;k++) {
						OBOi[k] = OB[k] - Oi[k];
					}
					dpij[i] = Math.pow(Math.pow(norm(OBOi), 2),0.5);					
				}
				
				return dpij;
				
				
			}
			
			
			//实现范数
			public static double norm(double R[]) {
				double no = 0;
				for(int i=0;i<R.length;i++) {
					no = no + Math.pow(R[i], 2.0);
				}
				return Math.pow(no, 0.5);
			}
			
			
			//得到每个点之间的角度
			public static double getAngle(double x[], double y[]) { //默认的起点为(0,0)
				double xy = 0.0;
				double angle = 0.0;
				for(int k=0;k<x.length;k++) {
					xy = xy + x[k]*y[k];
				}
				angle = Math.toDegrees(Math.acos(xy/(norm(x)*norm(y)))) ;//*(180/Math.PI)
			    return angle;
			}
			
			
			//更新memory,存储时间最近的solution,替换掉离当前时间久远的解。
			public void updateMemory() { 
				FastNonDominatedSortRanking2  fndsr = new FastNonDominatedSortRanking2();
				List<ArrayList<Individual>> ranking = fndsr.computeRanking(population);

				List<Individual> last = new ArrayList<>();
				List<Individual> pop = new ArrayList<>();
				List<Individual> memoryTemp = new ArrayList<>();
				
				int rankingIndex = 0;
				
					
				  last = fndsr.getSubFront(rankingIndex);					  
				  for(int j=0;j<last.size();j++) {
					  pop.add(last.get(j));
				  }
				  if (memory.size() + pop.size() <= memorySize) {
					  memoryTemp.addAll(pop);//更新的解保存在前面
					  memoryTemp.addAll(memory);
				  } else {
					  memoryTemp.addAll(pop);
					  int i = 0;
					  while(memoryTemp.size() < memorySize) {
						  memoryTemp.add(memory.get(i));
						  i = i + 1;
					  }
				  }
				  
				  Tool.memory = memoryTemp;
				   
				 		   
			}
			
			
			
			
			//Diversity maintenance / updateArchive
			public void estimatedSimilarity(List<Individual> jointPopulation) {
				//计算相似度
				
				int simpleSize = 20;
				double sim[][] = new double[jointPopulation.size()][simpleSize];
				double simMean[] = new double[jointPopulation.size()];//个体对其他的平均相似度
				double temp;
				int sortIndex[] = new int[jointPopulation.size()];
				int rankObjSpace[] = new int[jointPopulation.size()];
				int rankDecSpace[] = new int[jointPopulation.size()];
				double rankGlobal[] = new double[jointPopulation.size()];
				int rankGlobalSortIndex[] = new int[jointPopulation.size()]; 
				//计算平均相似度
				for(int i = 0; i < jointPopulation.size(); i++) {
					temp = 0.0;
					//选择sample个
					int popOrder[] = MyUtils.generateUqiInt(jointPopulation.size());
	
					for(int j = 0; j < simpleSize; j++) {
						sim[i][j] = MyUtils.similarity(jointPopulation.get(i).getAllVariables(), jointPopulation.get(popOrder[j]).getAllVariables());
						temp = temp + sim[i][j];
					}
					simMean[i] = temp/simpleSize;
				}
				sortIndex = MyUtils.sortIndex(simMean);
				for(int i = 0; i < jointPopulation.size(); i++) {
					rankDecSpace[sortIndex[i]] = i;
				}
				//计算拥挤距离
				rankObjSpace = crowdDistanceRank(jointPopulation);
				
				//计算总的rank
				for(int i = 0; i < jointPopulation.size(); i++) {
					rankGlobal[i] = (rankDecSpace[i] + rankObjSpace[i])/2.0;

				}
				rankGlobalSortIndex = MyUtils.sortIndex(rankGlobal);
				
				//设置容量大小
				int saveCanIndivi[];
				if (rankGlobalSortIndex.length > archiveSize) {
					saveCanIndivi = new int [archiveSize];
				} else {
					saveCanIndivi = new int [rankGlobalSortIndex.length];
				}
				
				for(int i = 0; i < saveCanIndivi.length; i++) {
					saveCanIndivi[i] = rankGlobalSortIndex[i];
				}
				//更新archive
				archive.clear();
				for(int i = 0; i < saveCanIndivi.length; i++) {
					archive.add(jointPopulation.get(saveCanIndivi[i]));
				}
			}
			
			
			
			//得到拥挤距离
			public static int[] crowdDistanceRank(List<Individual> jointPopulation) {
				
				int rank[] = new int[jointPopulation.size()];
				double fitnessArray[][] = new double[jointPopulation.size()][nobj];
				double fitnessArrayNew[][] = new double[jointPopulation.size()][nobj];
				double eps = 1.0e-14;
				double maxAndMinMatrix[][] = new double[2][nobj];//%第一行表示最大值，第二行最小值，第一列是第
				double maxValue = Double.MAX_VALUE;
				double minValue = Double.MIN_VALUE;
				//初始化
				for(int i=0;i<nobj;i++) {
					maxAndMinMatrix[0][i] = minValue;//这里是反着来的
					maxAndMinMatrix[1][i] = maxValue;
				}
			
				//先存入数组
				for(int i=0;i<jointPopulation.size();i++) {
					for(int j=0;j<nobj;j++) {
						fitnessArray[i][j] = jointPopulation.get(i).getObjectives(j);
					}
					
				}
				

				double oneObjec[] = new double[jointPopulation.size()];
				for(int i=0;i<jointPopulation.size();i++) {
					oneObjec[i] = fitnessArray[i][0];
				
				}
				
				int oriIndex[] = MyUtils.sortIndex(oneObjec);  //原有的顺序
//					
				//跟着变化
				for(int i=0;i<jointPopulation.size();i++) {
					for (int j=0;j<nobj;j++) {
						fitnessArrayNew[i][j] = fitnessArray[oriIndex[i]][j];
					}
				}
				
				//计算拥挤距离
				double crowdD[] = new double[jointPopulation.size()];
				//初始化
				for(int i=0;i<jointPopulation.size();i++) {
					crowdD[i] = 0.0;
				}
				

				//计算最小值和最大值

				for(int i=0;i<jointPopulation.size();i++) {
					for(int j=0;j<nobj;j++) {
						
						//比较最大值
						if (fitnessArray[i][j]>maxAndMinMatrix[0][j]) {
							maxAndMinMatrix[0][j] = fitnessArray[i][j];
						}
						//比较最小值
						if (fitnessArray[i][j]<maxAndMinMatrix[1][j]) {
							maxAndMinMatrix[1][j] = fitnessArray[i][j];
						} 


					}
				
			   }
				
				//拥挤距离计算
				
				for(int i=0;i<jointPopulation.size();i++) {
					if(i==0 || i+1 ==jointPopulation.size()) {
						crowdD[i] = maxValue;
					} else {
						for(int k=0;k<nobj;k++) {
							crowdD[i] = crowdD[i] +  Math.abs(fitnessArrayNew[i-1][k] - fitnessArrayNew[i+1][k]+eps)/(maxAndMinMatrix[0][k]-maxAndMinMatrix[1][k]+eps);
						}
						
					}
				}
				

				
				int oriIndex2[] = MyUtils.sortIndex(crowdD); 
				int needIndex = 0;
				for (int i=jointPopulation.size()-1;i>=0;i--) {
					rank[oriIndex[oriIndex2[i]]] = needIndex;
					needIndex = needIndex + 1;	
				}
				
				return rank;
			}
			
			
			
			//多样性学习策略
			public Individual multiSpaceDiversityLearningStrategy(Individual individual, List<Individual> archive) {
				Random random = new Random();
				int randIndiIndex = random.nextInt(archive.size());
				Individual t1 = archive.get(randIndiIndex);
				
				//t1和individual交叉
				double crossProb = 0.05;
				int crossLen = (int) Math.ceil(dimension*crossProb);
				Set<Integer> crossIndexArrayIntegers = null;
				int crossIndex = 0;
				
				crossIndexArrayIntegers = MyUtils.getRandoms(0, dimension, crossLen);
				//产生部分交换的gene
				Iterator<Integer> it = crossIndexArrayIntegers.iterator();
				while (it.hasNext()) {
					crossIndex = it.next().intValue();
					individual.setVariables(crossIndex, t1.getVariables(crossIndex));
					individual.setVariables2(crossIndex, t1.getVariables2(crossIndex));
				}
					
				
				return individual;
			}
			
			
			//去除重复的个体
			public List<Individual> removeDuplication(List<Individual> population) {
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
			
			
			/************************4GA**************************/
		  //初始化bestIndividual
			public void initBestIndividual() {
				bestIndividual.setObjectives(0, Double.MAX_VALUE);
				bestIndividual.setObjectives(1, Double.MAX_VALUE);
				bestIndividual.setObjectives(2, Double.MAX_VALUE);
			} 
			
			  //产生子代 Test OK
			public List<Individual> reproduction4GA(List<Individual> population, double crossoverProbability, double mutationProbability) {
				  List<Individual> offspringPopulation = new ArrayList<>(getPopSize());
				  
				  //选择交配
				  population = TournamentSelection4GA(population);
	
	
				  for (int i = 0; i < getPopSize(); i+=2) {//注意这里是+2
				    List<Individual> parents = new ArrayList<>(2);
				    
				    parents.add(population.get(i));
				    parents.add(population.get(Math.min(i + 1, getPopSize()-1)));
	
				    
				    //交叉变异
				    List<Individual> offspring = Intercrossover(crossoverProbability, parents.get(0), parents.get(1));
				    
				    //
				    offspring = IntercrossoverOrder(crossoverProbability, offspring.get(0), offspring.get(1));
	
				    Intermutation(mutationProbability, offspring.get(0));
				    //
				    IntermutationOrder(mutationProbability, offspring.get(0));
				    
				    Intermutation(mutationProbability, offspring.get(1));
				    //
				    IntermutationOrder(mutationProbability, offspring.get(1));
				    
	
	
				    //再计算下fitness
				    
				    evaluatePop4GA(offspring);
				    
				    offspringPopulation.add(offspring.get(0));
				    offspringPopulation.add(offspring.get(1));
				  }
			
		         return offspringPopulation ;
		  } 
			
			
			public List<Individual> replacement4GA(List<Individual> population, List<Individual> offspringPopulation) {
				
					List<Individual> jointPopulation = new ArrayList<>();
					List<Individual> pop = new ArrayList<>();
					
					//将父代和子代种群加入到混合种群中
					for(int i=0;i<population.size();i++) {
						jointPopulation.add(population.get(i)) ;
					}
					for(int i=0;i<offspringPopulation.size();i++) {
						jointPopulation.add(offspringPopulation.get(i)) ;
					}
					
					//挑选出适应值最小的N个个体
					double fitnessArray[] = new double[jointPopulation.size()];
					for(int i=0;i<fitnessArray.length;i++) {
						fitnessArray[i] = jointPopulation.get(i).getObjectives(Tool.objectiveNow);
					}
					int sortIndex[] = MyUtils.sortIndex(fitnessArray);
					//选择N个
					for(int i=0;i<population.size();i++) {
						pop.add(jointPopulation.get(sortIndex[i]));
					}
					
					//更新best
					if(pop.get(0).getObjectives(Tool.objectiveNow) < bestIndividual.getObjectives(Tool.objectiveNow)) {
						bestIndividual = pop.get(0).copy();
					}

					return pop;
				}
			
			
			//生成随机energy
			public double randEnergy(List<Individual> population) {
				double temp = 0.0;
				for (int i = 0; i < population.size(); i++ ) {
					temp = temp + population.get(i).getObjectives(2);
				}
				return temp / population.size();
			}
			
			
			
			public List<Individual> TournamentSelection4GA(List<Individual> population) {
		
				List<Individual> cpopulation = new ArrayList<>(getPopSize());
				Random rand = new Random();
				int induIndex1 = 0;
				int induIndex2 = 0;
		
				int selectIndex = 0;
		//		FastNonDominatedSortRanking2 fndsr = new FastNonDominatedSortRanking2 ();
		    	for (int i=0;i<getPopSize();i++) {
		    		//选俩
		    		induIndex1 = rand.nextInt(getPopSize());
		    		induIndex2 = rand.nextInt(getPopSize());
		
		    		if ((population.get(induIndex1).getObjectives(Tool.objectiveNow) < population.get(induIndex2).getObjectives(Tool.objectiveNow))) {
		    			selectIndex = induIndex1;
		    		} else {
		    			selectIndex = induIndex2;
		    		}
		
		    		//再放置
		    		cpopulation.add(population.get(selectIndex));
		        		
		    	}
		    	
		    	population = cpopulation;
		    	return population;
	    }
	
			
		public void evaluatePop4GA(List<Individual> population) {
		
			double makespan;
			double cost;
			double energy;

 
			int taskOrder[] = new int[Tool.TaskNum];
			double tempOrder[];
			int sortRank[] = new int[Tool.TaskNum];
			
			for(int i=0;i<population.size();i++) {
				double temp[] = new double[Tool.TaskNum];
				int assignment[] = new int[Tool.TaskNum];
				temp = population.get(i).getAllVariables().clone();
				for(int j = 0; j < temp.length; j++) {
					assignment[j] = (int)(0 + temp[j] *(Tool.VmNum));	
				}
				Tool.allot = assignment;
				
				tempOrder = population.get(i).getAllVariables2().clone();
				sortRank = MyUtils.sortIndex(tempOrder);
				for(int k = 0; k < sortRank.length; k++) {
					taskOrder[sortRank[k]] = k;
				}
				Tool.taskOrder = taskOrder;
				
				MyFitnessFunction myFitness = new MyFitnessFunction(); 
				myFitness.scheduleSimulation(assignment, taskOrder);

				makespan = myFitness.calMakespan();
				cost = myFitness.calCost();
				energy = myFitness.calEnergyConsumption();
				
				
				population.get(i).setObjectives(0, makespan);
				population.get(i).setObjectives(1, cost);				
				population.get(i).setObjectives(2, energy);
				
			    
			}
		
	}	
		
		
	//计算种群的平均适应值,但返回的是平均的maksspan
    public double calMeanFitness() {
    	double meanArr[] = new double[nobj];
    	//初始化
    	for(int i=0;i<nobj;i++) {
    		meanArr[i] = 0;
    	}
    	
    	
    	//累加
    	for (int i = 0; i <population.size(); i++) {
    		for(int j=0;j<nobj;j++) {
    			meanArr[j] = meanArr[j] + population.get(i).getObjectives(j);
    		}
    		
    	}
    	
    	//求平均
    	for(int i=0;i<nobj;i++) {
    		meanArr[i] = meanArr[i]/population.size();
    	}
    	return meanArr[0];
    }
		
}

	
	




