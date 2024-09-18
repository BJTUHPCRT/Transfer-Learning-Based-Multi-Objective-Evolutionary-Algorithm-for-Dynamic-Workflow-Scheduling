package TL4DMOEA;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import org.cloudbus.cloudsim.Vm;
import org.uma.jmetal.algorithm.multiobjective.tl4dmoea.Individual;
import org.workflowsim.CondorVM;
import org.workflowsim.Task;








public class Tool {
	
	//workflow序号
	public static int workflowIndex = 0;
    //两个约束
    public static double k1 = 0.25; //relaxed constraint--->0.25     tight constraint----->0.75 
    public static double k2 = 0.25;
    //当前求的目标函数的序号，仅用于单独的GA，获取makespan和cost的上下界
    public static int objectiveNow = 0;    
    //角度
    public static double angleBase = 54.7356;//2-->45  3 -->54.7356
	
	//host数目
	public static int hostNum = 0;
    //资源数量
    public static  int VmNum = 0;
    //任务数量
    public static  int TaskNum = 0;//文件中实际的task数目
    //
    public static int fileTaskNum = 0;//用于读取文件
    //迭代次数
    public static final int IterationNum = 0;
    //种群数量
    public static final int populationSize = 105;
    //记忆存储size
    public static final int memorySize = 105;
  
    public static final int archiveSize = 105;
    //标记是否多样性学习
    public static int isDiversityLearning = 0;
    
    public static double trainingThreshold = 0.1;//0.2
 
    //偏好点 
    public static final double g[] = {1,1,1};
    //优化目标个数
    public static final int nobj = 3;
    
    public static final int divisions = 13;// （2,99） （3,13）
    


    
    /*************************动态的一些*************************/

    public static List<Individual> lastEnvirPop = new ArrayList<>();
    
    public static List<Individual> lastPop = new ArrayList<>();//仅仅用于辅助计算变化程度
    
    public static  Individual bestSolution = new Individual();
    
    public static double zeta = 0.2;//用于环境变化后，初始化一部分种群
    
    public static final int nEpochs = 0;
    
    public static int batchSize = 0;
    
    public static int nSample = 0;
    
    public static List<Individual> netInputs = new ArrayList<>();
    public static List<Individual> netoutputs = new ArrayList<>();
    
    public static double points[][] = new double[populationSize][nobj];
    
    public static String featureName = new String(" ");
    public static String labelName = new String(" ");
    
    public static int env = 0;//当前所处的环境
    public static int severity[] = {6,6,6,6,6,6,6,6,6,6};//taskNum = 100  light severity: 20%------->3,  heavy severity:40%----->6   
    public static int severity2[] = {6,6,6,6,6,6,6,6,6,6};//taskNum = 1000   20%------->6, 40%----->12   
    
    public static int experIndex  = 0;
    
    public static List<DynamicVMsInfo> DynVMInfoList =  null;
    
    public static int ChangeFrequency[] = {300,300,300,300,300,300,300,300,300,300}; //high frequency---->300   low frequency----->500
    
    
    public static List<Integer> resourcePool = null;//资源池更新
    
    public static int dynamicType[] = {0,3,3,3,3,3,3,3,3,3};//0-->no change 3--->changed

    public static int subWorkflows[] = {2};//需要的workflow ,2,3,6,7,10,11,14,15,18,19
    
    public static int envNum = 10;//环境的总变化
    
    public static List<Individual> memory = null;
    
    public static double degreeChange = 0.0;
    

    
    //解析文件的名字 
    public static char wfFileName[] = {'1'};
    //workflow 名字
    public static char workflowName[] = {'1'};

    //vmInfolist
    public static List<VMInfo> VMInfoList =   null;

    public static List<HostInfo> HostInfoList =   null;
    //得到makespan和cost的两个区间
    public static List<JobMakespanInfo> JobMakespanInfoList =   null;
    public static List<JobCostInfo> JobCostInfoList = null;
    //三个文件中的数据
    public static List<JobBudgetInfo> JobBudgetInfoList =   null;
    public static List<JobDeadlineInfo> JobDeadlineInfoList =   null;
    public static List<JobEnergyInfo> JobEnergyInfoList =   null;


    public static double jobBudget = 0;
    public static double jobDeadline = 0;
    public static double jobEnergy = 0;
    
    

    //tasklist ,vmlist
    public static List<Task> tasktList ;
    public static List<?> vmList;

    
    //pipepair
    public static  List<ArrayList<Integer>> pipelineTaskList = new ArrayList<>();
    
    public static void setVmList(List<?> list) {
        Tool.vmList = list;
    }

    public static void setTaskList(List<Task> list) {
    	Tool.tasktList = list;
    }
    public List<Task> getTaskList() {
        return tasktList;
    }

    public List<?> getVmList() {
        return vmList;
    }


    //当前进行任务分配方案
    public static int[] allot= new int[Tool.TaskNum];
    public static int[] taskOrder = new int[Tool.TaskNum];

    
  //根据约束值  产生jobMakespan.txt   jobCost.txt文件
    public static void dataProcessMakespanCost() throws IOException {
    	//得到jobmakespan和jobcost
    	JobMakespanInfoList = JobMakespanInfo.readJobMakespanInfo(LoadInfo.jobMakespanFileName);
    	JobCostInfoList = JobCostInfo.readJobCostInfo(LoadInfo.jobCostFileName);
    	
    	FileWriter fw1 = new FileWriter("G:\\eclipseWorkspace\\2.tldmoea4wrokflowScheduling2\\WorkflowSim\\WorkflowSim\\WorkflowSim-1.0-master\\examples\\TL4DMOEA\\jobBudget"+".txt");
    	FileWriter fw2 = new FileWriter("G:\\eclipseWorkspace\\2.tldmoea4wrokflowScheduling2\\WorkflowSim\\WorkflowSim\\WorkflowSim-1.0-master\\examples\\TL4DMOEA\\jobDeadline"+".txt");
    	
    	//
    	char wfFileName[];
    	double makespanDiff = 0.0;
    	double costDiff = 0.0;
    	//计算
    	for(int i=0;i<JobMakespanInfoList.size();i++) {
    		JobMakespanInfo jobMakespanInfo = JobMakespanInfoList.get(i);
    		JobCostInfo jobCostInfo = JobCostInfoList.get(i);
    		wfFileName = jobMakespanInfo.wfFileName.clone();
    		makespanDiff = jobCostInfo.makespan - k1*(jobCostInfo.makespan - jobMakespanInfo.makespan);
    		costDiff = jobMakespanInfo.cost  - k2*(jobMakespanInfo.cost - jobCostInfo.cost);
    		
    		//写入文件
    		fw1.write(new String(wfFileName)+" ");	
    		fw1.write(costDiff+"\n");	
    		fw2.write(new String(wfFileName)+" ");	
    		fw2.write(makespanDiff+"\n");	
    		
    	}
    	fw1.close();
    	fw2.close();
    	
    	
    			
    }


    //配制基本的参数
    public static void configureParameters(int workflowIndex) throws IOException {

    	//产生jobmakespan.txt  jobcost.txt文件
    	dataProcessMakespanCost();
    	
    	List<JobInfo> JobInfoList = JobInfo.readJobInfo(LoadInfo.jobFileName);
    	JobInfo job = JobInfoList.get(workflowIndex);
    	
    	if (job.taskNum == 0) {
    		System.out.println("LoadInfo类里面的路径可能错了~");
    		return ;
    	}
    	Tool.fileTaskNum = job.taskNum;
    	Tool.wfFileName = job.wfFileName.clone();
    	Tool.workflowName = job.workflowName.clone();
    	
    	if (Tool.fileTaskNum > 0 && Tool.fileTaskNum<=30 ) {
    		Tool.VmNum = 6;
    		Tool.hostNum = 2;
    	}
    	if (Tool.fileTaskNum > 30 && Tool.fileTaskNum<=60 ) {
    		Tool.VmNum = 12;
    		Tool.hostNum = 4;
    	}
    	if (Tool.fileTaskNum > 60 && Tool.fileTaskNum<=100 ) {
    		Tool.VmNum = 18;
    		Tool.hostNum = 6;	
    	}
    	if (Tool.fileTaskNum > 100 && Tool.fileTaskNum<=1000 ) {
    		Tool.VmNum = 60;
    		Tool.hostNum = 20; 		
    	}
    	
    	
    	//VMs
    	List<VMInfo> VMInfoList1 = VMInfo.readVMInfo(LoadInfo.vmFileName);
    	List<VMInfo> VMInfoList2 =   new ArrayList<>();
    	for(int i=0;i<Tool.VmNum;i++) {
    		VMInfoList2.add(VMInfoList1.get(i));
    	}
    	VMInfoList = VMInfoList2;
    	
    	//Hosts
    	List<HostInfo> HostInfoList1 = HostInfo.readHostInfo(LoadInfo.hostFileName);
    	List<HostInfo> HostInfoList2 =   new ArrayList<>();
    	for(int i=0;i<Tool.hostNum;i++) {
    		HostInfoList2.add(HostInfoList1.get(i));
    	}
    	HostInfoList = HostInfoList2;
    	//
    	
    	//读取vm动态的数据
    	List<DynamicVMsInfo> DynVMInfoList1 = DynamicVMsInfo.readDynamicVMsInfoInfo(LoadInfo.dynamicVMsFileName);
    	List<DynamicVMsInfo> DynVMInfoList2 = new ArrayList<>();
    	DynVMInfoList2.add(DynVMInfoList1.get(workflowIndex));
    	DynVMInfoList = DynVMInfoList2;
    	
    	//读取budget deadline randAlgorithmEnergy
    	JobBudgetInfoList = JobBudgetInfo.readJobBudgetInfo(LoadInfo.jobBudgetFileName);
    	JobDeadlineInfoList = JobDeadlineInfo.readJobDeadlineInfo(LoadInfo.jobDeadlineFileName);
    	JobEnergyInfoList = JobEnergyInfo.readJobEnergyInfo(LoadInfo.jobEnergyFileName);
    	
    	jobBudget = JobBudgetInfoList.get(workflowIndex).budget;
    	jobDeadline = JobDeadlineInfoList.get(workflowIndex).deadline;
    	jobEnergy = JobEnergyInfoList.get(workflowIndex).energy;
    	
    }


}
