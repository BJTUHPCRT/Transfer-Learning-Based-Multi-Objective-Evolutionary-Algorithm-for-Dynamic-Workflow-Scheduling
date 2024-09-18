package org.uma.jmetal.algorithm.multiobjective.tl4dmoea;
import TL4DMOEA.Tool;

public class Bounds {
	//分别是vm 和 task执行序号对应的上下界
	public double lowBound1[] = new double[Tool.TaskNum];
	public double upBound1[] = new double[Tool.TaskNum];
	public double lowBound2[] = new double[Tool.TaskNum];
	public double upBound2[] = new double[Tool.TaskNum];
	
	//全部采用0-1的小数来初始化，然后把小数映射到整数中
	public Bounds() {
		for (int i=0;i<Tool.TaskNum;i++) {
			lowBound1[i] = 0;
			upBound1[i] = 1;//留意这里
			lowBound2[i] = 0;
			upBound2[i] = 1;//留意这里
		}
	}
	
	
	public double getLowerBound( int index) {
		return lowBound1[index];
	}
	
	public double getUpperBound(int index) {
		return upBound1[index];
	}
	public double getLowerBound2( int index) {
		return lowBound2[index];
	}
	
	public double getUpperBound2(int index) {
		return upBound2[index];
	}
}
