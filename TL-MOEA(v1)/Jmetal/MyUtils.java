package org.uma.jmetal.algorithm.multiobjective.tl4dmoea;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

import org.netlib.util.doubleW;

import TL4DMOEA.Tool;

// (a,b,c), 取值范围,[a,b),  如果[0,10),10就会每次结果一样
public class MyUtils {
	
    public static void main(String[] args) {
    	Random random = new Random();
//    	int a = 2;
//    	double b = a;
    	//System.out.print(b);
//    	for(int i=0;i<10;i++) {
//    		System.out.println(random.nextInt(10));
//	}
//    	
		
//    	int oriSortIndex[] = MyUtils.generateUqiInt(10);
//    	for(int i=0;i<10;i++) {
//            System.out.println(oriSortIndex[i]);
//    	}
    	double a[] = { 1, 1};
		double b[] = { 5, 5};
		System.out.println(similarity(a, b));
    	
    	
    }

    
  //计算欧式距离
  	public static double euclideanDistance(double g[],double point[]) {
  		double temp = 0.0;
  		double distance = 0.0;

  		for(int i=0;i<g.length;i++) {
  			temp = temp + Math.pow((g[i]- point[i]),2.0);
  		}
  		distance = Math.pow(temp, 0.5);
  		return distance;
  	}
    
  	//重写
  	public static double euclideanDistance(int g[],int point[]) {
  		double temp = 0.0;
  		double distance = 0.0;

  		for(int i=0;i<g.length;i++) {
  			temp = temp + Math.pow(((double)g[i]- (double)point[i]),2.0);
  		}
  		distance = Math.pow(temp, 0.5);
  		return distance;
  	}
  	
  	//得到每个点之间的角度
	public static double getAngle(double x[], double y[], double x1[], double y1[]) { //x1 和 y1为x,y对应的起点。
		//先处理下
		double x2[] = new double[x.length]; 
		double y2[] = new double[y.length]; 
		//向量尾巴 - 头
		for(int i=0; i<x.length; i++) {
			x2[i] = x[i] - x1[i];
			y2[i] = y[i] - y1[i];
		}
		
		double xy = 0.0;
		double angle = 0.0;
		for(int k=0;k<x2.length;k++) {
			xy = xy + x2[k]*y2[k];
		}
		angle = Math.toDegrees(Math.acos(xy/(norm(x2)*norm(y2)))) ;//*(180/Math.PI)
	    return angle;
	}
  	
	//实现范数
	public static double norm(double R[]) {
		double no = 0;
		for(int i=0;i<R.length;i++) {
			no = no + Math.pow(R[i], 2.0);
		}
		return Math.pow(no, 0.5);
	}
	
	//找最小值
	public static int findMinIndex(double R[]) {
		double maxValue = Double.POSITIVE_INFINITY;
		
		int minIndex = -1;
		double minValue = maxValue;
		
		for(int i=0;i<R.length;i++) {
			if (R[i] < minValue) {
				minValue = R[i];
				minIndex = i;
			}
		}
		return minIndex;
	}
	
	//找最小值
	public static double findMinValue(double R[]) {
		double maxValue = Double.POSITIVE_INFINITY;
		
		int minIndex = -1;
		double minValue = maxValue;
		
		for(int i=0;i<R.length;i++) {
			if (R[i] < minValue) {
				minValue = R[i];
				minIndex = i;
			}
		}
		return minValue;
	}
	
	//找最大值
	public static double findMaxValue(double R[]) {
			
			double minValue = Double.NEGATIVE_INFINITY ;
			double maxValue = minValue;
			int maxIndex = -1;
			
			
			for(int i=0;i<R.length;i++) {
				if (R[i] > maxValue) {
					maxValue = R[i];
					maxIndex = i;
				}
			}
			return maxValue;
		}
	//找最大值
	public static int findMaxIndex(double R[]) {
		
		double minValue = Double.NEGATIVE_INFINITY ;
		double maxValue = minValue;
		int maxIndex = -1;
		
		
		for(int i=0;i<R.length;i++) {
			if (R[i] > maxValue) {
				maxValue = R[i];
				maxIndex = i;
			}
		}
		return maxIndex;
	}
	
	//数据归一化
	public static double[] normalization(double data[]) {
		double norData[] = new double[data.length];
		double maxValue = findMaxValue(data);
		double minValue = findMinValue(data);
		for (int i = 0; i < data.length; i++) {
			norData[i]= (data[i]- minValue ) / (maxValue - minValue + 0.000001); 
		}
		return norData;
	}
	
	
	//矩阵的转换
	public static double[][] transposeMatrix(double oldMatrix[][]) {
		int row = oldMatrix.length;
		int col = oldMatrix[0].length;
		double newMatrix[][] = new double[col][row];
		for(int i = 0; i < row; i++) {
			for(int j = 0; j < col; j++) {
				newMatrix[j][i] = oldMatrix[i][j];
			}
		}
		return newMatrix;
	 }
	
    //Generate unique integers  
    public static int[] generateUqiInt(int len) {
    	Random ra = new Random();
    	double temp[] = new double[len];
    	for(int i=0;i<len;i++) {
    		temp[i] = ra.nextDouble();
    	}
    	int oriSortIndex[] = MyUtils.sortIndex(temp); 
    	return oriSortIndex;
    }
    
    
    /**
     * 生成一组不重复随机数 (a,b,c)  范围是 [a,b)
     *
     * @param start 开始位置：可以为负数
     * @param end   结束位置：end > start
     * @param count 数量 >= 0
     * @return
     */
	
	public static double randValue(double y,double low,double up) {

		Random ra = new Random();
		double result = y;
		if (y<low) {
			result = low + ra.nextDouble()*(up - low);
		}
		if (y>up) {
			result = low + ra.nextDouble()*(up - low);
		}
		return result;
	}
	
	
	
    public static Set<Integer> getRandoms(int start, int end, int count) {
        // 参数有效性检查
        if (start > end || count < 1) {
            count = 0;
        }
        // 结束值 与 开始值 的差小于 总数量
        if ((end - start) < count) {
            count = (end - start) > 0 ? (end - start) : 0;
        }

        // 定义存放集合
        Set<Integer> set = new HashSet<>(count);
        if (count > 0) {
            Random r = new Random();
            // 一直生成足够数量后再停止
            while (set.size() < count) {
                set.add(start + r.nextInt(end - start));
            }
        }
        return set;
    }
    
    
    //排序并返回序号，这是从小到大的排序
    public static int[] sortIndex(double a[]) {
		int count = 0;//用于加入到数组中
		int oriSortIndex[] = new int[a.length];
		Number sorted[] = new Number[a.length];
        for (int i = 0; i < a.length; ++i) {
            sorted[i] = new Number(a[i], i);
        }
        Arrays.sort(sorted);
    
        for (Number n: sorted){
//            System.out.print("" + n.index + ",");
            oriSortIndex[count++] = n.index;
        }
        return oriSortIndex;
        
	}
    
    
  //方差s^2=[(x1-x)^2 +...(xn-x)^2]/n 或者s^2=[(x1-x)^2 +...(xn-x)^2]/(n-1)
    public static double Variance(double[] x) {
        int m=x.length;
        double sum=0;
        for(int i=0;i<m;i++){//求和
            sum+=x[i];
        }
        double dAve=sum/m;//求平均值
        double dVar=0;
        for(int i=0;i<m;i++){//求方差
            dVar+=(x[i]-dAve)*(x[i]-dAve);
        }
        return dVar/m;
    }

    
    //标准差σ=sqrt(s^2)
    public static double StandardDiviation(double[] x) {
        int m=x.length;
        double sum=0;
        for(int i=0;i<m;i++){//求和
            sum+=x[i];
        }
        double dAve=sum/m;//求平均值
        double dVar=0;
        for(int i=0;i<m;i++){//求方差
            dVar+=(x[i]-dAve)*(x[i]-dAve);
        }
                //reture Math.sqrt(dVar/(m-1));
        return Math.sqrt(dVar/m);
    }
    
    
    
  //计算余弦相似度
    public static double similarity(double aOri0[], double bOri0[]) {
       
    	//先转换成整数
		double aOri[] = new double[aOri0.length];
		double bOri[] = new double[bOri0.length];
		for(int j = 0; j < aOri0.length; j++) {
			aOri[j] = (int)(0 + aOri0[j] *(Tool.VmNum));
			bOri[j] = (int)(0 + bOri0[j] *(Tool.VmNum));
		}
    	//转换成容器
		ArrayList<Double> va = new ArrayList<Double>();
		ArrayList<Double> vb = new ArrayList<Double>();
		double ar[] = new double[aOri.length];
		double br[] = new double[aOri.length];
		for (int i = 0; i < aOri.length; i++)
		{ 
			//进行转换
			if (aOri[i] == bOri[i]) {
				ar[i]= 1.0;
				br[i]= 1.0; 
			} else {
				ar[i]= 1.0;
				br[i]= -1.0; 
			}
			va.add(new Double(ar[i]));
			vb.add(new Double(br[i]));
		}
		
        int size = va.size();
        double simVal = 0;
        double num = 0;
        double den = 0;
        double powa_sum = 0;
        double powb_sum = 0;
        for (int i = 0; i < size; i++) {
            double a = Double.parseDouble(va.get(i).toString());
            double b = Double.parseDouble(vb.get(i).toString());
 
            num = num + a * b;
            powa_sum = powa_sum + (double) Math.pow(a, 2);
            powb_sum = powb_sum + (double) Math.pow(b, 2);
        }
        double sqrta = (double) Math.sqrt(powa_sum);
        double sqrtb = (double) Math.sqrt(powb_sum);
        den = sqrta * sqrtb;
 
        simVal = num / den;
 
        return simVal;
    }



    
    
}


//辅助，排序并返回原来序号的方法
class Number implements Comparable<Number>{
  Double data;
  int index;

  Number(double d, int i){
      this.data = d;
      this.index = i;
  }
  
  @Override
  public int compareTo(Number o) {
      return this.data.compareTo(o.data);
  }
}







