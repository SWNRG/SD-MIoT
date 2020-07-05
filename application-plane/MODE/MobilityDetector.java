package Controller;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.Arrays;
import java.io.*;

// Graph
import org.graphstream.graph.Graph;
import org.graphstream.graph.Node;
import org.graphstream.graph.implementations.MultiGraph;
import org.jfree.data.general.DatasetChangeEvent;

import static org.graphstream.algorithm.Toolkit.*;
// WEKA
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.DistanceFunction;
import weka.core.Instances;
import weka.clusterers.AbstractClusterer;

// Queue
import java.util.LinkedList; 
import java.util.Queue; 

//Eigen Values and Vectors
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Locale;


/* ***************************************
 * Class: AdjacencyMatrix 
 * Use: Maintain a Graph Adjacency Matrix
 * *************************************** */
class AdjacencyMatrix{
	private int n=0;         // Graph size
	private String nodes[];  // Node Names/Labels
	private double adjM[][]; // Adjacency Matrix Table
	
	AdjacencyMatrix(int n){ // Construct empty Adjacency Matrix 
		this.n = n;                 // Graph nodes
		this.nodes = new String[n]; // Create node names/labels Table
		this.adjM = new double[n][n];  // Create Adjacency Matrix Table
		for(int i=0;i<n;i++){
			this.nodes[i] = ""; // Initialize to empty
			for(int j=0;j<n;j++){
				this.adjM[i][j] = 0; // Initialize to 0 adj value
			}
		}	
	}
	AdjacencyMatrix(Graph g){ // Construct Adjacency Matrix from Graph g 
		this.n = g.getNodeCount();  // Graph nodes
		this.nodes = new String[n]; // Create node names/labels Table
		this.adjM = new double[n][n];  // Create Adjacency Matrix Table	
		
		int intAdjM[][] = new int[n][n]; // Int Adjacency Matrix because of graph lib requirement	
		intAdjM = getAdjacencyMatrix(g); // Read Graph Adjacency Matrix Table
		for(int i=0;i<n;i++){
			this.nodes[i] = g.getNode(i).toString(); // Read graph Node names/labels
			for(int j=0;j<n;j++){
				this.adjM[i][j] = (double)intAdjM[i][j]; // Copy adj value
			}
		}	
		sort(); // Sort Adjacency matrix in ascending order based on node labels
	}
	AdjacencyMatrix(AdjacencyMatrix srcAdjM){ // Construct Adjacency Matrix from Adjacency Matrix
		this.n = srcAdjM.size(); // Graph nodes
		this.nodes = new String[n]; // Create node names/labels Table
		this.adjM = new double[n][n];  // Create Adjacency Matrix Table
		
		for(int i=0;i<n;i++){
			this.nodes[i] = srcAdjM.nodes[i]; // Copy node label
			for(int j=0;j<n;j++){
				this.adjM[i][j] = srcAdjM.adjM[i][j]; // Copy adj value
			}
		}			
	}
	
	int size(){
		return this.n;
	}
	
	String[] getNodes(){
		return nodes;
	}
	
	void setNodes(String[] nodes){
		for(int i=0;i<n;i++) {
			this.nodes[i] = nodes[i];
		}
	}
	
	// Calculate the summation of adj matrix rows
	double[] getSumOfRows(){ 
		double sum[] = new double[this.n];
		for(int i=0;i<n;i++) 
			sum[i]=0; // Initialize array
		for(int i=0;i<n;i++){
			for(int j=0;j<n;j++){
				sum[i] += adjM[i][j]; // Sum values
			}
		}	
		return sum;
	}

	// Calculate Eigen Vectors
	double[] getEigenVectors(int vectorID){  
		double eigenVectors[] = new double[this.n];
		RealMatrix matrix = MatrixUtils.createRealMatrix(this.adjM);
	    EigenDecomposition decomposition = new EigenDecomposition(matrix);
	    //System.out.println("Eigenvector["+vectorID+"] = " + decomposition.getEigenvector(vectorID));
	    for(int i=0;i<n;i++){
	    	eigenVectors[i] = decomposition.getEigenvector(vectorID).getEntry(i);			
		}
	    return eigenVectors;
	}
	// Calculate Eigen Values
	double[] getEigenValues(){  
		RealMatrix matrix = MatrixUtils.createRealMatrix(this.adjM);
	    EigenDecomposition decomposition = new EigenDecomposition(matrix);
	    //System.out.println("EigenValues = " + Arrays.toString(decomposition.getRealEigenvalues()));
	    return decomposition.getRealEigenvalues();
	}
	
	// Subtract from adj matrix A - adj matrix B
	public static AdjacencyMatrix subtract(AdjacencyMatrix adjA, AdjacencyMatrix adjB){ 
		if(adjA.size() != adjB.size()){
			System.out.println("Mobile Engine error [subtract]: Substration of Adj Matrix with different size !!!");
			System.out.println("adjA.size="+adjA.size()+" adjB.size="+adjB.size());
			return null;
		}
		else{
			AdjacencyMatrix sub = new AdjacencyMatrix(adjA);
			for(int i=0;i<sub.size();i++){
				for(int j=0;j<sub.size();j++){
					sub.adjM[i][j] = Math.abs(sub.adjM[i][j] - adjB.adjM[i][j]); // Subtract values
				}
			}	
			return sub;
		}
	}	

	// Multiply values of a adj matrix by d (used to find the average)
	void multiply(double a){ 
		for(int i=0;i<n;i++){
			for(int j=0;j<n;j++){
				adjM[i][j] = adjM[i][j]*a; // multiply each value
			}
		}		
	}
	
	// Add values of a adj matrix to this
	void sum(AdjacencyMatrix adjMtoAdd){ 
		if(this.n != adjMtoAdd.size()){
			System.out.println("Mobile Engine error [sum]: Summation of Adj Matrix with different size !!!");
			System.out.println("adjM.size="+this.n+" adjMtoADD.size="+adjMtoAdd.size());
		}
		else{
			for(int i=0;i<n;i++){
				for(int j=0;j<n;j++){
					this.adjM[i][j] += adjMtoAdd.adjM[i][j]; // Sum values
				}
			}	
		}
	}	

	// Divide values of a adj matrix by d (used to find the average)
	void divide(int d){ 
		if(d!=0){
			for(int i=0;i<n;i++){
				for(int j=0;j<n;j++){
					adjM[i][j] = adjM[i][j]/d; // divide each value
				}
			}		
		}
	}
	
	// Copy values of a adj matrix to this
	void copy(AdjacencyMatrix srcAdjM){ 
		this.n = srcAdjM.size();
		for(int i=0;i<n;i++){
			this.nodes[i] = srcAdjM.nodes[i];
			for(int j=0;j<n;j++){
				this.adjM[i][j] = srcAdjM.adjM[i][j];
			}
		}	
	}
	
	// Print the adj Matrix
	void print(String msg){
		System.out.println(msg);
		System.out.print("AdjMt:");
		for(int i=0;i<n;i++){
			System.out.printf("%s,",nodes[i]);
		}
		System.out.println();
		for(int i=0;i<n;i++){
			System.out.print(nodes[i]+":");
			for(int j=0;j<n;j++){
				System.out.printf("%5.2f",adjM[i][j]);
				if(j<n-1) System.out.print(",");
			}		
			System.out.println();
		}		
	}

	// Write to File the adj Matrix
	void printToFile(BufferedWriter bw, double t){
		try {
			bw.write("T="+t); 
			bw.newLine();
			bw.write("AdjMt:,");
			bw.write(Arrays.toString(nodes));
			bw.newLine();
			for(int i=0;i<n;i++){
				bw.write(nodes[i]+",");
				bw.write(Arrays.toString(adjM[i]));
				bw.newLine();
			}	
		} catch (IOException e) {
			e.printStackTrace();
		}
	}    
    
	// Sort adj Matrix based on graph node labels in ascending order
	void sort(){								
		int sl=0; 
		double help=0;
		String shelp,min;
		for(int i=0;i<n;i++){
			//System.out.println("i="+i);
			sl = i;
			min = nodes[i];
			for(int j=i+1;j<n;j++){
				if(min.compareTo(nodes[j])>=1){
					sl=j;
					min = nodes[j];
				}
			}	

			if(i!=sl){
				shelp = nodes[i]; // Swap Node names
				nodes[i] = nodes[sl];
				nodes[sl] = shelp;
				for(int j=0;j<n;j++){
					help = adjM[i][j]; // Swap Line
					adjM[i][j] = adjM[sl][j];
					adjM[sl][j] = help;
				}
				for(int j=0;j<n;j++){
					help = adjM[j][i]; // Swap Column
					adjM[j][i] = adjM[j][sl];
					adjM[j][sl] = help;
				}
			}
		}	
	}
} // END: Class AdjacencyMatrix


/* *************************************************
 * Class: AdjacencyMatrixQueue 
 * Use: Maintain a Queue of Graph Adjacency Matrices
 * ************************************************* */
class AdjacencyMatrixQueue{
	int windowMaxSize;  //Depth of the summation queue	
	Queue<AdjacencyMatrix> queueOfAdjM = new LinkedList<>(); // Queue
	
	AdjacencyMatrixQueue(int windowMaxSize){
		if(windowMaxSize<2)
			this.windowMaxSize = 2;
		else
			this.windowMaxSize = windowMaxSize;
	}
	
	int size(){
		return queueOfAdjM.size();
	}
	
	void add(AdjacencyMatrix adjM){
		queueOfAdjM.add(adjM); // Add AdjacencyMatrix to Queue
		System.out.println("Add matrix in queue (new size:"+queueOfAdjM.size()+")");
		if(queueOfAdjM.size()>windowMaxSize){
			queueOfAdjM.remove(); // remove head to maintain window size (oldest queue value)
			System.out.println("Remove from queue");
		}
	}

	// Calculate the differences between the subsequent adjM and builds a queue of transition matrices
	Queue<AdjacencyMatrix> getTransitionMatrixQueue(){
		Queue<AdjacencyMatrix> TMqueue = new LinkedList<>(); // Transition Matrix Queue
		if(queueOfAdjM.size()>=2){ // If at least two adjM calculate transition matrix
			AdjacencyMatrix TM = new AdjacencyMatrix(queueOfAdjM.peek().size()); // Initialize Transition Matrix to 0os
			AdjacencyMatrix adjMprev = new AdjacencyMatrix(queueOfAdjM.peek()); // Set this matrix as the previous

			boolean first = true;
			TM.setNodes(adjMprev.getNodes()); // add node labels to the matrix
			for(AdjacencyMatrix adjM : queueOfAdjM){
				if(first){
					first = false; // Jump the first loop
				}
				else{ // Start from the second
					TM = AdjacencyMatrix.subtract(adjMprev,adjM); // Subtracted adjMatrices to find Transition Table				
					TMqueue.add(TM);
					adjMprev.copy(adjM);
				}
			}		
		} // end if not empty
		return TMqueue;	
	} // end getTransitionMatrixQueue	
	
	// Calculate the summation of the transition matrices queue
	AdjacencyMatrix getSumOfAdjMDifferences(){
		if(queueOfAdjM.size()<2){
			return null;
		}
		else{
			AdjacencyMatrix SD = new AdjacencyMatrix(queueOfAdjM.peek().size()); // Sum of differences initialize to 0os
			AdjacencyMatrix adjMprev = new AdjacencyMatrix(queueOfAdjM.peek()); // Initialize this matrix as the previous
			boolean first = true;
			SD.setNodes(adjMprev.getNodes());
			for(AdjacencyMatrix adjM : queueOfAdjM){
				if(first){
					first = false;
				}
				else{
					SD.sum(AdjacencyMatrix.subtract(adjMprev,adjM)); // Sum the subtracted tables				
					adjMprev.copy(adjM);
				}
			}
			return SD;
		}
	} // end sumOfDifferencesAdjM
	
	// Calculate the Simple Moving Average of the Transition Matrices Queue
	AdjacencyMatrix getSimpleMovingAverage(){
		Queue<AdjacencyMatrix> TMqueue = new LinkedList<>(); // Queue of Transition matrices
		TMqueue = getTransitionMatrixQueue();  // Calculate Queue of Transition matrices
		
		if(TMqueue.isEmpty()){
			return null;			
		}
		else{
			AdjacencyMatrix sma = new AdjacencyMatrix(queueOfAdjM.peek().size()); // Initialize Sum of differences initialize to 0os
			sma.setNodes(queueOfAdjM.peek().getNodes()); // add node labels to the matrix
			
			for(AdjacencyMatrix TM : TMqueue){
				sma.sum(TM); // Sum the transition matrices
			}
			sma.divide(TMqueue.size());
			return sma;		
		} // end if not empty
	} // end getSimpleMovingAverage
	
	// Calculate the Exponential Moving Average of the Transition Matrices Queue
	AdjacencyMatrix getExponentialMovingAverage(double alpha){
		Queue<AdjacencyMatrix> TMqueue = new LinkedList<>(); // Queue of Transition matrices
		TMqueue = getTransitionMatrixQueue();  // Calculate Queue of Transition matrices
		
		if(TMqueue.isEmpty()){
			return null;			
		}
		else{
			AdjacencyMatrix ema = null;
			AdjacencyMatrix emaOld = null;
			for(AdjacencyMatrix TM : TMqueue){
				if(emaOld == null){  // Initialize
					ema = new AdjacencyMatrix(TM);    //Create and copy TM
					emaOld = new AdjacencyMatrix(TM); //Create and copy TM
				}
				else{
					if(alpha>0){
						// newM = oldM + a*(new-old); Exponential Moving Average formula  
						ema = AdjacencyMatrix.subtract(TM, emaOld);
						ema.multiply(alpha);
						ema.sum(emaOld);
						emaOld.copy(ema);
					}
					else{ // SOTIRIS version
						//Ema(n)=(2/(p+1))X(n)+((p-1)/(p+1))*Ema(n-1)
						ema.copy(TM);
						//ema.multiply(2/(TMqueue.size()+1.0));
						//emaOld.multiply((TMqueue.size()-1.0)/(TMqueue.size()+1.0));
						ema.multiply(2/(windowMaxSize+1.0));
						emaOld.multiply((windowMaxSize-1.0)/(windowMaxSize+1.0));
						ema.sum(emaOld);
						emaOld.copy(ema);
					}
				}		
			}
			return ema;		
		} // end if not empty
	} // end getExponentialMovingAverage	
		
} // END: Class AdjacencyMatrixQueue


/* *************************************************
 * Class: MobilityEngine 
 * Use: Detects mobile nodes in a dynamic graph
 * ************************************************* */
public class MobilityEngine implements Runnable {
	//Configuration variables
	static String filter = "EMA"; // "SMA"=SimpleMovingAverage, "EMA"=ExponentialMovingAverage
	private int windowSize=11;
    static int clustersNum = 2; // Number of clusters for kmeans
    static double alpha = 0.5;  // Exponential moving average alpha value
    
    private Thread mbThread;
    private final AtomicBoolean running = new AtomicBoolean(false);
    private int interval;	 // Detect every xxx millisecond
    private static int N=0; 
	   
	public MobilityEngine(int interval, String filter, int windowSize) {
		this.interval = interval;
		this.filter= filter;
		this.windowSize = windowSize;
		System.out.println("Mobility Engnine established...");
		mbThread = new Thread(this);
		mbThread.start();
	}

	public void start(){
		this.start(60000);
	}
	
	public void start(int interval){
		this.interval = interval;
		System.out.println("Mobility Engnine starting... Interval:"+interval+" ms");
		running.set(true);	
	}
	public void stop(){
		System.out.println("Mobility Engnine stoping...");
		running.set(false);	
	}

	public int interval() {
		return interval;
	}

	public void setRefresh_time(int interval) {
		this.interval = interval;
	}

	// K means clustering
	// ==================================================
	int[] kmeansClustering(double[] values, int clusterNum, BufferedWriter bw4){
//		ArrayList<SilhouetteIndex> m_silhouetteIdx = new ArrayList<SilhouetteIndex>();
//		int m_bestK = 0;
		
		try {	
			// Prepare the dataset
			String arffString = "@relation SumOfDiff\n\n@attribute sum numeric\n\n@data\n";
			for(int i=0;i<values.length;i++){
				arffString = arffString + values[i]+ "\n";
			}
			Instances data = new Instances(new BufferedReader(new StringReader(arffString)));
			
			SimpleKMeans kmeans = new SimpleKMeans();
			kmeans.setSeed(10);
			kmeans.setPreserveInstancesOrder(true); //important parameter to set: preserver order, number of cluster.
			kmeans.setNumClusters(clusterNum);	
			kmeans.buildClusterer(data); // Calculate kmeans
			// This array returns the cluster number (starting with 0) for each instance
			// The array has as many elements as the number of instances
			int[] assignments = kmeans.getAssignments();
							
			// Centroids
			Instances centroids = kmeans.getClusterCentroids();
			for(int j=0; j<centroids.numInstances(); j++){
			    double cent = centroids.instance( j ).value( 0 );
				System.out.println("Centroid["+j+"]="+ cent);
				bw4.write(" Centroid["+j+"]="+ cent);
			} 
			bw4.newLine();
			return assignments;	
			
/*			// ELBOW number of clusters estimation
 *          //*************************************************************************			
			double logWk[] = new double[clusterNum];
			int[][] assignments = new int[clusterNum][];
			for (int i = 0; i < clusterNum; i++) {
				SimpleKMeans kmeans = new weka.clusterers.SimpleKMeans(); 
				DistanceFunction m_distanceFunction = new EuclideanDistance();
				kmeans.setSeed(10);
				kmeans.setPreserveInstancesOrder(true); //important parameter to set: preserver order, number of cluster.
				kmeans.setNumClusters(i+1);
				kmeans.buildClusterer(data); // Calculate kmeans
				// This array returns the cluster number (starting with 0) for each instance
				// The array has as many elements as the number of instances
				assignments[i] = kmeans.getAssignments();
				
				Instances m_instances = data;		
				m_silhouetteIdx.add(new SilhouetteIndex());	
				m_silhouetteIdx.get(i).evaluate(kmeans, kmeans.getClusterCentroids(),
					m_instances, m_distanceFunction);
						
				// This array gets the k value of the clustering using SSE calculation
				logWk[i] = Math.log(kmeans.getSquaredError() / data.numInstances()); // SSE calculation
				System.out.println("For groups:"+(i+1)+" Kval="+logWk[i]+" Kmeans:" + Arrays.toString(assignments[i]));
				bw4.write("For groups:"+(i+1)+" Kval="+logWk[i]+" Kmeans:" + Arrays.toString(assignments[i]));
				bw4.newLine();

			}
			System.out.println("All K-values:" + Arrays.toString(logWk));	
			bw4.write("All K-values:" + Arrays.toString(logWk));
			bw4.newLine();
			// Find the smallest value
			double min=logWk[0];
			int minpos = 0;
			for(int i=1; i<logWk.length; i++){
				if(logWk[i] > min){ // MAX
					min = logWk[i];
					minpos = i;
				}
			}
			System.out.println("Smallest K-value:" + min + " in pos:"+minpos);
			System.out.println("Return:"+Arrays.toString(assignments[minpos]));
			bw4.write("Smallest K-value:" + min + " in pos:"+minpos+" Return:"+Arrays.toString(assignments[minpos]));
			bw4.newLine();	
			bw4.flush();	
			return assignments[minpos];
*/	

/*			// Silhouette number of clusters estimation (problem fining one cluster)
 *          //**********************************************************************			
			double logWk[] = new double[clusterNum];
			int[][] assignments = new int[clusterNum][];
			for (int i = 0; i < clusterNum; i++) {
				SimpleKMeans kmeans = new weka.clusterers.SimpleKMeans(); 
				DistanceFunction m_distanceFunction = new EuclideanDistance();
				kmeans.setSeed(10);
				kmeans.setPreserveInstancesOrder(true); //important parameter to set: preserver order, number of cluster.
				kmeans.setNumClusters(i+1);
				kmeans.buildClusterer(data); // Calculate kmeans
				assignments[i] = kmeans.getAssignments();
				
				Instances m_instances = data;		
				m_silhouetteIdx.add(new SilhouetteIndex());	
				m_silhouetteIdx.get(i).evaluate(kmeans, kmeans.getClusterCentroids(),
					m_instances, m_distanceFunction);
			}					
			double si = 0;
			for (int i = 0; i < m_silhouetteIdx.size(); i++) {
				System.out.println("Silhouette ["+(i+1)+"]:"+m_silhouetteIdx.get(i).getGlobalSilhouette());
				bw4.write("Silhouette ["+(i+1)+"]:"+m_silhouetteIdx.get(i).getGlobalSilhouette());
				bw4.newLine();
				if (m_silhouetteIdx.get(i).getGlobalSilhouette() > si) {
					si  = m_silhouetteIdx.get(i).getGlobalSilhouette();
					m_bestK = i;
				}
			}
			System.out.println("Best Silhouette:"+(m_bestK+1));
			bw4.write("Best Silhouette:"+(m_bestK+1));
			bw4.newLine();
			return assignments[m_bestK+1];
*/
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
	} // End kmeansClustering method

	// Find the average of a cluster
	double calcAvg(double values[], int clusters[], int cluster){
		double sum=0;
		int avg_n=0;
		for(int i=0; i<clusters.length; i++) {
			if(clusters[i]==cluster){
				sum += values[i];
				avg_n++;
			}
		}
		return (avg_n!=0?sum/avg_n:0);
	}
	
	@Override
	public void run() {
		double t = 0;  // Time series counter
		AdjacencyMatrixQueue adjMQueue = new AdjacencyMatrixQueue(windowSize);  // Queue of adj Matrices
		int delayAdjustment=0; 

	    FileWriter fw1,fw2,fw3,fw4; //fileWriter
	    BufferedWriter bw1 = null, bw2 = null, bw3 = null, bw4 = null; // bufferedWriter
		try {
			fw1 = new FileWriter("mob-results1.csv");
	        bw1 = new BufferedWriter(fw1);
			fw2 = new FileWriter("mob-results2.csv");
	        bw2 = new BufferedWriter(fw2);
			fw3 = new FileWriter("mob-results3.csv");
	        bw3 = new BufferedWriter(fw3);
			fw4 = new FileWriter("mob-results4.txt");
	        bw4 = new BufferedWriter(fw4);
			bw3.write("Time,mob,sta,avg0,avg1,SumofRows");
			bw3.newLine();
		} catch (IOException e1) {
			e1.printStackTrace();
		}  
		
        running.set(false);
     
        while (true) {
            try { 
            	if(t==0){
            		Thread.sleep(60000); //Initial wait 
            		N = Controller.gns.getNodeCount();
            	}
            	else {
            		Thread.sleep(interval-delayAdjustment); //??? remove /2
            		System.out.println("Delay Adjustment:"+delayAdjustment);
            		delayAdjustment = 0;
            	}
            } catch (InterruptedException e){ 
                Thread.currentThread().interrupt();
                System.out.println("Mobility Thread was interrupted, Failed to complete operation");
            }
        	if(running.get()){
				try {
					System.out.println("Mobility sample t="+t+" window="+windowSize+" filter="+filter);
					bw4.write("Mobility sample t="+t); bw4.newLine();
					
					int n = Controller.gns.getNodeCount();
					System.out.println("Graph size: n="+n+" N="+N);

		            while(n!=30){
				//???	while(n!=N){
						try { 
			                Thread.sleep(150); 
			                delayAdjustment += 150;
			            } catch (InterruptedException e){ 
			                Thread.currentThread().interrupt();
			                System.out.println("Mobility Thread was interrupted, Failed to complete operation");
			            }
						n = Controller.gns.getNodeCount();
		            }    

		            // Get Graph Adjacency Matrix
		            AdjacencyMatrix adjM = new AdjacencyMatrix(Controller.gns.getGraph()); 
					adjM.print("Adjacency Matrix t="+t);  
					
					// Add adjacency matrix in the queue
					adjMQueue.add(adjM); 

					// Filter data series Simple or Exponential Moving Average Matrix
					AdjacencyMatrix maM = new AdjacencyMatrix(adjM.size());
					switch(filter){
						case "SMA":
							maM = adjMQueue.getSimpleMovingAverage(); // Calculate Simple Moving Average
							break;
						case "EMA":
							maM = adjMQueue.getExponentialMovingAverage(0); // Calculate Exponential Moving Average
							break;
						default: 
							System.out.println("Mobility: No Time Series filter defined!!!");
					}

					// Kmeans clustering
					if(maM!=null){
						maM.print(filter+" Moving Average at t="+t+"...");
						maM.printToFile(bw1, t);
						System.out.println(filter+" sumofrows="+Arrays.toString(maM.getSumOfRows()));

						int clusters[] = new int[maM.size()];
						clusters = kmeansClustering(maM.getSumOfRows(),clustersNum,bw4);

						for(int i=0; i<clusters.length; i++) {
						    System.out.printf("Node %s [%5.2f] -> Cluster %d \n", maM.getNodes()[i], maM.getSumOfRows()[i], clusters[i]);
						}

						// UPDATE the Graph
						if(clustersNum==2){
							double avg0 = calcAvg(maM.getSumOfRows(),clusters,0);
							double avg1 = calcAvg(maM.getSumOfRows(),clusters,1);			
							String str0="STA",str1="STA";
							if(avg1!=0){ // If two groups
								if(avg0>avg1){ // MOBILE nodes have larger avg
									str0 = "MOB";
								}
								else{
									str1 = "MOB";							
								}
							}
							if(Math.abs(avg0-avg1)<1.5){
								bw4.write("t="+t+" avg0="+avg0+" avg1=0"+avg1+" ACTIVATED");
								bw4.newLine();
								str0 = str1 = "STA";
							}
				            bw2.write(t+",");
							for(int i=0; i<clusters.length; i++) {
								switch(clusters[i]){
									case 0:
										Controller.gns.getNode(maM.getNodes()[i]).setAttribute("TYP", str0);
										break;
									case 1:
										Controller.gns.getNode(maM.getNodes()[i]).setAttribute("TYP", str1);
										break;
								}
					            bw2.write(Controller.gns.getNode(maM.getNodes()[i])+","+Controller.gns.getNode(maM.getNodes()[i]).getAttribute("TYP")+",");
							}
				            bw2.newLine();

							// SAVE RESULTS TO FILE
							System.out.println("Graph node types");
				            int mob=0,sta=0;
							for(int i=0; i<clusters.length; i++){ 
								System.out.println(Controller.gns.getNode(i)+" "+Controller.gns.getNode(i).getAttribute("TYP"));
					            if(Controller.gns.getNode(i).getAttribute("TYP").equals("MOB")) mob++; 
					            if(Controller.gns.getNode(i).getAttribute("TYP").equals("STA")) sta++; 
							}
												    
				            bw3.write(t+","+mob+","+sta+","+avg0+","+avg1+","+Arrays.toString(maM.getSumOfRows()));
				            bw3.newLine();
				            
				            bw1.flush();
				            bw2.flush();
				            bw3.flush();
				            bw4.flush();
						}
						else{
							System.out.println("Mobility cannot update graph because of clustersNum="+clustersNum);
						}
										
						// "FUTURE extension" find Eigen Values 
						//System.out.println("Eigen Values="+Arrays.toString(maM.getEigenValues()));
						//System.out.println("Eigen Vectors="+Arrays.toString(maM.getEigenVectors(0)));
					} // end if maM!=null
															
/*					// "FUTURE extension" use DEGREE Matrix
                    System.out.println("Degrees Matrix ...");
					for(Node node : Controller.graph){
						System.out.println(node.toString()+":"+node.getDegree());
					}
					// "FUTURE extension" use DISTANCE Matrix
					System.out.println("Distance Matrix ...");
					int disM[][] = new int[n][n];
					for(Node node : Controller.graph){
					    CALCULATE DIJKSTRA to find all paths to node
						for(int j=0;j<n;j++){
					        disM[node][j] = GET SHORTEST PATH from node to j
					}
					// "FUTURE extension" use LAPLACIAN Matrix					
		            // Get Graph Adjacency Matrix
		            AdjacencyMatrix adjM = new AdjacencyMatrix(Controller.graph);
					for(int i=0;i<adjM.size();i++){
						for(int j=0;j<adjM.size();j++){
						    if(i==j){
						        adjM[i][j] = Controller.graph.getNode(i).getDegree(); // Add node degree
						    }
						    else{
						    	adjM[i][j] = -adjM[i][j];  // Turn 1s to -1s
						    }
						}
					}		             
					adjM.print("Laplacian Adjacency Matrix t="+(t++));  					
*/					
				} catch(NullPointerException e) {
					System.out.println("Mobility Engine problem: "+e);
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				t = t + 1; // Increase timer
        	}
        	else{ // Restart initialize when running is false
        		t = 0;
        	}
		} // end While true	
	}
		
}

