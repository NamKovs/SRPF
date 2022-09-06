package algorithm;

import java.io.BufferedReader;
import java.io.FileReader;

import weka.core.Instance;
import weka.core.Instances;
import algorithm.SRPF;

public class Example {
	public static void main(String[] args) throws Exception {

		int numOfIterations = 20;  // The number of Iterations.
		int sizeOfSamplePool = 100; // The size of sample pool.
		int numOfKNeighbors = 2;   // The k-nearest neighbors. 
		int numOfExperiments = 10; // Number of experiments.
		double ration = 0.05;	   // The percentage of labeled datasets in the training set.
		for (int i = 0; i < numOfExperiments; i++) {
			try {;
				BufferedReader s = new BufferedReader(new FileReader("src/data/parkinsons.arff"));
				Instances data = new Instances(s);
//				SRDF.normalize(data);
				data.setClassIndex(data.numAttributes() - 1);
				s.close();

				// Divide the dataset into training set and test set.
				int[] tempRandArray = SRDP_1.getRandomOrder(data.numInstances());
				int tempTrainingSize = new Double(data.numInstances() * 0.7).intValue();

				Instance tempInstances = null; 
				Instances train = new Instances(data, 0);
				Instances test = new Instances(data, 0);

				for (int m = 0; m < tempTrainingSize; m++) {
					tempInstances = data.instance(tempRandArray[m]);
					train.add(tempInstances);
				} // Of for m
				for (int m = tempTrainingSize; m < data.numInstances(); m++) {
					tempInstances = data.instance(tempRandArray[m]);
					test.add(tempInstances);
				} // Of for m
				
				
				// model training.
				SRPF srpf = new SRPF(numOfIterations, sizeOfSamplePool, numOfKNeighbors, ration);
				srpf.buildClassifier(train);
				
				// Prediction.
				double rmse = 0;
				for (int n = 0; n < test.numInstances(); n++) {
					Instance inst = test.instance(n);
					double val = srpf.classifyInstance(inst);
					rmse += (inst.classValue() - val) * (inst.classValue() - val);
				}
				System.out.println(Math.sqrt(rmse/ test.numInstances()));
				} catch (Exception e) {
				e.printStackTrace();
			}//Of for Try
		}//Of for i
	}//Of main
}
