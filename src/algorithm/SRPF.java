package algorithm;

import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.WAODE;
import weka.classifiers.functions.SMO;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Description:
 * 
 * ATTN:??This package is free for academic usage. You can run it at your own
 * risk. For other purposes, please contact Jia-hui Zhang
 * (namkovs@sina.com).
 * 
 * Requirement: To use this package, the whole WEKA environment (ver 3.7.0) must
 * be available.
 * 
 * Data format: Both the input and output formats are the same as those used by
 * WEKA.
 * 
 * @author Zhang Jia-Hui<br>
 *         Email:namkovs@sina.com <br>
 *         Date Created : 2022/04/28 <br>
 *         Last Modifide: 2022/05/13 <br>
 * 
 * @version 2.0
 */


public class SRPF {
	/**
	 * A global random object.
	 */
	public static final Random random = new Random(19951004);

	/**
	 * The labeled instances.
	 */
	public Instances labeledInstances;

	/**
	 * The unLabeled instances.
	 */
	public Instances unlabeledInstances;

	/**
	 * The subset L1 of labeled instances.
	 */
	public Instances subLabeledInstances1;

	/**
	 * The subset L2 of labeled instances.
	 */
	public Instances subLabeledInstances2;

	/**
	 * The number of iterations in the training process.
	 */
	public int numOfIterations = 100;

	/**
	 * The size of sample pool.
	 */
	public int sizeOfSamplePool = 20;

	/**
	 * The k nearest neighbors.
	 */
	public int numOfKNeighbors = 3;

	/**
	 * The proportion of labeled data in the training set.
	 */
	public double proportion = 0.05;

	/**
	 * The separate regressor.
	 */
	public Classifier separationRegressor = new weka.classifiers.functions.SMOreg();
	/**
	 * The predictive regressors.
	 */
	public Classifier[] predictiveRegressors = { new weka.classifiers.functions.SMOreg(),
			new weka.classifiers.functions.SMOreg() };
	/**
	 * The classifier.
	 */
	public Classifier classifier = new weka.classifiers.functions.SMO();

	/**
	 ********************
	 * The constructor.
	 * 
	 * @param paraTrainingSet      The given training set.
	 * @param paraNumOfIterations  The training iterations..
	 * @param paraSizeOfSamplePool The given pool size of the unlableledSet.
	 * @param paraNumOfKeighbors   The given k-nearest neighbors.
	 * @param paraProportion       The given proportion of labeled data in the
	 *                             training set.
	 ********************
	 */
	public SRPF(int paraNumOfIterations, int paraSizeOfSamplePool, int paraNumOfKeighbors, double paraProportion) {
		// Step 1 Set the parameters of constructor.
		numOfIterations = paraNumOfIterations;
		sizeOfSamplePool = paraSizeOfSamplePool;
		numOfKNeighbors = paraNumOfKeighbors;
		proportion = paraProportion;
	}// Of SRDP

	/**
	 ********************************** 
	 * Get a random order index array.
	 * 
	 * @param paraLength The length of the array.
	 * @return A random order.
	 ********************************** 
	 */
	public static int[] getRandomOrder(int paraLength) {
		// Step 1. Initialize
		int[] resultArray = new int[paraLength];
		for (int i = 0; i < paraLength; i++) {
			resultArray[i] = i;
		} // Of for i

		// Step 2. Swap many times
		int tempFirst, tempSecond;
		int tempValue;
		for (int i = 0; i < paraLength * 10; i++) {
			tempFirst = random.nextInt(paraLength);
			tempSecond = random.nextInt(paraLength);

			tempValue = resultArray[tempFirst];
			resultArray[tempFirst] = resultArray[tempSecond];
			resultArray[tempSecond] = tempValue;
		} // Of for i

		return resultArray;
	}// Of getRandomOrder

	/**
	 * Perform min-max normalization
	 * 
	 * @param data: the data set.
	 */
	static void normalize(Instances data) {
		for (int iAtt = 0; iAtt < data.numAttributes(); iAtt++) {
			double[] vals = new double[data.numInstances()];
			double max = Double.MIN_VALUE;
			double min = Double.MAX_VALUE;
			for (int i = 0; i < data.numInstances(); i++) {
				vals[i] = data.instance(i).value(iAtt);

				if (max < vals[i]) {
					max = vals[i];
				}

				if (min > vals[i]) {
					min = vals[i];
				}
			}

			for (int i = 0; i < data.numInstances(); i++) {
				double newval = (vals[i] - min) / (max - min);
				data.instance(i).setValue(iAtt, newval);
			}
		}
	}
	/**
	 *************************
	 * Uses quadratic polynomial features to increase the dimensionality of the data.
	 * 
	 * @param paraData 			The given Instances.
	 * @param paraClassValue	The give class value.
	 * @throws InterruptedException 
	 *************************
	 **/
	public Instances constructSingleCLassifierData(Instance paraData) throws InterruptedException {
		
		// Get the dimension of the feature after adding dimension.
		int tempLengthForData = paraData.numAttributes() - 1;
		int featureLength = tempLengthForData * 2;
		
		double feature[] = new double[featureLength];
		
		for (int i = 0; i < tempLengthForData; i++) {
			feature[i] = paraData.value(i);
		}
		
		for (int i = tempLengthForData; i < tempLengthForData*2; i++) {
			if(i < tempLengthForData * 2 -1) {
				feature[i] = paraData.value(i-tempLengthForData) - paraData.value(i+1-tempLengthForData);
			}else {
				feature[i] = paraData.value(i-tempLengthForData) - paraData.value(0);
			}
		}

		// Restructure the instance.
		FastVector tempCreateAtts;
		FastVector tempCreateAttVals;
		tempCreateAtts = new FastVector();
		Instances tempLabeledInstances;
		for (int i = 0; i < featureLength; i++) {
			tempCreateAtts.addElement(new Attribute("" + i));
		} // Of for i
		tempCreateAttVals = new FastVector();
		for (int i = 0; i < 2; i++)
			tempCreateAttVals.addElement("" + i);
		tempCreateAtts.addElement(new Attribute("class", tempCreateAttVals));
		tempLabeledInstances = new Instances("labeledDataForClassification", tempCreateAtts, 0);

		double[] vals = new double[featureLength + 1];
		for (int i = 0; i < featureLength; i++) {
			vals[i] = feature[i];
		}
		vals[featureLength] = tempCreateAttVals.indexOf("0");
		tempLabeledInstances.add(new Instance(1.0, vals));
		tempLabeledInstances.setClassIndex(tempLabeledInstances.instance(0).numAttributes() - 1);
		
		return tempLabeledInstances;
	}


	/**
	 ************************* 
	 * Initialization parameters.
	 * 
	 * @param paraTrainingData The training data set.
	 ************************* 
	 **/
	public void initializeParameters(Instances paraTrainingSet) {
		// Step 1: Split the training set into two sub-set.
		int[] tempRandArray = getRandomOrder(paraTrainingSet.numInstances());

		int tempSizeOfLabeledSet = new Double(paraTrainingSet.numInstances() * proportion).intValue();

		Instance tempInstances = null;
		labeledInstances = new Instances(paraTrainingSet, 0);
		unlabeledInstances = new Instances(paraTrainingSet, 0);

		for (int i = 0; i < tempSizeOfLabeledSet; i++) {
			tempInstances = paraTrainingSet.instance(tempRandArray[i]);
			labeledInstances.add(tempInstances);
		} // Of for i
		for (int i = tempSizeOfLabeledSet; i < paraTrainingSet.numInstances(); i++) {
			tempInstances = paraTrainingSet.instance(tempRandArray[i]);
			unlabeledInstances.add(tempInstances);
		} // Of for i

		// Step 2: Construct separation regressors and classifier.
		try {
			subLabeledInstances1 = new Instances(paraTrainingSet, 0);
			subLabeledInstances2 = new Instances(paraTrainingSet, 0);
			separationRegressor.buildClassifier(labeledInstances);

			for (int i = 0; i < labeledInstances.numInstances(); i++) {
				double tempPredictValue = separationRegressor.classifyInstance(labeledInstances.instance(i));
				double tempActualValue = labeledInstances.instance(i).classValue();

				if (tempPredictValue < tempActualValue) {
					subLabeledInstances1.add(labeledInstances.instance(i));
				} else {
					subLabeledInstances2.add(labeledInstances.instance(i));
				} // Of if
			} // Of for i

			predictiveRegressors[0].buildClassifier(subLabeledInstances1);
			predictiveRegressors[1].buildClassifier(subLabeledInstances2);

		} catch (Exception e) {
			e.printStackTrace();
		} // Of try...catch
	}// Of InitializationData
	
	/**
	 *************************
	 * Compute the index of k neighbors.
	 * 
	 * @param paraTarget The unlabeled instance.
	 * @param paraData 	 The labeled instances.
	 * @throws InterruptedException 
	 *************************
	 **/
	public double [][] getKNeighbors(Instance paraTarget, Instances paraData) throws InterruptedException {
		
		// Step 1. Calculate the distance between unlabeled data and labeled data..
		double distance[][] = new double[paraData.numInstances()][2];
		for (int i = 0; i < paraData.numInstances(); i++) {
			Instance _tempInstance = constructSingleCLassifierData(paraData.instance(i)).instance(0);
			distance[i][0] = i;
			double tempDistance = 0;
			for (int j = 0; j < paraTarget.numAttributes()-1; j++) {
				tempDistance += Math.pow(paraTarget.value(j) - _tempInstance.value(j), 2);
			}//of for j
			distance[i][1] = Math.sqrt(tempDistance);
 		}//of for i
		
		// Step 2. Data is sorted in ascending order.
		double tempNumber = 0;
		for (int i = 0; i < distance.length - 1; i++) {
			for (int j = i; j < distance.length; j++) {
				if(distance[i][1] > distance[j][1]) {
					
					tempNumber = distance[i][1];
					distance[i][1] = distance[j][1];
					distance[j][1] = tempNumber;
					
					tempNumber = distance[i][0];
					distance[i][0] = distance[j][0];
					distance[j][0] = tempNumber;					
					
				}//of if
			}//of for j
		}//of for i
		return distance;
	}//of getKNeighbors
	
	/**
	 *************************
	 * CCalculate confidence
	 * 
	 * @param data The training data set
	 * @throws Exception 
	 *************************
	 **/
	public double calculateCondidence(Instances paraInstances, Instance paraInstance, int paraNumOfRegressor) throws Exception {
		
		Instances tempInstances = new Instances(paraInstances);
		tempInstances.add(paraInstance);
		Classifier tempregressor = new weka.classifiers.functions.SMOreg();
		tempregressor.buildClassifier(tempInstances);
				
		double confidence = 0;
		double kNeighbor[][] = getKNeighbors(paraInstance, paraInstances);
		
		if (paraNumOfRegressor < 2) {
			for (int j = 0; j < kNeighbor.length; j++) {
				Instance tempData = paraInstances.instance((int)kNeighbor[j][0]);
				double firstNum = Math.pow(predictiveRegressors[paraNumOfRegressor].classifyInstance(tempData) - tempData.classValue(), 2);
				double secondNum = Math.pow(tempregressor.classifyInstance(tempData) - tempData.classValue(), 2);
				confidence += firstNum - secondNum;
			} // Of for j
		}
		else {
			for (int j = 0; j < kNeighbor.length; j++) {
				Instance tempData = paraInstances.instance((int)kNeighbor[j][0]);
				double firstNum = Math.pow(separationRegressor.classifyInstance(tempData) - tempData.classValue(), 2);
				double secondNum = Math.pow(tempregressor.classifyInstance(tempData) - tempData.classValue(), 2);
				confidence += firstNum - secondNum;
			} // Of for j
		}
		return confidence;
	}//Of calculateCondidence
	
	/**
	 *************************
	 * Construction model
	 * 
	 * @param data The given training set
	 * @throws Exception 
	 *************************
	 **/
	public void buildClassifier(Instances paraTrainingSet) throws Exception {
		
		// Step 1. Initialize the data set and regression
		initializeParameters(paraTrainingSet);
		int tempIteration = 0;
		
		// Step 2. Iteratively enrich the labeled samples.
		while (tempIteration < numOfIterations && unlabeledInstances.numInstances() >= 100) {
			
			// Step 2.1. Construct a sample pool Index
			int[] tempRandArray = getRandomOrder(unlabeledInstances.numInstances());

			// Step 2.2. paritioning sample data and calculate the confidence.
			int tempMostConfidentIndex1 = 0;
			int tempMostConfidentIndex2 = 0;
			int tempMostConfidentIndex3 = 0;
			double tempMostConfidentNumber1 = Double.MIN_VALUE;
			double tempMostConfidentNumber2 = Double.MIN_VALUE;
			double tempMostConfidentNumber3 = Double.MIN_VALUE;
			
			for (int i = 0; i < sizeOfSamplePool; i++) {
				Instance tempUnlabelInstance = unlabeledInstances.instance(tempRandArray[i]);
				Instance tempDataInstance = constructSingleCLassifierData(tempUnlabelInstance).instance(0);
				double kNeighbor[][] = getKNeighbors(tempDataInstance, labeledInstances);
				
				double tempRmse[] = new double[3];
				for (int j = 0; j < 5; j++) {
					Instance tempInstance = labeledInstances.instance((int)kNeighbor[j][0]);
					tempRmse[0] += Math.pow(predictiveRegressors[0].classifyInstance(tempInstance) - tempInstance.classValue(), 2);
					tempRmse[1] += Math.pow(predictiveRegressors[1].classifyInstance(tempInstance) - tempInstance.classValue(), 2);
					tempRmse[2] += Math.pow(separationRegressor.classifyInstance(tempInstance) - tempInstance.classValue(), 2);
				}
				if (tempRmse[0] < tempRmse[1] && tempRmse[0] < tempRmse[2]){
					if(tempMostConfidentNumber1 < calculateCondidence(subLabeledInstances1, tempUnlabelInstance, 0)){
						tempMostConfidentNumber1 = calculateCondidence(subLabeledInstances1, tempUnlabelInstance, 0);
						tempMostConfidentIndex1 = i;
					}//Of if
				}
				if (tempRmse[1] < tempRmse[0] && tempRmse[1] < tempRmse[2]){
					if(tempMostConfidentNumber2 < calculateCondidence(subLabeledInstances1, tempUnlabelInstance, 1)){
						tempMostConfidentNumber2 = calculateCondidence(subLabeledInstances1, tempUnlabelInstance, 1);
						tempMostConfidentIndex2 = i;
					}//Of if
				}
				if (tempRmse[2] < tempRmse[0] && tempRmse[2] < tempRmse[1]){
					if(tempMostConfidentNumber3 < calculateCondidence(subLabeledInstances1, tempUnlabelInstance, 2)){
						tempMostConfidentNumber3 = calculateCondidence(subLabeledInstances1, tempUnlabelInstance, 2);
						tempMostConfidentIndex3 = i;
					}//Of if
				}	
			}//Of for i
			
			
			// Step 2.3. Add unlabeled data to the labeled data set and delete it from the unlabeled data set.
			Instance addInstance1 = unlabeledInstances.instance(tempRandArray[tempMostConfidentIndex1]);
			Instance addInstance2 = unlabeledInstances.instance(tempRandArray[tempMostConfidentIndex2]);
			Instance addInstance3 = unlabeledInstances.instance(tempRandArray[tempMostConfidentIndex3]);
			double tempPredictNum1 = predictiveRegressors[0].classifyInstance(addInstance1);
			double tempPredictNum2 = predictiveRegressors[1].classifyInstance(addInstance2);
			double tempPredictNum3 = separationRegressor.classifyInstance(addInstance3);
			subLabeledInstances1.add(addInstance1);
			subLabeledInstances1.instance(subLabeledInstances1.numInstances() - 1).setClassValue(tempPredictNum1);		
			subLabeledInstances2.add(addInstance2);
			subLabeledInstances2.instance(subLabeledInstances2.numInstances() - 1).setClassValue(tempPredictNum2);
			
			
			labeledInstances.add(addInstance1);
			labeledInstances.instance(labeledInstances.numInstances() - 1).setClassValue(tempPredictNum1);
			labeledInstances.add(addInstance2);
			labeledInstances.instance(labeledInstances.numInstances() - 1).setClassValue(tempPredictNum2);
			labeledInstances.add(addInstance3);
			labeledInstances.instance(labeledInstances.numInstances() - 1).setClassValue(tempPredictNum3);
		
			// Step 2.4. Delete most confidence data from the unlabeled data. 
			unlabeledInstances.delete(tempRandArray[tempMostConfidentIndex1]);
			
			if ((tempRandArray[tempMostConfidentIndex2] == 0) || 
					(tempRandArray[tempMostConfidentIndex2]== 1 && tempRandArray[tempMostConfidentIndex1] == 0)) {
				unlabeledInstances.delete(tempRandArray[tempMostConfidentIndex2]);
			}else{
				unlabeledInstances.delete(tempRandArray[tempMostConfidentIndex2] - 1);
			}
			
			if ((tempRandArray[tempMostConfidentIndex3] == 0) || 
					(tempRandArray[tempMostConfidentIndex3]== 1 && tempRandArray[tempMostConfidentIndex2] == 0) || 
					(tempRandArray[tempMostConfidentIndex3]== 1 && tempRandArray[tempMostConfidentIndex1] == 0) ||
					(tempRandArray[tempMostConfidentIndex3]== 2 && tempRandArray[tempMostConfidentIndex1] == 0 && 
					tempRandArray[tempMostConfidentIndex2] == 1) || (tempRandArray[tempMostConfidentIndex3]== 2 && 
					tempRandArray[tempMostConfidentIndex1] == 1 &&tempRandArray[tempMostConfidentIndex2] == 0)) {
				unlabeledInstances.delete(tempRandArray[tempMostConfidentIndex3]);
			}else{
				unlabeledInstances.delete(tempRandArray[tempMostConfidentIndex3] - 2);
			}
		
			// Step 2.5. Update the predictive regressors and classifier.
			predictiveRegressors[0].buildClassifier(subLabeledInstances1);
			predictiveRegressors[1].buildClassifier(subLabeledInstances2);
			separationRegressor.buildClassifier(labeledInstances);
			tempIteration += 1;
		}//Of while
	}// Of buildClassifier
	
	/**
	 *************************
	 * Data prediction
	 * 
	 * @param data The given testing instance.
	 * @throws Exception 
	 *************************
	 **/
	public double classifyInstance(Instance paraData) throws Exception {
		double val = 0;
		
		double kNeighbor[][] = getKNeighbors(constructSingleCLassifierData(paraData).instance(0), labeledInstances);
		double tempRmse[] = new double[3];
		for (int j = 0; j < 5; j++) {
			Instance tempInstance = labeledInstances.instance((int)kNeighbor[j][0]);
			tempRmse[0] += Math.pow(predictiveRegressors[0].classifyInstance(tempInstance) - tempInstance.classValue(), 2);
			tempRmse[1] += Math.pow(predictiveRegressors[1].classifyInstance(tempInstance) - tempInstance.classValue(), 2);
			tempRmse[2] += Math.pow(separationRegressor.classifyInstance(tempInstance) - tempInstance.classValue(), 2);
		}
		
		if (tempRmse[0] < tempRmse[1] && tempRmse[0] < tempRmse[2]){
			val = predictiveRegressors[0].classifyInstance(paraData);
		}
		if (tempRmse[1] < tempRmse[0] && tempRmse[1] < tempRmse[2]){
			val = predictiveRegressors[1].classifyInstance(paraData);
		}
		if (tempRmse[2] < tempRmse[0] && tempRmse[2] < tempRmse[1]){
			val = separationRegressor.classifyInstance(paraData);
		}
		return val;
	}
	
	
}// Of class SRDP
