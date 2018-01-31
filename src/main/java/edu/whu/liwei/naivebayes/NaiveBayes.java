package edu.whu.liwei.naivebayes;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.StringTokenizer;

import org.apache.commons.collections.functors.TruePredicate;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.broadcast.Broadcast;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


import scala.Tuple2;

public class NaiveBayes {
	private static Logger logger = LoggerFactory.getLogger(NaiveBayes.class);
	
	public static void main(String[] args) {
		
		if (args.length < 2) {
			System.out.println("Usage: java -cp *.jar edu.whu.liwei.naivebayes.NaiveBayes trainingData testData");
			System.out.println("trainingData: training data path, each line's format: docId:className word1 word2 word2 ...");
			System.out.println("testData: test data path, each line's format is the same as training data");
			return;
		}
		// get training and test data path
		String trainingDataPath = args[0];
		String testDataPath = args[1];
		
		// create Spark context with Spark configuration
		JavaSparkContext sc = new JavaSparkContext(new SparkConf().setAppName("NaiveBayesClassifier"));
		
		// read training and test data
		JavaRDD<String> trainingData = sc.textFile(trainingDataPath);
		JavaRDD<String> testData = sc.textFile(testDataPath);
		
		
		String dateFormat = "HH:mm:ss:SSS";
		SimpleDateFormat sdf = new SimpleDateFormat(dateFormat);
		// get begin time
		long startTime = System.currentTimeMillis();
		System.out.println("Strat training at " + sdf.format(new Date(startTime)));
		// get classDocNums, and convert to broadcast shared value
		logger.info("Counting classDocNums...");
		JavaPairRDD<String, Integer> classDocNums = classDocNumsCount(trainingData);
		Map<String, Integer> classDocNumsMap = classDocNums.collectAsMap();
		Broadcast<Map<String, Integer>> broadcastClassDocNums = sc.broadcast(classDocNumsMap);
		
		logger.info("Calculating prior probability...");
		// calculate prior probability, and convert to broadcast shared value
		Map<String, Double> priorProbability = calculatePriorProbability(classDocNumsMap);
		Broadcast<Map<String, Double>> broadcastPriorProbability = sc.broadcast(priorProbability);
		System.out.println(broadcastPriorProbability.getValue());
		// get classWordDocNums
		logger.info("Counting classWordDocNums...");
		JavaPairRDD<String, Integer> classWordDocNums = classWordDocNumsCount(trainingData);
		
		// calculate conditionProbably, and convert to broadcast shared value
		logger.info("Calculating condition probability");
		JavaPairRDD<String, Double> conditionProbability = calculateConditionProbability(broadcastClassDocNums, classWordDocNums, sc);
		Map<String, Double> conditionProbabilityMap = conditionProbability.collectAsMap();
		Broadcast<Map<String, Double>> broadcastconditionProbability = sc.broadcast(conditionProbabilityMap);
		
		// get end time
		long endTime = System.currentTimeMillis();
		System.out.println("Finished training at " + sdf.format(new Date(endTime)));
		System.out.println("Total cost " + (endTime - startTime)/1000.0 + "s");
		
		logger.info("Start evaluating...");
		logger.info("Getting doc labels...");
		JavaPairRDD<String, String> docLabels = getDocLabels(testData);
		// get begin time
		startTime = System.currentTimeMillis();
		// if jobCtrl doesn't all finished, block process
		System.out.println("Start predicting category at " + sdf.format(new Date(startTime)));
		JavaPairRDD<String, String> classPrediction = docClassPrediction(testData, broadcastPriorProbability, broadcastconditionProbability);
//		predict(testData, conditionProbabilityMap, conditionProbabilityMap);
		// get end time
		endTime = System.currentTimeMillis();
		System.out.println("Finished prediciton at " + sdf.format(new Date(endTime)));
		System.out.println("Total cost " + (endTime - startTime)/1000.0 + "s");
//		evaluation(docLabels, classPrediction);
	}
	
	/**
	 * count doc nums in each class
	 * @param trainingData input trainindData rdd
	 * @return class Doc Nums
	 */
	public static JavaPairRDD<String, Integer> classDocNumsCount(JavaRDD<String> trainingData) {
		@SuppressWarnings("serial")
		JavaPairRDD<String, Integer> classDocOnePairs = trainingData.mapToPair(
				new PairFunction<String, String, Integer>() {
					public Tuple2<String, Integer> call(String line) throws Exception {
						int colonIndex = line.indexOf(":");
						int commaIndex = line.indexOf(",");
						String className = line.substring(colonIndex + 1, commaIndex);
						return new Tuple2<String, Integer>(className, 1);
					}
				});
		
		@SuppressWarnings("serial")
		JavaPairRDD<String, Integer> classDocNums = classDocOnePairs.reduceByKey(
				new Function2<Integer, Integer, Integer>() {
					public Integer call(Integer v1, Integer v2) throws Exception {
						return v1 + v2;
					}
				});
		
		return classDocNums;
	}
	
	/**
	 * get word document counts in each class
	 * @param trainingData trainingData input trainindData rdd
	 * @return class Word Doc Nums 
	 */
	public static JavaPairRDD<String, Integer> classWordDocNumsCount(JavaRDD<String> trainingData) {
		@SuppressWarnings("serial")
		// append word to className, get (className:word) javaRDD
		JavaRDD<String> classWords = trainingData.flatMap(
				new FlatMapFunction<String, String>() {
					public Iterator<String> call(String line) throws Exception {
						int colonIndex = line.indexOf(":");
						int commaIndex = line.indexOf(",");
						String className = line.substring(colonIndex + 1, commaIndex);
						StringTokenizer itr = new StringTokenizer(line.toString().substring(commaIndex + 1));
						Set<String> wordSet = new HashSet<String>();
						while(itr.hasMoreTokens()){
							wordSet.add(itr.nextToken());
						}
						List<String> classWordsList = new ArrayList<String>();
						for (String word : wordSet) {
							classWordsList.add(className + ":" + word);
						}
						return classWordsList.iterator();
					}
				});
		
		@SuppressWarnings("serial")
		JavaPairRDD<String, Integer> classWordDocOnePairs = classWords.mapToPair(
				new PairFunction<String, String, Integer>() {
					public Tuple2<String, Integer> call(String classWord) throws Exception {
						return new Tuple2<String, Integer>(classWord, 1);
					}
				});
		
		@SuppressWarnings("serial")
		JavaPairRDD<String, Integer> classWordDocNums = classWordDocOnePairs.reduceByKey(
				new Function2<Integer, Integer, Integer>() {
					public Integer call(Integer v1, Integer v2) throws Exception {
						return v1 + v2;
					}
				});
		
		return classWordDocNums;
	}
	
	/**
	 * Calculate condition probability 
	 * @param classDocNums class Doc Nums javaRDD
	 * @param classWordDocNums class Word Doc Nums javaRDD
	 * @return
	 */
	public static JavaPairRDD<String, Double> calculateConditionProbability(final Broadcast<Map<String, Integer>> broadcastClassDocNums,
			JavaPairRDD<String, Integer> classWordDocNums, JavaSparkContext sc) {
		@SuppressWarnings("serial")
		JavaRDD<Tuple2<String, Double>> conditionProbabilityRDD = classWordDocNums.map(
				new Function<Tuple2<String, Integer>, Tuple2<String, Double>>() {
					public Tuple2<String, Double> call(Tuple2<String, Integer> tuple) throws Exception {
						String className = tuple._1().split(":")[0];
						double classDocNums = broadcastClassDocNums.getValue().get(className);
						return new Tuple2<String, Double>(tuple._1(), (tuple._2 + 1.0)/(classDocNums + 2.0));
					}
		});
		@SuppressWarnings("serial")
		JavaPairRDD<String, Double> conditionProbability = conditionProbabilityRDD.mapToPair(
				new PairFunction<Tuple2<String, Double>, String, Double>() {
					public Tuple2<String, Double> call(Tuple2<String, Double> t) throws Exception {
						return new Tuple2<String, Double>(t._1(), t._2());
					}
					
				});
		
		// add probability for words excluded in class 
		List<Tuple2<String, Double>> additionalProbability = new ArrayList<Tuple2<String, Double>>();
		for (Entry<String, Integer> entry :  broadcastClassDocNums.getValue().entrySet()) {
			double probability = 1.0 / (entry.getValue() + 2.0);
			additionalProbability.add(new Tuple2<String, Double>(entry.getKey(), probability));
		}
		JavaPairRDD<String, Double> additionProbabilityRDD = sc.parallelizePairs(additionalProbability);
		return conditionProbability.union(additionProbabilityRDD);
	}
	
	/**
	 * Calculate prior probability 
	 * @param classDocNumsMap
	 * @return
	 */
	public static Map<String, Double> calculatePriorProbability(Map<String, Integer> classDocNumsMap) {
		Map<String, Double> priorProbability= new HashMap<String, Double>();
		double totalDocNums = 0.0;
		for (Entry<String, Integer> entry : classDocNumsMap.entrySet()) {
			totalDocNums += entry.getValue();
		}
		for (Entry<String, Integer> entry : classDocNumsMap.entrySet()) {
			priorProbability.put(entry.getKey(), entry.getValue() / totalDocNums);
		}
		return priorProbability;
	}
	
	
	/**
	 * predict class for test data
	 * @param testData input test data rdd
 	 * @param broadcastPriorProbability broadcast shared PriorProbability 
	 * @param conditionProbability broadcast shared conditionProbability 
	 * @return map docid to predict result
	 */
	public static JavaPairRDD<String, String> docClassPrediction(JavaRDD<String> testData,
			final Broadcast<Map<String, Double>> broadcastPriorProbability,
			final Broadcast<Map<String, Double>> broadcastconditionProbability) {
		@SuppressWarnings("serial")
		JavaPairRDD<String, String> classProbability = testData.flatMapToPair(
				new PairFlatMapFunction<String, String, String>() {
					public Iterator<Tuple2<String, String>> call(String line) throws Exception {
						int colonIndex = line.indexOf(":");
						int commaIndex = line.indexOf(",");
						String docId = line.substring(0, colonIndex);
						Map<String, Double> conditionProbabilityMap = broadcastconditionProbability.getValue();
						List<Tuple2<String, String>> classProbability = new ArrayList<Tuple2<String, String>>();
						for (Entry<String, Double> entry : broadcastPriorProbability.getValue().entrySet()) {
							double tempValue = Math.log(entry.getValue()); // convert the predict value calculated by product to sum of log	
							String[] words = line.toString().substring(commaIndex + 1).split(" ");
							for (String word : words) {
								String tempkey = entry.getKey() + ":" + word;
								if (conditionProbabilityMap.containsKey(tempkey)) {
									// if <class:word> exists in wordsProbably, get the probability
									tempValue += Math.log(conditionProbabilityMap.get(tempkey));
								} else { // if doesn't exist, using the probability of class probability
									tempValue += Math.log(conditionProbabilityMap.get(entry.getKey()));						
								}
							}
							classProbability.add(new Tuple2<String, String>(docId, entry.getKey() + ":" + tempValue));
						}
						return classProbability.iterator();
					}
				});
		@SuppressWarnings("serial")
		JavaPairRDD<String, String> classPrediction = classProbability.reduceByKey(
				new Function2<String, String, String>() {
					public String call(String v1, String v2) throws Exception {
						Double probability1 = Double.parseDouble(v1.split(":")[1]);
						Double probability2 = Double.parseDouble(v2.split(":")[1]);
						if (probability1 > probability2)
							return v1;
						else 
							return v2;
					}
				});
		return classPrediction;
	}
	
	/**
	 * get doc labels 
	 */
	public static JavaPairRDD<String, String> getDocLabels(JavaRDD<String> data) {
		@SuppressWarnings("serial")
		JavaPairRDD<String, String> docLabels = data.mapToPair(
				new PairFunction<String, String, String>() {
					public Tuple2<String, String> call(String line) throws Exception {
						int colonIndex = line.indexOf(":");
						int commaIndex = line.indexOf(",");
						String docId = line.substring(0, colonIndex);
						String className = line.substring(colonIndex + 1, commaIndex);
						return new Tuple2<String, String>(docId, className);
					}
				});
		return docLabels;
	}
	
	/**
	 * predict on single node
	 */
	public static List<String> predictOnSingleNode(JavaRDD<String> testData, 
			Map<String, Double> conditionProbabilityMap, Map<String, Double> priorProbability ) {
		List<String> turePredict = new ArrayList<String>();
		List<String> lines = testData.collect();
		int right = 0, i = 0;
		for (String line : lines) {
			i++;
			int colonIndex = line.indexOf(":");
			int commaIndex = line.indexOf(",");
			String docId = line.substring(0, colonIndex);
			String className = line.substring(colonIndex + 1, commaIndex);
			double probability = 0.0;
			String predictLabel = "";
			for (Entry<String, Double> entry : priorProbability.entrySet()) {
				double tempValue = Math.log(entry.getValue()); // convert the predict value calculated by product to sum of log	
				String[] words = line.toString().substring(commaIndex + 1).split(" ");
				for (String word : words) {
					String tempkey = entry.getKey() + ":" + word;
					if (conditionProbabilityMap.containsKey(tempkey)) {
						// if <class:word> exists in wordsProbably, get the probability
						tempValue += Math.log(conditionProbabilityMap.get(tempkey));
					} else { // if doesn't exist, using the probability of class probability
						tempValue += Math.log(conditionProbabilityMap.get(entry.getKey()));						
					}
				}
				if (tempValue >= probability) {
					predictLabel = entry.getKey();
				}
			}
			if (predictLabel.equals(className)) {
				right++;
				turePredict.add(docId);
			}
		}
		System.out.println("Evaluating!\n Accuracy: " + right + "/" + i + " = " + i);
		System.out.println(turePredict);
		return turePredict;
	}
	/**
	 * evaluating predict result
	 * @param docLabels
	 * @param classPrediction
	 */
	public static void evaluation(JavaPairRDD<String, String> docLabels, JavaPairRDD<String, String> classPrediction) {
		Map<String, String> docLabelsMap = docLabels.collectAsMap();
		Map<String, String> classPredictionMap = classPrediction.collectAsMap();
		int right = 0;
		for (Entry<String, String> entry : classPredictionMap.entrySet()) {
			int colonIndex = entry.getValue().indexOf(":"); 
			String predictLabel = entry.getValue().substring(0, colonIndex);
			if (docLabelsMap.get(entry.getKey()).equals(predictLabel))
				right++;
		}
		double accuracy = right / (double)classPredictionMap.size();
		logger.info("Evaluating!\n Accuracy: " + right + "/" + classPredictionMap.size() + " = " + accuracy);
	}
	
	
}
