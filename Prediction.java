


import java.io.File;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AddClassification;


import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.supervised.attribute.AddClassification;




public class Prediction {
	public static void main(String[] args) throws Exception{
		/*load dataset*/
		
		DataSource source = new DataSource("/home/cse/Desktop/wed/data2set2 - (1).arff");
		Instances traindata = source.getDataSet();
		traindata.setClassIndex(traindata.numAttributes()-1);
		DataSource source2 = new DataSource("/home/cse/Desktop/wed/kyototest.arff");
		Instances testdata = source2.getDataSet();
		testdata.setClassIndex(testdata.numAttributes()-1);
		/**
		 * training the naive bayes classifier
		 */
		NaiveBayes nb = new NaiveBayes();
		
		AddClassification addClass = new AddClassification();
		addClass.setClassifier(nb);
		addClass.setRemoveOldClass(true);
		addClass.setOutputClassification(true);
		addClass.setInputFormat(traindata);
		Filter.useFilter(traindata, addClass);
		Instances newtestdata = Filter.useFilter(testdata, addClass);
		ArffSaver saver = new ArffSaver();
		saver.setInstances(newtestdata);
		//saver.setFile(new File("iris-new.arff"));
		saver.setFile(new File("/home/cse/Desktop/wed/result-test.arff"));
		saver.writeBatch();
		
	}

}
