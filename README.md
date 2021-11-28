# Data-Scientist-Classification
* Created a code that predict if a Data Scientist is willing to change his actual job.
* Based on a given dataset with information of Data Scientist, so no tools for data extraction or web scraping were needed.
* Feature engineering for data cleaning and feature selection.
* Optimized SVC, Kneighbor, XGBoost and Random Forest using GridsearchCV to reach the best model. 
* A neural network with Keras was used to built a classifier, but it didn't get best scores than conventional Maching Learning techniques.
* Built an HTTP server using Flask framework for receiving request of training and predicting. 
* Deployed a Cloudant database where all scores and information of current and old models are saved. 

## Feature engineering
I first divided data into training and test datasets in a 80/20 distribution. Then data is divided into categorical and numerical features, using most frequent and median as imputers. Finally, an one-hot encoder is used for categorial data.

## Model Building 

I tried SVC, Random Forest, Kneighbor and XGBoost classifiers and evaluated them using confusion matrix results and ROC. GridsearchCV were used for reaching the best hyperparameters for each model.  

## Model performance
The best model for training and validation data was the Random Forest Classifier.

ROC:

* XGBoost: 0.6697721994052237
* KNeighbor: 0.6707253873348893
* Random Forest: 0.7088193410859982
* SVC: 0.5
