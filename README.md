# machine_learning

Welcome to the Machine Learning Repository! This repository contains a collection of machine learning projects, algorithms, and resources to help you learn and implement various techniques in the field of machine learning.


Introduction
Machine learning is a rapidly evolving field that focuses on developing algorithms and models that enable computers to learn and make predictions or decisions without explicit programming. This repository aims to provide a comprehensive collection of machine learning projects and resources to assist both beginners and experienced practitioners in learning and applying machine learning techniques.


Algorithms
The algorithms directory contains implementations of various machine learning algorithms. These implementations are provided as code snippets, Jupyter notebooks, or Python scripts, depending on the complexity and requirements of each algorithm. You can find implementations of popular algorithms such as:

Data Sets
To facilitate the learning and experimentation process, we have included a collection of sample data sets in the data-sets directory. These data sets cover various domains and can be used for training, testing, and evaluating machine learning models. Each data set is accompanied by a README file providing details about the data, its format, and potential applications.

Contributing
We welcome contributions to this repository! If you would like to contribute your own machine learning projects, algorithms, or data sets, please follow the guidelines outlined in the CONTRIBUTING.md file. We appreciate your contributions and the opportunity to learn from each other.


# 1:linear regression 

This repository contains code for predicting home prices based on the area of the house using linear regression. It utilizes the `numpy`, `pandas`, `matplotlib`, and `scikit-learn` libraries in Python.

## Dataset

The `homeprices.csv` file contains the dataset used for training and testing the linear regression model. It consists of two columns: "area" (the area of the house in square feet) and "price" (the corresponding price of the house in dollars).

## Getting Started

To run the code locally, follow these steps:

1. Clone the repository:

git clone https://github.com/rahul-singh21/machine_learning.git

## homeprices.csv
    area,price
    2600,550000
    3000,565000
    3200,610000
    3600,680000
    4000,725000    
    

## Results

The script performs the following steps:

1. Loads the dataset from the `homeprices.csv` file.
2. Visualizes the data using a scatter plot.
3. Trains a linear regression model on the data

##graph

  ![image](https://github.com/rahul-singh2021/machine_learning/assets/95570957/8a2d9ed6-219e-490e-88d1-bae0f9ee29f0)



python
Copy code
import numpy as np
from gradient_descent import gradient_descent

x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])

gradient_descent(x, y)
The function will print the updated values of w (slope), b (y-intercept), and the current iteration number during the optimization process.

Parameters
The gradient_descent function has the following parameters:

x (NumPy array): Input data points (independent variable).
y (NumPy array): Output data points (dependent variable).
iterations (int): Number of iterations for the optimization process (default: 1000).
alpha (float): Learning rate for gradient descent (default: 0.001).
Feel free to modify the default values of iterations and alpha as per your requirements.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Contributing
If you have any suggestions, improvements, or bug fixes, please feel free to contribute! You can open an issue or submit a pull request.

# 2:Linear Regression Multivariate
This program demonstrates multivariate linear regression using the scikit-learn library in Python. It predicts home prices based on the area, number of bedrooms, and age of the property. The dataset used for training and prediction is loaded from a CSV file named "homeprices.csv."
## Dependencies
- numpy
- pandas
- matplotlib
- scikit-learn

## Usage
- Ensure that the "homeprices.csv" file is in the same directory as the script.
- Modify the script if necessary to adapt to your dataset or analysis requirements.
- Run the script to perform multivariate linear regression and obtain price predictions.

## homeprices.csv
    area,bedrooms,age,price
    2600,3,20,550000
    3000,4,15,565000
    3200,,18,610000
    3600,3,30,595000
    4000,5,8,760000
    4100,6,8,810000


# 3:Gradient Descent 

This is an implementation of the gradient descent algorithm using numpy. It calculates the best-fit line for a given set of input data points.

## Usage

The `gradient_descent` function takes two parameters, `x` and `y`, representing the input data points and their corresponding target values.
import numpy as np

def gradient_descent(x, y):
    # Function code here

x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])

gradient_descent(x, y)


# 4:Model Saving with Pickle

This program demonstrates how to save a machine learning model using the Python `pickle` library. The saved model can be later loaded and used for making predictions.

## Prerequisites

Before running this program, make sure you have the following dependencies installed:

- pandas
- numpy
- scikit-learn

You can install these dependencies using `pip`:
## Usage

1. Clone the repository:
  https://github.com/rahul-singh21/machine_learning.git
      

# 5:Model Saving with jobliib

This program demonstrates how to save a machine learning model using the Python `joblib` library. The saved model can be later loaded and used for making predictions.

## Prerequisites

Before running this program, make sure you have the following dependencies installed:

- pandas
- numpy
- scikit-learn

You can install these dependencies using `pip`:
## Usage

1. Clone the repository:
  https://github.com/rahul-singh21/machine_learning.git
  
  
# 6:stochastic gradient descent

This repository contains an implementation of SVM (Support Vector Machine) using SGD (Stochastic Gradient Descent) for classification. It includes a function called `svm_sgd_plot` that trains an SVM model on the given input data `X` and corresponding labels `Y`. The function updates the model's weights using SGD for a specified number of epochs.

## Requirements

The following dependencies are required to run the code:
- NumPy
- Matplotlib

Install them using the following command:

pip install numpy matplotlib

perl
Copy code

  ## Usage

   To use the `svm_sgd_plot` function, pass the input data `X` and labels `Y`. It will return the learned weights of the SVM model.          Additionally, it will plot the misclassification errors over epochs.

    ```python
      w = svm_sgd_plot(X, Y)  
      


# 7:Logistic Regression with Newton Method

This program demonstrates logistic regression using the Newton method for parameter estimation. It generates a dataset and fits a logistic regression model using the Newton-Raphson algorithm to estimate the coefficients.

## Purpose

The purpose of this program is to showcase the implementation of logistic regression with the Newton method. Logistic regression is a popular classification algorithm used to model the relationship between a dependent variable and one or more independent variables. The Newton method is an iterative optimization algorithm used to estimate the parameters of the logistic regression model.

## Key Features

- Generates a synthetic dataset with specified parameters
- Fits a logistic regression model using the Newton method
- Demonstrates the calculation of the sigmoid function
- Uses the patsy library for formula specification
Provide instructions for running the program and installing any necessary dependencies:


## Getting Started

### Prerequisites

- Python 3.x
- Required libraries: numpy, pandas, patsy

### Installation

# 8:dummy variable

This program demonstrates the use of dummy variables in a linear regression model for predicting home prices.

## Requirements

- Python 3.x
- pandas
- scikit-learn

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/rahul-singh21/machine_learning.git
    
    ## homeprices.csv
    
    town            ,area  ,price
    monroe township ,2600  ,550000
    monroe township ,3000  ,565000
    monroe township ,3200  ,610000
    monroe township ,3600  ,680000
    monroe township ,4000  ,725000
    west windsor    ,2600  ,585000
    west windsor    ,2800  ,615000
    west windsor    ,3300  ,650000
    west windsor    ,3600  ,710000
    robinsville     ,2600  ,575000
    robinsville     ,2900  ,600000
    robinsville     ,3100  ,620000
    robinsville     ,3600  ,695000


# 9:One-Hot-Encoder

This program demonstrates how to use One-Hot Encoding in machine learning using the `LabelEncoder`, `OneHotEncoder`, and `ColumnTransformer` classes from the scikit-learn library.

## Dependencies

- pandas
- scikit-learn

  ## Installation

        Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required dependencies.
    
        ```bash
        pip install pandas scikit-learn
    
        ## homeprices.csv
        
        town            ,area  ,price
        monroe township ,2600  ,550000
        monroe township ,3000  ,565000
        monroe township ,3200  ,610000
        monroe township ,3600  ,680000
        monroe township ,4000  ,725000
        west windsor    ,2600  ,585000
        west windsor    ,2800  ,615000
        west windsor    ,3300  ,650000
        west windsor    ,3600  ,710000
        robinsville     ,2600  ,575000
        robinsville     ,2900  ,600000
        robinsville     ,3100  ,620000
        robinsville     ,3600  ,695000

# 10:Train_Test_Split
This program uses linear regression to predict car prices based on mileage and age. It demonstrates how to split the data into training and testing sets using train_test_split from the scikit-learn library and then trains a linear regression model on the training data.

## Dependencies
    pandas
    scikit-learn
## Usage
    Install the required dependencies:

    pip install pandas scikit-learn
    Clone the repository:
    git clone https://github.com/rahul-singh/machine_learning.git
    
## carprices.csv
       ```bash
       Mileage,Age(yrs),Sell Price($)
        69000 ,6       ,18000
        35000 ,3       ,34000
        57000 ,5       ,26100
        22500 ,2       ,40000
        46000 ,4       ,31500
        59000 ,5       ,26750
        52000 ,5       ,32000
        72000 ,6       ,19300
        91000 ,8       ,12000
        67000 ,6       ,22000
        83000 ,7       ,18700
        79000 ,7       ,19500
        59000 ,5       ,26000
        58780 ,4       ,27500
        82450 ,7       ,19400
        25400 ,3       ,35000
        28000 ,2       ,35500
        69000 ,5       ,19700
        87600 ,8       ,12800
        52000 ,5       ,28200
    
    
# 11:Logistic Regression
This program demonstrates the use of logistic regression to predict insurance purchases based on age. The dataset used is stored in a CSV file called insurance_data.csv.

## Dependencies
   To run this program, the following dependencies are required:

    pandas
    scikit-learn
    Install the dependencies using pip:
    pip install pandas scikit-learn

## Usage
    Clone the repository or download the insurance_data.csv file.
    Make sure the dataset file (insurance_data.csv) is placed in the same directory as the Python script.
    Execute the Python script.
    Program Explanation
    Import the necessary libraries: pandas, train_test_split from sklearn.model_selection, and LogisticRegression from        sklearn.linear_model.
    Read the insurance dataset from the CSV file using pd.read_csv.
    Split the dataset into training and testing sets using train_test_split, where the 'age' column is the feature (x) and the 'bought_insurance' column is the target (y).
    Create an instance of the LogisticRegression model.
    Train the model using the training data using the fit method.
    Predict the insurance purchases for the test data using the predict method.
    You can use the predicted values (y_pred) for further analysis or evaluation.
    Note: This program assumes that the CSV file contains a header row with column names.    

## insurance.csv
       ```bash
       age,bought_insurance
        22,0
        25,0
        47,1
        52,0
        46,1
        56,1
        55,0
        60,1
        62,1
        61,1
        18,0
        28,0
        27,0
        29,0
        49,1
        55,1
        25,1
        58,1
        19,0
        18,0
        21,0
        26,0
        40,1
        45,1
        50,1
        54,1
        23,0
        
# 12:Handwritten Digit Classification
  This program uses logistic regression to classify handwritten digits from the MNIST dataset. It demonstrates how to load the dataset,     split it into training and testing sets, train a logistic regression model, and make predictions.

## Prerequisites
   To run this program, you need to have the following dependencies installed:

   Python 3
   Matplotlib
   Scikit-learn
   You can install the required packages by running the following command:

   pip install matplotlib scikit-learn
## Usage
   Clone the repository or download the program file.
   Open a terminal or command prompt and navigate to the program directory.
   Run the following command to execute the program:
   python digit_classification.py
   
## Program Description
   The program starts by importing the necessary libraries: matplotlib.pyplot, sklearn.datasets.load_digits,                  sklearn.model_selection.train_test_split, and sklearn.linear_model.LogisticRegression.

   The MNIST digit dataset is loaded using the load_digits function from sklearn.datasets. This dataset contains grayscale images of  handwritten digits and their corresponding labels.

   The grayscale plot is set using plt.gray().

   The program prints the labels of the first five digits using digits.target[0:5].

   The dataset is split into training and testing sets using train_test_split from sklearn.model_selection. The training set contains 80%  of the data, while the testing set contains 20%.

   An instance of logistic regression is created using LogisticRegression from sklearn.linear_model.

   The logistic regression model is trained using the training data and labels using the fit method.

   The model makes predictions on the first five digits in the dataset using model.predict(digits.data[0:5]).  
   
   
# 12:Decision Tree Classifier

This program implements a decision tree classifier using Python's pandas and scikit-learn libraries.

## Prerequisites

- Python (version X.X)
- pandas (version X.X)
- scikit-learn (version X.X)

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/your-repository.git   
## salaries.csv
    
    company     ,job                   ,degree      ,salary_more_then_100k
    abc pharma  ,business manager      ,bachelors   ,0
    google      ,computer programmer   ,bachelors   ,0
    abc pharma  ,computer programmer   ,bachelors   ,0
    google      ,sales executive       ,bachelors   ,0
    google      ,sales executive       ,masters     ,0
    abc pharma  ,sales executive       ,masters     ,0
    google      ,business manager      ,bachelors   ,1
    google      ,business manager      ,masters     ,1
    abc pharma  ,business manager      ,masters     ,1
    facebook    ,business manager      ,bachelors   ,1
    facebook    ,business manager      ,masters     ,1
    google      ,computer programmer   ,masters     ,1
    facebook    ,computer programmer   ,bachelors   ,1
    facebook    ,computer programmer   ,masters     ,1
    facebook    ,sales executive       ,bachelors   ,1
    facebook    ,sales executive       ,masters     ,1


# 13:Iris Classification using Support Vector Machines (SVM)

This program demonstrates the classification of Iris flowers using Support Vector Machines (SVM) algorithm. It utilizes the Iris dataset from the scikit-learn library.

## Dataset Description

The Iris dataset contains measurements of sepal length, sepal width, petal length, and petal width of three different Iris flower species: setosa, versicolor, and virginica.

## Installation

To run this program, you need to have the following dependencies installed:

- pandas
- scikit-learn
- matplotlib

You can install these dependencies using pip:

## Program Usage

1. The program loads the Iris dataset using the `load_iris` function from scikit-learn.

2. It creates a pandas DataFrame to store the dataset and adds column names.

3. The target variable is added to the DataFrame, which represents the species of each Iris flower.

4. The DataFrame is filtered to create separate DataFrames for each species (setosa, versicolor, and virginica).

5. The flower names are added to the DataFrame using the `apply` method.

6. The program visualizes the sepal length and width as well as the petal length and width using scatter plots.

7. The dataset is split into training and testing sets using the `train_test_split` function from scikit-learn.

8. A Support Vector Machines (SVM) model is created and trained using the training data.

9. The model predicts the species of an unseen sample.

## Result

The SVM model successfully predicts the species of an unseen sample.


# 14:Digit Classification using Random Forest

This program uses the Random Forest algorithm to classify digits from the sklearn digits dataset.

## Requirements

- Python 3.x
- pandas
- scikit-learn
- matplotlib

## Usage

1. Install the required dependencies by running the following command:

   ```shell
   pip install pandas scikit-learn matplotlib


     git clone https://github.com/your-username/your-repository.git

# 15:K-Means Clustering Program
    This program demonstrates the implementation of K-Means clustering using the scikit-learn library in Python. It uses the K-Means     
    algorithm to cluster a dataset based on age and income.

## Prerequisites
   To run this program, you need the following:

    Python (version 3.6 or higher)
    Jupyter Notebook or any Python IDE
    Installation
    Clone the repository or download the income.csv file to your local machine.

    Install the required libraries by running the following command:

    pip install scikit-learn pandas matplotlib
## Usage
    Open the Jupyter Notebook or your Python IDE.

    Import the required libraries and load the dataset:


    from sklearn.cluster import KMeans
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.pyplot as plt
    %matplotlib inline

    df = pd.read_csv('income.csv')
    Visualize the data by plotting the scatter plot:

    plt.scatter(df['Age'], df['Income($)'])
    plt.xlabel('Age')
    plt.ylabel('Income($)')
    plt.show()
    Perform K-Means clustering on the dataset:

   
    km = KMeans(n_clusters=3)
    y_predicted = km.fit_predict(df[['Age', 'Income($)']])
    df['cluster'] = y_predicted
    Plot the clustered data:
    
    
    df1 = df[df.cluster == 0]
    df2 = df[df.cluster == 1]
    df3 = df[df.cluster == 2]
    
    plt.scatter(df1['Age'], df1['Income($)'], color='green')
    plt.scatter(df2['Age'], df2['Income($)'], color='blue')
    plt.scatter(df3['Age'], df3['Income($)'], color='black')
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='purple', marker='*', label='centroid')
    
    plt.xlabel('Age')
    plt.ylabel('Income($)')
    plt.legend()
    plt.show()
    Perform feature scaling on the dataset:
    
   
    scaler = MinMaxScaler()
    scaler.fit(df[['Income($)']])
    df['Income($)'] = scaler.transform(df[['Income($)']])
    
    scaler.fit(df[['Age']])
    df['Age'] = scaler.transform(df[['Age']])
    Repeat the K-Means clustering with the scaled dataset:
    
   
    km = KMeans(n_clusters=3)
    y_predicted = km.fit_predict(df[['Age', 'Income($)']])
    df['cluster'] = y_predicted
    Plot the clustered data with the scaled features:
    
    
    df1 = df[df.cluster == 0]
    df2 = df[df.cluster == 1]
    df3 = df[df.cluster == 2]
    
    plt.scatter(df1['Age'], df1['Income($)'], color='green')
    plt.scatter(df2['Age'], df2['Income($)'], color='blue')
    plt.scatter(df3['Age'], df3['Income($)'], color='black')
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='purple', marker='*', label='centroid')
    
    plt.xlabel('Age')
    plt.ylabel('Income($)')
    plt.legend()
    plt.show()

# 16:Naive Bayes Classifier for Titanic Survival Prediction
     This program demonstrates the use of a Naive Bayes classifier to predict survival outcomes for passengers aboard the Titanic. It 
     utilizes the pandas library to read a CSV file containing the Titanic dataset and performs various operations on the data before 
     training and testing the classifier.

## Prerequisites
    Before running the program, make sure you have the following dependencies installed:

    pandas
    scikit-learn
    You can install these dependencies using pip:


    pip install pandas scikit-learn
## Usage
    Clone the repository or download the source code files.

    Place the titanic.csv file in the same directory as the program.

    Run the program using a Python interpreter:

    python titanic_classifier.py
    The program will read the CSV file, remove unwanted columns, create dummy variables for the "Sex" column, fill missing values in the 
    "Age" column with the mean, split the dataset into training and test sets, train a Naive Bayes classifier, and predict survival 
    outcomes for the test set.

    The program will output the predicted survival outcomes and the accuracy score of the classifier.

## Description
    The program performs the following steps:

    Imports the required libraries, including pandas and scikit-learn.

    Reads the titanic.csv file using pandas and displays the initial rows of the dataset.

    Removes unwanted columns (PassengerId, Name, SibSp, Parch, Ticket, Cabin, Embarked) from the dataset.

    Creates dummy variables for the Sex column using one-hot encoding and drops the original Sex column.

    Fills missing values in the Age column with the mean value of the column.

    Splits the dataset into training and test sets using the train_test_split function from scikit-learn.

    Trains a Gaussian Naive Bayes classifier using the training set.

    Predicts survival outcomes for the test set using the trained classifier.

    Compares the predicted outcomes with the actual outcomes (stored in the Survived column of the test set).

    Calculates and displays the accuracy score of the classifier.
    

# 17:Removing Outliers using Percentile Method 
     This Python program is designed to remove outliers from a dataset using the percentile method. The dataset is read from a CSV file, 
     and the outliers are identified based on the price per square foot column.

## Prerequisites
    Python 3.x
    pandas library
    Usage
    Make sure you have the necessary prerequisites installed.
    Place the dataset file (bhp.csv) in the same directory as the Python script.
    Open the Python script in a text editor or an Integrated Development Environment (IDE).
    Run the script.
## Program Flow
    The pandas library is imported.
    The dataset is loaded from the CSV file into a DataFrame using the read_csv() function.
    The shape of the DataFrame is printed using the shape attribute to display the number of rows and columns in the dataset.
    The describe() method is called on the DataFrame to generate summary statistics of the dataset, including count, mean, standard 
    deviation, minimum, quartiles, and maximum values.
    The minimum and maximum thresholds for outliers are calculated using the quantile() method on the price_per_sqft column, with 
    percentiles set to 0.001 and 0.999, respectively.
    Rows with price per square foot values below the minimum threshold are filtered using boolean indexing and printed.
    A new DataFrame (df2) is created by filtering rows where the price per square foot is greater than the minimum threshold and less 
    than the maximum threshold.
    The shape of the new DataFrame (df2) is printed to display the number of rows and columns after removing outliers.
## bhp.csv
    location,size,total_sqft,bath,price,bhk,price_per_sqft
Electronic City Phase II,2 BHK,1056.0,2.0,39.07,2,3699
Chikka Tirupathi,4 Bedroom,2600.0,5.0,120.0,4,4615
Uttarahalli,3 BHK,1440.0,2.0,62.0,3,4305
Lingadheeranahalli,3 BHK,1521.0,3.0,95.0,3,6245
Kothanur,2 BHK,1200.0,2.0,51.0,2,4250
Whitefield,2 BHK,1170.0,2.0,38.0,2,3247
Old Airport Road,4 BHK,2732.0,4.0,204.0,4,7467
Rajaji Nagar,4 BHK,3300.0,4.0,600.0,4,18181
Marathahalli,3 BHK,1310.0,3.0,63.25,3,4828
other,6 Bedroom,1020.0,6.0,370.0,6,36274
Whitefield,3 BHK,1800.0,2.0,70.0,3,3888
Whitefield,4 Bedroom,2785.0,5.0,295.0,4,10592
7th Phase JP Nagar,2 BHK,1000.0,2.0,38.0,2,3800
Gottigere,2 BHK,1100.0,2.0,40.0,2,3636
Sarjapur,3 Bedroom,2250.0,3.0,148.0,3,6577
Mysore Road,2 BHK,1175.0,2.0,73.5,2,6255
Bisuvanahalli,3 BHK,1180.0,3.0,48.0,3,4067
Raja Rajeshwari Nagar,3 BHK,1540.0,3.0,60.0,3,3896
other,3 BHK,2770.0,4.0,290.0,3,10469
other,2 BHK,1100.0,2.0,48.0,2,4363
Kengeri,1 BHK,600.0,1.0,15.0,1,2500
Binny Pete,3 BHK,1755.0,3.0,122.0,3,6951
Thanisandra,4 Bedroom,2800.0,5.0,380.0,4,13571
Bellandur,3 BHK,1767.0,3.0,103.0,3,5829
Thanisandra,1 RK,510.0,1.0,25.25,1,4950
other,3 BHK,1250.0,3.0,56.0,3,4480
Electronic City,2 BHK,660.0,1.0,23.1,2,3500
Whitefield,3 BHK,1610.0,3.0,81.0,3,5031
Ramagondanahalli,2 BHK,1151.0,2.0,48.77,2,4237
Electronic City,3 BHK,1025.0,2.0,47.0,3,4585
Yelahanka,4 BHK,2475.0,4.0,186.0,4,7515
Bisuvanahalli,3 BHK,1075.0,2.0,35.0,3,3255
Hebbal,3 BHK,1760.0,2.0,123.0,3,6988
Raja Rajeshwari Nagar,3 BHK,1693.0,3.0,57.39,3,3389
Kasturi Nagar,3 BHK,1925.0,3.0,125.0,3,6493
Kanakpura Road,2 BHK,700.0,2.0,36.0,2,5142
Electronics City Phase 1,2 BHK,1070.0,2.0,45.5,2,4252
Kundalahalli,3 BHK,1724.0,3.0,125.0,3,7250
Chikkalasandra,3 BHK,1290.0,2.0,56.12,3,4350
Uttarahalli,2 BHK,1143.0,2.0,45.0,2,3937
Murugeshpalya,2 BHK,1296.0,2.0,81.0,2,6250
Sarjapur  Road,3 BHK,1254.0,3.0,38.0,3,3030
other,1 BHK,600.0,1.0,38.0,1,6333
Yelahanka,1 Bedroom,660.0,1.0,48.0,1,7272
Kanakpura Road,2 BHK,1330.74,2.0,91.79,2,6897
HSR Layout,8 Bedroom,600.0,9.0,200.0,8,33333
Doddathoguru,2 BHK,970.0,2.0,33.0,2,3402
Whitefield,2 BHK,1459.0,2.0,94.82,2,6498
KR Puram,2 Bedroom,800.0,1.0,130.0,2,16250
other,2 BHK,869.0,2.0,36.0,2,4142
other,2 BHK,1270.0,2.0,50.0,2,3937
Bhoganhalli,3 BHK,1670.0,3.0,99.0,3,5928
Whitefield,3 BHK,2010.0,3.0,91.0,3,4527
Lakshminarayana Pura,2 BHK,1185.0,2.0,75.0,2,6329
Yelahanka,3 BHK,1600.0,2.0,75.0,3,4687
Begur Road,2 BHK,1200.0,2.0,44.0,2,3666
other,2 BHK,1500.0,2.0,185.0,2,12333
Murugeshpalya,6 Bedroom,1407.0,4.0,150.0,6,10660
other,2 BHK,840.0,2.0,45.0,2,5357
other,3 Bedroom,4395.0,3.0,240.0,3,5460
other,2 BHK,845.0,2.0,55.0,2,6508
Whitefield,4 Bedroom,5700.0,5.0,650.0,4,11403
Varthur,2 BHK,1160.0,2.0,44.0,2,3793
Bommanahalli,8 Bedroom,3000.0,8.0,140.0,8,4666
Doddathoguru,2 BHK,1100.0,2.0,62.0,2,5636
Gunjur,2 BHK,1140.0,2.0,43.0,2,3771
Marathahalli,2 BHK,1220.0,2.0,57.0,2,4672
Devarachikkanahalli,8 Bedroom,1350.0,7.0,85.0,8,6296
Kanakpura Road,2 BHK,1005.0,2.0,36.68,2,3649
other,3 Bedroom,500.0,3.0,100.0,3,20000
Begur Road,2 BHK,1358.0,2.0,80.58,2,5933
Hegde Nagar,3 BHK,1569.0,3.0,101.0,3,6437
Haralur Road,2 BHK,1240.0,2.0,70.0,2,5645
Hennur Road,3 BHK,2089.0,3.0,140.0,3,6701
Kothannur,2 BHK,1206.0,2.0,48.23,2,3999
Kalena Agrahara,2 BHK,1150.0,2.0,40.0,2,3478
other,3 BHK,2511.0,3.0,205.0,3,8164
Kaval Byrasandra,2 BHK,460.0,1.0,22.0,2,4782
ISRO Layout,6 Bedroom,4400.0,6.0,250.0,6,5681
other,3 BHK,1660.0,2.0,105.0,3,6325
Yelahanka,2 BHK,1326.0,2.0,78.0,2,5882
Garudachar Palya,3 BHK,1325.0,2.0,60.8,3,4588
EPIP Zone,3 BHK,1499.0,5.0,102.0,3,6804
Hegde Nagar,6 Bedroom,3000.0,7.0,210.0,6,7000
Kanakpura Road,3 BHK,1665.0,3.0,88.0,3,5285
Dasanapura,2 BHK,708.0,2.0,37.0,2,5225
Kasavanhalli,2 BHK,1060.0,2.0,58.06,2,5477
Rajaji Nagar,6 Bedroom,710.0,6.0,160.0,6,22535
Sanjay nagar,2 BHK,1000.0,2.0,70.0,2,7000
Electronic City,2 BHK,1000.0,2.0,28.88,2,2888
other,3 BHK,1450.0,2.0,70.0,3,4827
ISRO Layout,4 Bedroom,1200.0,4.0,155.0,4,12916
Thanisandra,2 BHK,1296.0,2.0,80.0,2,6172
Domlur,3 BHK,1540.0,3.0,90.0,3,5844
Kengeri,4 Bedroom,2894.0,4.0,245.0,4,8465
Sarjapura - Attibele Road,3 BHK,1330.0,2.0,48.0,3,3609
other,2 BHK,1200.0,2.0,65.0,2,5416
other,3 Bedroom,1200.0,3.0,90.0,3,7500
Yeshwanthpur,3 BHK,2502.0,3.0,138.0,3,5515
Chandapura,2 BHK,650.0,1.0,17.0,2,2615
Kothanur,3 Bedroom,2400.0,2.0,150.0,3,6250
other,2 BHK,1007.0,2.0,43.0,2,4270
other,2 BHK,1200.0,2.0,50.0,2,4166
other,2 BHK,966.0,2.0,49.9,2,5165
Nagarbhavi,3 BHK,1630.0,2.0,98.0,3,6012
Rajaji Nagar,3 BHK,1640.0,3.0,229.0,3,13963
other,2 BHK,782.0,2.0,55.68,2,7120
Devanahalli,2 BHK,1260.0,2.0,66.78,2,5300
other,3 BHK,1800.0,3.0,120.0,3,6666
other,3 BHK,1413.0,2.0,75.0,3,5307
Whitefield,2 BHK,1116.0,2.0,51.91,2,4651
Electronic City,3 BHK,1530.0,2.0,45.9,3,3000
Ramamurthy Nagar,4 Bedroom,3700.0,4.0,225.0,4,6081
Sarjapur  Road,3 Bedroom,2497.0,3.0,140.0,3,5606
Kengeri,3 BHK,1540.0,2.0,64.0,3,4155
Thanisandra,3 BHK,1436.0,3.0,74.75,3,5205
Malleshwaram,2 BHK,1100.0,2.0,75.0,2,6818
Hennur Road,2 Bedroom,276.0,3.0,23.0,2,8333
Thanisandra,3 BHK,1427.0,3.0,120.0,3,8409
Akshaya Nagar,3 BHK,2061.0,3.0,200.0,3,9704
Hebbal,4 BHK,5611.5,4.0,477.0,4,8500
Shampura,4 BHK,2650.0,4.0,150.0,4,5660
Devanahalli,3 BHK,1282.0,2.0,68.52,3,5344
Kadugodi,2 BHK,1050.0,2.0,34.0,2,3238
LB Shastri Nagar,3 BHK,1600.0,3.0,65.0,3,4062
other,2 BHK,945.0,2.0,54.0,2,5714
Hormavu,2 BHK,1500.0,2.0,78.0,2,5200
Vishwapriya Layout,7 Bedroom,950.0,7.0,115.0,7,12105
other,3 BHK,1870.0,3.0,110.0,3,5882
Sarjapur,3 Bedroom,1600.0,3.0,75.0,3,4687
Electronic City,2 BHK,880.0,1.0,16.5,2,1875
other,4 Bedroom,1200.0,4.0,210.0,4,17500
Kudlu Gate,3 BHK,1535.0,3.0,83.0,3,5407
Kanakpura Road,2 BHK,950.0,2.0,57.0,2,6000
Devanahalli,2 BHK,1360.0,2.0,65.0,2,4779
8th Phase JP Nagar,2 BHK,1073.5,2.0,54.005,2,5030
Bommasandra Industrial Area,3 BHK,1280.0,3.0,50.4,3,3937
other,3 BHK,1260.0,2.0,85.05,3,6750
Hennur Road,8 Bedroom,5000.0,8.0,250.0,8,5000
Yelahanka,5 BHK,3050.0,5.0,213.0,5,6983
Kasavanhalli,3 BHK,1563.05,3.0,105.0,3,6717
ISRO Layout,2 BHK,1000.0,2.0,60.0,2,6000
Anandapura,2 BHK,1167.0,2.0,43.76,2,3749
Vishveshwarya Layout,7 BHK,4000.0,7.0,225.0,7,5625
Kothanur,3 BHK,1828.0,3.0,110.0,3,6017
Kengeri Satellite Town,2 BHK,890.0,2.0,35.0,2,3932
other,3 BHK,1612.0,3.0,46.74,3,2899
other,6 Bedroom,1034.0,5.0,185.0,6,17891
Mysore Road,3 BHK,1710.0,4.0,91.31,3,5339
Kannamangala,2 BHK,957.0,2.0,58.0,2,6060
Devarachikkanahalli,3 BHK,1250.0,2.0,44.0,3,3520
other,3 BHK,2795.0,4.0,235.0,3,8407
Hulimavu,2 BHK,1125.0,2.0,50.0,2,4444
Electronic City Phase II,2 BHK,1020.0,2.0,30.6,2,3000
Kalena Agrahara,2 BHK,1200.0,2.0,50.0,2,4166
Thanisandra,3 BHK,1735.0,3.0,135.0,3,7780
Thanisandra,3 BHK,2050.0,3.0,145.0,3,7073
Mahalakshmi Layout,4 Bedroom,3750.0,4.0,760.0,4,20266
other,3 BHK,1350.0,2.0,48.0,3,3555
Hosa Road,2 BHK,1063.0,2.0,32.0,2,3010
other,3 BHK,1904.0,3.0,150.0,3,7878
Whitefield,4 Bedroom,4200.0,4.0,420.0,4,10000
other,3 BHK,2000.0,3.0,175.0,3,8750
Sarjapur,2 BHK,1242.5,2.0,43.49,2,3500
other,3 BHK,1425.0,2.0,75.0,3,5263
Electronic City,3 BHK,1500.0,2.0,64.5,3,4300
Electronic City,2 BHK,1060.0,2.0,60.0,2,5660
other,3 BHK,1470.0,2.0,70.0,3,4761
other,6 BHK,1300.0,6.0,99.0,6,7615
Attibele,1 BHK,450.0,1.0,11.0,1,2444
Electronic City,2 BHK,1152.0,2.0,64.5,2,5598
Electronic City,3 BHK,1350.0,2.0,56.0,3,4148
CV Raman Nagar,3 BHK,1550.0,3.0,65.0,3,4193
other,3 Bedroom,1500.0,3.0,100.0,3,6666
Kumaraswami Layout,5 Bedroom,600.0,3.0,85.0,5,14166
Nagavara,1 Bedroom,400.0,1.0,14.0,1,3500
Malleshwaram,1 BHK,705.0,1.0,67.0,1,9503
Electronic City,2 BHK,770.0,1.0,36.0,2,4675
Hulimavu,2 BHK,1242.0,2.0,51.0,2,4106
Hebbal Kempapura,3 BHK,1700.0,3.0,155.0,3,9117
Thanisandra,3 BHK,2144.0,3.0,145.0,3,6763
Vijayanagar,3 BHK,1704.0,3.0,110.0,3,6455
Electronic City,2 BHK,1070.0,2.0,52.0,2,4859
other,3 Bedroom,1846.0,3.0,300.0,3,16251
other,2 BHK,1340.0,2.0,55.7,2,4156
Electronic City,2 BHK,1025.0,2.0,46.0,2,4487
KR Puram,2 BHK,1277.5,2.0,56.8,2,4446
Marathahalli,2 BHK,1200.0,2.0,52.0,2,4333
Kanakpura Road,4 Bedroom,2250.0,4.0,110.0,4,4888
Pattandur Agrahara,3 BHK,1550.0,2.0,80.0,3,5161
Bellandur,4 Bedroom,1200.0,5.0,325.0,4,27083
other,7 Bedroom,1800.0,7.0,250.0,7,13888
Marathahalli,2 BHK,1200.0,2.0,60.0,2,5000
Yelahanka,2 BHK,1327.0,2.0,98.0,2,7385
Kothanur,2 BHK,1186.0,2.0,58.0,2,4890
other,3 Bedroom,1783.0,3.0,115.0,3,6449
HSR Layout,3 BHK,1400.0,3.0,56.0,3,4000
Nagasandra,2 BHK,980.0,2.0,48.0,2,4897
EPIP Zone,2 BHK,1285.0,2.0,82.0,2,6381
other,3 BHK,912.0,2.0,70.0,3,7675
Whitefield,2 BHK,1225.0,2.0,47.6,2,3885
Whitefield,2 BHK,1075.0,2.0,53.0,2,4930
Kadugodi,3 BHK,1260.0,2.0,54.0,3,4285
Yelahanka,3 BHK,1282.0,2.0,48.7,3,3798
Kogilu,3 BHK,1909.0,3.0,88.0,3,4609
Panathur,2 BHK,1359.0,2.0,87.0,2,6401
other,2 BHK,1207.0,2.0,43.5,2,3603
Padmanabhanagar,4 Bedroom,1736.0,6.0,190.0,4,10944
1st Block Jayanagar,4 BHK,2850.0,4.0,428.0,4,15017
Kammasandra,3 BHK,1595.0,3.0,65.0,3,4075
other,3 BHK,1798.0,3.0,89.0,3,4949
Electronics City Phase 1,3 BHK,1475.0,3.0,78.29,3,5307
other,3 BHK,1580.0,3.0,100.0,3,6329
Dasarahalli,2 BHK,1295.0,2.0,65.0,2,5019
Magadi Road,6 Bedroom,3600.0,6.0,141.0,6,3916
Electronic City,1 BHK,589.0,1.0,27.0,1,4584
other,3 BHK,1415.0,3.0,78.0,3,5512
Sarjapur  Road,3 BHK,1787.0,3.0,98.29,3,5500
Sarjapur  Road,3 BHK,1787.0,3.0,125.0,3,6994
Koramangala,3 BHK,1475.0,2.0,90.0,3,6101
other,4 BHK,2000.0,4.0,75.5,4,3775
Sarjapur  Road,2 BHK,984.0,2.0,44.28,2,4500
other,3 BHK,2405.0,4.0,260.0,3,10810
7th Phase JP Nagar,2 BHK,1080.0,2.0,72.0,2,6666
other,2 Bedroom,1500.0,2.0,115.0,2,7666
Hebbal,3 BHK,1900.0,3.0,119.0,3,6263
Dommasandra,2 BHK,805.0,2.0,22.14,2,2750
Budigere,2 BHK,1153.0,2.0,56.4,2,4891
other,2 BHK,1148.0,2.0,60.0,2,5226
other,2 BHK,1110.0,2.0,55.0,2,4954
Electronic City,2 BHK,1100.0,2.0,42.0,2,3818
Kalyan nagar,2 BHK,1290.0,2.0,110.0,2,8527
other,3 BHK,1500.0,2.0,65.0,3,4333
other,2 BHK,1080.0,2.0,39.0,2,3611
Vijayanagar,3 BHK,1933.0,3.0,129.0,3,6673
Ramamurthy Nagar,5 Bedroom,3500.0,5.0,150.0,5,4285
other,2 BHK,1060.0,2.0,60.0,2,5660
Chandapura,1 BHK,645.0,1.0,16.45,1,2550
other,8 Bedroom,2600.0,8.0,180.0,8,6923
Chandapura,1 BHK,645.0,1.0,16.45,1,2550
other,3 BHK,1644.0,3.0,84.0,3,5109
other,2 BHK,1285.0,2.0,72.5,2,5642
Electronic City,3 Bedroom,1200.0,3.0,90.0,3,7500
Electronic City,2 BHK,910.0,2.0,40.0,2,4395
Kothanur,3 BHK,1577.0,3.0,70.0,3,4438
other,3 Bedroom,4050.0,3.0,280.0,3,6913
OMBR Layout,3 BHK,2420.0,3.0,185.0,3,7644
Chandapura,2 BHK,800.0,1.0,20.0,2,2500
other,2 BHK,1060.0,2.0,55.0,2,5188
7th Phase JP Nagar,2 BHK,1270.0,2.0,93.0,2,7322
other,2 BHK,900.0,2.0,22.5,2,2500
other,2 Bedroom,1280.0,2.0,69.0,2,5390
Horamavu Agara,4 Bedroom,1200.0,2.0,95.0,4,7916
Sarjapur  Road,2 BHK,1025.0,2.0,36.0,2,3512
Electronic City,2 BHK,1108.0,2.0,63.0,2,5685
other,5 Bedroom,1200.0,5.0,170.0,5,14166
other,3 BHK,3045.0,3.0,170.0,3,5582
Ambedkar Nagar,4 Bedroom,2900.0,3.0,300.0,4,10344
Vijayanagar,3 BHK,1500.0,3.0,75.5,3,5033
Raja Rajeshwari Nagar,2 BHK,1295.0,2.0,43.9,2,3389
Budigere,2 BHK,1162.0,2.0,59.9,2,5154
Sarjapur  Road,2 BHK,1035.0,2.0,49.2,2,4753
Attibele,4 Bedroom,1600.0,4.0,52.8,4,3300
other,3 BHK,1464.0,3.0,46.0,3,3142
Hennur Road,3 BHK,1866.0,2.0,61.58,3,3300
other,2 BHK,700.0,1.0,26.0,2,3714
Kalena Agrahara,3 BHK,1804.0,3.0,120.0,3,6651
Sarjapur,2 BHK,913.0,2.0,32.0,2,3504
Talaghattapura,3 BHK,1868.0,3.0,131.0,3,7012
Kengeri Satellite Town,2 BHK,883.0,2.0,45.0,2,5096
other,2 Bedroom,900.0,2.0,110.0,2,12222
Kogilu,3 BHK,1664.0,2.0,73.95,3,4444
Hegde Nagar,3 BHK,2026.0,3.0,132.0,3,6515
Balagere,2 BHK,1210.0,2.0,80.9,2,6685
other,4 Bedroom,4111.0,4.0,250.0,4,6081
Kadugodi,3 BHK,1762.0,3.0,91.45,3,5190
Jigani,3 BHK,1252.0,3.0,55.0,3,4392
Gollarapalya Hosahalli,2 BHK,861.0,2.0,34.5,2,4006
7th Phase JP Nagar,3 BHK,1420.0,2.0,100.0,3,7042
other,6 Bedroom,1450.0,6.0,250.0,6,17241
Electronics City Phase 1,3 BHK,1490.0,3.0,78.8,3,5288
Bisuvanahalli,3 BHK,1075.0,2.0,46.0,3,4279
Old Madras Road,3 BHK,1425.0,2.0,94.0,3,6596
Kalena Agrahara,2 BHK,1200.0,2.0,40.0,2,3333
Kaggadasapura,3 BHK,1280.0,2.0,56.0,3,4375
other,2 BHK,1084.0,2.0,50.0,2,4612
Chandapura,2 BHK,1015.0,2.0,25.88,2,2549
Kanakpura Road,2 BHK,1017.0,2.0,51.25,2,5039
Haralur Road,2 BHK,1027.0,2.0,44.0,2,4284
Electronic City Phase II,2 BHK,1069.0,2.0,45.0,2,4209
Hebbal,2 BHK,1349.0,2.0,98.2,2,7279
9th Phase JP Nagar,2 BHK,1005.0,2.0,50.0,2,4975
Jakkur,2 BHK,1417.0,2.0,75.0,2,5292
TC Palaya,3 Bedroom,1475.0,3.0,98.0,3,6644
other,2 BHK,950.0,2.0,47.0,2,4947
other,4 BHK,2000.0,4.0,120.0,4,6000
Giri Nagar,3 Bedroom,880.0,3.0,140.0,3,15909
Sarjapur  Road,4 BHK,1863.0,3.0,104.0,4,5582
Singasandra,2 BHK,1010.0,2.0,29.5,2,2920
Kalena Agrahara,3 BHK,1425.0,2.0,70.0,3,4912
Vijayanagar,3 BHK,1450.0,3.0,100.0,3,6896
Kothanur,3 BHK,1847.0,3.0,105.0,3,5684
AECS Layout,2 BHK,1100.0,2.0,45.0,2,4090
Kanakpura Road,1 BHK,525.0,1.0,26.0,1,4952
Mallasandra,3 BHK,1665.0,3.0,95.0,3,5705
Begur,4 BHK,1664.0,4.0,65.0,4,3906
JP Nagar,3 BHK,1850.0,3.0,150.0,3,8108
Panathur,2 BHK,1438.0,2.0,100.0,2,6954
other,3 BHK,1560.0,3.0,115.0,3,7371
Yelahanka,2 BHK,1350.0,2.0,55.55,2,4114
Kanakpura Road,3 BHK,1550.0,3.0,67.0,3,4322
Malleshpalya,2 BHK,1140.0,2.0,46.5,2,4078
other,5 Bedroom,1200.0,5.0,180.0,5,15000
other,2 BHK,850.0,2.0,27.0,2,3176
Whitefield,2 BHK,1280.0,2.0,75.0,2,5859
Munnekollal,2 BHK,1170.0,2.0,52.0,2,4444
other,2 BHK,1113.0,2.0,51.0,2,4582
Hennur Road,2 BHK,1385.0,2.0,83.09,2,5999
Electronic City,2 BHK,1128.0,2.0,65.0,2,5762
Marathahalli,2 Bedroom,1200.0,2.0,128.0,2,10666
other,4 Bedroom,1200.0,4.0,165.0,4,13750
Rajaji Nagar,3 BHK,2390.0,3.0,415.0,3,17364
Giri Nagar,4 Bedroom,2400.0,4.0,400.0,4,16666
Haralur Road,3 BHK,1464.0,3.0,56.0,3,3825
other,2 BHK,1200.0,2.0,55.0,2,4583
Kaval Byrasandra,2 BHK,1200.0,2.0,52.0,2,4333
Kaggalipura,3 BHK,1150.0,2.0,55.0,3,4782
other,3 BHK,1645.0,3.0,95.0,3,5775
Sarjapur  Road,2 BHK,1000.0,2.0,67.0,2,6700
Haralur Road,4 Bedroom,2650.0,4.0,198.0,4,7471
6th Phase JP Nagar,2 BHK,1192.0,2.0,74.0,2,6208
Ulsoor,3 BHK,2135.0,3.0,215.0,3,10070
Whitefield,2 BHK,1173.0,2.0,58.0,2,4944
Uttarahalli,2 BHK,1020.0,2.0,45.0,2,4411
Thigalarapalya,4 BHK,3122.0,6.0,230.0,4,7367
Somasundara Palya,3 BHK,1600.0,3.0,64.0,3,4000
other,2 Bedroom,1200.0,2.0,79.0,2,6583
other,2 BHK,1230.0,2.0,48.0,2,3902
Devarachikkanahalli,2 BHK,1250.0,2.0,40.0,2,3200
Kalena Agrahara,2 BHK,1325.0,2.0,76.0,2,5735
7th Phase JP Nagar,3 BHK,1850.0,3.0,150.0,3,8108
Basaveshwara Nagar,2 BHK,1200.0,2.0,65.0,2,5416
other,3 Bedroom,1350.0,2.0,140.0,3,10370
Bommasandra,3 BHK,1260.0,3.0,49.36,3,3917
other,3 BHK,1800.0,3.0,150.0,3,8333
other,3 Bedroom,11.0,3.0,74.0,3,672727
Ardendale,2 BHK,1100.0,2.0,43.25,2,3931
Harlur,2 BHK,1508.0,2.0,77.0,2,5106
other,3 BHK,1592.0,3.0,75.0,3,4711
Akshaya Nagar,2 BHK,1388.0,2.0,57.0,2,4106
Electronic City Phase II,1 BHK,630.0,1.0,28.35,1,4500
other,3 Bedroom,2000.0,3.0,365.0,3,18250
other,3 BHK,1762.0,3.0,125.0,3,7094
other,2 BHK,950.0,2.0,50.0,2,5263
Kodihalli,4 BHK,3252.0,5.0,335.0,4,10301
Magadi Road,2 BHK,1116.0,2.0,50.0,2,4480
Narayanapura,2 BHK,1308.0,2.0,89.04,2,6807
other,3 BHK,1200.0,3.0,90.0,3,7500
other,4 Bedroom,1500.0,4.0,230.0,4,15333
Bannerghatta Road,1 BHK,500.0,1.0,18.5,1,3700
Hennur,2 BHK,1075.0,2.0,52.0,2,4837
Chandapura,1 BHK,530.0,1.0,11.66,1,2200
Bellandur,2 BHK,1205.0,2.0,66.0,2,5477
5th Phase JP Nagar,2 BHK,1075.0,2.0,60.0,2,5581
KR Puram,2 BHK,930.0,2.0,39.0,2,4193
Balagere,2 BHK,1380.0,2.0,60.0,2,4347
Hebbal,4 BHK,2483.0,5.0,212.0,4,8538
Kodigehaali,2 BHK,1166.0,2.0,55.0,2,4716
Bannerghatta Road,2 BHK,1050.0,2.0,65.0,2,6190
other,3 BHK,2023.71,3.0,275.0,3,13588
Begur Road,2 BHK,1200.0,2.0,46.8,2,3900
Lakshminarayana Pura,3 BHK,1600.0,2.0,108.0,3,6750
Billekahalli,3 BHK,1935.0,3.0,110.0,3,5684
8th Phase JP Nagar,1 BHK,451.0,1.0,29.9,1,6629
Electronic City,3 BHK,1800.0,3.0,95.0,3,5277
Jalahalli,3 BHK,1400.0,3.0,77.0,3,5500
Hegde Nagar,3 BHK,1801.0,3.0,115.0,3,6385
Mahadevpura,2 BHK,1451.0,2.0,90.0,2,6202
Anekal,2 Bedroom,1200.0,1.0,36.0,2,3000
Begur Road,2 BHK,1160.0,2.0,44.0,2,3793
Kengeri,2 BHK,1000.0,2.0,25.0,2,2500
Kanakpura Road,2 BHK,950.0,2.0,40.0,2,4210
other,3 BHK,1629.0,3.0,79.0,3,4849
Vijayanagar,3 BHK,1580.0,2.0,134.0,3,8481
Vijayanagar,3 BHK,1826.0,3.0,121.0,3,6626
7th Phase JP Nagar,2 BHK,1245.0,2.0,94.0,2,7550
Raja Rajeshwari Nagar,2 BHK,1145.0,2.0,45.79,2,3999
Sompura,2 BHK,825.0,2.0,33.0,2,4000
Yeshwanthpur,3 BHK,1693.0,3.0,108.0,3,6379
Electronics City Phase 1,2 BHK,1113.27,2.0,53.0,2,4760
other,2 BHK,1460.0,2.0,58.0,2,3972
Doddathoguru,2 BHK,1050.0,2.0,32.0,2,3047
Dodda Nekkundi,2 BHK,1260.0,2.0,55.0,2,4365
other,6 BHK,700.0,3.0,120.0,6,17142
Hosur Road,2 BHK,1656.0,2.0,120.0,2,7246
Chandapura,3 BHK,1208.0,3.0,45.0,3,3725
Whitefield,3 BHK,1910.0,3.0,161.0,3,8429
Whitefield,4 BHK,3252.0,4.0,230.0,4,7072
Battarahalli,2 Bedroom,1200.0,2.0,65.0,2,5416
other,4 BHK,1200.0,4.0,175.0,4,14583
7th Phase JP Nagar,2 BHK,1175.0,2.0,82.0,2,6978
other,2 Bedroom,1000.0,2.0,165.0,2,16500
Bannerghatta Road,4 Bedroom,1200.0,2.0,125.0,4,10416
Rajaji Nagar,3 BHK,2390.0,3.0,410.0,3,17154
Rajaji Nagar,7 BHK,12000.0,6.0,2200.0,7,18333
Sultan Palaya,2 Bedroom,550.0,1.0,62.0,2,11272
other,2 BHK,1185.0,2.0,38.0,2,3206
Kengeri,2 BHK,750.0,2.0,38.0,2,5066
Mahalakshmi Layout,6 Bedroom,1200.0,7.0,250.0,6,20833
Kanakpura Road,3 BHK,1550.0,3.0,64.5,3,4161
other,3 BHK,1760.0,3.0,88.0,3,5000
Billekahalli,2 BHK,1125.0,2.0,62.0,2,5511
Nagarbhavi,3 Bedroom,1350.0,3.0,150.0,3,11111
Anandapura,2 Bedroom,1000.0,2.0,55.0,2,5500
Bommasandra Industrial Area,2 BHK,1090.0,2.0,31.48,2,2888
TC Palaya,2 Bedroom,1200.0,2.0,60.0,2,5000
Budigere,3 BHK,1991.0,4.0,103.0,3,5173
Bommasandra,2 BHK,1060.0,2.0,26.5,2,2500
Ambalipura,2 BHK,1105.0,2.0,75.0,2,6787
Hoodi,2 BHK,985.0,2.0,67.0,2,6802
Chandapura,6 Bedroom,1533.0,5.0,85.0,6,5544
Balagere,3 BHK,1590.0,3.0,79.0,3,4968
CV Raman Nagar,2 BHK,1120.0,2.0,55.0,2,4910
other,2 BHK,1069.0,2.0,55.0,2,5144
Marathahalli,3 BHK,1933.0,3.0,140.0,3,7242
Hosur Road,2 BHK,1194.0,2.0,69.0,2,5778
Brookefield,2 BHK,1150.0,2.0,69.0,2,6000
Yelenahalli,2 BHK,1240.0,2.0,47.12,2,3800
Raja Rajeshwari Nagar,2 BHK,1419.0,2.0,48.1,2,3389
7th Phase JP Nagar,8 Bedroom,1200.0,8.0,250.0,8,20833
other,3 BHK,2150.0,4.0,240.0,3,11162
other,3 BHK,1450.0,3.0,55.0,3,3793
other,3 BHK,1630.0,3.0,68.29,3,4189
Dasanapura,2 BHK,708.0,1.0,40.0,2,5649
Kasavanhalli,3 Bedroom,1000.0,4.0,110.0,3,11000
Whitefield,4 Bedroom,11890.0,4.0,700.0,4,5887
Bellandur,2 BHK,1250.0,2.0,76.0,2,6080
7th Phase JP Nagar,3 BHK,1400.0,2.0,95.0,3,6785
Sarjapur  Road,3 BHK,1670.0,2.0,57.0,3,3413
Ardendale,3 BHK,1750.0,3.0,100.0,3,5714
Vittasandra,2 BHK,1404.0,2.0,71.0,2,5056
Hoodi,3 BHK,1715.0,2.0,95.0,3,5539
Electronic City,1 BHK,630.0,1.0,34.65,1,5500
Harlur,3 BHK,1752.12,3.0,135.0,3,7704
Hulimavu,3 BHK,1650.0,3.0,78.0,3,4727
Raja Rajeshwari Nagar,2 BHK,1295.0,2.0,55.47,2,4283
Giri Nagar,3 Bedroom,1800.0,3.0,252.0,3,14000
Sarjapur  Road,2 BHK,1346.0,2.0,74.03,2,5500
other,4 Bedroom,1200.0,4.0,200.0,4,16666
Hormavu,2 BHK,1150.0,2.0,55.0,2,4782
other,4 Bedroom,3309.0,4.0,325.0,4,9821
Chandapura,3 BHK,1190.0,2.0,30.35,3,2550
AECS Layout,3 BHK,1620.0,3.0,98.0,3,6049
Marathahalli,2 BHK,950.0,2.0,46.96,2,4943
other,11 BHK,5000.0,9.0,360.0,11,7200
Harlur,3 BHK,2450.0,3.0,120.0,3,4897
Electronic City Phase II,3 BHK,1220.0,2.0,35.23,3,2887
other,2 BHK,850.0,2.0,30.0,2,3529
Padmanabhanagar,2 BHK,1020.0,2.0,63.0,2,6176
Chikkalasandra,2 BHK,1100.0,2.0,50.0,2,4545
other,2 BHK,1150.0,2.0,126.0,2,10956
Hebbal,3 BHK,1760.0,2.0,130.0,3,7386
Ambalipura,3 BHK,1800.0,3.0,110.0,3,6111
Akshaya Nagar,2 BHK,1070.0,2.0,54.0,2,5046
2nd Stage Nagarbhavi,4 Bedroom,1500.0,4.0,220.0,4,14666
Vidyaranyapura,2 Bedroom,1440.0,2.0,95.0,2,6597
Amruthahalli,2 BHK,900.0,2.0,40.0,2,4444
other,4 BHK,1200.0,4.0,230.0,4,19166
Whitefield,2 BHK,1130.0,2.0,42.16,2,3730
Koramangala,2 BHK,1320.0,2.0,165.0,2,12500
Kodigehalli,2 BHK,1270.0,2.0,98.0,2,7716
other,2 BHK,1200.0,2.0,62.0,2,5166
other,4 BHK,4800.0,4.0,235.0,4,4895
other,2 BHK,1200.0,2.0,74.0,2,6166
Subramanyapura,2 BHK,929.0,2.0,50.17,2,5400
other,4 BHK,1150.0,4.0,260.0,4,22608
Basavangudi,4 Bedroom,1125.0,4.0,180.0,4,16000
Kenchenahalli,5 Bedroom,500.0,3.0,65.0,5,13000
other,5 Bedroom,1200.0,5.0,190.0,5,15833
other,3 BHK,1464.0,3.0,56.0,3,3825
7th Phase JP Nagar,4 BHK,3600.0,4.0,400.0,4,11111
other,4 Bedroom,2000.0,4.0,99.0,4,4950
Chikkalasandra,2 BHK,1208.0,2.0,49.0,2,4056
Hebbal Kempapura,2 BHK,1130.0,2.0,60.0,2,5309
other,3 BHK,1753.0,3.0,120.0,3,6845
Old Madras Road,5 BHK,4500.0,7.0,337.0,5,7488
Banjara Layout,3 Bedroom,600.0,3.0,58.0,3,9666
Hoodi,2 BHK,1196.0,2.0,94.42,2,7894
5th Phase JP Nagar,2 BHK,1150.0,3.0,52.5,2,4565
Electronic City,2 BHK,1128.0,2.0,63.0,2,5585
other,2 BHK,1000.0,2.0,53.0,2,5300
Yelahanka,2 BHK,1035.0,2.0,45.0,2,4347
Kereguddadahalli,2 BHK,950.0,2.0,32.0,2,3368
Bisuvanahalli,3 BHK,1075.0,2.0,36.0,3,3348
Whitefield,2 BHK,1040.0,2.0,48.5,2,4663
other,2 Bedroom,720.0,2.0,45.0,2,6250
Lingadheeranahalli,3 BHK,1511.0,3.0,95.0,3,6287
other,3 BHK,1300.0,3.0,58.0,3,4461
Hoodi,3 BHK,1545.0,3.0,68.75,3,4449
Shampura,1 BHK,375.0,2.0,26.0,1,6933
Vidyaranyapura,2 BHK,1062.0,2.0,38.0,2,3578
other,2 BHK,1115.0,2.0,40.0,2,3587
JP Nagar,2 BHK,1000.0,2.0,35.0,2,3500
Kambipura,2 BHK,883.0,2.0,45.0,2,5096
Sarjapur  Road,2 BHK,1195.0,2.0,70.0,2,5857
other,4 Bedroom,1200.0,3.0,165.0,4,13750
other,1 Bedroom,700.0,1.0,41.0,1,5857
Vittasandra,2 BHK,1246.0,2.0,67.0,2,5377
8th Phase JP Nagar,4 Bedroom,660.0,4.0,90.0,4,13636
Banashankari Stage III,4 Bedroom,8500.0,4.0,145.0,4,1705
other,4 BHK,1600.0,4.0,80.0,4,5000
Haralur Road,4 BHK,2805.0,5.0,154.0,4,5490
Begur Road,3 BHK,1584.0,3.0,65.0,3,4103
Sector 7 HSR Layout,4 Bedroom,3000.0,5.0,275.0,4,9166
Hosur Road,3 BHK,1175.0,2.0,48.0,3,4085
other,3 BHK,1595.0,3.0,115.0,3,7210
other,2 BHK,1353.0,2.0,110.0,2,8130
Electronic City,3 BHK,1599.0,3.0,125.0,3,7817
Whitefield,2 BHK,1150.0,2.0,70.0,2,6086
Jakkur,4 BHK,5230.0,6.0,465.0,4,8891
KR Puram,2 BHK,1155.0,2.0,42.74,2,3700
other,1 BHK,1200.0,2.0,85.0,1,7083
Electronic City Phase II,2 BHK,1000.0,2.0,25.0,2,2500
Thanisandra,4 BHK,3000.0,4.0,120.0,4,4000
Rajiv Nagar,3 BHK,1867.0,3.0,160.0,3,8569
Ramagondanahalli,2 BHK,1251.0,2.0,47.0,2,3756
other,2 BHK,1028.0,2.0,45.5,2,4426
Arekere,2 BHK,1222.0,2.0,45.0,2,3682
Bannerghatta Road,3 BHK,1170.0,2.0,66.0,3,5641
Hebbal,3 BHK,2400.0,4.0,245.0,3,10208
other,3 BHK,1385.0,2.0,48.48,3,3500
HSR Layout,2 BHK,1372.0,2.0,68.0,2,4956
Magadi Road,3 BHK,1282.0,2.0,49.0,3,3822
Mico Layout,9 BHK,5000.0,9.0,210.0,9,4200
other,5 Bedroom,3000.0,3.0,528.0,5,17600
Kanakpura Road,2 BHK,1135.0,2.0,39.73,2,3500
Whitefield,3 BHK,1768.0,3.0,101.0,3,5712
other,3 BHK,1325.0,2.0,90.0,3,6792
Electronic City,3 BHK,1599.0,3.0,99.0,3,6191
Whitefield,3 Bedroom,1500.0,3.0,61.95,3,4130
other,3 Bedroom,2610.0,3.0,499.0,3,19118
Uttarahalli,2 BHK,1286.0,2.0,63.0,2,4898
Kammanahalli,5 BHK,2845.0,5.0,140.0,5,4920
other,3 BHK,1600.0,2.0,50.0,3,3125
Hennur Road,2 BHK,1317.5,2.0,63.77,2,4840
Hebbal,3 BHK,3450.0,5.0,260.0,3,7536
Munnekollal,2 BHK,1102.0,2.0,53.67,2,4870
Yelahanka,2 BHK,1350.0,2.0,54.0,2,4000
Banashankari,3 BHK,1200.0,2.0,42.0,3,3500
Hosa Road,6 Bedroom,1300.0,6.0,145.0,6,11153
other,2 Bedroom,1800.0,1.0,80.0,2,4444
Chikkabanavar,2 BHK,950.0,2.0,40.0,2,4210
Kanakpura Road,1 BHK,525.0,1.0,27.25,1,5190
Attibele,2 BHK,656.0,2.0,25.0,2,3810
Marathahalli,4 Bedroom,1780.0,4.0,175.0,4,9831
other,2 BHK,1056.0,2.0,80.0,2,7575
Bannerghatta Road,1 BHK,595.0,1.0,33.32,1,5600
other,2 BHK,1080.0,2.0,50.0,2,4629
7th Phase JP Nagar,3 BHK,2225.0,3.0,160.0,3,7191
Bommasandra,2 BHK,1126.0,2.0,28.15,2,2500
Electronics City Phase 1,3 BHK,1490.0,3.0,84.0,3,5637
HRBR Layout,4 Bedroom,2000.0,4.0,300.0,4,15000
Sarjapur  Road,3 BHK,1550.0,3.0,100.0,3,6451
other,2 BHK,1160.0,2.0,55.0,2,4741
Whitefield,4 Bedroom,4144.0,4.0,315.0,4,7601
Sarjapur  Road,3 BHK,2100.0,3.0,125.0,3,5952
Vittasandra,2 BHK,1404.0,2.0,69.3,2,4935
Sarjapur  Road,4 Bedroom,2230.0,4.0,225.0,4,10089
other,3 BHK,1544.0,3.0,120.0,3,7772
Banashankari Stage III,3 BHK,1305.0,2.0,58.7,3,4498
Marathahalli,2 BHK,1230.0,2.0,80.0,2,6504
other,2 Bedroom,1200.0,2.0,65.0,2,5416
Harlur,3 BHK,1460.0,3.0,73.0,3,5000
Gottigere,2 BHK,967.0,2.0,45.0,2,4653
other,1 Bedroom,540.0,1.0,36.0,1,6666
Hormavu,1 BHK,715.0,1.0,46.0,1,6433
Whitefield,3 BHK,2500.0,3.0,313.0,3,12520
other,3 BHK,1578.0,3.0,180.0,3,11406
Mysore Road,2 BHK,1020.0,2.0,48.95,2,4799
Jigani,2 BHK,1020.0,2.0,40.0,2,3921
Koramangala,2 BHK,1253.0,2.0,102.0,2,8140
other,2 BHK,1180.0,2.0,42.0,2,3559
Nehru Nagar,2 BHK,961.0,2.0,38.0,2,3954
Kanakapura,3 BHK,1419.0,3.0,75.0,3,5285
Harlur,3 BHK,1709.0,3.0,150.0,3,8777
other,2 Bedroom,1600.0,2.0,115.0,2,7187
other,1 BHK,416.0,1.0,18.5,1,4447
Sanjay nagar,2 BHK,1100.0,2.0,98.0,2,8909
HSR Layout,3 BHK,1430.0,2.0,65.0,3,4545
other,2 BHK,1260.0,2.0,45.0,2,3571
other,3 BHK,1630.0,3.0,85.0,3,5214
Sarjapur  Road,3 BHK,1249.0,3.0,44.71,3,3579
Kalena Agrahara,3 BHK,1450.0,2.0,85.0,3,5862
Konanakunte,3 BHK,2791.0,3.0,223.0,3,7989
other,1 BHK,600.0,1.0,90.0,1,15000
other,1 BHK,834.0,1.0,60.0,1,7194
other,3 Bedroom,1125.0,4.0,70.0,3,6222
Arekere,3 BHK,2060.0,3.0,140.0,3,6796
Malleshwaram,7 BHK,12000.0,7.0,2200.0,7,18333
Gottigere,2 BHK,891.0,2.0,25.0,2,2805
Raja Rajeshwari Nagar,2 BHK,1133.0,2.0,49.07,2,4330
Bisuvanahalli,3 BHK,1075.0,2.0,37.0,3,3441
other,4 BHK,3000.0,5.0,260.0,4,8666
Brookefield,4 Bedroom,2440.0,5.0,425.0,4,17418
Vidyaranyapura,2 BHK,1200.0,2.0,55.0,2,4583
Thanisandra,2 BHK,1075.0,2.0,45.8,2,4260
Kaggadasapura,2 BHK,1140.0,2.0,40.0,2,3508
Margondanahalli,5 Bedroom,940.0,4.0,150.0,5,15957
other,6 Bedroom,2160.0,5.0,250.0,6,11574
Raja Rajeshwari Nagar,2 BHK,1090.0,2.0,42.0,2,3853
R.T. Nagar,4 Bedroom,1500.0,4.0,70.0,4,4666
Whitefield,4 BHK,4104.0,4.0,360.0,4,8771
Whitefield,3 BHK,1790.0,3.0,100.0,3,5586
Hebbal,3 BHK,1920.0,3.0,134.0,3,6979
HRBR Layout,2 BHK,1374.0,2.0,95.0,2,6914
Bannerghatta Road,3 BHK,1445.0,3.0,95.0,3,6574
Kanakapura,1 BHK,711.0,1.0,36.0,1,5063
Tumkur Road,3 BHK,1500.0,3.0,95.0,3,6333
Whitefield,3 BHK,1720.0,3.0,100.0,3,5813
Vasanthapura,3 BHK,1400.0,3.0,60.0,3,4285
Kundalahalli,2 BHK,1030.0,2.0,49.0,2,4757
Mysore Road,2 BHK,1200.0,2.0,85.0,2,7083
other,3 BHK,1375.0,3.0,70.0,3,5090
Hegde Nagar,2 Bedroom,1050.0,1.0,56.0,2,5333
Jalahalli,3 BHK,2250.0,3.0,160.0,3,7111
Bellandur,2 BHK,1000.0,2.0,50.0,2,5000
other,1 BHK,469.0,1.0,25.0,1,5330
Marathahalli,4 BHK,3800.0,4.0,250.0,4,6578
Kambipura,2 BHK,883.0,2.0,29.0,2,3284
GM Palaya,3 BHK,1820.0,2.0,86.0,3,4725
Singasandra,3 BHK,1440.0,2.0,65.0,3,4513
other,6 BHK,3600.0,4.0,160.0,6,4444
Malleshwaram,4 BHK,2500.0,5.0,325.0,4,13000
Mahadevpura,2 BHK,1225.0,2.0,48.0,2,3918
other,3 Bedroom,4000.0,3.0,660.0,3,16500
Chikkalasandra,2 BHK,875.0,2.0,52.8,2,6034
Jalahalli East,1 BHK,750.0,1.0,40.0,1,5333
6th Phase JP Nagar,2 BHK,1180.0,2.0,80.0,2,6779
Haralur Road,2 BHK,1194.0,2.0,47.0,2,3936
Hennur Road,3 BHK,1500.0,2.0,62.0,3,4133
other,2 BHK,1160.0,2.0,52.2,2,4500
Hosakerehalli,3 BHK,2378.0,3.0,262.0,3,11017
Sarjapur,2 BHK,1128.0,2.0,60.0,2,5319
Hebbal,3 BHK,1500.0,3.0,150.0,3,10000
other,3 BHK,1220.0,2.0,35.23,3,2887
Sarjapur  Road,2 BHK,1210.0,2.0,58.0,2,4793
Kanakpura Road,3 BHK,1100.0,2.0,58.0,3,5272
Bommasandra,3 BHK,1200.0,3.0,46.0,3,3833
Haralur Road,4 BHK,3385.0,5.0,260.0,4,7680
other,3 BHK,1641.0,3.0,66.0,3,4021
Horamavu Agara,2 BHK,1200.0,2.0,45.0,2,3750
Raja Rajeshwari Nagar,2 BHK,1260.0,2.0,67.39,2,5348
Mahadevpura,3 BHK,1720.0,3.0,78.0,3,4534
Yelahanka,2 BHK,1132.5,2.0,48.13,2,4249
Yelahanka,2 BHK,1210.0,2.0,45.98,2,3800
Indira Nagar,4 Bedroom,2200.0,4.0,200.0,4,9090
Bisuvanahalli,3 BHK,1075.0,2.0,45.0,3,4186
Thanisandra,3 BHK,1702.0,3.0,130.0,3,7638
other,2 BHK,1630.0,2.0,200.0,2,12269
Raja Rajeshwari Nagar,2 BHK,1141.0,2.0,38.55,2,3378
other,3 BHK,2072.0,3.0,108.0,3,5212
other,3 Bedroom,1200.0,2.0,70.0,3,5833
other,3 BHK,1350.0,3.0,77.0,3,5703
other,4 Bedroom,4046.0,4.0,445.0,4,10998
other,3 BHK,1180.0,2.0,78.34,3,6638
Yelahanka,3 BHK,35000.0,3.0,130.0,3,371
other,2 BHK,1355.0,2.0,115.0,2,8487
other,1 Bedroom,600.0,1.0,50.0,1,8333
Hormavu,7 Bedroom,1200.0,7.0,150.0,7,12500
Bannerghatta Road,3 BHK,1640.0,2.0,120.0,3,7317
other,5 Bedroom,1200.0,5.0,100.0,5,8333
Kodichikkanahalli,3 BHK,1190.0,2.0,57.0,3,4789
OMBR Layout,4 Bedroom,2400.0,4.0,375.0,4,15625
Marathahalli,2 BHK,1019.0,2.0,49.85,2,4892
other,3 BHK,1875.0,3.0,165.0,3,8800
Akshaya Nagar,3 BHK,1250.0,3.0,40.0,3,3200
Brookefield,3 Bedroom,1200.0,3.0,140.0,3,11666
Whitefield,2 BHK,1140.0,2.0,56.0,2,4912
Lingadheeranahalli,3 BHK,1683.0,3.0,109.0,3,6476
Kaggadasapura,3 BHK,1515.0,3.0,83.3,3,5498
Hoodi,4 BHK,2118.0,4.0,111.0,4,5240
Bannerghatta Road,4 Bedroom,1380.0,3.0,58.0,4,4202
Varthur Road,2 BHK,1083.0,2.0,61.75,2,5701
other,2 Bedroom,600.0,2.0,55.0,2,9166
other,3 BHK,2300.0,3.0,120.0,3,5217
other,2 BHK,1240.0,2.0,60.0,2,4838
other,3 BHK,1750.0,3.0,120.0,3,6857
other,2 BHK,1230.0,2.0,43.48,2,3534
Anjanapura,3 Bedroom,1500.0,3.0,139.0,3,9266
Abbigere,2 BHK,985.0,2.0,40.39,2,4100
Tindlu,3 Bedroom,1500.0,2.0,87.0,3,5800
Electronic City,2 BHK,1125.0,2.0,32.49,2,2888
other,2 BHK,950.0,2.0,44.0,2,4631
Kothanur,3 BHK,1580.0,3.0,76.0,3,4810
other,3 Bedroom,1092.0,2.0,98.5,3,9020
Electronic City Phase II,2 BHK,1090.0,2.0,28.34,2,2600
Hennur Road,3 Bedroom,2264.0,3.0,159.0,3,7022
Dommasandra,3 BHK,1033.0,2.0,25.53,3,2471
Kammasandra,2 BHK,810.0,2.0,24.5,2,3024
other,2 BHK,1045.0,2.0,45.0,2,4306
Hebbal,2 BHK,1337.0,2.0,82.0,2,6133
Chandapura,2 BHK,1200.0,2.0,32.0,2,2666
Koramangala,3 BHK,1580.0,3.0,160.0,3,10126
Begur Road,3 BHK,1500.0,2.0,54.0,3,3600
Whitefield,3 BHK,1640.0,4.0,91.0,3,5548
other,3 BHK,1570.0,3.0,62.0,3,3949
Gubbalala,3 BHK,1470.0,2.0,82.0,3,5578
Kengeri,2 BHK,1160.0,2.0,40.61,2,3500
7th Phase JP Nagar,2 BHK,1050.0,2.0,71.0,2,6761
other,3 BHK,1855.0,3.0,225.0,3,12129
Bannerghatta Road,3 BHK,1460.0,2.0,90.0,3,6164
Hulimavu,3 BHK,1823.0,3.0,100.0,3,5485
Electronic City,2 BHK,1094.0,2.0,42.0,2,3839
Budigere,2 BHK,1153.0,2.0,57.0,2,4943
other,3 BHK,1325.0,3.0,57.0,3,4301
Parappana Agrahara,4 Bedroom,1200.0,3.0,85.0,4,7083
Jakkur,3 BHK,1590.0,2.0,125.0,3,7861
other,5 Bedroom,1210.0,4.0,160.0,5,13223
Raja Rajeshwari Nagar,4 Bedroom,1200.0,4.0,220.0,4,18333
other,2 BHK,1202.0,2.0,48.0,2,3993
Haralur Road,2 BHK,1202.0,2.0,53.0,2,4409
other,2 BHK,1200.0,2.0,62.5,2,5208
Panathur,3 BHK,1688.0,3.0,118.0,3,6990
Kaval Byrasandra,2 BHK,1020.0,2.0,52.0,2,5098
Sarjapur,2 BHK,1185.0,2.0,47.0,2,3966
Ramagondanahalli,2 BHK,1235.0,2.0,52.04,2,4213
Hosakerehalli,4 BHK,3205.0,5.0,500.0,4,15600
Banashankari,2 BHK,1077.0,2.0,37.64,2,3494
Kengeri Satellite Town,3 BHK,1415.0,2.0,66.0,3,4664
EPIP Zone,3 BHK,2330.0,3.0,162.0,3,6952
Whitefield,2 BHK,805.0,2.0,35.0,2,4347
other,1 BHK,425.0,1.0,20.03,1,4712
Whitefield,2 BHK,1155.0,2.0,59.95,2,5190
Cunningham Road,4 BHK,5270.0,4.0,1250.0,4,23719
Anekal,2 BHK,656.0,2.0,22.0,2,3353
other,2 BHK,1100.0,2.0,48.0,2,4363
Mahadevpura,2 BHK,1150.0,2.0,52.0,2,4521
other,1 Bedroom,600.0,1.0,60.0,1,10000
other,2 BHK,1468.0,2.0,140.0,2,9536
other,4 Bedroom,4300.0,5.0,550.0,4,12790
Whitefield,3 BHK,2280.0,4.0,126.0,3,5526
Hegde Nagar,2 BHK,1341.0,2.0,97.0,2,7233
Jakkur,2 BHK,1279.0,2.0,77.0,2,6020
Electronic City Phase II,4 BHK,2225.0,4.0,98.88,4,4444
Thanisandra,2 BHK,1185.0,2.0,43.51,2,3671
other,3 BHK,1750.0,3.0,85.0,3,4857
Kudlu,2 BHK,1152.0,2.0,53.5,2,4644
other,5 Bedroom,1000.0,4.0,160.0,5,16000
Hulimavu,2 BHK,1300.0,2.0,36.0,2,2769
Varthur Road,4 Bedroom,2760.0,4.0,155.0,4,5615
5th Phase JP Nagar,2 BHK,1070.0,2.0,39.0,2,3644
Uttarahalli,2 BHK,1101.0,2.0,45.0,2,4087
other,2 BHK,775.0,2.0,55.0,2,7096
Yeshwanthpur,1 BHK,667.0,1.0,36.85,1,5524
Electronic City,2 BHK,1070.0,2.0,55.0,2,5140
Old Airport Road,3 BHK,1875.0,2.0,110.0,3,5866
Kundalahalli,2 BHK,735.0,2.0,57.99,2,7889
Sarjapur,5 Bedroom,4360.0,4.0,90.0,5,2064
Koramangala,3 BHK,1750.0,3.0,130.0,3,7428
Marathahalli,2 BHK,1215.0,2.0,67.0,2,5514
other,9 Bedroom,600.0,9.0,190.0,9,31666
Vishwapriya Layout,2 BHK,820.0,2.0,30.0,2,3658
Banashankari Stage VI,2 BHK,1177.5,2.0,59.935,2,5090
Battarahalli,3 BHK,1779.0,3.0,89.61,3,5037
other,2 BHK,1105.0,2.0,58.0,2,5248
Whitefield,3 BHK,1650.0,3.0,60.0,3,3636
Jalahalli,2 BHK,1694.0,2.0,150.0,2,8854
Hosakerehalli,3 BHK,2376.0,3.0,203.0,3,8543
Cox Town,3 BHK,1975.0,3.0,150.0,3,7594
Kammasandra,2 BHK,674.0,2.0,33.0,2,4896
Kaggadasapura,2 BHK,1185.0,2.0,41.0,2,3459
Thanisandra,1 RK,445.0,1.0,28.0,1,6292
other,6 Bedroom,900.0,7.0,76.0,6,8444
HSR Layout,3 BHK,1475.0,3.0,75.0,3,5084
Gottigere,3 BHK,1618.0,3.0,82.5,3,5098
Koramangala,2 BHK,1120.0,2.0,65.0,2,5803
Marathahalli,4 BHK,2181.0,3.0,152.0,4,6969
Vidyaranyapura,4 Bedroom,1200.0,4.0,90.0,4,7500
other,6 Bedroom,600.0,4.0,65.0,6,10833
Bannerghatta Road,3 BHK,1556.0,2.0,59.13,3,3800
Bannerghatta Road,2 BHK,1179.0,2.0,66.02,2,5599
Thanisandra,2 BHK,1296.0,2.0,83.0,2,6404
Kaggadasapura,2 BHK,1275.0,2.0,52.0,2,4078
other,7 Bedroom,1875.0,2.0,300.0,7,16000
Kathriguppe,3 BHK,1400.0,2.0,77.0,3,5500
Gottigere,2 BHK,1222.0,2.0,63.0,2,5155
Electronic City,2 BHK,940.0,2.0,41.0,2,4361
Thanisandra,2 BHK,1185.0,2.0,42.6,2,3594
Tumkur Road,3 BHK,1779.0,3.0,112.0,3,6295
Begur Road,3 BHK,1615.0,3.0,59.76,3,3700
Haralur Road,2 BHK,1056.0,2.0,61.0,2,5776
7th Phase JP Nagar,2 BHK,1100.0,2.0,46.0,2,4181
Arekere,2 BHK,920.0,2.0,40.0,2,4347
other,3 BHK,1602.0,2.0,75.0,3,4681
Padmanabhanagar,2 BHK,1176.0,2.0,62.0,2,5272
HBR Layout,4 Bedroom,675.0,3.0,59.0,4,8740
other,3 BHK,1352.0,3.0,135.0,3,9985
Kasavanhalli,3 BHK,1717.0,3.0,99.0,3,5765
other,4 Bedroom,10961.0,4.0,80.0,4,729
Bhoganhalli,4 BHK,2119.0,4.0,111.0,4,5238
Vittasandra,2 BHK,1246.0,2.0,67.3,2,5401
Anandapura,2 BHK,1141.0,2.0,42.79,2,3750
Hennur Road,2 BHK,1157.0,2.0,84.0,2,7260
Chandapura,2 BHK,1025.0,2.0,27.68,2,2700
Yelahanka New Town,1 BHK,650.0,1.0,33.0,1,5076
Hebbal,2 BHK,1349.0,2.0,98.0,2,7264
Narayanapura,3 BHK,1566.0,3.0,83.3,3,5319
Sahakara Nagar,4 BHK,2830.0,3.0,195.0,4,6890
Electronic City,3 BHK,1320.0,2.0,38.13,3,2888
other,3 BHK,1780.0,3.0,84.83,3,4765
Varthur,2 BHK,1091.0,2.0,34.9,2,3198
Rachenahalli,4 BHK,3670.0,4.0,300.0,4,8174
7th Phase JP Nagar,2 BHK,918.0,2.0,50.49,2,5500
Yeshwanthpur,4 BHK,1950.0,4.0,130.0,4,6666
Panathur,3 BHK,1695.0,3.0,90.0,3,5309
Yelahanka,3 BHK,1705.0,3.0,95.0,3,5571
Kalyan nagar,3 BHK,1375.0,3.0,75.0,3,5454
Kanakpura Road,1 BHK,525.0,1.0,30.0,1,5714
Whitefield,2 BHK,1447.0,2.0,83.0,2,5736
Thanisandra,2 BHK,1114.0,2.0,39.0,2,3500
7th Phase JP Nagar,3 BHK,1450.0,2.0,100.0,3,6896
other,2 BHK,460.0,1.0,15.0,2,3260
Vasanthapura,2 BHK,1022.0,2.0,40.0,2,3913
Hebbal,2 BHK,1000.0,2.0,45.0,2,4500
Electronic City,2 BHK,1128.0,2.0,68.75,2,6094
Kanakpura Road,2 BHK,1180.0,2.0,57.0,2,4830
TC Palaya,6 Bedroom,1000.0,6.0,69.0,6,6900
Nagarbhavi,2 Bedroom,1200.0,2.0,150.0,2,12500
other,3 BHK,3761.0,3.0,660.0,3,17548
Kanakpura Road,2 BHK,1339.0,2.0,75.0,2,5601
Yelachenahalli,3 BHK,1400.0,2.0,55.0,3,3928
7th Phase JP Nagar,2 BHK,1040.0,2.0,39.52,2,3800
Kadugodi,2 BHK,1198.0,2.0,58.0,2,4841
other,4 Bedroom,1200.0,3.0,95.0,4,7916
Sarjapur  Road,3 BHK,1691.0,3.0,113.0,3,6682
Bisuvanahalli,3 BHK,1075.0,3.0,36.0,3,3348
1st Block Jayanagar,3 BHK,1630.0,3.0,194.0,3,11901
8th Phase JP Nagar,3 BHK,1240.0,3.0,43.5,3,3508
Bannerghatta Road,2 BHK,1122.5,2.0,58.935,2,5250
Mallasandra,3 BHK,1665.0,3.0,86.58,3,5200
Marathahalli,3 BHK,2489.0,3.0,94.0,3,3776
Brookefield,2 BHK,1142.0,2.0,70.0,2,6129
other,3 BHK,1976.0,3.0,184.0,3,9311
other,4 Bedroom,5500.0,4.0,600.0,4,10909
Kothannur,3 BHK,1853.0,3.0,82.0,3,4425
Vijayanagar,8 Bedroom,600.0,4.0,72.0,8,12000
Kasturi Nagar,2 BHK,1567.0,2.0,92.0,2,5871
Talaghattapura,2 BHK,1090.0,2.0,27.24,2,2499
Raja Rajeshwari Nagar,6 Bedroom,1200.0,4.0,125.0,6,10416
other,4 Bedroom,2400.0,4.0,640.0,4,26666
Electronics City Phase 1,2 BHK,1175.0,2.0,51.47,2,4380
Subramanyapura,2 BHK,1200.0,2.0,52.0,2,4333
Kodichikkanahalli,2 BHK,995.0,2.0,41.0,2,4120
Magadi Road,2 BHK,884.0,2.0,41.1,2,4649
other,5 Bedroom,900.0,3.0,80.0,5,8888
Yeshwanthpur,2 BHK,1170.0,2.0,57.0,2,4871
KR Puram,2 BHK,1225.0,2.0,46.55,2,3800
Electronic City,2 BHK,1342.0,2.0,73.3,2,5461
Uttarahalli,2 BHK,1345.0,2.0,67.0,2,4981
Kodigehaali,3 BHK,1320.0,3.0,65.0,3,4924
Raja Rajeshwari Nagar,2 BHK,1140.0,2.0,49.39,2,4332
Battarahalli,2 BHK,1300.0,2.0,50.0,2,3846
Hosa Road,3 BHK,1652.0,3.0,85.0,3,5145
Thigalarapalya,3 BHK,2072.0,4.0,142.0,3,6853
Arekere,3 BHK,1740.0,2.0,95.0,3,5459
Electronic City,3 BHK,1360.0,2.0,71.0,3,5220
Hosur Road,3 BHK,1145.0,2.0,60.0,3,5240
Uttarahalli,3 BHK,1540.0,2.0,40.0,3,2597
Sarjapur  Road,2 BHK,1278.0,2.0,95.0,2,7433
Electronic City Phase II,1 BHK,630.0,1.0,40.0,1,6349
Kanakpura Road,3 BHK,1100.0,3.0,58.0,3,5272
other,3 Bedroom,1200.0,3.0,95.0,3,7916
other,7 Bedroom,2500.0,6.0,115.0,7,4600
Chandapura,1 BHK,582.5,1.0,15.135,1,2598
Chandapura,2 BHK,1015.0,2.0,25.88,2,2549
Hosur Road,5 Bedroom,3300.0,5.0,240.0,5,7272
Basaveshwara Nagar,5 Bedroom,4500.0,5.0,415.0,5,9222
Gunjur,3 BHK,1356.0,2.0,66.0,3,4867
other,2 BHK,823.0,2.0,28.0,2,3402
TC Palaya,3 Bedroom,1200.0,2.0,66.0,3,5500
Ramamurthy Nagar,2 BHK,1050.0,2.0,37.0,2,3523
other,2 BHK,1180.0,2.0,69.0,2,5847
Whitefield,2 BHK,1116.0,2.0,51.91,2,4651
7th Phase JP Nagar,2 BHK,1070.0,2.0,42.79,2,3999
Varthur,2 BHK,1385.0,2.0,80.0,2,5776
Varthur,1 Bedroom,1000.0,1.0,70.0,1,7000
Whitefield,3 BHK,1897.0,3.0,130.0,3,6852
Magadi Road,2 BHK,1005.0,2.0,55.78,2,5550
Green Glen Layout,3 BHK,1575.0,3.0,125.0,3,7936
Subramanyapura,2 BHK,975.0,2.0,65.0,2,6666
Hennur Road,3 BHK,1904.0,3.0,120.0,3,6302
Kengeri Satellite Town,1 BHK,930.0,1.0,30.0,1,3225
Kanakpura Road,1 BHK,525.0,1.0,26.0,1,4952
other,1 BHK,686.0,1.0,17.0,1,2478
Electronic City Phase II,2 Bedroom,2400.0,3.0,150.0,2,6250
other,2 Bedroom,1410.0,1.0,233.0,2,16524
Sarjapur  Road,3 Bedroom,2238.0,3.0,140.0,3,6255
Harlur,2 BHK,1174.0,2.0,69.0,2,5877
Thubarahalli,4 Bedroom,3800.0,4.0,239.0,4,6289
Whitefield,3 BHK,2225.0,2.0,125.0,3,5617
other,2 BHK,1250.0,2.0,200.0,2,16000
Jigani,2 BHK,918.0,2.0,52.0,2,5664
Mahadevpura,2 BHK,1250.0,2.0,52.0,2,4160
Sarjapur  Road,3 BHK,1800.0,3.0,110.0,3,6111
Marathahalli,2 BHK,1170.0,2.0,53.0,2,4529
other,1 BHK,1300.0,1.0,200.0,1,15384
Bannerghatta Road,2 BHK,793.0,2.0,45.0,2,5674
Jakkur,3 BHK,1710.0,3.0,107.0,3,6257
Kambipura,3 BHK,1082.0,2.0,57.5,3,5314
other,2 BHK,1001.0,2.0,33.03,2,3299
other,4 Bedroom,1500.0,4.0,180.0,4,12000
Yelahanka,3 BHK,1590.0,2.0,54.0,3,3396
other,3 BHK,2400.0,2.0,80.0,3,3333
Horamavu Banaswadi,3 BHK,1554.0,3.0,55.0,3,3539
Thanisandra,2 BHK,1142.5,2.0,43.415,2,3800
Bellandur,5 BHK,4239.0,5.0,423.0,5,9978
Konanakunte,2 Bedroom,884.0,1.0,58.0,2,6561
Hosa Road,2 BHK,1045.0,2.0,98.89,2,9463
Marathahalli,2 BHK,1019.0,2.0,49.86,2,4893
Ambedkar Nagar,3 BHK,1935.0,4.0,130.0,3,6718
Uttarahalli,3 BHK,1135.0,2.0,39.73,3,3500
Lakshminarayana Pura,3 BHK,1680.0,3.0,150.0,3,8928
Hebbal,4 BHK,2470.0,5.0,247.0,4,10000
1st Phase JP Nagar,4 BHK,2825.0,4.0,250.0,4,8849
Hosakerehalli,3 BHK,2480.0,3.0,330.0,3,13306
5th Phase JP Nagar,9 Bedroom,1260.0,11.0,290.0,9,23015
Ramamurthy Nagar,5 Bedroom,1800.0,4.0,187.0,5,10388
Kasavanhalli,3 BHK,1799.0,3.0,90.0,3,5002
Attibele,1 BHK,400.0,1.0,11.0,1,2750
Kundalahalli,2 BHK,1047.0,2.0,82.0,2,7831
other,2 BHK,1282.0,2.0,87.0,2,6786
NGR Layout,2 BHK,1020.0,2.0,45.9,2,4500
Kammanahalli,4 Bedroom,1080.0,3.0,155.0,4,14351
Whitefield,2 BHK,1495.0,2.0,79.5,2,5317
Kasavanhalli,4 Bedroom,3260.0,4.0,240.0,4,7361
other,3 BHK,1611.0,2.0,64.44,3,4000
Koramangala,4 BHK,3500.0,5.0,425.0,4,12142
other,3 Bedroom,1500.0,2.0,95.0,3,6333
Sahakara Nagar,3 BHK,1500.0,2.0,95.0,3,6333
Yelahanka,4 Bedroom,3206.0,5.0,270.0,4,8421
Uttarahalli,3 BHK,1540.0,3.0,50.0,3,3246
Hosa Road,3 BHK,1639.0,3.0,80.0,3,4881
Marathahalli,4 Bedroom,700.0,4.0,72.0,4,10285
Raja Rajeshwari Nagar,2 BHK,1303.0,2.0,55.79,2,4281
Koramangala,2 BHK,1005.0,1.0,110.0,2,10945
Kammasandra,1 BHK,657.5,1.0,18.41,1,2800
Chandapura,3 BHK,1305.0,3.0,33.28,3,2550
Kothannur,3 Bedroom,600.0,4.0,95.0,3,15833
other,2 BHK,1300.0,1.0,115.0,2,8846
Brookefield,3 BHK,1440.0,2.0,68.0,3,4722
Seegehalli,2 BHK,901.0,2.0,35.14,2,3900
Rajaji Nagar,3 BHK,1725.0,3.0,200.0,3,11594
Sarjapur,3 BHK,900.0,2.0,42.0,3,4666
BEML Layout,2 BHK,1060.0,2.0,55.0,2,5188
NGR Layout,2 BHK,1020.0,2.0,48.45,2,4750
Varthur,3 BHK,1535.0,3.0,65.99,3,4299
other,1 BHK,530.0,1.0,20.0,1,3773
Haralur Road,3 BHK,1464.0,3.0,56.0,3,3825
other,2 BHK,1350.0,2.0,80.0,2,5925
Sarjapur  Road,2 BHK,1115.0,2.0,50.0,2,4484
Uttarahalli,2 BHK,1175.0,2.0,47.0,2,4000
Whitefield,2 BHK,1113.0,2.0,62.0,2,5570
Uttarahalli,6 Bedroom,1200.0,4.0,225.0,6,18750
Ramamurthy Nagar,5 Bedroom,1640.0,4.0,240.0,5,14634
Giri Nagar,2 BHK,650.0,2.0,42.0,2,6461
Marathahalli,2 BHK,1325.0,2.0,71.55,2,5400
Electronics City Phase 1,2 BHK,950.0,2.0,31.0,2,3263
Jakkur,3 BHK,1396.0,2.0,86.51,3,6196
Rachenahalli,2 BHK,1050.0,2.0,52.07,2,4959
Thanisandra,3 BHK,1825.0,3.0,100.0,3,5479
7th Phase JP Nagar,2 BHK,1190.0,2.0,49.98,2,4200
Marathahalli,2 BHK,1220.0,2.0,56.5,2,4631
NRI Layout,3 BHK,1565.0,2.0,75.0,3,4792
Hennur Road,3 BHK,1891.0,3.0,109.0,3,5764
other,2 BHK,900.0,2.0,27.0,2,3000
Hosa Road,2 BHK,1161.0,2.0,55.15,2,4750
Sarjapur,1 BHK,649.5,1.0,17.535,1,2699
other,8 Bedroom,2400.0,6.0,125.0,8,5208
Rajaji Nagar,4 Bedroom,315.0,4.0,90.0,4,28571
Yeshwanthpur,1 BHK,665.0,1.0,36.85,1,5541
Mysore Road,2 BHK,1175.0,2.0,73.5,2,6255
Hoodi,2 BHK,1425.0,2.0,80.0,2,5614
Chikkabanavar,1 Bedroom,1200.0,1.0,20.0,1,1666
ITPL,3 Bedroom,1200.0,3.0,56.12,3,4676
Somasundara Palya,2 BHK,1255.0,2.0,67.0,2,5338
Hegde Nagar,3 BHK,2112.95,4.0,145.0,3,6862
Raja Rajeshwari Nagar,1 BHK,600.0,1.0,22.0,1,3666
Haralur Road,3 BHK,1810.0,3.0,100.0,3,5524
Indira Nagar,2 BHK,1200.0,2.0,93.0,2,7750
Harlur,2 BHK,1033.0,2.0,49.0,2,4743
ITPL,3 BHK,1548.0,3.0,89.5,3,5781
Horamavu Banaswadi,3 BHK,1611.0,3.0,66.0,3,4096
Electronic City Phase II,3 BHK,1400.0,2.0,40.43,3,2887
Vidyaranyapura,3 BHK,1485.0,2.0,67.0,3,4511
Margondanahalli,2 Bedroom,2400.0,2.0,82.0,2,3416
Pattandur Agrahara,2 BHK,1025.0,2.0,44.5,2,4341
other,4 Bedroom,600.0,3.0,65.0,4,10833
other,2 BHK,1225.0,2.0,53.0,2,4326
5th Phase JP Nagar,2 BHK,1256.0,2.0,62.8,2,5000
Whitefield,4 BHK,2268.0,3.0,163.0,4,7186
Kanakpura Road,3 BHK,1100.0,3.0,53.0,3,4818
other,2 BHK,1141.0,2.0,62.0,2,5433
Attibele,1 BHK,400.0,1.0,12.0,1,3000
Chandapura,2 Bedroom,1400.0,2.0,60.0,2,4285
other,7 BHK,4100.0,7.0,140.0,7,3414
other,1 BHK,15.0,1.0,30.0,1,200000
other,2 BHK,1175.0,2.0,48.27,2,4108
Sahakara Nagar,2 BHK,1100.0,2.0,56.0,2,5090
Kanakpura Road,3 BHK,1843.0,3.0,95.84,3,5200
1st Phase JP Nagar,8 Bedroom,1200.0,7.0,240.0,8,20000
Horamavu Banaswadi,2 BHK,1460.0,2.0,80.5,2,5513
Rachenahalli,3 BHK,1530.0,2.0,74.5,3,4869
HSR Layout,2 BHK,1467.0,2.0,55.0,2,3749
other,1 BHK,950.0,1.0,40.0,1,4210
Vittasandra,2 BHK,1246.0,2.0,64.5,2,5176
Kanakapura,1 BHK,711.0,1.0,38.0,1,5344
Haralur Road,2 BHK,1194.0,2.0,46.0,2,3852
Babusapalaya,3 BHK,1358.0,2.0,43.4,3,3195
Nagasandra,6 Bedroom,1500.0,5.0,130.0,6,8666
Hoodi,2 BHK,1420.0,2.0,75.0,2,5281
Kathriguppe,3 BHK,1250.0,2.0,68.75,3,5500
Akshaya Nagar,2 BHK,1280.0,2.0,58.0,2,4531
other,2 BHK,1209.0,2.0,50.0,2,4135
other,5 Bedroom,4200.0,5.0,225.0,5,5357
other,4 Bedroom,1215.0,4.0,205.0,4,16872
Panathur,3 BHK,1315.0,2.0,54.8,3,4167
other,3 BHK,1800.0,3.0,90.0,3,5000
other,5 Bedroom,1600.0,4.0,125.0,5,7812
TC Palaya,3 BHK,1600.0,2.0,65.0,3,4062
other,6 Bedroom,3968.0,5.0,900.0,6,22681
other,3 BHK,1563.0,3.0,52.0,3,3326
Whitefield,3 BHK,1560.0,2.0,72.0,3,4615
Brookefield,3 BHK,2169.0,4.0,115.0,3,5301
other,9 Bedroom,900.0,10.0,170.0,9,18888
Rachenahalli,2 BHK,1050.0,2.0,55.5,2,5285
Iblur Village,3 BHK,3235.0,3.0,235.0,3,7264
TC Palaya,3 Bedroom,1200.0,3.0,66.0,3,5500
HSR Layout,2 BHK,1140.0,2.0,46.0,2,4035
7th Phase JP Nagar,3 Bedroom,1500.0,3.0,170.0,3,11333
Horamavu Agara,3 BHK,1560.0,3.0,75.0,3,4807
JP Nagar,2 BHK,1100.0,2.0,70.0,2,6363
Kasavanhalli,3 BHK,1260.0,2.0,40.32,3,3200
Hebbal,2 BHK,1036.0,2.0,50.0,2,4826
Akshaya Nagar,3 BHK,1662.0,3.0,85.1,3,5120
Yelachenahalli,3 BHK,1464.0,2.0,105.0,3,7172
Yelahanka,2 BHK,1234.0,2.0,80.0,2,6482
Sarjapur  Road,3 BHK,1403.0,3.0,55.0,3,3920
other,4 Bedroom,915.0,4.0,54.9,4,6000
2nd Stage Nagarbhavi,6 Bedroom,3000.0,8.0,451.0,6,15033
Seegehalli,3 BHK,1150.0,2.0,42.9,3,3730
Hebbal,4 BHK,3900.0,4.0,410.0,4,10512
Yeshwanthpur,5 Bedroom,850.0,4.0,90.0,5,10588
Kanakapura,2 BHK,929.0,2.0,46.0,2,4951
2nd Stage Nagarbhavi,6 Bedroom,2400.0,8.0,450.0,6,18750
TC Palaya,3 Bedroom,1350.0,2.0,75.0,3,5555
Vasanthapura,2 BHK,1135.0,2.0,39.73,2,3500
other,2 BHK,1405.0,2.0,70.0,2,4982
Whitefield,2 Bedroom,1200.0,2.0,45.84,2,3820
Haralur Road,4 BHK,2805.0,4.0,154.0,4,5490
Yeshwanthpur,2 BHK,1161.0,2.0,64.08,2,5519
Kanakpura Road,2 BHK,1290.0,2.0,74.0,2,5736
Yeshwanthpur,3 BHK,2557.0,3.0,141.0,3,5514
other,3 BHK,1480.0,3.0,90.0,3,6081
other,9 Bedroom,3300.0,14.0,500.0,9,15151
Old Madras Road,3 BHK,1350.0,3.0,54.54,3,4040
Kadugodi,3 BHK,1580.0,2.0,65.0,3,4113
Banashankari,3 BHK,1470.0,2.0,88.64,3,6029
Ananth Nagar,1 BHK,500.0,1.0,14.0,1,2800
Sarjapur  Road,2 BHK,1346.0,2.0,61.92,2,4600
other,3 BHK,1600.0,3.0,90.0,3,5625
other,2 BHK,720.0,2.0,120.0,2,16666
other,4 Bedroom,1200.0,4.0,135.0,4,11250
other,2 BHK,1200.0,2.0,35.0,2,2916
Channasandra,3 BHK,1340.0,2.0,48.33,3,3606
Devanahalli,4 Bedroom,6136.0,4.0,560.0,4,9126
Jigani,2 BHK,918.0,2.0,48.0,2,5228
other,3 Bedroom,1500.0,3.0,115.0,3,7666
Yelahanka,2 BHK,1315.0,2.0,85.0,2,6463
other,4 Bedroom,2400.0,5.0,775.0,4,32291
Uttarahalli,3 BHK,1390.0,2.0,62.55,3,4500
Uttarahalli,3 BHK,1330.0,2.0,56.0,3,4210
Jalahalli,5 BHK,3100.0,4.0,265.0,5,8548
other,3 BHK,1824.0,3.0,125.0,3,6853
7th Phase JP Nagar,3 BHK,1420.0,2.0,96.0,3,6760
other,1 Bedroom,750.0,1.0,56.25,1,7500
Jalahalli East,3 BHK,1260.0,2.0,60.0,3,4761
Hennur Road,3 BHK,1561.0,3.0,95.0,3,6085
5th Phase JP Nagar,9 Bedroom,812.0,6.0,165.0,9,20320
other,2 BHK,1060.0,2.0,42.0,2,3962
Hoodi,8 Bedroom,1120.0,8.0,145.0,8,12946
Haralur Road,2 BHK,1225.0,2.0,67.38,2,5500
Thanisandra,3 BHK,2144.0,3.0,135.0,3,6296
Kengeri,7 Bedroom,1200.0,6.0,140.0,7,11666
Electronic City,2 BHK,880.0,2.0,29.0,2,3295
Hoodi,3 BHK,1639.0,3.0,110.0,3,6711
Thanisandra,3 BHK,1240.0,2.0,62.0,3,5000
Kanakpura Road,2 BHK,1500.0,2.0,69.0,2,4600
Haralur Road,2 BHK,1243.0,2.0,46.0,2,3700
Marathahalli,3 BHK,1575.0,3.0,78.0,3,4952
Haralur Road,2 BHK,1027.0,2.0,44.0,2,4284
Kambipura,3 BHK,1082.0,2.0,56.0,3,5175
Konanakunte,3 BHK,1637.0,3.0,73.0,3,4459
Kanakpura Road,3 BHK,1665.0,3.0,86.58,3,5200
Electronic City,2 BHK,1265.0,2.0,56.93,2,4500
Electronic City,3 BHK,1691.0,2.0,102.0,3,6031
Thanisandra,2 BHK,1100.0,2.0,38.0,2,3454
other,5 Bedroom,24.0,2.0,150.0,5,625000
other,3 Bedroom,1500.0,3.0,62.0,3,4133
Hosakerehalli,4 BHK,3205.0,5.0,380.0,4,11856
Kammanahalli,5 Bedroom,1666.0,4.0,150.0,5,9003
Choodasandra,2 BHK,1115.0,2.0,50.0,2,4484
Mysore Road,2 BHK,1070.0,2.0,45.0,2,4205
Indira Nagar,2 BHK,1128.0,2.0,105.0,2,9308
Budigere,2 BHK,1162.0,2.0,57.0,2,4905
other,2 BHK,1339.0,2.0,55.0,2,4107
Mahadevpura,2 BHK,1192.0,2.0,75.0,2,6291
Haralur Road,3 BHK,1920.0,3.0,115.0,3,5989
Hennur,2 BHK,1255.0,2.0,52.32,2,4168
Rajaji Nagar,2 BHK,1357.0,2.0,115.0,2,8474
other,3 BHK,1600.0,3.0,75.0,3,4687
other,2 BHK,1225.0,2.0,55.11,2,4498
Whitefield,4 Bedroom,2400.0,4.0,200.0,4,8333
Kasturi Nagar,3 BHK,1570.0,3.0,80.0,3,5095
Thanisandra,2 BHK,1093.0,2.0,70.0,2,6404
Basaveshwara Nagar,3 BHK,1865.0,3.0,145.0,3,7774
R.T. Nagar,6 Bedroom,1200.0,7.0,165.0,6,13750
Kaikondrahalli,2 BHK,1039.0,2.0,80.0,2,7699
Vittasandra,2 BHK,1246.0,2.0,64.5,2,5176
Whitefield,3 BHK,1985.0,3.0,75.0,3,3778
Hosakerehalli,3 BHK,1590.0,3.0,49.0,3,3081
other,4 Bedroom,4000.0,4.0,675.0,4,16875
Electronic City Phase II,2 BHK,775.0,2.0,38.0,2,4903
other,2 BHK,1000.0,2.0,50.0,2,5000
Hebbal Kempapura,3 BHK,1800.0,3.0,175.0,3,9722
Bommasandra,2 BHK,1035.0,2.0,41.0,2,3961
Dasarahalli,3 BHK,1520.0,2.0,75.0,3,4934
Whitefield,2 BHK,1360.0,2.0,95.0,2,6985
other,4 Bedroom,6000.0,5.0,400.0,4,6666
Kodigehaali,2 BHK,1117.0,2.0,56.0,2,5013
KR Puram,4 Bedroom,1200.0,4.0,120.0,4,10000
EPIP Zone,2 BHK,1330.0,2.0,92.5,2,6954
Marathahalli,3 BHK,1937.0,3.0,160.0,3,8260
Electronic City,2 BHK,1128.0,2.0,65.49,2,5805
Neeladri Nagar,2 BHK,1053.0,2.0,45.0,2,4273
Panathur,2 BHK,1198.0,2.0,77.0,2,6427
Marathahalli,2 BHK,1152.0,2.0,56.0,2,4861
1st Block Jayanagar,6 BHK,1200.0,6.0,125.0,6,10416
Frazer Town,2 BHK,1625.0,2.0,75.0,2,4615
Kundalahalli,3 BHK,1397.0,3.0,105.0,3,7516
Yeshwanthpur,3 BHK,1523.0,3.0,170.0,3,11162
Kambipura,2 BHK,883.0,2.0,45.0,2,5096
Kaggalipura,1 BHK,700.0,1.0,38.0,1,5428
Yelahanka,2 BHK,1170.0,2.0,62.5,2,5341
Kanakpura Road,3 BHK,1200.0,2.0,42.0,3,3500
other,3 BHK,2400.0,3.0,170.0,3,7083
other,2 BHK,1480.0,2.0,65.0,2,4391
Akshaya Nagar,3 BHK,1897.0,3.0,120.0,3,6325
Rachenahalli,2 BHK,996.0,2.0,67.0,2,6726
Haralur Road,3 BHK,1875.0,3.0,110.0,3,5866
other,3 BHK,1500.0,2.0,98.0,3,6533
other,3 BHK,3095.0,3.0,350.0,3,11308
Ambalipura,3 BHK,1390.0,2.0,175.0,3,12589
Yelahanka,3 BHK,1450.0,3.0,65.255,3,4500
Yelahanka,1 BHK,697.0,1.0,45.0,1,6456
Sarjapur  Road,1 BHK,710.0,1.0,27.0,1,3802
other,2 BHK,910.0,2.0,30.5,2,3351
Sompura,3 BHK,1025.0,2.0,48.0,3,4682
Magadi Road,2 BHK,1000.0,2.0,46.5,2,4650
Harlur,4 Bedroom,1200.0,4.0,244.0,4,20333
Horamavu Agara,3 BHK,1453.0,2.0,46.48,3,3198
Sarjapur  Road,4 BHK,3335.0,5.0,300.0,4,8995
Thanisandra,2 BHK,971.5,2.0,36.435,2,3750
Bellandur,2 BHK,1281.0,2.0,74.0,2,5776
Cooke Town,1 BHK,565.0,1.0,35.0,1,6194
Varthur,2 BHK,1024.0,2.0,42.0,2,4101
Electronic City,3 BHK,1620.0,3.0,62.0,3,3827
Whitefield,2 BHK,1254.0,2.0,40.0,2,3189
Banashankari,3 BHK,1300.0,2.0,75.0,3,5769
Doddakallasandra,2 BHK,1233.0,2.0,43.0,2,3487
other,1 Bedroom,1200.0,1.0,28.0,1,2333
Hosur Road,3 BHK,1510.0,2.0,105.0,3,6953
Kaggalipura,2 BHK,950.0,2.0,65.0,2,6842
Battarahalli,3 BHK,1945.0,3.0,97.42,3,5008
Uttarahalli,2 BHK,1008.0,2.0,45.0,2,4464
Thanisandra,3 BHK,1265.0,2.0,79.0,3,6245
Hebbal,3 BHK,1255.0,2.0,77.68,3,6189
Kanakpura Road,4 BHK,2689.0,6.0,220.0,4,8181
Yelenahalli,2 BHK,1200.0,2.0,46.17,2,3847
Sarjapur  Road,3 BHK,1403.0,3.0,51.63,3,3679
Kudlu Gate,3 BHK,1535.0,3.0,85.0,3,5537
other,3 Bedroom,1500.0,3.0,100.0,3,6666
Chamrajpet,3 BHK,1475.0,2.0,120.0,3,8135
Uttarahalli,3 BHK,1390.0,2.0,60.47,3,4350
Raja Rajeshwari Nagar,3 BHK,1278.0,2.0,53.76,3,4206
other,2 BHK,1133.0,2.0,55.0,2,4854
other,5 Bedroom,1560.0,5.0,145.0,5,9294
other,5 Bedroom,2900.0,5.0,90.0,5,3103
Kogilu,2 BHK,1170.0,2.0,51.99,2,4443
Sahakara Nagar,3 Bedroom,1200.0,3.0,200.0,3,16666
Whitefield,3 BHK,1697.0,3.0,94.79,3,5585
Thanisandra,3 BHK,1651.0,3.0,97.38,3,5898
other,3 BHK,1560.0,3.0,188.0,3,12051
Hennur Road,3 BHK,1192.0,2.0,60.0,3,5033
OMBR Layout,2 BHK,1078.0,2.0,65.0,2,6029
Kumaraswami Layout,7 Bedroom,800.0,7.0,105.0,7,13125
Kasavanhalli,2 BHK,1349.0,2.0,78.0,2,5782
Ulsoor,9 Bedroom,2300.0,8.0,210.0,9,9130
7th Phase JP Nagar,3 BHK,1680.0,3.0,120.0,3,7142
Rayasandra,3 BHK,1314.0,2.0,63.0,3,4794
Garudachar Palya,2 BHK,1060.0,2.0,48.7,2,4594
Kalyan nagar,2 BHK,820.0,2.0,39.0,2,4756
Whitefield,4 Bedroom,5000.0,4.0,400.0,4,8000
BEML Layout,3 BHK,2000.0,3.0,85.0,3,4250
other,2 BHK,856.0,2.0,40.0,2,4672
Rajaji Nagar,3 BHK,1621.0,4.0,130.0,3,8019
ITPL,3 Bedroom,1200.0,3.0,68.4,3,5700
Bannerghatta Road,3 BHK,1484.0,3.0,52.5,3,3537
other,4 Bedroom,600.0,5.0,60.0,4,10000
other,4 Bedroom,14000.0,3.0,800.0,4,5714
Yelahanka New Town,1 BHK,500.0,1.0,20.0,1,4000
other,1 BHK,485.0,1.0,15.0,1,3092
7th Phase JP Nagar,2 BHK,1050.0,2.0,42.0,2,4000
Uttarahalli,3 BHK,1617.0,3.0,66.68,3,4123
Electronic City,2 BHK,550.0,1.0,16.0,2,2909
other,3 BHK,1384.0,3.0,62.28,3,4500
Devanahalli,2 BHK,1408.0,2.0,85.0,2,6036
Uttarahalli,2 BHK,1150.0,2.0,49.0,2,4260
Akshaya Nagar,3 BHK,1410.0,2.0,75.0,3,5319
Kaggadasapura,4 BHK,2150.0,4.0,100.0,4,4651
Koramangala,3 BHK,2292.0,4.0,275.0,3,11998
Malleshwaram,3 BHK,2006.0,3.0,297.0,3,14805
other,3 BHK,1508.0,3.0,70.25,3,4658
Ramamurthy Nagar,2 BHK,1040.0,2.0,50.0,2,4807
TC Palaya,2 Bedroom,1240.0,2.0,60.0,2,4838
Kudlu Gate,3 BHK,1535.0,3.0,78.0,3,5081
Lakshminarayana Pura,2 BHK,1180.0,2.0,75.0,2,6355
Abbigere,6 Bedroom,2200.0,6.0,68.0,6,3090
Bannerghatta Road,2 BHK,1400.0,2.0,78.0,2,5571
Raja Rajeshwari Nagar,1 BHK,500.0,1.0,25.0,1,5000
Whitefield,4 Bedroom,4800.0,4.0,525.0,4,10937
Sarjapur  Road,3 BHK,1984.0,4.0,146.0,3,7358
Whitefield,4 Bedroom,3000.0,5.0,250.0,4,8333
Garudachar Palya,3 BHK,1960.0,3.0,160.0,3,8163
Sarjapur  Road,2 BHK,1035.0,2.0,45.0,2,4347
other,4 Bedroom,2150.0,4.0,80.0,4,3720
Whitefield,3 BHK,1700.0,3.0,115.0,3,6764
Hosakerehalli,4 BHK,3024.0,5.0,248.0,4,8201
Tumkur Road,3 BHK,1586.0,3.0,100.0,3,6305
Chikka Tirupathi,4 Bedroom,2325.0,4.0,120.0,4,5161
Yelahanka,2 BHK,1120.0,2.0,55.0,2,4910
Sarjapur  Road,2 BHK,1128.0,2.0,59.0,2,5230
5th Block Hbr Layout,6 Bedroom,1200.0,6.0,250.0,6,20833
Jigani,2 BHK,924.0,2.0,65.0,2,7034
other,3 BHK,1600.0,3.0,52.0,3,3250
Pai Layout,2 BHK,1006.0,2.0,35.0,2,3479
Amruthahalli,2 BHK,1200.0,2.0,55.0,2,4583
other,2 BHK,1250.0,2.0,62.0,2,4960
other,3 Bedroom,1200.0,3.0,130.0,3,10833
Kannamangala,2 BHK,957.0,2.0,56.0,2,5851
Whitefield,2 BHK,1140.0,2.0,41.0,2,3596
other,2 BHK,1050.0,2.0,67.0,2,6380
Kasturi Nagar,3 BHK,1565.0,3.0,88.5,3,5654
Whitefield,2 BHK,1190.0,2.0,57.0,2,4789
Whitefield,3 Bedroom,2842.0,3.0,162.0,3,5700
Kanakpura Road,3 BHK,1558.67,2.0,65.0,3,4170
Electronics City Phase 1,1 BHK,630.0,1.0,39.0,1,6190
Whitefield,3 BHK,1403.0,2.0,60.81,3,4334
Kanakpura Road,3 BHK,1542.0,2.0,85.0,3,5512
Haralur Road,4 Bedroom,2750.0,4.0,220.0,4,8000
Iblur Village,4 BHK,3596.0,5.0,290.0,4,8064
Kengeri Satellite Town,3 BHK,1635.0,2.0,78.0,3,4770
Mysore Road,2 BHK,1239.0,2.0,53.0,2,4277
Haralur Road,2 BHK,1194.0,2.0,47.0,2,3936
other,3 BHK,1596.0,3.0,105.0,3,6578
Channasandra,2 BHK,1009.0,2.0,38.34,2,3799
Hosa Road,3 BHK,1726.0,3.0,81.99,3,4750
Kudlu Gate,3 BHK,1535.0,3.0,86.0,3,5602
Banashankari Stage V,3 BHK,1300.0,2.0,58.5,3,4500
Whitefield,4 Bedroom,3100.0,4.0,465.0,4,15000
other,3 BHK,1800.0,3.0,80.0,3,4444
Binny Pete,1 BHK,665.0,1.0,50.75,1,7631
Chamrajpet,9 Bedroom,4050.0,7.0,1200.0,9,29629
TC Palaya,3 Bedroom,1500.0,2.0,100.0,3,6666
other,2 BHK,925.0,2.0,48.0,2,5189
Old Airport Road,4 BHK,3356.0,4.0,251.0,4,7479
other,2 BHK,1395.0,2.0,63.59,2,4558
Sonnenahalli,3 BHK,1415.0,2.0,55.0,3,3886
Begur,3 BHK,1419.0,2.0,59.0,3,4157
Kasavanhalli,2 BHK,1121.0,2.0,78.0,2,6958
Kambipura,3 BHK,1082.0,2.0,45.0,3,4158
KR Puram,2 BHK,1100.0,2.0,47.0,2,4272
Hoodi,3 BHK,1606.0,3.0,89.55,3,5575
Billekahalli,3 BHK,1650.0,3.0,88.0,3,5333
other,3 BHK,4634.0,4.0,1015.0,3,21903
Old Airport Road,2 BHK,1206.0,2.0,75.0,2,6218
Sarjapur  Road,3 BHK,1495.0,2.0,64.0,3,4280
Malleshwaram,8 Bedroom,840.0,7.0,195.0,8,23214
Electronic City,2 BHK,710.0,2.0,16.0,2,2253
Balagere,2 BHK,1210.0,2.0,72.0,2,5950
Nagarbhavi,3 BHK,1523.0,2.0,53.4,3,3506
Whitefield,2 BHK,1232.0,2.0,45.0,2,3652
Rachenahalli,1 BHK,680.0,1.0,44.0,1,6470
Hebbal,3 BHK,1355.0,3.0,83.87,3,6189
Thanisandra,3 BHK,1430.0,2.0,51.46,3,3598
Kothannur,3 BHK,1275.0,2.0,43.85,3,3439
Electronic City,2 BHK,1070.0,2.0,56.1,2,5242
other,3 Bedroom,1200.0,3.0,85.0,3,7083
Electronic City,4 Bedroom,2000.0,4.0,120.0,4,6000
Bommanahalli,2 BHK,1200.0,2.0,40.0,2,3333
Marathahalli,3 BHK,1610.0,3.0,90.0,3,5590
Haralur Road,2 BHK,1140.0,2.0,43.0,2,3771
Raja Rajeshwari Nagar,2 BHK,1143.0,2.0,38.65,2,3381
Jakkur,4 BHK,3467.86,6.0,249.0,4,7180
Rachenahalli,3 BHK,1530.0,2.0,74.4,3,4862
7th Phase JP Nagar,2 BHK,980.0,2.0,69.0,2,7040
Whitefield,3 BHK,1485.0,3.0,65.0,3,4377
Whitefield,3 Bedroom,1500.0,3.0,130.0,3,8666
Whitefield,2 BHK,1280.0,2.0,75.0,2,5859
Thanisandra,2 BHK,1265.0,2.0,85.0,2,6719
other,2 BHK,1132.0,2.0,67.0,2,5918
other,2 BHK,1262.0,2.0,63.0,2,4992
JP Nagar,2 BHK,940.0,2.0,49.0,2,5212
other,3 BHK,1515.0,3.0,88.0,3,5808
Jigani,3 BHK,1300.0,3.0,150.0,3,11538
Kothanur,3 Bedroom,1840.0,3.0,69.0,3,3750
Electronic City,2 BHK,890.0,2.0,45.0,2,5056
other,3 BHK,1655.0,3.0,86.06,3,5200
Hennur Road,3 BHK,2060.0,3.0,149.0,3,7233
Bommanahalli,3 BHK,1730.0,3.0,96.0,3,5549
Yelahanka,3 BHK,2195.0,4.0,127.0,3,5785
Kathriguppe,3 BHK,1650.0,3.0,125.0,3,7575
KR Puram,2 Bedroom,1200.0,2.0,71.0,2,5916
Hosa Road,2 BHK,1079.0,2.0,32.37,2,2999
Panathur,2 BHK,1085.0,2.0,36.0,2,3317
other,7 Bedroom,5000.0,7.0,299.0,7,5980
Yelahanka,2 BHK,1075.0,2.0,51.5,2,4790
other,2 BHK,1380.0,2.0,80.0,2,5797
other,3 BHK,1450.0,3.0,65.0,3,4482
Thubarahalli,2 BHK,1200.0,2.0,78.0,2,6500
Benson Town,3 BHK,3200.0,4.0,350.0,3,10937
Hoodi,3 BHK,1639.0,3.0,98.32,3,5998
other,2 BHK,1200.0,2.0,54.0,2,4500
Vijayanagar,3 BHK,1739.0,3.0,116.0,3,6670
other,4 Bedroom,4346.0,5.0,500.0,4,11504
Old Madras Road,2 BHK,935.0,2.0,32.72,2,3499
Thanisandra,1 RK,510.0,1.0,25.25,1,4950
other,6 Bedroom,1200.0,6.0,180.0,6,15000
other,2 BHK,1178.0,2.0,47.0,2,3989
Chikkalasandra,3 BHK,1375.0,2.0,50.0,3,3636
Chandapura,3 BHK,1065.0,2.0,33.0,3,3098
2nd Phase Judicial Layout,3 BHK,1450.0,2.0,50.75,3,3500
BEML Layout,2 BHK,999.0,2.0,45.0,2,4504
Amruthahalli,2 BHK,4400.0,3.0,475.0,2,10795
Marathahalli,5 Bedroom,1350.0,4.0,135.0,5,10000
Kothannur,2 BHK,1100.0,2.0,40.0,2,3636
Koramangala,3 BHK,1835.0,2.0,155.0,3,8446
Hoodi,4 BHK,2090.0,4.0,95.1,4,4550
Lingadheeranahalli,3 BHK,1519.0,3.0,93.0,3,6122
Dommasandra,3 BHK,1267.0,3.0,56.0,3,4419
Sanjay nagar,2 BHK,1150.0,2.0,72.2,2,6278
Marathahalli,2 BHK,1146.0,2.0,70.0,2,6108
other,8 Bedroom,1200.0,7.0,235.0,8,19583
Kannamangala,3 BHK,1550.0,3.0,65.0,3,4193
Chandapura,2 BHK,850.0,1.0,18.5,2,2176
other,3 BHK,2200.0,3.0,187.0,3,8500
other,3 Bedroom,3329.0,3.0,330.0,3,9912
Electronics City Phase 1,1 BHK,785.0,1.0,55.0,1,7006
Hebbal Kempapura,3 BHK,2900.0,4.0,300.0,3,10344
Raja Rajeshwari Nagar,3 BHK,1610.0,3.0,74.0,3,4596
Bellandur,2 BHK,921.0,2.0,40.0,2,4343
Kengeri Satellite Town,3 Bedroom,850.0,3.0,95.0,3,11176
Bannerghatta Road,5 Bedroom,2700.0,5.0,158.0,5,5851
Sarjapur  Road,3 BHK,1700.0,3.0,110.0,3,6470
Sarjapur,2 BHK,1020.0,2.0,35.0,2,3431
Billekahalli,2 BHK,1090.0,2.0,52.0,2,4770
other,4 Bedroom,3600.0,4.0,300.0,4,8333
Haralur Road,2 BHK,1027.0,2.0,44.0,2,4284
Thanisandra,2 BHK,1185.0,2.0,42.1,2,3552
Doddathoguru,2 BHK,850.0,2.0,25.0,2,2941
Harlur,2 BHK,1197.0,2.0,84.0,2,7017
other,4 BHK,3500.0,3.0,450.0,4,12857
Poorna Pragna Layout,2 BHK,920.0,2.0,39.55,2,4298
Raja Rajeshwari Nagar,2 BHK,1105.0,2.0,65.0,2,5882
Yeshwanthpur,3 BHK,1850.0,3.0,100.0,3,5405
Judicial Layout,3 BHK,1300.0,3.0,64.0,3,4923
other,3 BHK,1595.0,3.0,76.0,3,4764
Kodihalli,4 BHK,3073.0,5.0,696.0,4,22648
Horamavu Agara,2 BHK,755.0,2.0,37.49,2,4965
Nagasandra,5 Bedroom,4050.0,5.0,140.0,5,3456
Kanakpura Road,3 BHK,1300.0,2.0,69.0,3,5307
Bannerghatta Road,2 BHK,1215.0,2.0,50.0,2,4115
Kanakpura Road,2 BHK,900.0,2.0,46.0,2,5111
Banashankari Stage III,2 BHK,1145.0,2.0,58.4,2,5100
other,2 BHK,1093.0,2.0,65.0,2,5946
Babusapalaya,3 BHK,1213.0,2.0,41.39,3,3412
Bannerghatta Road,3 BHK,1654.0,3.0,100.0,3,6045
NGR Layout,2 BHK,1105.0,2.0,52.48,2,4749
Babusapalaya,3 BHK,1611.0,2.0,66.0,3,4096
other,2 BHK,1367.0,2.0,62.0,2,4535
Devanahalli,2 BHK,1010.0,2.0,51.0,2,5049
other,3 Bedroom,750.0,2.0,75.0,3,10000
other,3 BHK,1535.0,3.0,95.0,3,6188
Anekal,2 BHK,620.0,2.0,22.0,2,3548
Horamavu Agara,2 BHK,982.0,2.0,47.65,2,4852
Banashankari Stage II,2 BHK,1700.0,2.0,150.0,2,8823
Whitefield,2 BHK,1330.0,2.0,71.0,2,5338
JP Nagar,3 Bedroom,2500.0,3.0,135.0,3,5400
Poorna Pragna Layout,3 BHK,1475.0,2.0,58.99,3,3999
Bhoganhalli,3 BHK,1410.0,3.0,65.0,3,4609
Yelahanka New Town,1 BHK,440.0,1.0,16.5,1,3750
Yelahanka,3 BHK,1614.0,3.0,95.0,3,5885
Neeladri Nagar,2 BHK,1363.0,2.0,68.0,2,4988
Kogilu,2 Bedroom,650.0,2.0,28.0,2,4307
other,2 BHK,1015.0,2.0,45.0,2,4433
Horamavu Banaswadi,2 BHK,1156.0,2.0,46.0,2,3979
other,2 BHK,1100.0,2.0,62.7,2,5700
KR Puram,3 BHK,1455.0,3.0,46.0,3,3161
other,2 BHK,930.0,2.0,45.0,2,4838
other,3 BHK,1650.0,2.0,80.0,3,4848
KR Puram,2 Bedroom,1200.0,2.0,90.0,2,7500
other,1 Bedroom,700.0,1.0,150.0,1,21428
Karuna Nagar,4 BHK,2900.0,3.0,135.0,4,4655
Kadugodi,3 BHK,1890.0,4.0,127.0,3,6719
other,3 BHK,2197.0,4.0,280.0,3,12744
Hebbal Kempapura,3 BHK,3522.0,3.0,380.0,3,10789
other,3 BHK,1240.0,2.0,46.0,3,3709
HRBR Layout,2 BHK,1440.0,2.0,98.0,2,6805
Kodichikkanahalli,1 BHK,650.0,1.0,38.0,1,5846
Budigere,1 BHK,764.0,1.0,32.0,1,4188
Sector 7 HSR Layout,5 Bedroom,900.0,4.0,90.0,5,10000
Raja Rajeshwari Nagar,3 BHK,1693.0,3.0,57.39,3,3389
Hoodi,3 BHK,1445.0,3.0,65.0,3,4498
Kanakpura Road,2 BHK,900.0,1.0,45.0,2,5000
Singasandra,2 BHK,1030.0,2.0,55.0,2,5339
Yelahanka,3 BHK,1381.0,2.0,52.48,3,3800
other,2 BHK,1100.0,2.0,60.0,2,5454
Hoodi,3 BHK,1575.0,3.0,66.39,3,4215
Thigalarapalya,2 Bedroom,1200.0,2.0,65.0,2,5416
Begur Road,2 BHK,1200.0,2.0,43.2,2,3600
Kasturi Nagar,2 BHK,1080.0,2.0,80.0,2,7407
Hebbal Kempapura,5 Bedroom,2280.0,5.0,200.0,5,8771
Akshaya Nagar,3 BHK,1360.0,2.0,85.0,3,6250
Jakkur,3 BHK,1865.0,3.0,124.0,3,6648
Sarjapur  Road,3 BHK,1157.0,2.0,75.0,3,6482
Kammasandra,2 BHK,674.0,2.0,29.0,2,4302
Whitefield,3 BHK,1405.0,2.0,78.0,3,5551
Begur Road,2 BHK,1160.0,2.0,44.08,2,3800
other,4 Bedroom,1600.0,3.0,172.0,4,10750
Hebbal,3 BHK,1790.0,3.0,127.0,3,7094
Electronic City,2 BHK,1128.0,2.0,65.65,2,5820
Panathur,2 BHK,1180.0,2.0,70.0,2,5932
other,3 BHK,1450.0,2.0,85.0,3,5862
Kengeri,1 BHK,340.0,1.0,10.0,1,2941
other,4 Bedroom,2000.0,3.0,166.0,4,8300
Rachenahalli,2 BHK,1050.0,2.0,53.5,2,5095
Varthur,2 BHK,1125.0,2.0,40.0,2,3555
Vittasandra,2 BHK,1246.0,2.0,68.0,2,5457
HSR Layout,2 BHK,1372.0,2.0,61.0,2,4446
Uttarahalli,3 BHK,1330.0,2.0,56.0,3,4210
Binny Pete,3 BHK,2465.0,5.0,234.0,3,9492
Dasarahalli,2 BHK,1300.0,2.0,55.0,2,4230
other,2 Bedroom,1420.0,1.0,220.0,2,15492
Bannerghatta,4 Bedroom,810.0,4.0,50.0,4,6172
Channasandra,2 BHK,1065.0,2.0,35.0,2,3286
other,2 BHK,700.0,2.0,27.0,2,3857
Hebbal,2 BHK,687.325,2.0,42.72,2,6215
other,2 BHK,1050.0,2.0,45.0,2,4285
other,3 BHK,1250.0,3.0,39.5,3,3160
Electronic City,2 BHK,865.0,2.0,42.0,2,4855
Uttarahalli,2 BHK,1200.0,2.0,44.0,2,3666
other,3 BHK,1756.0,3.0,100.0,3,5694
Malleshwaram,9 Bedroom,750.0,8.0,150.0,9,20000
Yelahanka,1 BHK,628.0,1.0,24.0,1,3821
9th Phase JP Nagar,7 Bedroom,1200.0,6.0,195.0,7,16250
5th Phase JP Nagar,2 BHK,1207.0,2.0,63.0,2,5219
Kalena Agrahara,3 BHK,1765.0,3.0,125.0,3,7082
Electronic City,2 BHK,1130.0,2.0,32.64,2,2888
other,3 BHK,1532.0,3.0,92.0,3,6005
other,3 BHK,1650.0,3.0,110.0,3,6666
Horamavu Banaswadi,4 Bedroom,1200.0,4.0,105.0,4,8750
Bhoganhalli,2 BHK,804.1,2.0,69.09,2,8592
Doddathoguru,2 BHK,1200.0,2.0,75.0,2,6250
Kengeri,2 BHK,1197.0,2.0,50.0,2,4177
other,2 BHK,1200.0,2.0,66.0,2,5500
Uttarahalli,3 BHK,1385.0,2.0,48.48,3,3500
other,2 BHK,1010.0,2.0,148.0,2,14653
Whitefield,3 BHK,2225.0,3.0,115.0,3,5168
other,2 BHK,1180.0,2.0,48.0,2,4067
other,3 BHK,1525.0,2.0,122.0,3,8000
Ananth Nagar,2 BHK,1100.0,2.0,31.5,2,2863
other,3 BHK,1435.0,3.0,70.0,3,4878
Attibele,1 BHK,400.0,1.0,14.0,1,3500
Dommasandra,2 BHK,1200.0,2.0,50.0,2,4166
Whitefield,2 BHK,1272.0,2.0,95.05,2,7472
Whitefield,2 BHK,1205.0,2.0,39.0,2,3236
Kalena Agrahara,2 BHK,980.0,2.0,40.0,2,4081
Malleshpalya,2 BHK,1225.0,2.0,46.86,2,3825
Whitefield,3 BHK,1541.0,3.0,72.0,3,4672
Ramagondanahalli,3 BHK,1610.0,2.0,111.0,3,6894
Jalahalli,2 BHK,1694.0,2.0,125.0,2,7378
other,3 Bedroom,600.0,3.0,75.0,3,12500
Kaggalipura,2 BHK,950.0,2.0,39.0,2,4105
other,3 BHK,1560.0,2.0,66.0,3,4230
Chandapura,2 BHK,1201.0,2.0,27.0,2,2248
Chandapura,2 Bedroom,1500.0,2.0,60.0,2,4000
Malleshpalya,3 BHK,1785.0,3.0,75.0,3,4201
Raja Rajeshwari Nagar,2 BHK,1258.0,2.0,63.53,2,5050
Electronic City,2 BHK,910.0,2.0,42.0,2,4615
Thigalarapalya,2 BHK,1297.0,2.0,112.0,2,8635
Kundalahalli,2 BHK,1291.0,2.0,49.0,2,3795
other,2 BHK,1070.0,2.0,42.0,2,3925
Varthur,2 BHK,1012.0,2.0,66.5,2,6571
Mahadevpura,2 BHK,1260.0,2.0,57.6,2,4571
Sarjapur  Road,2 BHK,1282.0,2.0,72.0,2,5616
Kanakpura Road,3 BHK,1843.0,3.0,92.15,3,5000
other,4 Bedroom,2400.0,3.0,200.0,4,8333
Chikka Tirupathi,5 Bedroom,3356.0,5.0,105.0,5,3128
Hebbal,4 BHK,2790.0,5.0,198.0,4,7096
EPIP Zone,2 BHK,1330.0,2.0,86.98,2,6539
Konanakunte,3 BHK,1400.0,2.0,73.0,3,5214
Whitefield,2 BHK,1192.0,2.0,71.0,2,5956
other,2 BHK,1000.0,2.0,130.0,2,13000
Thigalarapalya,2 BHK,1297.0,2.0,103.0,2,7941
Sarjapur  Road,1 BHK,615.0,1.0,17.835,1,2900
Vijayanagar,3 BHK,1490.0,3.0,100.0,3,6711
Rayasandra,3 BHK,1373.0,3.0,72.0,3,5243
7th Phase JP Nagar,2 BHK,1035.0,2.0,41.39,2,3999
Kadugodi,4 Bedroom,3750.0,4.0,210.0,4,5600
Doddakallasandra,2 BHK,1010.0,2.0,40.39,2,3999
Sarjapur,2 BHK,1195.0,2.0,42.0,2,3514
HSR Layout,3 BHK,1750.0,3.0,95.0,3,5428
Malleshpalya,2 BHK,1052.0,2.0,50.0,2,4752
Horamavu Banaswadi,2 BHK,1180.0,2.0,58.0,2,4915
Padmanabhanagar,5 Bedroom,3680.0,4.0,440.0,5,11956
Hormavu,3 BHK,2100.0,3.0,73.0,3,3476
Hoodi,4 Bedroom,2863.0,4.0,140.0,4,4889
Raja Rajeshwari Nagar,2 BHK,1128.0,2.0,38.24,2,3390
Ulsoor,3 BHK,2700.0,3.0,450.0,3,16666
JP Nagar,2 BHK,1120.0,2.0,33.6,2,3000
HSR Layout,3 BHK,2424.0,3.0,90.0,3,3712
other,2 BHK,1300.0,2.0,60.0,2,4615
Kanakapura,2 BHK,1090.0,2.0,32.69,2,2999
Yelahanka,3 Bedroom,2710.0,3.0,251.0,3,9261
other,2 Bedroom,3040.0,1.0,180.0,2,5921
Uttarahalli,3 BHK,1330.0,2.0,46.55,3,3500
Hosur Road,3 BHK,1427.0,2.0,80.63,3,5650
Banjara Layout,3 BHK,1294.0,2.0,48.0,3,3709
Begur,3 BHK,1580.0,3.0,75.0,3,4746
Uttarahalli,5 Bedroom,400.0,5.0,200.0,5,50000
Chikka Tirupathi,3 Bedroom,1808.0,4.0,80.27,3,4439
Rajaji Nagar,4 BHK,3516.0,4.0,540.0,4,15358
Parappana Agrahara,2 BHK,1194.0,2.0,47.0,2,3936
Kanakpura Road,2 BHK,900.0,2.0,46.0,2,5111
other,3 BHK,1310.0,2.0,50.0,3,3816
Kanakpura Road,2 BHK,900.0,2.0,46.0,2,5111
Singasandra,4 Bedroom,3850.0,6.0,195.0,4,5064
Kogilu,7 Bedroom,925.0,7.0,88.0,7,9513
Yelachenahalli,2 BHK,2400.0,1.0,150.0,2,6250
Koramangala,2 BHK,1290.0,2.0,100.0,2,7751
Yelahanka,6 Bedroom,3770.0,6.0,110.0,6,2917
other,3 BHK,1390.0,3.0,120.0,3,8633
other,4 Bedroom,1200.0,4.0,72.0,4,6000
Bisuvanahalli,2 BHK,845.0,2.0,33.0,2,3905
Yelahanka,3 BHK,1537.0,3.0,120.0,3,7807
other,2 BHK,1135.0,2.0,68.0,2,5991
Hoodi,3 BHK,1512.0,3.0,75.77,3,5011
other,2 BHK,900.0,2.0,55.0,2,6111
Jigani,3 BHK,1230.0,3.0,54.0,3,4390
other,3 BHK,1378.0,2.0,42.33,3,3071
Whitefield,2 BHK,1277.0,2.0,61.3,2,4800
Hebbal,3 BHK,1985.0,4.0,189.0,3,9521
Kengeri,5 Bedroom,8000.0,5.0,500.0,5,6250
other,2 BHK,1210.0,2.0,62.0,2,5123
Yelahanka,2 BHK,890.0,2.0,40.0,2,4494
JP Nagar,9 Bedroom,2550.0,9.0,360.0,9,14117
Sarjapur,4 Bedroom,1200.0,5.0,110.0,4,9166
Ardendale,4 BHK,3198.0,4.0,200.0,4,6253
other,3 BHK,1464.0,3.0,56.0,3,3825
Hebbal,4 BHK,2483.0,5.0,215.0,4,8658
Begur,4 Bedroom,1550.0,4.0,130.0,4,8387
Sarjapur  Road,4 BHK,4111.0,4.0,310.0,4,7540
Electronic City,2 BHK,975.0,2.0,37.5,2,3846
other,3 BHK,2500.0,3.0,250.0,3,10000
Rajiv Nagar,3 BHK,1690.0,3.0,125.0,3,7396
Hosa Road,2 BHK,1063.0,2.0,31.89,2,3000
Cooke Town,3 Bedroom,2572.0,3.0,390.0,3,15163
other,3 BHK,2200.0,4.0,350.0,3,15909
other,3 BHK,1600.0,3.0,80.0,3,5000
other,3 BHK,1250.0,3.0,39.5,3,3160
Bommanahalli,1 BHK,520.0,1.0,19.5,1,3750
Marathahalli,4 BHK,4000.0,5.0,212.0,4,5300
Munnekollal,3 BHK,1390.0,3.0,68.0,3,4892
Whitefield,2 BHK,1185.0,2.0,40.0,2,3375
Sahakara Nagar,2 BHK,1219.0,2.0,48.0,2,3937
Kudlu Gate,3 BHK,1564.0,2.0,92.0,3,5882
Sarjapur  Road,4 BHK,3430.0,6.0,228.5,4,6661
Hosa Road,2 BHK,1100.0,2.0,37.1,2,3372
Kothannur,3 BHK,1300.0,2.0,60.0,3,4615
Chikkabanavar,3 Bedroom,1600.0,2.0,80.0,3,5000
Ramamurthy Nagar,3 Bedroom,1600.0,3.0,75.0,3,4687
Hennur Road,3 BHK,1482.0,2.0,92.0,3,6207
Marsur,3 Bedroom,1000.0,4.0,66.0,3,6600
Shampura,2 BHK,1200.0,2.0,53.0,2,4416
Gubbalala,3 Bedroom,600.0,3.0,75.0,3,12500
Kengeri,2 BHK,1200.0,2.0,58.0,2,4833
Kereguddadahalli,6 Bedroom,600.0,4.0,60.0,6,10000
Nagarbhavi,3 Bedroom,2400.0,3.0,160.0,3,6666
Marathahalli,3 BHK,1469.0,3.0,89.0,3,6058
Kanakapura,1 BHK,551.0,1.0,30.0,1,5444
Electronics City Phase 1,2 BHK,1160.0,2.0,52.0,2,4482
Balagere,2 BHK,1205.0,2.0,78.5,2,6514
Kasavanhalli,3 BHK,1550.0,2.0,52.0,3,3354
Gubbalala,3 BHK,1745.0,3.0,125.0,3,7163
AECS Layout,2 BHK,1028.0,2.0,42.0,2,4085
Frazer Town,3 BHK,1700.0,3.0,180.0,3,10588
Karuna Nagar,3 BHK,1200.0,2.0,34.0,3,2833
Kodihalli,3 BHK,2700.0,4.0,270.0,3,10000
other,2 BHK,1100.0,2.0,50.0,2,4545
Ramamurthy Nagar,2 Bedroom,1200.0,3.0,72.0,2,6000
other,2 BHK,1269.72,2.0,68.0,2,5355
Electronic City Phase II,2 BHK,1266.0,2.0,59.0,2,4660
Mysore Road,2 BHK,1060.0,2.0,62.0,2,5849
other,3 Bedroom,1168.0,3.0,70.0,3,5993
other,2 BHK,1552.0,2.0,51.0,2,3286
Talaghattapura,3 BHK,2038.5,3.0,120.0,3,5886
Attibele,1 BHK,500.0,1.0,17.0,1,3400
Hulimavu,2 BHK,1080.0,2.0,43.2,2,4000
other,4 Bedroom,2100.0,4.0,155.0,4,7380
other,2 BHK,1300.0,2.0,65.0,2,5000
Electronics City Phase 1,2 BHK,920.0,2.0,33.0,2,3586
other,4 Bedroom,817.0,2.0,120.0,4,14687
Harlur,2 BHK,1310.0,2.0,50.0,2,3816
Horamavu Agara,3 BHK,1650.0,2.0,75.0,3,4545
other,3 BHK,1515.0,3.0,93.93,3,6200
Raja Rajeshwari Nagar,2 BHK,1157.0,2.0,53.8,2,4649
other,4 BHK,2400.0,5.0,85.0,4,3541
Konanakunte,4 BHK,3000.0,4.0,160.0,4,5333
Basavangudi,3 BHK,2200.0,2.0,162.0,3,7363
other,2 BHK,1243.0,2.0,58.0,2,4666
Rachenahalli,3 BHK,1530.0,2.0,74.39,3,4862
Judicial Layout,3 BHK,2070.0,3.0,162.0,3,7826
Yelahanka,3 BHK,1815.0,3.0,82.0,3,4517
Sarjapur  Road,4 BHK,3900.0,4.0,300.0,4,7692
Electronic City Phase II,2 BHK,1000.0,2.0,28.88,2,2888
Ananth Nagar,2 BHK,982.0,2.0,24.55,2,2500
Hennur Road,2 BHK,1232.0,2.0,89.0,2,7224
Binny Pete,3 BHK,1282.0,3.0,178.0,3,13884
Yelahanka,1 BHK,654.0,1.0,38.4,1,5871
Bannerghatta Road,2 BHK,1181.0,2.0,55.0,2,4657
Bisuvanahalli,3 BHK,1075.0,2.0,45.0,3,4186
2nd Stage Nagarbhavi,4 Bedroom,1200.0,4.0,245.0,4,20416
other,3 BHK,1616.0,3.0,75.0,3,4641
other,5 Bedroom,1800.0,6.0,44.5,5,2472
other,6 Bedroom,1200.0,4.0,135.0,6,11250
Whitefield,3 BHK,1306.0,3.0,54.65,3,4184
Whitefield,3 BHK,1626.0,3.0,89.0,3,5473
other,6 BHK,2700.0,4.0,214.0,6,7925
Whitefield,3 BHK,1585.0,3.0,88.0,3,5552
7th Phase JP Nagar,2 BHK,1370.0,2.0,56.17,2,4100
Whitefield,5 Bedroom,3250.0,5.0,900.0,5,27692
Kanakpura Road,2 BHK,1495.0,2.0,90.0,2,6020
other,3 BHK,2000.0,3.0,210.0,3,10500
other,5 Bedroom,2400.0,4.0,300.0,5,12500
Bannerghatta,3 Bedroom,1375.0,3.0,57.0,3,4145
Old Madras Road,4 BHK,3715.0,6.0,224.5,4,6043
Kereguddadahalli,3 BHK,1400.0,2.0,48.0,3,3428
Kadugodi,5 BHK,1100.0,5.0,135.0,5,12272
Hebbal Kempapura,4 Bedroom,1200.0,4.0,215.0,4,17916
Yelahanka,2 BHK,1150.0,2.0,46.0,2,4000
Hebbal,2 BHK,1080.0,2.0,54.0,2,5000
Kaikondrahalli,3 BHK,1645.0,3.0,100.0,3,6079
other,3 BHK,1760.0,3.0,176.0,3,10000
other,1 BHK,595.0,1.0,42.0,1,7058
Bommenahalli,3 Bedroom,1200.0,3.0,125.0,3,10416
JP Nagar,4 BHK,4624.5,4.0,314.5,4,6800
other,2 BHK,1410.0,2.0,67.0,2,4751
other,2 BHK,1090.0,2.0,31.48,2,2888
Ramagondanahalli,1 Bedroom,540.0,1.0,30.0,1,5555
Sarjapur  Road,2 BHK,1300.0,2.0,60.0,2,4615
9th Phase JP Nagar,3 BHK,1890.0,2.0,85.0,3,4497
other,2 BHK,1254.0,2.0,53.0,2,4226
other,7 Bedroom,2800.0,7.0,125.0,7,4464
Ulsoor,2 BHK,1275.0,2.0,120.0,2,9411
Kathriguppe,3 BHK,1245.0,2.0,68.48,3,5500
other,2 BHK,1170.0,2.0,86.0,2,7350
Panathur,2 BHK,1193.0,2.0,86.0,2,7208
Chandapura,2 BHK,850.0,2.0,25.4,2,2988
Marathahalli,3 BHK,1678.0,3.0,70.0,3,4171
Gunjur,3 BHK,1600.0,3.0,75.0,3,4687
Electronics City Phase 1,3 BHK,1750.0,3.0,84.0,3,4800
Whitefield,2 BHK,1125.0,2.0,62.0,2,5511
other,4 Bedroom,2200.0,4.0,235.0,4,10681
HBR Layout,3 BHK,2800.0,2.0,200.0,3,7142
R.T. Nagar,2 Bedroom,1200.0,2.0,120.0,2,10000
other,2 BHK,1000.0,2.0,50.0,2,5000
other,4 BHK,2780.0,5.0,480.0,4,17266
Electronic City Phase II,3 BHK,1320.0,2.0,38.13,3,2888
other,27 BHK,8000.0,27.0,230.0,27,2875
Ardendale,4 BHK,2422.0,4.0,160.0,4,6606
Kasavanhalli,2 BHK,1100.0,2.0,50.0,2,4545
Uttarahalli,2 BHK,882.0,2.0,30.0,2,3401
Bisuvanahalli,2 BHK,845.0,2.0,37.0,2,4378
Kambipura,2 BHK,883.0,2.0,39.0,2,4416
Brookefield,3 BHK,1595.0,3.0,88.0,3,5517
Whitefield,2 BHK,1280.0,2.0,85.0,2,6640
other,4 Bedroom,1700.0,2.0,550.0,4,32352
other,2 BHK,920.0,2.0,35.0,2,3804
Choodasandra,2 BHK,1300.0,2.0,57.0,2,4384
other,2 BHK,1450.0,1.0,230.0,2,15862
Harlur,3 BHK,1757.0,3.0,132.0,3,7512
Anekal,1 BHK,600.0,1.0,16.5,1,2750
other,4 Bedroom,2400.0,3.0,142.0,4,5916
Sarjapur,2 BHK,1240.0,2.0,44.0,2,3548
other,10 Bedroom,750.0,10.0,90.0,10,12000
EPIP Zone,2 BHK,1330.0,2.0,93.36,2,7019
8th Phase JP Nagar,2 BHK,1062.0,2.0,42.47,2,3999
JP Nagar,2 BHK,1078.0,2.0,45.0,2,4174
Bannerghatta Road,3 BHK,1460.0,2.0,72.0,3,4931
Whitefield,3 Bedroom,1200.0,3.0,68.5,3,5708
Laggere,1 BHK,1200.0,1.0,48.0,1,4000
Kammasandra,3 BHK,912.0,2.0,39.0,3,4276
other,5 Bedroom,2400.0,5.0,625.0,5,26041
Prithvi Layout,4 Bedroom,2028.0,4.0,162.0,4,7988
Mysore Road,2 BHK,980.0,2.0,45.47,2,4639
Sarjapur  Road,2 BHK,1000.0,2.0,40.0,2,4000
Sarjapur  Road,2 BHK,1050.0,2.0,62.0,2,5904
Bannerghatta,3 Bedroom,2611.0,3.0,239.0,3,9153
Domlur,3 BHK,1875.0,2.0,150.0,3,8000
Hennur Road,2 BHK,1065.0,2.0,42.6,2,4000
Sanjay nagar,3 BHK,1800.0,3.0,180.0,3,10000
Sector 7 HSR Layout,2 BHK,1163.0,2.0,98.86,2,8500
Tumkur Road,2 BHK,1098.0,2.0,72.0,2,6557
5th Phase JP Nagar,3 BHK,1725.0,2.0,100.0,3,5797
Whitefield,4 Bedroom,4000.0,4.0,330.0,4,8250
Thubarahalli,2 BHK,1128.0,2.0,75.0,2,6648
other,8 Bedroom,1320.0,6.0,200.0,8,15151
Yelahanka,3 BHK,1705.0,3.0,74.16,3,4349
Thigalarapalya,2 BHK,1418.0,2.0,95.0,2,6699
Ramamurthy Nagar,3 Bedroom,1500.0,3.0,160.0,3,10666
Hebbal,2 BHK,1100.0,2.0,50.0,2,4545
other,3 BHK,1464.0,3.0,56.0,3,3825
Hebbal,2 BHK,812.0,2.0,55.0,2,6773
other,2 BHK,1280.0,2.0,40.0,2,3125
Whitefield,2 BHK,1420.0,2.0,136.0,2,9577
other,11 Bedroom,1200.0,11.0,170.0,11,14166
other,6 Bedroom,727.0,6.0,66.0,6,9078
Banashankari Stage II,4 Bedroom,1200.0,2.0,150.0,4,12500
Old Airport Road,4 BHK,2774.0,4.0,207.0,4,7462
other,3 BHK,2000.0,3.0,85.0,3,4250
HBR Layout,3 BHK,1722.0,3.0,95.0,3,5516
Indira Nagar,3 BHK,2800.0,3.0,330.0,3,11785
other,3 BHK,1650.0,3.0,175.0,3,10606
other,2 BHK,840.0,2.0,27.0,2,3214
Jigani,2 BHK,914.0,2.0,47.0,2,5142
Hennur Road,3 BHK,1482.0,2.0,83.73,3,5649
Kundalahalli,3 BHK,1724.0,3.0,127.0,3,7366
other,3 BHK,1670.0,3.0,91.86,3,5500
Banaswadi,2 BHK,1184.0,2.0,53.0,2,4476
Kodigehaali,2 BHK,1200.0,2.0,55.0,2,4583
TC Palaya,2 Bedroom,1200.0,2.0,65.0,2,5416
Jigani,2 BHK,927.0,2.0,55.0,2,5933
other,2 BHK,1242.0,2.0,39.7,2,3196
Electronic City Phase II,3 BHK,1220.0,3.0,35.23,3,2887
Electronic City,3 BHK,1360.0,2.0,75.0,3,5514
Brookefield,4 Bedroom,2800.0,4.0,240.0,4,8571
7th Phase JP Nagar,2 BHK,1180.0,2.0,72.0,2,6101
Whitefield,3 BHK,1697.0,3.0,108.0,3,6364
Yelahanka New Town,2 BHK,550.0,2.0,26.0,2,4727
Hennur,2 BHK,1225.0,2.0,57.0,2,4653
other,3 BHK,1690.0,2.0,55.0,3,3254
other,10 Bedroom,1660.0,10.0,475.0,10,28614
other,3 BHK,1488.0,3.0,57.0,3,3830
other,3 BHK,1250.0,2.0,80.0,3,6400
JP Nagar,1 BHK,1050.0,1.0,44.0,1,4190
HSR Layout,2 BHK,1027.0,2.0,44.0,2,4284
Kanakpura Road,3 BHK,1699.0,3.0,81.94,3,4822
Whitefield,2 BHK,1205.0,2.0,58.0,2,4813
other,4 Bedroom,712.0,2.0,67.5,4,9480
Horamavu Agara,2 BHK,950.0,2.0,39.0,2,4105
TC Palaya,2 Bedroom,1200.0,2.0,70.0,2,5833
Akshaya Nagar,3 BHK,1476.0,3.0,120.0,3,8130
CV Raman Nagar,2 BHK,1070.0,2.0,55.0,2,5140
EPIP Zone,3 BHK,2160.0,4.0,172.0,3,7962
Sector 2 HSR Layout,2 BHK,1296.0,2.0,75.0,2,5787
Chikkalasandra,2 Bedroom,1800.0,1.0,128.0,2,7111
Kanakpura Road,3 BHK,1570.0,3.0,64.5,3,4108
Koramangala,2 BHK,1300.0,2.0,110.0,2,8461
Sarjapur,4 BHK,1550.0,2.0,65.0,4,4193
Electronic City,2 BHK,770.0,1.0,40.0,2,5194
Jalahalli,3 BHK,1870.0,3.0,110.0,3,5882
other,3 Bedroom,1650.0,4.0,160.0,3,9696
Domlur,3 BHK,1429.0,3.0,86.0,3,6018
Babusapalaya,3 BHK,1167.0,2.0,39.98,3,3425
Bommasandra,2 BHK,1035.0,2.0,41.73,2,4031
Kengeri,3 BHK,1188.0,2.0,38.0,3,3198
2nd Stage Nagarbhavi,5 Bedroom,1200.0,4.0,285.0,5,23750
Nagarbhavi,1 BHK,884.0,2.0,36.0,1,4072
Haralur Road,2 BHK,1194.0,2.0,46.0,2,3852
Hoodi,2 BHK,1026.0,2.0,60.87,2,5932
Shivaji Nagar,7 Bedroom,1400.0,6.0,200.0,7,14285
Kaikondrahalli,2 BHK,1115.0,2.0,50.0,2,4484
Hebbal Kempapura,2 Bedroom,1200.0,2.0,125.0,2,10416
Old Airport Road,4 BHK,2690.0,4.0,191.0,4,7100
Subramanyapura,2 BHK,975.0,2.0,58.0,2,5948
Hosa Road,3 Bedroom,675.0,3.0,72.5,3,10740
Chamrajpet,4 BHK,1900.0,2.0,170.0,4,8947
Banashankari,3 Bedroom,600.0,3.0,100.0,3,16666
OMBR Layout,2 BHK,1101.0,2.0,65.5,2,5949
Sarjapur  Road,3 BHK,1700.0,3.0,135.0,3,7941
Whitefield,4 BHK,3606.0,4.0,312.0,4,8652
Padmanabhanagar,2 BHK,1258.0,2.0,115.0,2,9141
Kanakpura Road,3 BHK,1560.0,3.0,92.0,3,5897
Old Madras Road,4 BHK,3630.0,6.0,195.0,4,5371
JP Nagar,3 BHK,2000.0,3.0,250.0,3,12500
Hebbal,4 BHK,2790.0,4.0,198.0,4,7096
Badavala Nagar,3 BHK,1494.0,2.0,94.55,3,6328
Hennur Road,3 BHK,1482.0,2.0,87.0,3,5870
other,2 BHK,800.0,2.0,50.0,2,6250
Raja Rajeshwari Nagar,2 BHK,1185.0,2.0,47.39,2,3999
Hulimavu,2 BHK,1100.0,2.0,71.5,2,6500
NRI Layout,2 Bedroom,1300.0,2.0,120.0,2,9230
other,2 BHK,1225.0,2.0,150.0,2,12244
Hulimavu,3 BHK,1758.0,3.0,65.0,3,3697
Banaswadi,2 BHK,1145.0,2.0,55.0,2,4803
Hormavu,2 BHK,1180.0,2.0,50.0,2,4237
other,2 BHK,1260.0,2.0,58.0,2,4603
Arekere,2 BHK,1190.0,2.0,55.0,2,4621
Sarjapur  Road,3 BHK,1691.0,3.0,119.0,3,7037
other,4 BHK,3675.0,4.0,367.0,4,9986
Frazer Town,3 BHK,1305.0,3.0,95.45,3,7314
Basavangudi,3 BHK,2337.0,4.0,292.0,3,12494
other,2 Bedroom,1200.0,2.0,150.0,2,12500
Domlur,2 BHK,1206.0,2.0,217.0,2,17993
other,5 Bedroom,1350.0,4.0,130.0,5,9629
Kogilu,2 BHK,1140.0,2.0,50.66,2,4443
other,2 BHK,1250.0,2.0,65.0,2,5200
other,3 Bedroom,600.0,2.0,60.0,3,10000
other,2 BHK,1225.0,2.0,48.0,2,3918
Kereguddadahalli,2 BHK,800.0,2.0,28.0,2,3500
Jalahalli East,2 BHK,1020.0,2.0,58.0,2,5686
TC Palaya,4 Bedroom,2100.0,2.0,50.0,4,2380
other,2 Bedroom,1350.0,2.0,80.0,2,5925
Anekal,2 BHK,925.0,2.0,40.0,2,4324
Yelachenahalli,4 Bedroom,1800.0,3.0,220.0,4,12222
other,3 Bedroom,2000.0,3.0,100.0,3,5000
Frazer Town,3 BHK,1510.0,2.0,75.0,3,4966
Kasavanhalli,5 Bedroom,5800.0,7.0,1200.0,5,20689
other,3 BHK,1614.0,3.0,110.0,3,6815
Yeshwanthpur,3 Bedroom,682.0,2.0,130.0,3,19061
Gottigere,2 Bedroom,1200.0,2.0,95.0,2,7916
TC Palaya,2 Bedroom,900.0,2.0,55.0,2,6111
Bommenahalli,4 Bedroom,1632.0,3.0,145.0,4,8884
OMBR Layout,2 BHK,1041.0,2.0,65.5,2,6292
7th Phase JP Nagar,6 BHK,5080.0,7.0,450.0,6,8858
Bannerghatta,4 BHK,3012.0,6.0,250.0,4,8300
Marathahalli,3 BHK,1530.0,3.0,74.5,3,4869
Sector 2 HSR Layout,3 BHK,1450.0,2.0,135.0,3,9310
Bannerghatta Road,2 BHK,1200.0,2.0,75.0,2,6250
Sahakara Nagar,4 Bedroom,1200.0,4.0,130.0,4,10833
Hosur Road,3 BHK,1508.0,2.0,110.0,3,7294
other,7 BHK,600.0,7.0,106.0,7,17666
other,2 BHK,1188.0,2.0,73.0,2,6144
Old Airport Road,4 BHK,2690.0,4.0,199.0,4,7397
Harlur,2 BHK,1174.0,2.0,76.0,2,6473
Old Madras Road,3 BHK,1350.0,3.0,58.99,3,4369
Frazer Town,3 BHK,1870.0,3.0,185.0,3,9893
Kengeri Satellite Town,2 BHK,1072.0,2.0,25.0,2,2332
other,3 Bedroom,52272.0,2.0,140.0,3,267
Kengeri,2 BHK,726.0,2.0,31.0,2,4269
Hosur Road,2 BHK,1345.0,2.0,106.0,2,7881
Begur Road,2 BHK,1200.0,2.0,45.0,2,3750
Nagavarapalya,2 BHK,980.0,2.0,37.0,2,3775
Hormavu,2 BHK,1050.0,2.0,77.0,2,7333
Uttarahalli,2 BHK,993.0,2.0,39.71,2,3998
7th Phase JP Nagar,3 BHK,1300.0,2.0,52.0,3,4000
Vittasandra,3 BHK,1743.0,3.0,85.0,3,4876
Whitefield,4 Bedroom,1204.0,4.0,125.0,4,10382
Hormavu,2 Bedroom,1200.0,2.0,67.0,2,5583
other,3 BHK,1480.0,3.0,90.0,3,6081
Sarjapur,2 BHK,950.0,2.0,32.5,2,3421
Domlur,1 BHK,640.0,1.0,55.0,1,8593
Electronic City Phase II,2 BHK,972.0,2.0,44.0,2,4526
Marathahalli,3 BHK,1840.0,3.0,98.0,3,5326
Electronic City,2 BHK,1128.0,2.0,64.0,2,5673
other,2 Bedroom,680.0,2.0,42.0,2,6176
other,2 Bedroom,2400.0,2.0,95.0,2,3958
other,2 BHK,1045.0,2.0,42.0,2,4019
other,2 BHK,955.0,2.0,53.55,2,5607
Doddakallasandra,3 BHK,1360.0,2.0,54.39,3,3999
BTM Layout,2 BHK,1150.0,2.0,63.0,2,5478
Hennur Road,4 Bedroom,3450.0,4.0,294.0,4,8521
other,2 BHK,1115.0,2.0,56.87,2,5100
Jigani,2 BHK,920.0,2.0,46.0,2,5000
Bommasandra Industrial Area,2 BHK,1030.0,2.0,41.0,2,3980
Thanisandra,3 BHK,1564.0,3.0,101.0,3,6457
Uttarahalli,2 BHK,1045.0,2.0,45.0,2,4306
Hegde Nagar,4 Bedroom,1200.0,4.0,83.0,4,6916
Subramanyapura,2 BHK,1277.0,2.0,55.0,2,4306
other,3 BHK,2479.13,3.0,215.0,3,8672
Sarjapur  Road,3 BHK,1846.0,3.0,140.0,3,7583
other,2 BHK,1100.0,2.0,106.0,2,9636
other,3 BHK,1690.0,3.0,75.0,3,4437
Hormavu,2 Bedroom,1200.0,3.0,70.0,2,5833
Yelahanka,4 Bedroom,3042.0,4.0,131.0,4,4306
Rajaji Nagar,3 BHK,1640.0,3.0,223.0,3,13597
other,2 BHK,1068.0,2.0,40.05,2,3750
Kanakpura Road,2 BHK,1135.0,2.0,58.0,2,5110
other,3 BHK,1925.0,3.0,120.0,3,6233
Begur Road,3 BHK,1615.0,3.0,60.0,3,3715
other,1 BHK,650.0,1.0,26.0,1,4000
other,9 Bedroom,1200.0,9.0,120.0,9,10000
Kaggadasapura,2 BHK,1150.0,2.0,42.0,2,3652
Kothannur,6 Bedroom,930.0,6.0,135.0,6,14516
other,3 BHK,1410.0,2.0,45.12,3,3200
6th Phase JP Nagar,3 Bedroom,600.0,2.0,100.0,3,16666
other,3 BHK,2000.0,3.0,207.0,3,10350
Tumkur Road,3 BHK,1354.0,3.0,90.0,3,6646
other,2 Bedroom,1050.0,2.0,110.0,2,10476
Arekere,1 BHK,600.0,1.0,28.0,1,4666
Varthur,4 Bedroom,1320.0,2.0,110.0,4,8333
Koramangala,3 BHK,1600.0,3.0,88.0,3,5500
Kanakpura Road,4 Bedroom,3500.0,6.0,225.0,4,6428
GM Palaya,2 BHK,1000.0,2.0,36.0,2,3600
other,3 BHK,1130.0,3.0,38.0,3,3362
Vittasandra,2 BHK,1238.0,2.0,67.0,2,5411
other,3 BHK,2204.0,2.0,305.0,3,13838
KR Puram,8 Bedroom,1200.0,12.0,110.0,8,9166
Kudlu Gate,3 BHK,1364.0,2.0,46.38,3,3400
other,2 BHK,900.0,2.0,45.0,2,5000
Gubbalala,3 BHK,1745.0,3.0,104.0,3,5959
Kogilu,2 BHK,1200.0,2.0,53.33,2,4444
Kanakapura,2 BHK,1020.0,2.0,42.83,2,4199
Banashankari Stage III,3 BHK,1650.0,3.0,118.0,3,7151
other,3 BHK,1550.0,2.0,70.0,3,4516
5th Phase JP Nagar,2 BHK,1440.0,2.0,60.0,2,4166
Sarjapura - Attibele Road,3 BHK,1329.0,2.0,45.95,3,3457
other,2 BHK,916.0,2.0,42.0,2,4585
Hulimavu,1 BHK,688.0,1.0,50.0,1,7267
other,3 BHK,1358.0,3.0,50.0,3,3681
Kanakpura Road,2 BHK,1331.0,2.0,107.0,2,8039
Malleshwaram,1 BHK,686.0,1.0,62.5,1,9110
Hoodi,2 BHK,1050.0,2.0,50.0,2,4761
Talaghattapura,3 BHK,2106.0,3.0,126.0,3,5982
Bhoganhalli,3 BHK,1718.0,3.0,90.2,3,5250
Raja Rajeshwari Nagar,3 BHK,1530.0,3.0,57.0,3,3725
5th Block Hbr Layout,2 BHK,1206.0,2.0,56.0,2,4643
Hebbal,2 BHK,1075.0,2.0,52.0,2,4837
Sarjapur  Road,3 Bedroom,1200.0,3.0,95.0,3,7916
other,3 BHK,1936.0,2.0,54.21,3,2800
Nagavara,4 BHK,2172.65,3.0,120.0,4,5523
Sahakara Nagar,3 BHK,1100.0,2.0,75.0,3,6818
BTM 2nd Stage,3 Bedroom,1260.0,5.0,185.0,3,14682
other,8 Bedroom,990.0,12.0,120.0,8,12121
Jalahalli East,3 Bedroom,800.0,2.0,80.0,3,10000
Sarjapur  Road,2 BHK,1117.0,2.0,29.99,2,2684
other,3 BHK,1685.0,2.0,110.0,3,6528
CV Raman Nagar,2 BHK,740.0,2.0,37.0,2,5000
Chamrajpet,4 Bedroom,1080.0,4.0,250.0,4,23148
Magadi Road,4 Bedroom,1500.0,2.0,93.0,4,6200
Banashankari,3 BHK,1800.0,3.0,115.0,3,6388
Yelachenahalli,3 BHK,1700.0,3.0,130.0,3,7647
Uttarahalli,2 BHK,850.0,2.0,37.93,2,4462
Hennur Road,3 BHK,1891.0,3.0,139.0,3,7350
Jalahalli,2 BHK,1400.0,1.0,80.0,2,5714
Lakshminarayana Pura,2 BHK,1172.0,2.0,82.0,2,6996
Electronic City Phase II,3 BHK,1400.0,2.0,40.43,3,2887
Sonnenahalli,2 BHK,896.0,2.0,43.75,2,4882
Vijayanagar,2 BHK,1180.0,2.0,65.0,2,5508
Basaveshwara Nagar,7 Bedroom,2460.0,7.0,350.0,7,14227
other,2 BHK,850.0,2.0,33.0,2,3882
Anekal,1 BHK,530.0,1.0,18.0,1,3396
other,3 BHK,1550.0,3.0,110.0,3,7096
Kanakapura,3 BHK,1699.0,3.0,97.0,3,5709
Bellandur,4 BHK,2025.0,4.0,109.0,4,5382
Ananth Nagar,1 BHK,500.0,2.0,14.0,1,2800
other,1 BHK,500.0,1.0,20.0,1,4000
other,2 BHK,1146.0,2.0,38.5,2,3359
Malleshwaram,3 BHK,2475.0,4.0,332.0,3,13414
Haralur Road,2 BHK,1455.0,2.0,145.0,2,9965
Seegehalli,2 Bedroom,800.0,1.0,50.0,2,6250
Abbigere,1 BHK,765.0,2.0,30.0,1,3921
Lingadheeranahalli,3 BHK,1521.0,3.0,94.99,3,6245
Gottigere,3 BHK,1600.0,2.0,55.0,3,3437
other,3 BHK,3155.0,4.0,370.0,3,11727
Hebbal Kempapura,3 BHK,2900.0,4.0,240.0,3,8275
other,3 BHK,1765.0,3.0,105.0,3,5949
Kasavanhalli,3 BHK,1980.0,3.0,88.0,3,4444
Amruthahalli,3 Bedroom,1900.0,3.0,135.0,3,7105
other,2 BHK,1265.0,2.0,70.0,2,5533
Bommasandra,2 BHK,1034.0,2.0,40.0,2,3868
other,2 BHK,1012.0,2.0,59.0,2,5830
Electronic City,2 BHK,1070.0,2.0,54.0,2,5046
Mahadevpura,4 Bedroom,3555.0,4.0,230.0,4,6469
5th Block Hbr Layout,3 BHK,1270.0,2.0,70.0,3,5511
Yeshwanthpur,2 Bedroom,585.0,1.0,70.0,2,11965
other,3 BHK,2045.0,4.0,154.0,3,7530
Bellandur,3 BHK,1490.0,2.0,98.0,3,6577
Banaswadi,7 Bedroom,6150.0,6.0,830.0,7,13495
Mysore Road,2 BHK,883.0,2.0,40.0,2,4530
Yeshwanthpur,1 BHK,671.0,1.0,36.85,1,5491
Kammasandra,2 BHK,1156.0,2.0,32.0,2,2768
Haralur Road,3 BHK,2017.0,3.0,125.0,3,6197
Thigalarapalya,4 BHK,3122.0,6.0,235.0,4,7527
Ramagondanahalli,3 BHK,1635.0,3.0,62.0,3,3792
other,7 Bedroom,4400.0,9.0,120.0,7,2727
Kanakpura Road,2 BHK,1296.0,2.0,89.0,2,6867
other,6 Bedroom,2400.0,4.0,280.0,6,11666
Budigere,2 BHK,1139.0,2.0,62.0,2,5443
CV Raman Nagar,2 BHK,1230.0,3.0,47.0,2,3821
other,1 BHK,581.91,2.0,25.0,1,4296
Binny Pete,2 BHK,1245.0,2.0,86.28,2,6930
other,3 Bedroom,1158.0,2.0,85.0,3,7340
Marathahalli,4 BHK,3951.0,4.0,220.0,4,5568
Electronic City,2 BHK,550.0,1.0,15.0,2,2727
Ramamurthy Nagar,1 BHK,360.0,1.0,26.0,1,7222
Electronics City Phase 1,3 BHK,1350.0,3.0,55.0,3,4074
Green Glen Layout,3 BHK,1670.0,3.0,120.0,3,7185
Akshaya Nagar,3 BHK,1600.0,2.0,65.0,3,4062
Horamavu Banaswadi,2 BHK,1307.0,2.0,51.6,2,3947
7th Phase JP Nagar,3 BHK,1050.0,2.0,57.0,3,5428
Hennur Road,3 BHK,1703.0,3.0,101.0,3,5930
Kaggalipura,3 BHK,1210.0,2.0,58.0,3,4793
Electronic City,2 BHK,1025.0,2.0,29.6,2,2887
Haralur Road,2 BHK,1140.0,2.0,43.0,2,3771
7th Phase JP Nagar,3 BHK,1343.0,2.0,56.41,3,4200
R.T. Nagar,3 BHK,1560.0,3.0,85.0,3,5448
other,2 BHK,925.0,2.0,51.0,2,5513
Kanakpura Road,3 BHK,1420.0,2.0,75.0,3,5281
other,2 BHK,1415.0,2.0,110.0,2,7773
other,2 BHK,1210.0,2.0,60.0,2,4958
Yeshwanthpur,3 BHK,1852.0,3.0,160.0,3,8639
Hoskote,3 BHK,1250.0,3.0,50.0,3,4000
9th Phase JP Nagar,2 BHK,1100.0,2.0,65.0,2,5909
2nd Phase Judicial Layout,2 BHK,1150.0,2.0,40.25,2,3500
Jakkur,2 BHK,1100.0,2.0,53.35,2,4850
Magadi Road,2 Bedroom,1350.0,1.0,100.0,2,7407
Panathur,3 BHK,1546.0,3.0,99.0,3,6403
Ambedkar Nagar,2 BHK,1424.0,2.0,90.0,2,6320
Bellandur,3 BHK,1398.0,3.0,68.0,3,4864
Indira Nagar,3 BHK,1650.0,3.0,200.0,3,12121
HBR Layout,4 Bedroom,1200.0,4.0,120.0,4,10000
Brookefield,3 BHK,1476.0,3.0,105.0,3,7113
other,2 Bedroom,500.0,2.0,70.0,2,14000
Magadi Road,3 BHK,1322.0,2.0,58.82,3,4449
other,2 Bedroom,432.0,2.0,40.0,2,9259
other,2 BHK,1100.0,2.0,55.0,2,5000
other,4 BHK,4750.0,5.0,600.0,4,12631
Sarjapur,3 BHK,1285.0,2.0,68.0,3,5291
Ambalipura,2 BHK,1150.0,2.0,80.0,2,6956
Thanisandra,3 BHK,1430.0,2.0,56.0,3,3916
Marathahalli,3 BHK,1350.0,2.0,62.0,3,4592
other,3 Bedroom,1350.0,3.0,195.0,3,14444
Thubarahalli,2 BHK,1185.0,2.0,65.0,2,5485
Bannerghatta Road,3 BHK,1760.0,3.0,67.0,3,3806
other,4 Bedroom,2400.0,5.0,350.0,4,14583
Kanakpura Road,2 BHK,900.0,2.0,41.0,2,4555
Kanakpura Road,2 BHK,1297.0,2.0,105.0,2,8095
other,3 BHK,1840.0,3.0,155.0,3,8423
Ramamurthy Nagar,2 BHK,950.0,2.0,50.79,2,5346
Chikkabanavar,5 Bedroom,1200.0,4.0,70.0,5,5833
BTM 2nd Stage,2 BHK,950.0,2.0,52.0,2,5473
Yeshwanthpur,2 BHK,1277.5,2.0,95.815,2,7500
Electronic City,3 BHK,1571.0,3.0,89.0,3,5665
Electronic City Phase II,2 BHK,1031.0,2.0,55.0,2,5334
Whitefield,2 BHK,1485.0,2.0,90.0,2,6060
Ananth Nagar,2 BHK,1109.0,2.0,60.0,2,5410
Koramangala,3 BHK,2200.0,3.0,180.0,3,8181
Whitefield,1 BHK,950.0,1.0,50.0,1,5263
Laggere,3 Bedroom,1200.0,2.0,150.0,3,12500
Hosa Road,2 BHK,1161.0,2.0,55.15,2,4750
Basavangudi,3 BHK,2350.0,3.0,300.0,3,12765
Electronic City Phase II,2 BHK,1003.0,2.0,40.6,2,4047
other,2 Bedroom,1350.0,2.0,185.0,2,13703
9th Phase JP Nagar,2 BHK,1080.0,2.0,37.0,2,3425
Hulimavu,2 BHK,1255.0,2.0,73.0,2,5816
Banaswadi,2 BHK,1250.0,2.0,60.0,2,4800
Brookefield,4 Bedroom,1700.0,4.0,152.0,4,8941
Ramamurthy Nagar,3 Bedroom,1200.0,2.0,66.0,3,5500
other,2 BHK,824.0,2.0,60.0,2,7281
1st Phase JP Nagar,3 BHK,1875.0,3.0,167.0,3,8906
Bommasandra Industrial Area,3 BHK,1320.0,2.0,38.12,3,2887
Kothanur,5 Bedroom,9600.0,5.0,550.0,5,5729
Uttarahalli,3 BHK,1330.0,2.0,47.0,3,3533
Kaikondrahalli,3 BHK,1220.0,3.0,56.0,3,4590
Electronic City,3 BHK,1111.0,3.0,50.0,3,4500
Kothannur,2 BHK,1197.0,2.0,47.88,2,4000
Marathahalli,4 BHK,2519.0,5.0,185.0,4,7344
other,3 BHK,1340.0,2.0,100.0,3,7462
Old Madras Road,4 Bedroom,1450.0,3.0,132.0,4,9103
Hoodi,3 BHK,1837.0,3.0,145.0,3,7893
Haralur Road,4 Bedroom,1200.0,4.0,245.0,4,20416
Kudlu Gate,2 BHK,1238.0,2.0,55.0,2,4442
7th Phase JP Nagar,3 BHK,1405.0,2.0,59.01,3,4200
Doddaballapur,4 BHK,1690.0,3.0,80.0,4,4733
other,3 BHK,1620.0,2.0,110.0,3,6790
Kanakpura Road,3 BHK,1673.0,3.0,115.0,3,6873
Margondanahalli,2 Bedroom,1090.0,2.0,58.0,2,5321
Varthur,2 BHK,1112.0,2.0,40.0,2,3597
Marathahalli,3 BHK,1385.0,2.0,66.65,3,4812
Electronic City Phase II,2 BHK,545.0,1.0,27.0,2,4954
Subramanyapura,2 BHK,985.0,2.0,58.0,2,5888
Kanakpura Road,1 BHK,825.0,1.0,36.29,1,4398
Kanakpura Road,2 BHK,1080.0,2.0,37.8,2,3499
Hosa Road,2 BHK,1430.0,2.0,97.52,2,6819
Kengeri Satellite Town,2 BHK,1050.0,2.0,43.0,2,4095
Kumaraswami Layout,4 Bedroom,2000.0,4.0,139.0,4,6950
Banashankari Stage VI,4 Bedroom,4800.0,4.0,190.0,4,3958
other,3 BHK,1533.0,3.0,115.0,3,7501
other,4 Bedroom,2800.0,3.0,500.0,4,17857
Malleshwaram,3 BHK,2610.0,3.0,399.0,3,15287
Lakshminarayana Pura,3 BHK,1750.0,3.0,150.0,3,8571
TC Palaya,3 Bedroom,1200.0,2.0,66.0,3,5500
Prithvi Layout,2 BHK,1352.0,2.0,87.5,2,6471
Harlur,3 BHK,1754.0,3.0,125.0,3,7126
other,3 BHK,1560.0,3.0,85.0,3,5448
Bannerghatta,4 BHK,3012.0,6.0,268.0,4,8897
Whitefield,4 BHK,3584.0,4.0,199.0,4,5552
Channasandra,3 BHK,1470.0,2.0,90.0,3,6122
9th Phase JP Nagar,2 BHK,1035.0,2.0,45.0,2,4347
Whitefield,4 Bedroom,2400.0,5.0,329.0,4,13708
Kammanahalli,2 BHK,1160.0,2.0,52.0,2,4482
Jakkur,2 BHK,1100.0,2.0,52.0,2,4727
Horamavu Agara,2 BHK,1170.0,2.0,38.0,2,3247
Bannerghatta Road,1 BHK,595.0,1.0,31.83,1,5349
HSR Layout,3 BHK,1650.0,3.0,110.0,3,6666
1st Block Jayanagar,3 BHK,1875.0,2.0,235.0,3,12533
other,4 BHK,800.0,4.0,80.0,4,10000
other,3 BHK,1375.0,2.0,48.06,3,3495
Ramagondanahalli,2 BHK,1235.0,2.0,46.8,2,3789
other,1 BHK,750.0,1.0,25.0,1,3333
7th Phase JP Nagar,3 BHK,1400.0,3.0,115.0,3,8214
Poorna Pragna Layout,3 BHK,1270.0,2.0,50.79,3,3999
Bannerghatta Road,2 BHK,1105.0,2.0,51.0,2,4615
other,3 BHK,1375.0,2.0,85.0,3,6181
Attibele,4 Bedroom,2168.0,4.0,95.0,4,4381
Gottigere,8 Bedroom,1200.0,8.0,90.0,8,7500
Yelahanka,2 BHK,1065.0,2.0,40.84,2,3834
other,3 Bedroom,3800.0,3.0,580.0,3,15263
Raja Rajeshwari Nagar,9 Bedroom,3125.0,9.0,350.0,9,11200
Whitefield,1 BHK,825.0,1.0,45.0,1,5454
other,2 BHK,1095.0,2.0,40.0,2,3652
other,3 Bedroom,720.0,3.0,65.0,3,9027
other,3 BHK,1440.0,2.0,60.33,3,4189
Electronic City,2 BHK,870.0,2.0,39.5,2,4540
Panathur,2 BHK,1195.0,2.0,84.0,2,7029
other,2 BHK,1200.0,2.0,49.5,2,4125
Raja Rajeshwari Nagar,3 BHK,1530.0,3.0,61.2,3,4000
Padmanabhanagar,3 BHK,1708.0,2.0,90.5,3,5298
other,2 BHK,975.0,2.0,43.0,2,4410
KR Puram,2 BHK,750.0,2.0,25.0,2,3333
other,8 Bedroom,1200.0,8.0,190.0,8,15833
Iblur Village,3 BHK,1920.0,3.0,130.0,3,6770
Nagarbhavi,6 Bedroom,1200.0,5.0,240.0,6,20000
Thanisandra,2 BHK,1183.0,2.0,58.56,2,4950
Kothanur,4 Bedroom,1140.0,4.0,90.0,4,7894
Mico Layout,3 BHK,1550.0,2.0,65.0,3,4193
Kaggadasapura,2 BHK,1110.0,2.0,53.0,2,4774
other,8 Bedroom,1500.0,10.0,165.0,8,11000
Kaggadasapura,2 BHK,1105.0,2.0,38.67,2,3499
Anekal,3 Bedroom,1800.0,3.0,68.0,3,3777
other,5 BHK,2800.0,4.0,200.0,5,7142
Bommanahalli,2 BHK,1000.0,2.0,29.9,2,2990
Prithvi Layout,2 BHK,1190.0,2.0,68.0,2,5714
2nd Stage Nagarbhavi,4 Bedroom,1350.0,4.0,200.0,4,14814
Hebbal,4 BHK,3067.0,4.0,230.0,4,7499
Sarjapur  Road,3 BHK,1929.0,4.0,103.0,3,5339
Ramamurthy Nagar,3 BHK,1525.0,3.0,100.0,3,6557
Nehru Nagar,3 BHK,1775.0,3.0,110.0,3,6197
Banashankari,2 BHK,1125.0,2.0,63.0,2,5600
other,2 BHK,900.0,2.0,42.0,2,4666
Sarjapur  Road,2 BHK,1155.0,2.0,58.0,2,5021
Hulimavu,4 Bedroom,1200.0,4.0,75.0,4,6250
Sonnenahalli,2 BHK,1268.0,2.0,73.0,2,5757
other,3 BHK,1500.0,3.0,88.0,3,5866
Electronic City,1 BHK,630.0,1.0,47.0,1,7460
Seegehalli,3 BHK,1525.0,3.0,65.0,3,4262
Kaggadasapura,2 BHK,1060.0,2.0,48.0,2,4528
Lingadheeranahalli,3 BHK,1687.0,3.0,127.0,3,7528
Thanisandra,3 BHK,1847.0,3.0,123.0,3,6659
Begur Road,3 BHK,1410.0,2.0,53.0,3,3758
7th Phase JP Nagar,2 BHK,1075.0,2.0,42.99,2,3999
Rajaji Nagar,4 BHK,3526.0,4.0,492.0,4,13953
BTM Layout,3 Bedroom,3000.0,3.0,338.0,3,11266
Basavangudi,3 BHK,1500.0,2.0,143.0,3,9533
other,2 BHK,1010.0,2.0,46.0,2,4554
Haralur Road,3 BHK,1875.0,3.0,110.0,3,5866
other,5 Bedroom,1200.0,6.0,137.0,5,11416
Kanakpura Road,2 BHK,983.0,2.0,50.6,2,5147
Kasturi Nagar,2 BHK,1101.0,2.0,65.0,2,5903
other,2 BHK,1140.0,2.0,76.0,2,6666
Sarjapur  Road,2 BHK,1215.0,2.0,110.0,2,9053
Kengeri Satellite Town,2 BHK,1007.0,2.0,42.0,2,4170
Bellandur,3 BHK,2039.0,3.0,168.0,3,8239
Ulsoor,3 BHK,1020.0,3.0,110.0,3,10784
other,2 BHK,1198.0,2.0,84.0,2,7011
other,4 BHK,4000.0,4.0,520.0,4,13000
other,4 Bedroom,1470.0,4.0,34.5,4,2346
other,2 BHK,1370.0,2.0,75.0,2,5474
other,2 BHK,1316.0,2.0,55.5,2,4217
Kaggadasapura,3 BHK,1350.0,3.0,55.5,3,4111
other,2 BHK,1050.0,2.0,43.0,2,4095
KR Puram,3 BHK,1128.76,3.0,47.405,3,4199
Electronic City,2 BHK,921.0,2.0,34.0,2,3691
Kundalahalli,2 BHK,1010.0,2.0,49.27,2,4878
KR Puram,2 BHK,1015.0,2.0,43.0,2,4236
TC Palaya,3 Bedroom,1200.0,2.0,66.5,3,5541
Lakshminarayana Pura,2 BHK,1179.0,2.0,75.0,2,6361
Anjanapura,4 BHK,1850.0,4.0,86.0,4,4648
Kammasandra,2 BHK,870.0,2.0,24.0,2,2758
Bellandur,3 BHK,1605.0,3.0,85.0,3,5295
Banashankari Stage VI,4 Bedroom,2800.0,4.0,89.0,4,3178
Malleshwaram,3 BHK,2475.0,4.0,320.0,3,12929
other,2 Bedroom,3000.0,2.0,510.0,2,17000
other,6 Bedroom,2400.0,7.0,110.0,6,4583
other,2 BHK,1000.0,1.0,55.0,2,5500
Ramamurthy Nagar,2 Bedroom,1200.0,2.0,80.0,2,6666
other,2 Bedroom,1200.0,2.0,125.0,2,10416
Kammasandra,2 Bedroom,1200.0,2.0,66.0,2,5500
other,2 BHK,1200.0,2.0,50.0,2,4166
Hennur,2 BHK,1255.0,2.0,57.5,2,4581
Hosur Road,2 BHK,1085.0,2.0,30.38,2,2800
Kanakpura Road,3 BHK,1592.0,3.0,125.0,3,7851
other,2 BHK,1313.0,2.0,58.27,2,4437
Kogilu,3 BHK,1934.0,3.0,150.0,3,7755
Harlur,2 BHK,1200.0,2.0,49.5,2,4125
Uttarahalli,6 BHK,3600.0,6.0,120.0,6,3333
Old Madras Road,2 BHK,935.0,2.0,45.0,2,4812
Bommasandra,2 BHK,877.0,2.0,30.0,2,3420
other,2 BHK,936.0,2.0,42.0,2,4487
Nagasandra,3 BHK,1470.0,2.0,110.0,3,7482
other,1 Bedroom,1050.0,1.0,70.0,1,6666
Hennur Road,2 BHK,1232.0,2.0,69.61,2,5650
Ambedkar Nagar,3 BHK,2225.0,4.0,169.0,3,7595
other,3 Bedroom,1200.0,2.0,80.0,3,6666
Dasarahalli,3 BHK,1901.0,3.0,119.0,3,6259
Sarjapur  Road,3 BHK,1819.0,3.0,100.0,3,5497
other,3 BHK,1464.0,3.0,56.0,3,3825
Old Madras Road,4 BHK,3715.0,6.0,200.5,4,5397
9th Phase JP Nagar,4 BHK,5000.0,4.0,290.0,4,5800
other,4 BHK,1200.0,4.0,169.0,4,14083
other,2 BHK,1256.0,2.0,60.0,2,4777
7th Phase JP Nagar,2 BHK,1100.0,2.0,44.0,2,4000
TC Palaya,4 Bedroom,4800.0,3.0,302.0,4,6291
Kothanur,3 BHK,1435.0,3.0,70.0,3,4878
6th Phase JP Nagar,4 BHK,3245.0,4.0,250.0,4,7704
Sarjapur  Road,3 BHK,1845.0,4.0,110.0,3,5962
Jigani,3 Bedroom,2400.0,4.0,130.0,3,5416
HSR Layout,2 BHK,1140.0,2.0,46.0,2,4035
Ulsoor,2 BHK,1200.0,2.0,60.0,2,5000
Talaghattapura,3 BHK,2254.0,3.0,170.0,3,7542
Frazer Town,3 BHK,1750.0,3.0,150.0,3,8571
other,2 BHK,1020.0,2.0,37.0,2,3627
Bommanahalli,1 BHK,515.0,1.0,25.0,1,4854
Lingadheeranahalli,3 BHK,1893.0,3.0,140.0,3,7395
other,3 BHK,3025.0,3.0,500.0,3,16528
other,2 BHK,830.0,2.0,34.0,2,4096
Singasandra,2 Bedroom,600.0,2.0,36.5,2,6083
Bisuvanahalli,3 BHK,1075.0,2.0,90.0,3,8372
other,4 Bedroom,3200.0,4.0,1200.0,4,37500
Kasavanhalli,2 BHK,1200.0,2.0,78.0,2,6500
Sarjapur  Road,2 BHK,1233.0,2.0,77.65,2,6297
Bannerghatta Road,2 BHK,1255.0,2.0,65.26,2,5200
Begur Road,2 BHK,1200.0,2.0,37.8,2,3149
Varthur,3 BHK,2145.0,3.0,175.0,3,8158
1st Phase JP Nagar,5 Bedroom,1500.0,5.0,85.0,5,5666
Kothannur,2 BHK,740.0,2.0,34.0,2,4594
Kengeri,3 BHK,1250.0,3.0,48.0,3,3840
other,2 BHK,1210.0,2.0,40.0,2,3305
Sarjapur  Road,3 BHK,1525.0,2.0,60.0,3,3934
Kogilu,1 Bedroom,540.0,1.0,36.0,1,6666
Tumkur Road,3 BHK,1240.0,2.0,84.0,3,6774
Subramanyapura,3 BHK,1245.0,2.0,68.0,3,5461
other,3 Bedroom,1865.0,3.0,350.0,3,18766
Hosa Road,2 BHK,1104.0,2.0,36.43,2,3299
Bisuvanahalli,3 BHK,1075.0,2.0,35.99,3,3347
Nagavara,2 BHK,2200.0,2.0,55.0,2,2500
Whitefield,2 BHK,1155.0,2.0,60.0,2,5194
other,2 BHK,1520.0,2.0,62.0,2,4078
Thigalarapalya,4 BHK,3122.0,6.0,245.0,4,7847
Harlur,2 BHK,1290.0,2.0,85.0,2,6589
other,3 BHK,2300.0,3.0,80.0,3,3478
Kannamangala,4 Bedroom,3000.0,4.0,200.0,4,6666
other,1 BHK,400.0,1.0,14.0,1,3500
Banashankari,3 BHK,1340.0,2.0,53.6,3,4000
Malleshwaram,2 BHK,900.0,2.0,95.0,2,10555
Rajaji Nagar,6 Bedroom,1800.0,4.0,324.0,6,18000
Hennur Road,2 BHK,987.0,2.0,49.0,2,4964
Old Madras Road,3 BHK,1720.0,3.0,100.0,3,5813
Hebbal,2 BHK,1040.0,2.0,50.0,2,4807
Anandapura,2 Bedroom,900.0,2.0,64.0,2,7111
Raja Rajeshwari Nagar,2 BHK,1030.0,2.0,50.0,2,4854
Neeladri Nagar,3 BHK,2556.0,3.0,133.0,3,5203
Basavangudi,3 BHK,1850.0,3.0,168.0,3,9081
Kaval Byrasandra,4 Bedroom,750.0,4.0,110.0,4,14666
Balagere,2 BHK,1028.0,2.0,31.86,2,3099
7th Phase JP Nagar,2 BHK,1270.0,2.0,83.0,2,6535
Sahakara Nagar,3 BHK,1655.0,3.0,115.0,3,6948
Electronics City Phase 1,2 BHK,940.0,2.0,40.0,2,4255
Domlur,2 BHK,1050.0,2.0,85.0,2,8095
other,4 Bedroom,600.0,4.0,40.0,4,6666
other,2 BHK,946.0,2.0,38.0,2,4016
CV Raman Nagar,3 BHK,1726.0,3.0,145.0,3,8400
Sarakki Nagar,3 BHK,2121.0,4.0,253.0,3,11928
Vittasandra,2 BHK,1404.0,2.0,75.0,2,5341
Sarjapur,3 BHK,1525.0,2.0,48.0,3,3147
Kengeri,2 BHK,1025.0,2.0,57.0,2,5560
other,5 Bedroom,1575.0,4.0,150.0,5,9523
other,2 BHK,860.0,1.0,43.0,2,5000
Talaghattapura,3 BHK,1223.0,2.0,42.81,3,3500
other,4 BHK,3750.0,3.0,550.0,4,14666
Kothannur,2 BHK,1197.0,2.0,47.8,2,3993
Garudachar Palya,2 BHK,1150.0,2.0,52.75,2,4586
other,4 Bedroom,4800.0,5.0,629.0,4,13104
R.T. Nagar,2 BHK,1040.0,2.0,46.0,2,4423
Banashankari,2 BHK,1310.0,2.0,80.0,2,6106
Uttarahalli,3 BHK,1788.0,3.0,98.34,3,5500
Raja Rajeshwari Nagar,2 BHK,1144.0,2.0,38.55,2,3369
Anjanapura,2 BHK,1070.0,2.0,39.0,2,3644
other,2 BHK,1200.0,2.0,70.0,2,5833
Ambedkar Nagar,3 BHK,1862.0,3.0,120.0,3,6444
Margondanahalli,2 Bedroom,1200.0,2.0,58.5,2,4875
Begur Road,2 BHK,970.0,2.0,27.0,2,2783
Kasavanhalli,3 BHK,1646.0,3.0,79.5,3,4829
Sarjapur  Road,4 BHK,4395.0,4.0,242.0,4,5506
other,5 Bedroom,1200.0,4.0,100.0,5,8333
Iblur Village,3 BHK,1995.0,3.0,135.0,3,6766
1st Block Jayanagar,4 Bedroom,2400.0,4.0,450.0,4,18750
Hosakerehalli,2 BHK,1085.0,2.0,47.74,2,4400
Budigere,2 BHK,1139.0,2.0,56.8,2,4986
Haralur Road,2 BHK,1092.0,2.0,44.0,2,4029
Pattandur Agrahara,3 BHK,1767.0,3.0,98.0,3,5546
other,3 BHK,1700.0,3.0,80.0,3,4705
Bellandur,3 BHK,1692.0,3.0,88.0,3,5200
6th Phase JP Nagar,3 BHK,1645.0,3.0,100.0,3,6079
Uttarahalli,3 BHK,1345.0,2.0,57.0,3,4237
other,2 BHK,1100.0,2.0,60.0,2,5454
Gollarapalya Hosahalli,3 BHK,1408.0,3.0,62.0,3,4403
Banashankari Stage III,4 Bedroom,2400.0,4.0,240.0,4,10000
Kadugodi,4 Bedroom,1920.0,4.0,200.0,4,10416
other,5 Bedroom,6040.0,4.0,170.0,5,2814
Thigalarapalya,3 BHK,2072.0,4.0,150.0,3,7239
Malleshwaram,2 BHK,1420.0,2.0,158.0,2,11126
Electronics City Phase 1,3 Bedroom,2040.0,3.0,137.0,3,6715
Sarjapur  Road,3 BHK,1220.0,3.0,56.0,3,4590
8th Phase JP Nagar,4 Bedroom,1200.0,4.0,145.0,4,12083
other,2 BHK,1300.0,2.0,79.0,2,6076
Electronics City Phase 1,2 BHK,1150.0,2.0,39.0,2,3391
KR Puram,2 BHK,1470.0,2.0,86.0,2,5850
Hosa Road,2 BHK,1063.0,2.0,31.9,2,3000
other,2 BHK,1311.0,2.0,48.0,2,3661
other,4 BHK,3500.0,4.0,350.0,4,10000
Nagarbhavi,4 Bedroom,600.0,2.0,100.0,4,16666
other,3 BHK,1400.0,2.0,55.0,3,3928
Kanakpura Road,2 Bedroom,2200.0,2.0,62.0,2,2818
Thanisandra,3 BHK,1430.0,2.0,56.0,3,3916
Vidyaranyapura,2 BHK,1060.0,2.0,55.0,2,5188
other,3 BHK,1000.0,2.0,110.0,3,11000
Sector 7 HSR Layout,3 BHK,1760.0,3.0,145.0,3,8238
Whitefield,3 BHK,2010.0,3.0,92.0,3,4577
Kanakpura Road,3 BHK,1480.0,2.0,64.23,3,4339
8th Phase JP Nagar,3 BHK,1230.0,2.0,43.5,3,3536
Gottigere,3 BHK,1300.0,3.0,65.0,3,5000
other,4 Bedroom,800.0,4.0,120.0,4,15000
Frazer Town,4 BHK,4850.0,6.0,385.0,4,7938
Singasandra,2 BHK,1100.0,2.0,52.0,2,4727
Harlur,3 BHK,1752.12,3.0,116.0,3,6620
Marathahalli,2 BHK,957.0,2.0,46.9,2,4900
Balagere,1 BHK,790.5,1.0,41.9,1,5300
other,3 BHK,1942.0,3.0,102.0,3,5252
other,2 BHK,1000.0,2.0,53.0,2,5300
Yeshwanthpur,1 BHK,668.0,1.0,36.85,1,5516
other,5 Bedroom,3000.0,6.0,1000.0,5,33333
Nagarbhavi,3 Bedroom,600.0,2.0,80.0,3,13333
Uttarahalli,2 BHK,1089.0,2.0,43.55,2,3999
Thanisandra,3 BHK,1800.0,3.0,90.0,3,5000
Yelahanka New Town,1 BHK,960.0,2.0,18.0,1,1875
Yelahanka,2 BHK,1036.0,2.0,39.35,2,3798
Marathahalli,3 BHK,1583.0,3.0,105.0,3,6632
7th Phase JP Nagar,4 Bedroom,3300.0,4.0,150.0,4,4545
Padmanabhanagar,2 BHK,1100.0,2.0,65.0,2,5909
other,1 Bedroom,800.0,1.0,100.0,1,12500
Vittasandra,2 BHK,1246.0,2.0,65.0,2,5216
Dodda Nekkundi,2 BHK,850.0,2.0,28.0,2,3294
Raja Rajeshwari Nagar,2 BHK,1196.0,2.0,51.51,2,4306
Thanisandra,3 BHK,1697.0,3.0,114.0,3,6717
TC Palaya,2 Bedroom,1000.0,2.0,64.0,2,6400
HRBR Layout,3 BHK,1567.0,3.0,90.0,3,5743
Anekal,2 Bedroom,1200.0,1.0,36.0,2,3000
Sarjapur  Road,4 Bedroom,3750.0,6.0,290.0,4,7733
other,2 BHK,950.0,2.0,45.0,2,4736
other,2 BHK,1022.0,2.0,52.0,2,5088
Kammasandra,2 BHK,1018.0,2.0,23.45,2,2303
other,4 Bedroom,2000.0,3.0,25.0,4,1250
Bharathi Nagar,2 BHK,1322.0,2.0,68.0,2,5143
Marsur,4 Bedroom,1800.0,4.0,145.0,4,8055
Doddakallasandra,3 BHK,1460.0,2.0,58.39,3,3999
other,2 BHK,990.0,2.0,44.55,2,4500
other,3 BHK,2400.0,3.0,260.0,3,10833
Whitefield,3 BHK,1635.0,2.0,85.0,3,5198
Rajaji Nagar,4 BHK,3526.0,4.0,558.0,4,15825
Padmanabhanagar,4 Bedroom,3400.0,3.0,210.0,4,6176
other,3 BHK,2770.0,3.0,460.0,3,16606
EPIP Zone,4 BHK,3035.0,5.0,200.0,4,6589
other,3 BHK,1533.0,3.0,85.0,3,5544
Akshaya Nagar,3 BHK,1720.0,3.0,134.0,3,7790
Frazer Town,3 BHK,2000.0,3.0,200.0,3,10000
Yeshwanthpur,2 BHK,1185.0,2.0,75.84,2,6400
Varthur Road,2 BHK,900.0,2.0,27.0,2,3000
Attibele,1 BHK,395.0,1.0,10.25,1,2594
Electronic City Phase II,2 BHK,1031.0,2.0,56.35,2,5465
other,6 Bedroom,2400.0,8.0,650.0,6,27083
Thanisandra,4 BHK,2650.0,4.0,240.0,4,9056
Basaveshwara Nagar,5 Bedroom,2400.0,5.0,415.0,5,17291
Uttarahalli,3 BHK,1330.0,2.0,46.55,3,3500
Hennur,2 BHK,1259.0,2.0,57.0,2,4527
Raja Rajeshwari Nagar,3 BHK,1570.0,3.0,88.0,3,5605
Bisuvanahalli,3 BHK,1075.0,2.0,36.0,3,3348
Whitefield,2 BHK,1216.0,2.0,79.0,2,6496
Sarjapur  Road,3 BHK,1428.0,2.0,66.0,3,4621
Begur Road,2 BHK,1160.0,2.0,40.59,2,3499
other,3 Bedroom,600.0,2.0,110.0,3,18333
other,3 BHK,1475.0,3.0,70.0,3,4745
Harlur,3 BHK,1754.0,3.0,135.0,3,7696
Rayasandra,3 BHK,1639.0,3.0,83.0,3,5064
Hennur Road,3 BHK,1804.0,3.0,102.0,3,5654
Whitefield,3 Bedroom,1200.0,3.0,56.2,3,4683
other,2 BHK,1135.0,2.0,55.0,2,4845
Begur Road,2 BHK,1260.0,2.0,46.62,2,3700
other,5 Bedroom,1440.0,5.0,110.0,5,7638
Hennur Road,3 BHK,2160.0,3.0,160.0,3,7407
Rayasandra,7 Bedroom,700.0,6.0,85.0,7,12142
Rachenahalli,2 BHK,1220.0,2.0,59.5,2,4877
Margondanahalli,2 Bedroom,1200.0,2.0,67.0,2,5583
Yelahanka,2 BHK,1300.0,2.0,65.0,2,5000
Cox Town,1 BHK,605.0,1.0,29.0,1,4793
Vijayanagar,6 Bedroom,1230.0,4.0,210.0,6,17073
Bhoganhalli,2 BHK,1444.0,2.0,75.97,2,5261
Amruthahalli,2 BHK,924.0,2.0,45.0,2,4870
Kaikondrahalli,2 BHK,1000.0,2.0,30.0,2,3000
Vishveshwarya Layout,4 Bedroom,600.0,4.0,95.0,4,15833
other,3 Bedroom,1200.0,3.0,225.0,3,18750
Bisuvanahalli,2 BHK,845.0,2.0,29.0,2,3431
Kanakpura Road,2 BHK,700.0,2.0,35.0,2,5000
Kothanur,2 BHK,1187.0,2.0,58.0,2,4886
Kaggadasapura,3 BHK,1625.0,3.0,55.0,3,3384
Devarachikkanahalli,2 Bedroom,1200.0,2.0,83.0,2,6916
other,2 BHK,1452.55,2.0,110.0,2,7572
other,3 BHK,2357.0,3.0,151.0,3,6406
Kaval Byrasandra,2 BHK,1200.0,2.0,49.5,2,4125
Budigere,2 BHK,1139.0,2.0,56.85,2,4991
Basavangudi,1 Bedroom,600.0,1.0,97.5,1,16250
Kanakpura Road,3 BHK,1660.0,3.0,75.0,3,4518
other,2 BHK,1170.0,2.0,105.0,2,8974
Somasundara Palya,2 BHK,1448.0,2.0,68.0,2,4696
Kadugodi,3 BHK,1762.0,3.0,110.0,3,6242
Kaggadasapura,2 BHK,1105.0,2.0,38.65,2,3497
Yelahanka,3 BHK,1825.0,3.0,105.0,3,5753
Bhoganhalli,1 RK,296.0,1.0,22.89,1,7733
other,5 Bedroom,1200.0,4.0,85.0,5,7083
Thanisandra,2 BHK,1185.0,2.0,43.0,2,3628
Electronic City Phase II,3 BHK,1320.0,2.0,38.12,3,2887
Hulimavu,2 BHK,1058.0,2.0,48.0,2,4536
Bellandur,2 BHK,1220.0,2.0,60.0,2,4918
Sarjapur,2 BHK,925.0,2.0,25.0,2,2702
Old Madras Road,2 BHK,1200.0,2.0,72.0,2,6000
Kengeri,3 BHK,1160.0,2.0,40.6,3,3500
Kengeri Satellite Town,3 Bedroom,600.0,4.0,72.0,3,12000
other,3 BHK,1411.0,3.0,91.72,3,6500
Bannerghatta Road,3 BHK,1550.0,3.0,78.0,3,5032
Green Glen Layout,4 BHK,3250.0,4.0,230.0,4,7076
Haralur Road,3 BHK,2475.0,4.0,130.0,3,5252
7th Phase JP Nagar,4 BHK,2503.0,4.0,188.0,4,7510
Marathahalli,4 BHK,2524.0,5.0,190.0,4,7527
Kaggadasapura,2 BHK,1063.0,2.0,42.0,2,3951
Kanakpura Road,2 BHK,1077.0,2.0,37.7,2,3500
other,2 BHK,1200.0,3.0,44.0,2,3666
Hosakerehalli,4 BHK,1500.0,3.0,70.0,4,4666
Jalahalli,2 BHK,1045.0,2.0,76.77,2,7346
Kanakapura,3 BHK,1290.0,2.0,45.15,3,3500
Hoodi,2 BHK,1400.0,2.0,78.0,2,5571
Begur Road,3 BHK,1600.0,3.0,58.14,3,3633
other,3 BHK,3095.0,3.0,375.0,3,12116
Nagavara,3 BHK,1545.0,2.0,58.0,3,3754
Sarjapur  Road,3 BHK,1680.0,3.0,112.0,3,6666
other,2 BHK,1350.0,2.0,67.5,2,5000
Gottigere,2 BHK,950.0,2.0,38.0,2,4000
Thanisandra,2 BHK,1050.0,2.0,45.0,2,4285
Banashankari Stage V,3 BHK,1305.0,2.0,49.0,3,3754
Hosakerehalli,4 Bedroom,800.0,4.0,120.0,4,15000
Kodichikkanahalli,5 BHK,2700.0,7.0,125.0,5,4629
other,3 Bedroom,1100.0,3.0,78.0,3,7090
Thanisandra,6 Bedroom,3500.0,5.0,160.0,6,4571
Begur Road,3 BHK,1634.0,3.0,83.33,3,5099
Prithvi Layout,2 BHK,1352.0,2.0,85.0,2,6286
Balagere,2 BHK,1145.0,2.0,65.0,2,5676
other,2 BHK,1240.0,2.0,43.79,2,3531
7th Phase JP Nagar,3 BHK,2200.0,3.0,190.0,3,8636
Bommasandra Industrial Area,2 BHK,1020.0,2.0,39.0,2,3823
other,4 BHK,3600.0,4.0,365.0,4,10138
Raja Rajeshwari Nagar,2 BHK,1168.0,2.0,61.0,2,5222
Electronic City Phase II,2 BHK,1205.0,2.0,75.0,2,6224
Yeshwanthpur,2 BHK,1195.0,2.0,100.0,2,8368
CV Raman Nagar,2 BHK,1028.0,2.0,49.0,2,4766
other,2 BHK,1175.0,2.0,48.27,2,4108
Pai Layout,3 BHK,1550.0,2.0,90.0,3,5806
Yeshwanthpur,3 BHK,1826.0,3.0,165.0,3,9036
Hoodi,5 Bedroom,2400.0,5.0,160.0,5,6666
Yelahanka,3 BHK,1603.0,3.0,96.0,3,5988
Marathahalli,3 BHK,1350.0,3.0,72.0,3,5333
KR Puram,1 BHK,714.0,1.0,28.0,1,3921
Whitefield,2 BHK,1215.0,2.0,65.0,2,5349
Malleshpalya,3 BHK,1405.0,2.0,77.0,3,5480
KR Puram,2 BHK,1142.0,2.0,51.0,2,4465
other,4 Bedroom,4000.0,4.0,170.0,4,4250
Whitefield,3 BHK,1870.0,3.0,90.0,3,4812
Subramanyapura,2 BHK,1313.0,2.0,73.0,2,5559
other,3 BHK,1945.0,3.0,120.0,3,6169
Thanisandra,3 BHK,1573.0,3.0,95.0,3,6039
Sarjapur  Road,2 BHK,1255.0,2.0,92.0,2,7330
Benson Town,4 BHK,1400.0,3.0,91.15,4,6510
Kannamangala,2 BHK,1235.0,2.0,43.63,2,3532
Sarjapur  Road,3 BHK,1380.0,2.0,52.0,3,3768
Ramamurthy Nagar,7 Bedroom,1500.0,9.0,250.0,7,16666
Vittasandra,2 BHK,1238.0,2.0,67.0,2,5411
other,3 BHK,1465.0,2.0,74.72,3,5100
Thigalarapalya,2 BHK,1418.0,2.0,103.0,2,7263
other,3 BHK,1650.0,3.0,165.0,3,10000
Anekal,1 RK,351.0,1.0,16.0,1,4558
Uttarahalli,2 BHK,1279.0,2.0,48.0,2,3752
Kasavanhalli,3 Bedroom,1870.0,3.0,200.0,3,10695
other,4 Bedroom,1500.0,5.0,230.0,4,15333
Hebbal,3 BHK,2526.0,2.0,250.0,3,9897
other,3 Bedroom,1550.0,2.0,150.0,3,9677
Uttarahalli,3 BHK,1845.0,2.0,68.27,3,3700
other,3 BHK,1360.0,2.0,59.27,3,4358
other,3 BHK,1150.0,2.0,75.0,3,6521
Whitefield,2 BHK,1260.0,2.0,49.5,2,3928
other,3 Bedroom,1200.0,3.0,74.0,3,6166
Dasanapura,2 BHK,965.0,2.0,42.5,2,4404
Hoodi,3 BHK,1350.0,2.0,48.0,3,3555
Bhoganhalli,2 BHK,1447.0,2.0,75.97,2,5250
other,3 BHK,1229.0,3.0,86.03,3,7000
Kudlu,2 BHK,1027.0,2.0,43.0,2,4186
Gottigere,2 BHK,1153.0,2.0,48.5,2,4206
Whitefield,4 BHK,2856.0,5.0,145.5,4,5094
Malleshpalya,2 BHK,1065.0,2.0,50.0,2,4694
other,2 BHK,1030.0,2.0,61.0,2,5922
Mallasandra,2 BHK,1340.0,2.0,61.0,2,4552
Marathahalli,4 BHK,4000.0,5.0,212.0,4,5300
other,1 Bedroom,600.0,1.0,32.0,1,5333
Begur Road,3 BHK,1410.0,2.0,44.42,3,3150
Varthur,2 BHK,1055.0,2.0,45.35,2,4298
Yelahanka,2 BHK,1304.0,2.0,79.0,2,6058
other,4 BHK,2720.0,4.0,485.0,4,17830
Kammasandra,2 BHK,990.0,2.0,31.5,2,3181
7th Phase JP Nagar,2 BHK,1035.0,2.0,39.33,2,3800
Tumkur Road,1 BHK,700.0,1.0,30.09,1,4298
Sonnenahalli,2 BHK,1120.0,2.0,43.0,2,3839
Haralur Road,2 BHK,1300.0,2.0,73.0,2,5615
Tumkur Road,2 BHK,1239.0,2.0,82.0,2,6618
Sarjapur  Road,3 BHK,1403.0,3.0,51.63,3,3679
Hosa Road,2 BHK,1365.0,2.0,92.0,2,6739
other,2 BHK,1165.0,2.0,135.0,2,11587
Subramanyapura,3 BHK,1130.0,2.0,45.19,3,3999
Green Glen Layout,3 BHK,1776.42,3.0,105.0,3,5910
Vidyaranyapura,4 Bedroom,900.0,4.0,175.0,4,19444
Haralur Road,2 BHK,1194.0,2.0,47.0,2,3936
other,2 BHK,1107.0,2.0,33.0,2,2981
Nagavara,3 BHK,2319.0,3.0,180.0,3,7761
other,3 BHK,1520.0,3.0,80.0,3,5263
Kannamangala,3 BHK,1536.0,3.0,89.0,3,5794
other,4 BHK,3000.0,4.0,240.0,4,8000
other,2 BHK,1050.0,2.0,38.0,2,3619
Hoodi,3 BHK,1850.0,3.0,146.0,3,7891
Billekahalli,2 BHK,1112.0,2.0,62.0,2,5575
Marathahalli,3 BHK,1535.0,3.0,72.5,3,4723
Kanakpura Road,2 BHK,1050.0,2.0,35.0,2,3333
Banashankari,4 Bedroom,2400.0,3.0,370.0,4,15416
7th Phase JP Nagar,3 BHK,1250.0,3.0,65.0,3,5200
Kalena Agrahara,2 BHK,1187.0,2.0,52.0,2,4380
9th Phase JP Nagar,2 BHK,1164.0,2.0,56.0,2,4810
Electronic City Phase II,3 BHK,1395.0,2.0,62.78,3,4500
Arekere,3 BHK,2072.0,3.0,108.0,3,5212
Kanakpura Road,3 BHK,1450.0,3.0,60.91,3,4200
Bannerghatta Road,3 BHK,1550.0,3.0,82.0,3,5290
other,2 BHK,1315.0,2.0,90.0,2,6844
Hebbal,4 BHK,3067.0,4.0,230.0,4,7499
other,2 BHK,1210.0,2.0,72.0,2,5950
Bisuvanahalli,3 BHK,1075.0,2.0,47.0,3,4372
other,6 BHK,11338.0,9.0,1000.0,6,8819
other,3 BHK,1215.0,3.0,42.2,3,3473
Kaggadasapura,3 BHK,1470.0,2.0,75.0,3,5102
other,4 Bedroom,30000.0,4.0,2100.0,4,7000
Hosur Road,3 BHK,1590.0,2.0,125.0,3,7861
Kanakpura Road,2 BHK,1050.0,2.0,36.75,2,3500
Raja Rajeshwari Nagar,2 BHK,1306.0,2.0,55.91,2,4281
Malleshwaram,3 BHK,2000.0,3.0,250.0,3,12500
other,3 BHK,1445.0,3.0,64.0,3,4429
Sarjapur  Road,3 BHK,1220.0,3.0,58.0,3,4754
Sarjapur,4 Bedroom,3190.0,3.0,160.0,4,5015
Ramagondanahalli,2 BHK,1215.0,2.0,51.26,2,4218
Budigere,3 BHK,1920.0,2.0,88.0,3,4583
Dodda Nekkundi,2 BHK,1390.0,2.0,60.0,2,4316
Battarahalli,3 Bedroom,800.0,2.0,75.0,3,9375
Nagarbhavi,3 Bedroom,600.0,3.0,100.0,3,16666
Hoodi,2 BHK,1196.0,2.0,71.74,2,5998
Bannerghatta Road,3 BHK,1550.0,2.0,75.0,3,4838
Thigalarapalya,2 BHK,1418.0,2.0,101.0,2,7122
Varthur Road,3 BHK,1655.0,3.0,115.0,3,6948
KR Puram,2 BHK,1115.0,2.0,33.45,2,3000
other,2 BHK,1170.0,2.0,68.0,2,5811
Anekal,2 BHK,1035.0,2.0,38.3,2,3700
Singasandra,3 BHK,1306.0,2.0,58.0,3,4441
Kengeri,2 BHK,750.0,2.0,22.0,2,2933
Thanisandra,3 BHK,1917.0,3.0,122.0,3,6364
Kaggadasapura,3 BHK,1615.0,3.0,58.0,3,3591
Marathahalli,2 BHK,1071.0,2.0,52.25,2,4878
Koramangala,2 BHK,1325.0,2.0,119.0,2,8981
Malleshwaram,7 Bedroom,3000.0,4.0,900.0,7,30000
5th Phase JP Nagar,3 BHK,1400.0,2.0,88.0,3,6285
Tumkur Road,2 BHK,1246.0,2.0,82.0,2,6581
Sarjapur,2 BHK,1215.0,2.0,40.0,2,3292
Munnekollal,2 BHK,1200.0,2.0,49.5,2,4125
other,4 Bedroom,450.0,3.0,53.0,4,11777
Kodigehaali,2 BHK,1200.0,2.0,60.0,2,5000
Kaval Byrasandra,2 BHK,945.0,2.0,50.0,2,5291
Benson Town,4 BHK,4460.0,5.0,650.0,4,14573
Hormavu,2 BHK,1175.0,2.0,52.0,2,4425
Whitefield,2 BHK,920.0,2.0,63.0,2,6847
Hosakerehalli,4 BHK,3024.0,5.0,350.0,4,11574
other,2 Bedroom,1600.0,2.0,110.0,2,6875
Sanjay nagar,4 Bedroom,750.0,2.0,110.0,4,14666
other,4 Bedroom,2400.0,4.0,700.0,4,29166
other,2 BHK,955.0,1.0,34.01,2,3561
Ardendale,3 BHK,1750.0,3.0,100.0,3,5714
Uttarahalli,3 BHK,1425.0,2.0,64.0,3,4491
Basavangudi,3 BHK,2150.0,3.0,265.0,3,12325
Whitefield,2 BHK,1165.0,2.0,40.5,2,3476
Kaggadasapura,2 BHK,1050.0,2.0,48.0,2,4571
other,3 BHK,1370.0,2.0,75.0,3,5474
Electronic City,3 BHK,1374.0,2.0,72.0,3,5240
Harlur,3 BHK,1754.0,3.0,124.0,3,7069
Kothanur,3 BHK,1847.0,3.0,110.0,3,5955
Ambedkar Nagar,3 BHK,1850.0,4.0,139.0,3,7513
Chikka Tirupathi,3 Bedroom,3297.0,3.0,135.0,3,4094
other,2 BHK,1187.0,2.0,45.0,2,3791
Bommasandra Industrial Area,3 BHK,1400.0,2.0,40.44,3,2888
other,3 Bedroom,600.0,2.0,30.0,3,5000
Chamrajpet,2 BHK,1350.0,2.0,180.0,2,13333
Doddaballapur,3 Bedroom,3000.0,2.0,120.0,3,4000
Thanisandra,1 BHK,693.0,1.0,34.3,1,4949
other,3 BHK,1980.0,3.0,224.0,3,11313
KR Puram,2 BHK,1021.0,2.0,48.0,2,4701
Thigalarapalya,2 BHK,1165.0,2.0,95.0,2,8154
Horamavu Agara,3 BHK,1756.0,3.0,92.0,3,5239
Bharathi Nagar,3 BHK,1739.0,3.0,105.0,3,6037
Varthur,2 BHK,1280.0,2.0,65.0,2,5078
Marathahalli,2 BHK,1215.0,2.0,58.86,2,4844
Munnekollal,2 BHK,950.0,2.0,46.5,2,4894
Kengeri,3 BHK,1082.0,2.0,49.0,3,4528
7th Phase JP Nagar,2 BHK,1050.0,2.0,42.0,2,4000
Devanahalli,2 BHK,1080.0,2.0,53.35,2,4939
Arekere,2 BHK,1100.0,2.0,55.0,2,5000
Ardendale,4 Bedroom,3200.0,4.0,205.0,4,6406
Sarjapur  Road,2 BHK,1311.0,2.0,78.0,2,5949
Raja Rajeshwari Nagar,3 BHK,1608.0,3.0,67.98,3,4227
Begur Road,2 BHK,1160.0,2.0,36.0,2,3103
Whitefield,2 BHK,1170.0,2.0,71.0,2,6068
Kengeri,2 BHK,1155.0,2.0,40.6,2,3515
Hebbal,3 BHK,1255.0,2.0,77.68,3,6189
Rayasandra,2 BHK,1065.0,2.0,44.95,2,4220
Neeladri Nagar,1 BHK,674.0,1.0,21.0,1,3115
Ambedkar Nagar,3 BHK,1936.0,4.0,126.0,3,6508
Sarakki Nagar,3 BHK,2289.0,3.0,260.0,3,11358
other,2 BHK,1100.0,2.0,38.45,2,3495
Ramagondanahalli,3 BHK,2257.0,4.0,157.0,3,6956
Nagarbhavi,2 BHK,1055.0,2.0,55.0,2,5213
Raja Rajeshwari Nagar,3 BHK,1530.0,3.0,51.87,3,3390
other,3 BHK,1700.0,3.0,97.0,3,5705
Raja Rajeshwari Nagar,3 BHK,1575.0,2.0,78.0,3,4952
Frazer Town,3 BHK,3500.0,4.0,330.0,3,9428
Rajiv Nagar,2 BHK,1263.0,2.0,93.32,2,7388
KR Puram,3 BHK,1559.0,3.0,69.315,3,4446
Kengeri Satellite Town,2 BHK,1043.0,2.0,65.0,2,6232
Hennur,2 BHK,1255.0,2.0,53.5,2,4262
JP Nagar,2 BHK,1352.5,2.0,93.865,2,6940
Chandapura,1 BHK,590.0,1.0,13.57,1,2300
Yelahanka,3 Bedroom,1550.0,3.0,120.0,3,7741
Sarjapur  Road,3 BHK,1220.0,3.0,56.0,3,4590
other,3 BHK,1900.0,3.0,160.0,3,8421
other,3 BHK,1360.0,3.0,55.0,3,4044
other,2 BHK,1100.0,2.0,75.0,2,6818
8th Phase JP Nagar,3 BHK,1455.0,3.0,73.31,3,5038
KR Puram,2 BHK,1035.0,2.0,40.5,2,3913
Harlur,2 BHK,1152.0,2.0,69.0,2,5989
Bellandur,2 BHK,1299.0,2.0,82.0,2,6312
Sarjapur  Road,3 BHK,1157.0,2.0,75.0,3,6482
Rachenahalli,2 BHK,1050.0,2.0,52.5,2,5000
Pai Layout,2 BHK,1050.0,2.0,40.0,2,3809
Vittasandra,2 BHK,1246.0,2.0,67.4,2,5409
other,3 BHK,1929.0,3.0,115.0,3,5961
Hennur,2 BHK,1285.0,2.0,60.0,2,4669
Margondanahalli,3 Bedroom,1200.0,2.0,65.0,3,5416
other,4 BHK,3161.0,4.0,214.0,4,6770
BTM 2nd Stage,3 BHK,1850.0,3.0,170.0,3,9189
Sompura,3 BHK,1025.0,2.0,37.0,3,3609
Malleshwaram,2 BHK,1124.0,3.0,80.0,2,7117
Balagere,2 BHK,1012.0,2.0,54.59,2,5394
Giri Nagar,7 BHK,4500.0,5.0,250.0,7,5555
8th Phase JP Nagar,2 BHK,871.0,2.0,55.0,2,6314
other,2 BHK,1020.0,2.0,40.0,2,3921
Rachenahalli,3 BHK,1550.0,3.0,73.5,3,4741
other,3 Bedroom,3515.0,3.0,420.0,3,11948
Akshaya Nagar,3 BHK,1410.0,2.0,80.0,3,5673
Chikka Tirupathi,4 Bedroom,3500.0,5.0,150.0,4,4285
Whitefield,2 BHK,1118.0,2.0,75.0,2,6708
Thanisandra,2 BHK,1098.0,2.0,68.0,2,6193
Kanakapura,2 BHK,1130.0,2.0,45.2,2,4000
Sarjapur  Road,2 BHK,1145.0,2.0,75.0,2,6550
Kadugodi,3 Bedroom,950.0,3.0,80.0,3,8421
other,2 BHK,1051.0,2.0,33.0,2,3139
other,7 Bedroom,600.0,7.0,108.0,7,18000
Marathahalli,2 BHK,1102.0,2.0,53.67,2,4870
Jalahalli,2 BHK,1478.0,2.0,125.0,2,8457
Chandapura,1 BHK,645.0,1.0,16.45,1,2550
Hebbal,2 BHK,823.0,2.0,50.63,2,6151
Abbigere,2 BHK,880.0,2.0,35.0,2,3977
KR Puram,4 Bedroom,600.0,5.0,75.0,4,12500
Budigere,3 BHK,1991.0,4.0,100.0,3,5022
TC Palaya,3 BHK,1100.0,3.0,50.0,3,4545
Gubbalala,3 BHK,1745.0,3.0,115.0,3,6590
HAL 2nd Stage,2 BHK,1226.0,3.0,135.0,2,11011
Begur Road,3 Bedroom,680.0,4.0,68.0,3,10000
other,2 BHK,1100.0,2.0,70.0,2,6363
Banashankari Stage VI,4 Bedroom,1200.0,4.0,195.0,4,16250
5th Phase JP Nagar,2 BHK,1450.0,2.0,58.0,2,4000
Singasandra,4 BHK,3366.0,4.0,120.0,4,3565
Sahakara Nagar,2 BHK,1150.0,2.0,54.0,2,4695
Kanakpura Road,3 BHK,1843.0,3.0,95.84,3,5200
Banashankari Stage VI,4 Bedroom,2200.0,4.0,195.0,4,8863
Bommanahalli,2 BHK,1090.0,3.0,44.0,2,4036
Kaggadasapura,2 BHK,1132.0,2.0,42.0,2,3710
Hulimavu,2 BHK,1375.0,2.0,80.0,2,5818
7th Phase JP Nagar,4 Bedroom,1500.0,5.0,220.0,4,14666
Thanisandra,1 BHK,760.0,1.0,50.3,1,6618
Kaggadasapura,2 BHK,1180.0,2.0,65.0,2,5508
Jigani,2 BHK,918.0,2.0,45.0,2,4901
Rachenahalli,5 Bedroom,600.0,5.0,50.0,5,8333
Margondanahalli,2 Bedroom,1200.0,1.0,53.0,2,4416
HSR Layout,2 BHK,1372.0,2.0,61.0,2,4446
other,7 Bedroom,2400.0,7.0,355.0,7,14791
Ardendale,3 BHK,1728.0,3.0,95.0,3,5497
Malleshwaram,3 BHK,2215.0,3.0,275.0,3,12415
BTM 2nd Stage,2 BHK,1200.0,2.0,80.0,2,6666
6th Phase JP Nagar,2 BHK,1280.0,2.0,56.0,2,4375
Hulimavu,4 Bedroom,800.0,4.0,110.0,4,13750
other,3 BHK,1455.0,2.0,56.0,3,3848
Rachenahalli,1 RK,440.0,1.0,28.0,1,6363
Kanakpura Road,3 Bedroom,1500.0,3.0,97.0,3,6466
Whitefield,3 BHK,1800.0,3.0,110.0,3,6111
Mysore Road,3 BHK,1525.0,3.0,96.0,3,6295
Rachenahalli,2 BHK,1204.0,2.0,38.5,2,3197
Old Madras Road,2 BHK,1225.0,2.0,53.55,2,4371
Kalyan nagar,6 Bedroom,800.0,6.0,98.0,6,12250
Bannerghatta Road,3 BHK,1650.0,3.0,122.0,3,7393
Electronic City,3 BHK,880.0,2.0,18.0,3,2045
other,4 Bedroom,4200.0,4.0,255.0,4,6071
other,2 Bedroom,1200.0,2.0,55.0,2,4583
Sarjapur  Road,4 Bedroom,1152.0,4.0,230.0,4,19965
9th Phase JP Nagar,2 BHK,1005.0,2.0,42.0,2,4179
Whitefield,3 BHK,1562.0,3.0,103.0,3,6594
Devanahalli,4 Bedroom,5000.0,5.0,395.0,4,7900
Babusapalaya,6 Bedroom,810.0,5.0,95.0,6,11728
Ramamurthy Nagar,2 BHK,1200.0,2.0,42.0,2,3500
Kasavanhalli,3 BHK,1555.0,3.0,82.5,3,5305
KR Puram,2 Bedroom,1550.0,2.0,71.0,2,4580
Whitefield,2 BHK,1190.0,2.0,67.0,2,5630
Chikkalasandra,2 BHK,1070.0,2.0,50.0,2,4672
Jalahalli East,2 BHK,1020.0,2.0,42.48,2,4164
Yelahanka,3 BHK,1610.0,3.0,115.0,3,7142
other,1 BHK,700.0,1.0,41.0,1,5857
other,3 BHK,1300.0,2.0,49.0,3,3769
Hebbal,2 BHK,1420.0,2.0,123.0,2,8661
Vidyaranyapura,9 Bedroom,1200.0,9.0,100.0,9,8333
Hebbal,3 BHK,1069.0,3.0,65.4,3,6117
Marathahalli,3 BHK,1900.0,3.0,123.0,3,6473
Old Airport Road,2 BHK,1655.0,2.0,97.65,2,5900
Uttarahalli,2 BHK,1155.0,2.0,40.42,2,3499
Sarjapur  Road,3 BHK,1300.0,2.0,70.0,3,5384
Electronic City,2 BHK,1100.0,2.0,45.0,2,4090
Doddathoguru,2 BHK,1030.0,2.0,42.0,2,4077
Vittasandra,2 BHK,2105.0,2.0,70.2,2,3334
Bhoganhalli,3 BHK,1700.0,3.0,126.0,3,7411
Jakkur,3 BHK,1760.0,3.0,139.0,3,7897
Kadugodi,9 Bedroom,6200.0,9.0,200.0,9,3225
other,2 BHK,1400.0,2.0,67.0,2,4785
other,2 BHK,1200.0,2.0,53.0,2,4416
Bannerghatta Road,2 BHK,912.0,2.0,55.0,2,6030
Whitefield,2 BHK,1102.0,2.0,48.0,2,4355
other,3 BHK,1535.0,3.0,55.0,3,3583
Haralur Road,2 BHK,1243.0,2.0,46.0,2,3700
Jigani,2 BHK,918.0,2.0,63.2,2,6884
Begur Road,3 BHK,1500.0,2.0,70.0,3,4666
other,3 Bedroom,3000.0,3.0,250.0,3,8333
Kasavanhalli,3 BHK,1476.0,3.0,89.0,3,6029
Mysore Road,2 BHK,1200.0,2.0,65.0,2,5416
Vijayanagar,1 BHK,606.0,1.0,34.78,1,5739
other,2 BHK,1100.0,2.0,61.0,2,5545
Garudachar Palya,3 BHK,1295.0,2.0,59.5,3,4594
Neeladri Nagar,1 BHK,527.0,1.0,26.0,1,4933
OMBR Layout,2 BHK,1300.0,2.0,80.0,2,6153
Bommanahalli,3 Bedroom,680.0,4.0,68.0,3,10000
other,1 Bedroom,1260.0,1.0,41.0,1,3253
Koramangala,2 BHK,1140.0,2.0,75.2,2,6596
other,1 Bedroom,1500.0,1.0,37.5,1,2500
Whitefield,3 BHK,2321.0,4.0,128.0,3,5514
Sarjapur  Road,2 BHK,1390.0,2.0,62.0,2,4460
Sarjapur  Road,3 BHK,1725.0,3.0,90.0,3,5217
Hebbal,4 BHK,3067.0,4.0,310.0,4,10107
Hennur Road,3 BHK,1186.0,2.0,63.0,3,5311
Kanakpura Road,2 BHK,1230.0,2.0,85.0,2,6910
other,1 BHK,600.0,1.0,17.0,1,2833
Yeshwanthpur,3 BHK,1313.0,3.0,96.0,3,7311
TC Palaya,2 Bedroom,900.0,2.0,55.0,2,6111
Ambalipura,2 BHK,1250.0,2.0,66.25,2,5300
HBR Layout,9 Bedroom,1200.0,6.0,280.0,9,23333
Binny Pete,1 BHK,660.0,1.0,54.0,1,8181
Basavangudi,2 BHK,1036.0,2.0,105.0,2,10135
Koramangala,4 BHK,2700.0,4.0,450.0,4,16666
Old Madras Road,2 BHK,1100.0,2.0,70.0,2,6363
BTM 2nd Stage,4 Bedroom,1500.0,2.0,450.0,4,30000
Singasandra,4 Bedroom,525.0,5.0,56.0,4,10666
Electronics City Phase 1,2 BHK,1015.0,2.0,56.0,2,5517
Gottigere,1 Bedroom,812.0,1.0,26.0,1,3201
other,3 BHK,1360.0,2.0,65.0,3,4779
Nehru Nagar,3 BHK,2167.0,3.0,170.0,3,7844
Raja Rajeshwari Nagar,2 BHK,1200.0,2.0,35.0,2,2916
other,2 BHK,1165.0,2.0,75.0,2,6437
BTM 2nd Stage,2 BHK,1274.0,2.0,70.0,2,5494
Raja Rajeshwari Nagar,2 BHK,1151.0,2.0,39.02,2,3390
Vidyaranyapura,6 Bedroom,1060.0,6.0,110.0,6,10377
Thubarahalli,3 BHK,1626.0,2.0,120.0,3,7380
Banashankari Stage V,2 BHK,1035.0,2.0,35.77,2,3456
other,3 Bedroom,1272.0,4.0,75.0,3,5896
other,2 BHK,965.0,2.0,45.0,2,4663
other,5 Bedroom,1600.0,5.0,140.0,5,8750
Kothannur,2 BHK,1070.0,2.0,42.79,2,3999
Raja Rajeshwari Nagar,3 BHK,1570.0,2.0,65.0,3,4140
other,2 BHK,780.0,2.0,25.0,2,3205
other,3 BHK,1000.0,2.0,65.0,3,6500
9th Phase JP Nagar,4 Bedroom,1200.0,3.0,188.0,4,15666
Tindlu,7 Bedroom,7500.0,7.0,400.0,7,5333
5th Phase JP Nagar,3 BHK,1700.0,2.0,100.0,3,5882
Haralur Road,3 BHK,1850.0,3.0,110.0,3,5945
Kereguddadahalli,2 BHK,904.0,2.0,30.0,2,3318
other,2 BHK,1121.0,2.0,43.72,2,3900
7th Phase JP Nagar,3 BHK,1385.0,2.0,58.17,3,4200
Sahakara Nagar,2 BHK,1270.0,2.0,69.85,2,5499
Kanakpura Road,3 BHK,1450.0,3.0,56.0,3,3862
other,3 Bedroom,2400.0,4.0,325.0,3,13541
Mahadevpura,1 BHK,730.0,1.0,35.0,1,4794
Raja Rajeshwari Nagar,3 BHK,1580.0,3.0,53.56,3,3389
Thanisandra,2 BHK,1111.0,2.0,50.0,2,4500
Kengeri,4 Bedroom,1125.0,2.0,88.0,4,7822
Varthur,2 BHK,977.0,2.0,36.0,2,3684
other,2 BHK,674.0,1.0,19.9,2,2952
Nagarbhavi,3 Bedroom,600.0,3.0,83.0,3,13833
Sarjapur,4 Bedroom,2540.0,4.0,115.0,4,4527
Jakkur,2 BHK,1290.0,2.0,100.0,2,7751
Kumaraswami Layout,3 Bedroom,600.0,3.0,98.0,3,16333
Abbigere,2 BHK,795.0,2.0,32.54,2,4093
Kanakpura Road,2 BHK,700.0,1.0,35.0,2,5000
other,2 BHK,1463.0,1.0,142.0,2,9706
Vijayanagar,3 Bedroom,1000.0,3.0,140.0,3,14000
other,3 BHK,1306.0,2.0,58.0,3,4441
other,3 BHK,2401.0,3.0,185.0,3,7705
Bommenahalli,4 Bedroom,1670.0,3.0,135.0,4,8083
other,3 BHK,1875.0,2.0,235.0,3,12533
Marsur,3 Bedroom,1200.0,3.0,90.0,3,7500
Dodda Nekkundi,3 BHK,1804.0,3.0,121.0,3,6707
1st Phase JP Nagar,3 BHK,2065.0,4.0,210.0,3,10169
Hoskote,3 BHK,1225.0,3.0,40.0,3,3265
Thanisandra,2 BHK,1185.5,2.0,58.68,2,4949
Bellandur,2 BHK,900.0,2.0,56.0,2,6222
Binny Pete,2 BHK,1350.0,2.0,91.0,2,6740
Kaggalipura,1 BHK,700.0,1.0,36.0,1,5142
Mysore Road,2 BHK,1005.0,2.0,35.175,2,3499
Poorna Pragna Layout,2 BHK,1160.0,2.0,46.39,2,3999
Hennur Road,3 BHK,1680.0,3.0,99.27,3,5908
LB Shastri Nagar,2 BHK,1043.0,2.0,55.0,2,5273
Hoskote,4 Bedroom,1200.0,4.0,120.0,4,10000
Hebbal,2 BHK,1040.0,2.0,46.0,2,4423
Tindlu,3 BHK,1357.0,2.0,65.0,3,4789
other,7 Bedroom,1600.0,7.0,104.0,7,6500
other,2 BHK,1225.0,2.0,47.0,2,3836
other,9 Bedroom,1500.0,6.0,80.0,9,5333
Yelachenahalli,2 BHK,1130.0,2.0,40.0,2,3539
Basavangudi,4 Bedroom,2500.0,4.0,150.0,4,6000
Sultan Palaya,3 BHK,1700.0,3.0,120.0,3,7058
Doddathoguru,3 BHK,1549.0,3.0,65.0,3,4196
Yeshwanthpur,2 BHK,1400.0,2.0,100.0,2,7142
Dodda Nekkundi,2 BHK,1080.0,2.0,50.0,2,4629
Ulsoor,2 BHK,1180.0,2.0,100.0,2,8474
other,7 Bedroom,780.0,7.0,150.0,7,19230
Domlur,3 BHK,1800.0,3.0,150.0,3,8333
Haralur Road,2 BHK,1225.0,2.0,67.38,2,5500
other,1 BHK,890.0,1.0,75.0,1,8426
Abbigere,2 BHK,1005.0,2.0,39.59,2,3939
Whitefield,2 BHK,1320.0,2.0,95.0,2,7196
Anandapura,4 BHK,1475.0,2.0,50.0,4,3389
other,2 BHK,1110.0,2.0,51.0,2,4594
other,2 Bedroom,1300.0,2.0,54.0,2,4153
other,5 Bedroom,600.0,3.0,86.0,5,14333
Electronic City,2 BHK,1128.0,2.0,65.35,2,5793
Jakkur,2 BHK,1452.19,2.0,100.0,2,6886
other,3 Bedroom,2400.0,3.0,290.0,3,12083
Banashankari,3 BHK,2600.0,4.0,135.0,3,5192
Sarakki Nagar,3 BHK,2500.0,4.0,325.0,3,13000
Sanjay nagar,2 BHK,630.0,1.0,35.0,2,5555
Hosa Road,2 BHK,1243.0,2.0,48.5,2,3901
other,2 BHK,960.0,2.0,32.0,2,3333
other,3 BHK,1690.0,3.0,100.0,3,5917
Whitefield,2 BHK,735.0,2.0,35.0,2,4761
Banaswadi,2 BHK,1365.0,2.0,73.0,2,5347
Rajaji Nagar,5 BHK,2849.0,5.0,300.0,5,10530
Hebbal,2 BHK,1420.0,2.0,99.39,2,6999
Somasundara Palya,3 BHK,1570.0,3.0,68.0,3,4331
Bisuvanahalli,2 BHK,850.0,1.0,32.0,2,3764
7th Phase JP Nagar,3 BHK,1575.0,2.0,110.0,3,6984
Thigalarapalya,2 BHK,1418.0,2.0,104.0,2,7334
Uttarahalli,2 BHK,1050.0,2.0,42.0,2,4000
Hennur Road,2 BHK,1065.0,2.0,42.6,2,4000
Ulsoor,3 BHK,1340.0,3.0,90.0,3,6716
Yelahanka New Town,3 BHK,1610.0,3.0,92.0,3,5714
Electronic City Phase II,2 BHK,1160.0,2.0,33.5,2,2887
Kudlu,2 BHK,1092.0,2.0,44.0,2,4029
Parappana Agrahara,2 BHK,1194.0,2.0,47.0,2,3936
Yelahanka,3 BHK,1455.0,3.0,61.5,3,4226
Abbigere,1 BHK,734.0,1.0,19.82,1,2700
other,2 BHK,799.0,2.0,38.0,2,4755
Whitefield,3 BHK,1896.0,3.0,75.0,3,3955
Ardendale,2 BHK,1100.0,2.0,48.0,2,4363
Bellandur,2 BHK,1350.0,2.0,79.0,2,5851
other,3 BHK,1355.0,3.0,61.0,3,4501
Electronic City,2 BHK,1128.0,2.0,65.0,2,5762
other,2 Bedroom,1350.0,2.0,90.0,2,6666
Cunningham Road,3 BHK,3875.0,3.0,864.0,3,22296
Kasavanhalli,3 BHK,1493.0,3.0,82.0,3,5492
8th Phase JP Nagar,2 BHK,1098.0,2.0,43.91,2,3999
Banashankari,3 BHK,1900.0,2.0,170.0,3,8947
other,3 BHK,1460.0,3.0,88.0,3,6027
Frazer Town,3 BHK,2350.0,4.0,285.0,3,12127
Horamavu Banaswadi,5 Bedroom,1600.0,5.0,140.0,5,8750
Varthur,2 BHK,1091.0,2.0,32.0,2,2933
Yelahanka,2 BHK,1180.0,2.0,42.0,2,3559
Kanakpura Road,2 BHK,1366.0,3.0,76.85,2,5625
other,3 BHK,1685.0,3.0,95.0,3,5637
Koramangala,2 BHK,1260.0,2.0,100.0,2,7936
Whitefield,3 Bedroom,3010.0,3.0,200.0,3,6644
Ramamurthy Nagar,3 Bedroom,1330.0,4.0,140.0,3,10526
Mallasandra,2 Bedroom,1200.0,2.0,130.0,2,10833
Whitefield,2 BHK,1346.0,2.0,96.17,2,7144
Chandapura,1 BHK,630.0,1.0,13.86,1,2200
7th Phase JP Nagar,3 BHK,1850.0,3.0,140.0,3,7567
Frazer Town,5 Bedroom,2350.0,2.0,423.0,5,18000
Electronic City,2 BHK,755.0,1.0,34.0,2,4503
other,5 Bedroom,800.0,5.0,98.0,5,12250
Parappana Agrahara,2 BHK,1194.0,2.0,46.0,2,3852
Whitefield,3 BHK,3850.0,5.0,316.0,3,8207
Hebbal,4 BHK,4235.0,4.0,365.0,4,8618
other,1 Bedroom,800.0,1.0,52.0,1,6500
Rachenahalli,3 BHK,2250.0,3.0,195.0,3,8666
Vijayanagar,2 BHK,1352.0,2.0,68.0,2,5029
Thanisandra,3 BHK,2002.0,3.0,112.0,3,5594
other,4 Bedroom,4000.0,5.0,280.0,4,7000
Kaval Byrasandra,2 BHK,1060.0,2.0,45.0,2,4245
Kanakpura Road,3 BHK,1419.59,2.0,59.0,3,4156
Raja Rajeshwari Nagar,2 BHK,1196.0,2.0,51.15,2,4276
Electronic City,3 BHK,1220.0,2.0,35.24,3,2888
Electronics City Phase 1,3 BHK,1595.0,3.0,75.0,3,4702
Banashankari,2 BHK,1260.0,2.0,75.0,2,5952
Hebbal,2 BHK,1100.0,2.0,47.0,2,4272
Frazer Town,2 BHK,1550.0,2.0,140.0,2,9032
ISRO Layout,2 BHK,1180.0,2.0,52.0,2,4406
Electronic City,4 Bedroom,4000.0,5.0,150.0,4,3750
Akshaya Nagar,3 BHK,1662.0,3.0,85.0,3,5114
Hennur,2 BHK,1285.0,2.0,60.0,2,4669
Akshaya Nagar,4 Bedroom,1080.0,4.0,75.0,4,6944
Ambedkar Nagar,3 BHK,1862.0,3.0,119.0,3,6390
Raja Rajeshwari Nagar,2 BHK,1419.0,2.0,48.0,2,3382
Whitefield,3 BHK,1410.0,3.0,43.71,3,3100
Raja Rajeshwari Nagar,3 BHK,1560.0,2.0,62.4,3,4000
other,5 Bedroom,1100.0,4.0,150.0,5,13636
Thanisandra,3 BHK,1634.0,3.0,78.0,3,4773
Ambedkar Nagar,5 Bedroom,5530.0,5.0,627.0,5,11338
Whitefield,3 BHK,1150.0,2.0,44.0,3,3826
other,2 BHK,1100.0,2.0,53.0,2,4818
Yelahanka,3 BHK,1616.0,3.0,65.0,3,4022
Nagavarapalya,2 BHK,1392.0,2.0,99.0,2,7112
Jakkur,3 BHK,1798.0,3.0,120.0,3,6674
Babusapalaya,3 BHK,1675.0,3.0,75.0,3,4477
other,2 Bedroom,792.0,2.0,70.0,2,8838
other,5 BHK,2000.0,4.0,103.0,5,5150
other,7 Bedroom,600.0,7.0,89.0,7,14833
Hebbal,3 BHK,1700.0,3.0,124.0,3,7294
Hulimavu,2 BHK,1248.0,2.0,69.0,2,5528
other,2 BHK,1147.0,2.0,54.0,2,4707
Sultan Palaya,2 BHK,1100.0,2.0,40.0,2,3636
Vijayanagar,3 BHK,1527.0,3.0,115.0,3,7531
other,5 Bedroom,2400.0,5.0,235.0,5,9791
Kanakpura Road,1 BHK,525.0,1.0,25.0,1,4761
Kaggadasapura,2 BHK,1100.0,2.0,58.0,2,5272
Seegehalli,2 BHK,1118.0,2.0,37.5,2,3354
Kanakapura,3 BHK,1476.0,3.0,75.0,3,5081
other,3 BHK,2100.0,3.0,75.0,3,3571
other,3 BHK,1215.0,2.0,46.5,3,3827
Ambalipura,2 BHK,1332.0,2.0,79.0,2,5930
Hennur Road,2 BHK,1350.0,2.0,49.0,2,3629
Kothannur,4 Bedroom,600.0,4.0,81.0,4,13500
Haralur Road,2 BHK,1243.0,2.0,46.0,2,3700
NGR Layout,2 BHK,1105.0,2.0,49.73,2,4500
Padmanabhanagar,3 Bedroom,1500.0,2.0,200.0,3,13333
Kasavanhalli,2 BHK,1200.0,2.0,50.0,2,4166
Ramagondanahalli,2 BHK,1251.0,2.0,47.4,2,3788
Kambipura,2 BHK,883.0,2.0,45.0,2,5096
Pai Layout,3 BHK,1400.0,2.0,67.0,3,4785
Rajaji Nagar,3 BHK,1800.0,3.0,260.0,3,14444
Hegde Nagar,3 BHK,1703.0,3.0,113.0,3,6635
NRI Layout,5 BHK,2500.0,5.0,130.0,5,5200
Varthur,3 BHK,1250.0,3.0,71.0,3,5680
Jigani,2 BHK,933.0,2.0,43.0,2,4608
other,1 Bedroom,461.82,1.0,31.7,1,6864
Brookefield,2 BHK,1262.0,2.0,75.0,2,5942
Yelahanka,2 BHK,915.0,2.0,48.0,2,5245
Panathur,3 BHK,1370.0,2.0,61.5,3,4489
Malleshpalya,2 BHK,1210.0,2.0,55.0,2,4545
Kodihalli,3 BHK,2392.0,4.0,260.0,3,10869
other,3 Bedroom,1500.0,2.0,85.0,3,5666
other,2 BHK,1065.0,2.0,50.0,2,4694
Rajaji Nagar,4 BHK,3100.0,4.0,570.0,4,18387
Ramamurthy Nagar,3 BHK,1515.0,3.0,65.0,3,4290
Sarjapura - Attibele Road,4 Bedroom,1800.0,4.0,110.0,4,6111
Thanisandra,3 BHK,1445.0,2.0,40.31,3,2789
Uttarahalli,3 BHK,1300.0,2.0,55.0,3,4230
Vijayanagar,2 BHK,989.0,2.0,60.0,2,6066
Jakkur,2 BHK,1300.0,2.0,85.0,2,6538
Chandapura,3 BHK,1185.0,2.0,30.22,3,2550
Battarahalli,5 Bedroom,1350.0,3.0,110.0,5,8148
Marathahalli,2 BHK,1045.0,2.0,60.0,2,5741
other,2 BHK,1205.0,2.0,42.18,2,3500
other,2 BHK,1260.0,2.0,75.0,2,5952
other,5 Bedroom,1320.0,5.0,199.0,5,15075
Ulsoor,3 BHK,1500.0,2.0,125.0,3,8333
Bommanahalli,3 BHK,1250.0,2.0,42.5,3,3400
other,2 BHK,1199.0,2.0,50.0,2,4170
Hennur Road,2 BHK,1385.0,2.0,83.09,2,5999
other,6 Bedroom,1222.0,5.0,250.0,6,20458
Lakshminarayana Pura,2 BHK,1149.0,2.0,75.0,2,6527
Sarjapur  Road,3 BHK,2180.0,3.0,170.0,3,7798
TC Palaya,8 Bedroom,1200.0,7.0,110.0,8,9166
other,10 BHK,12000.0,12.0,525.0,10,4375
other,5 Bedroom,1200.0,5.0,195.0,5,16250
Hulimavu,3 BHK,1818.0,3.0,135.0,3,7425
other,2 BHK,1070.0,2.0,33.0,2,3084
Rajaji Nagar,3 BHK,1640.0,3.0,241.0,3,14695
Nagarbhavi,3 BHK,1350.0,2.0,54.6,3,4044
Chandapura,2 BHK,630.0,1.0,27.0,2,4285
Hoodi,2 BHK,1430.0,2.0,113.0,2,7902
Nagavara,3 BHK,2430.0,4.0,180.0,3,7407
other,6 Bedroom,1500.0,7.0,220.0,6,14666
Sarjapur  Road,2 BHK,1084.0,2.0,55.0,2,5073
Hennur Road,3 BHK,1186.0,2.0,56.0,3,4721
Kengeri,2 BHK,1255.0,2.0,50.0,2,3984
Haralur Road,4 BHK,3400.0,5.0,240.0,4,7058
Hosa Road,3 BHK,1541.0,3.0,69.84,3,4532
Iblur Village,4 BHK,3596.0,5.0,282.0,4,7842
Raja Rajeshwari Nagar,2 BHK,1295.0,2.0,70.0,2,5405
Kasavanhalli,3 Bedroom,1000.0,4.0,110.0,3,11000
other,3 BHK,2820.0,3.0,285.0,3,10106
7th Phase JP Nagar,3 BHK,1370.0,2.0,54.79,3,3999
Electronic City,2 BHK,1100.0,2.0,40.0,2,3636
Hebbal,4 BHK,4772.0,6.0,510.0,4,10687
Rajaji Nagar,2 BHK,1370.0,2.0,170.0,2,12408
7th Phase JP Nagar,3 BHK,1370.0,2.0,54.79,3,3999
JP Nagar,2 BHK,1200.0,2.0,62.0,2,5166
Gunjur,4 Bedroom,2000.0,3.0,95.0,4,4750
9th Phase JP Nagar,4 BHK,3500.0,4.0,300.0,4,8571
2nd Stage Nagarbhavi,5 Bedroom,1200.0,4.0,240.0,5,20000
Kengeri,4 Bedroom,2400.0,5.0,253.0,4,10541
Sarjapur  Road,3 BHK,1220.0,3.0,62.0,3,5081
other,3 Bedroom,1950.0,3.0,130.0,3,6666
Somasundara Palya,2 BHK,1178.0,2.0,73.0,2,6196
Dasanapura,3 BHK,1286.0,3.0,61.44,3,4777
Begur Road,2 BHK,1160.0,2.0,42.0,2,3620
Jigani,2 BHK,918.0,2.0,55.0,2,5991
Electronic City,3 BHK,1563.0,3.0,91.84,3,5875
other,2 BHK,1326.0,2.0,62.17,2,4688
other,3 BHK,1350.0,2.0,140.0,3,10370
Hebbal,4 BHK,3960.0,5.0,386.0,4,9747
Begur Road,2 BHK,1240.0,2.0,39.06,2,3150
Kaggadasapura,2 BHK,1325.0,2.0,50.0,2,3773
Sonnenahalli,2 BHK,1100.0,2.0,44.0,2,4000
Cunningham Road,3 BHK,3875.0,3.0,800.0,3,20645
Gollarapalya Hosahalli,3 BHK,1605.0,3.0,50.0,3,3115
Hennur Road,5 BHK,5600.0,5.0,515.0,5,9196
Somasundara Palya,2 BHK,1260.0,2.0,60.0,2,4761
Hoodi,3 BHK,1553.0,3.0,83.09,3,5350
Whitefield,2 BHK,1280.0,2.0,52.0,2,4062
Hebbal,3 BHK,1550.0,2.0,75.0,3,4838
Arekere,3 BHK,1445.0,2.0,125.0,3,8650
Murugeshpalya,3 BHK,1550.0,3.0,74.0,3,4774
Electronics City Phase 1,3 BHK,1515.0,2.0,80.91,3,5340
Hosa Road,3 BHK,1332.0,2.0,86.12,3,6465
Electronic City,3 BHK,1652.0,3.0,70.0,3,4237
Hegde Nagar,3 BHK,1570.0,3.0,105.0,3,6687
Sarjapur  Road,4 Bedroom,5400.0,4.0,725.0,4,13425
Sector 2 HSR Layout,1 BHK,794.0,1.0,60.0,1,7556
Seegehalli,3 Bedroom,1200.0,2.0,62.0,3,5166
7th Phase JP Nagar,4 Bedroom,3600.0,3.0,410.0,4,11388
Hosakerehalli,3 Bedroom,1200.0,3.0,175.0,3,14583
JP Nagar,4 BHK,1650.0,3.0,220.0,4,13333
other,4 Bedroom,600.0,2.0,85.0,4,14166
HRBR Layout,3 BHK,2800.0,3.0,250.0,3,8928
Sarjapur  Road,3 BHK,1750.0,3.0,110.0,3,6285
Hebbal,2 BHK,1053.0,2.0,48.0,2,4558
Kanakpura Road,3 BHK,1100.0,3.0,53.0,3,4818
Kadubeesanahalli,3 BHK,1665.0,3.0,70.0,3,4204
other,4 Bedroom,4051.0,4.0,345.0,4,8516
Malleshwaram,4 Bedroom,5500.0,5.0,1500.0,4,27272
Bommasandra,3 BHK,1447.0,3.0,80.0,3,5528
Balagere,2 BHK,1012.0,2.0,75.0,2,7411
Malleshpalya,3 BHK,1900.0,3.0,60.0,3,3157
Narayanapura,2 BHK,1308.0,2.0,92.66,2,7084
Green Glen Layout,4 BHK,3250.0,4.0,230.0,4,7076
Hormavu,1 BHK,1351.0,2.0,53.0,1,3923
Electronic City,2 BHK,1000.0,2.0,22.0,2,2200
other,4 Bedroom,1200.0,4.0,350.0,4,29166
Gunjur,2 BHK,1140.0,2.0,49.11,2,4307
Jakkur,4 BHK,3181.0,4.0,260.5,4,8189
Uttarahalli,3 BHK,1250.0,2.0,55.0,3,4400
Kanakapura,3 BHK,1290.0,3.0,45.15,3,3500
Ambedkar Nagar,2 BHK,1424.0,2.0,90.0,2,6320
Garudachar Palya,3 BHK,1320.0,2.0,60.6,3,4590
Akshaya Nagar,3 BHK,1662.0,3.0,85.5,3,5144
other,5 BHK,8321.0,5.0,2700.0,5,32448
Begur Road,2 BHK,1200.0,2.0,45.0,2,3750
Koramangala,3 BHK,1975.0,3.0,210.0,3,10632
Garudachar Palya,3 BHK,1295.0,2.0,59.5,3,4594
Battarahalli,3 BHK,1777.0,3.0,100.0,3,5627
Kadugodi,3 BHK,1900.0,3.0,108.0,3,5684
Anandapura,3 BHK,1576.0,3.0,59.1,3,3750
Sompura,2 BHK,1020.0,2.0,41.0,2,4019
other,3 BHK,2100.0,3.0,170.0,3,8095
Raja Rajeshwari Nagar,1 BHK,648.0,1.0,31.75,1,4899
Neeladri Nagar,3 BHK,1167.0,2.0,38.0,3,3256
other,7 Bedroom,1000.0,7.0,160.0,7,16000
Hoodi,2 BHK,1220.0,2.0,50.89,2,4171
Kogilu,3 BHK,2350.0,5.0,327.0,3,13914
Vittasandra,2 BHK,1259.0,2.0,67.4,2,5353
Gottigere,4 Bedroom,2000.0,4.0,82.0,4,4100
8th Phase JP Nagar,3 BHK,1800.0,3.0,80.0,3,4444
other,3 BHK,1991.0,3.0,160.0,3,8036
Bannerghatta,3 BHK,1665.0,3.0,110.0,3,6606
other,2 BHK,1234.0,2.0,61.0,2,4943
Mysore Road,3 BHK,1340.0,2.0,64.31,3,4799
Banashankari,2 BHK,1170.0,2.0,46.79,2,3999
Electronics City Phase 1,2 BHK,995.0,2.0,46.0,2,4623
Old Airport Road,2 BHK,1075.0,2.0,60.0,2,5581
Electronic City,2 BHK,1025.0,2.0,54.5,2,5317
Rachenahalli,3 BHK,2721.0,3.0,113.0,3,4152
Thanisandra,3 BHK,2000.0,3.0,99.0,3,4950
Whitefield,4 BHK,2500.0,4.0,155.0,4,6200
Mallasandra,2 BHK,1325.0,2.0,70.0,2,5283
Kadugodi,2 BHK,1314.0,2.0,68.2,2,5190
Kanakpura Road,3 BHK,2265.0,3.0,136.0,3,6004
7th Phase JP Nagar,7 Bedroom,4800.0,7.0,170.0,7,3541
Yelahanka,2 BHK,1026.0,2.0,41.0,2,3996
other,3 BHK,1200.0,3.0,66.0,3,5500
other,3 BHK,2220.0,4.0,65.0,3,2927
KR Puram,2 BHK,1192.5,2.0,51.22,2,4295
Hebbal,2 BHK,1420.0,2.0,99.39,2,6999
Sarjapur  Road,3 BHK,1888.0,4.0,95.0,3,5031
JP Nagar,2 BHK,1300.0,2.0,90.87,2,6990
Rajaji Nagar,2 BHK,1216.0,2.0,97.16,2,7990
other,4 Bedroom,3000.0,4.0,342.0,4,11400
Begur,3 BHK,1559.0,3.0,63.0,3,4041
Harlur,2 BHK,1290.0,2.0,87.0,2,6744
other,6 Bedroom,2400.0,6.0,245.0,6,10208
Whitefield,1 BHK,905.0,1.0,55.0,1,6077
other,2 BHK,1069.0,2.0,55.0,2,5144
Sarjapura - Attibele Road,2 BHK,829.0,2.0,22.8,2,2750
Koramangala,2 BHK,1900.0,2.0,200.0,2,10526
Sarjapur  Road,3 BHK,1300.0,3.0,60.0,3,4615
Channasandra,2 BHK,1115.0,2.0,33.45,2,3000
Rajaji Nagar,3 BHK,2409.0,3.0,360.0,3,14943
Ambedkar Nagar,3 BHK,2150.0,4.0,125.0,3,5813
Marathahalli,3 BHK,1305.0,2.0,69.0,3,5287
Kanakpura Road,2 BHK,1252.0,2.0,56.32,2,4498
Hosakerehalli,5 Bedroom,2400.0,5.0,250.0,5,10416
Whitefield,2 Bedroom,1200.0,2.0,46.13,2,3844
Koramangala,3 BHK,1750.0,3.0,130.0,3,7428
Electronic City Phase II,2 BHK,545.0,1.0,35.0,2,6422
other,2 Bedroom,1375.0,2.0,105.0,2,7636
Subramanyapura,2 BHK,1313.0,2.0,66.5,2,5064
Attibele,4 Bedroom,3640.0,4.0,275.0,4,7554
other,3 BHK,1693.0,3.0,108.0,3,6379
other,2 BHK,1160.0,2.0,41.75,2,3599
other,3 Bedroom,1200.0,3.0,200.0,3,16666
Subramanyapura,3 BHK,1255.0,2.0,79.0,3,6294
KR Puram,4 Bedroom,1000.0,4.0,75.0,4,7500
other,2 BHK,900.0,2.0,45.0,2,5000
Yelahanka New Town,3 BHK,1700.0,3.0,90.0,3,5294
Indira Nagar,3 BHK,1875.0,3.0,180.0,3,9600
Uttarahalli,3 BHK,1700.0,3.0,88.0,3,5176
Harlur,3 BHK,1749.0,3.0,115.0,3,6575
Dasanapura,2 Bedroom,1500.0,3.0,65.0,2,4333
other,2 BHK,1175.0,2.0,48.0,2,4085
Akshaya Nagar,2 BHK,1280.0,2.0,60.0,2,4687
other,2 BHK,1200.0,2.0,50.0,2,4166
Ramagondanahalli,3 BHK,1610.0,2.0,115.0,3,7142
Electronic City,2 BHK,1070.0,2.0,46.0,2,4299
TC Palaya,2 Bedroom,1200.0,2.0,66.0,2,5500
9th Phase JP Nagar,2 BHK,1127.0,2.0,50.0,2,4436
Indira Nagar,2 BHK,1210.0,2.0,102.0,2,8429
other,2 BHK,600.0,2.0,23.0,2,3833
Thigalarapalya,4 BHK,4190.0,4.0,380.0,4,9069
other,2 BHK,1364.0,2.0,70.0,2,5131
Whitefield,4 BHK,2268.0,3.0,146.0,4,6437
other,3 BHK,1150.0,2.0,77.5,3,6739
other,2 BHK,1146.0,2.0,39.5,2,3446
Magadi Road,2 BHK,1345.0,2.0,55.0,2,4089
other,2 BHK,1110.0,2.0,41.0,2,3693
Kengeri,2 BHK,750.0,2.0,36.0,2,4800
other,2 BHK,1156.0,2.0,49.5,2,4282
Pattandur Agrahara,2 BHK,1247.0,2.0,59.8,2,4795
other,2 BHK,1058.0,2.0,60.0,2,5671
Kaggalipura,2 BHK,950.0,2.0,48.0,2,5052
Vasanthapura,2 BHK,420.0,2.0,35.0,2,8333
other,4 Bedroom,600.0,2.0,90.0,4,15000
Vijayanagar,3 BHK,1520.0,2.0,103.0,3,6776
Marathahalli,2 BHK,1215.0,2.0,58.86,2,4844
other,1 BHK,900.0,1.0,30.0,1,3333
Hennur Road,3 BHK,2350.0,3.0,174.0,3,7404
Thanisandra,2 BHK,1050.0,2.0,46.0,2,4380
other,3 BHK,1672.0,3.0,150.0,3,8971
Laggere,5 Bedroom,2800.0,5.0,125.0,5,4464
other,3 BHK,1500.0,2.0,65.0,3,4333
other,2 BHK,1400.0,2.0,65.0,2,4642
8th Phase JP Nagar,4 Bedroom,2700.0,4.0,130.0,4,4814
Bommasandra,2 BHK,920.0,2.0,37.46,2,4071
Sarjapur  Road,3 Bedroom,1200.0,3.0,70.0,3,5833
other,3 BHK,1250.0,3.0,139.0,3,11120
Hosakerehalli,3 BHK,1817.0,4.0,153.0,3,8420
Cooke Town,5 Bedroom,2043.0,6.0,375.0,5,18355
HSR Layout,2 BHK,1140.0,2.0,46.0,2,4035
Yelahanka,4 Bedroom,4346.0,4.0,450.0,4,10354
Haralur Road,2 BHK,1464.0,2.0,56.0,2,3825
other,2 BHK,1468.0,3.0,130.0,2,8855
Electronics City Phase 1,3 BHK,1555.0,2.0,82.0,3,5273
Munnekollal,6 Bedroom,1200.0,4.0,110.0,6,9166
Whitefield,2 BHK,1150.0,2.0,55.0,2,4782
Kadubeesanahalli,2 BHK,970.0,2.0,35.0,2,3608
Banashankari,4 BHK,2500.0,4.0,95.0,4,3800
other,2 BHK,1166.0,2.0,53.0,2,4545
Konanakunte,3 Bedroom,2779.0,3.0,317.0,3,11406
Electronic City,3 BHK,1350.0,3.0,58.0,3,4296
other,2 BHK,900.0,2.0,46.0,2,5111
7th Phase JP Nagar,3 BHK,1300.0,2.0,52.0,3,4000
other,3 BHK,1190.0,3.0,55.0,3,4621
Kodichikkanahalli,3 BHK,1495.0,2.0,75.0,3,5016
Indira Nagar,2 BHK,1224.0,2.0,105.0,2,8578
Banashankari Stage III,3 BHK,1350.0,2.0,68.0,3,5037
Green Glen Layout,2 BHK,940.0,2.0,56.8,2,6042
Raja Rajeshwari Nagar,2 BHK,1100.0,2.0,55.0,2,5000
Kaikondrahalli,2 BHK,1300.0,2.0,61.0,2,4692
other,3 BHK,1480.0,2.0,145.0,3,9797
Hoodi,2 BHK,1105.0,2.0,46.37,2,4196
Sarjapur  Road,3 Bedroom,2600.0,3.0,200.0,3,7692
Yelahanka,3 BHK,1517.0,3.0,76.3,3,5029
Horamavu Banaswadi,6 Bedroom,3800.0,8.0,200.0,6,5263
Jalahalli,3 BHK,1615.0,3.0,89.75,3,5557
other,3 BHK,1500.0,3.0,65.0,3,4333
Mysore Road,3 BHK,1082.0,3.0,50.5,3,4667
Marathahalli,2 BHK,1365.0,2.0,73.0,2,5347
Electronic City,2 BHK,1000.0,2.0,25.0,2,2500
CV Raman Nagar,2 BHK,1392.0,2.0,98.0,2,7040
Kodigehalli,2 BHK,1086.0,2.0,33.66,2,3099
Hennur Road,3 BHK,1586.0,3.0,93.94,3,5923
Hennur,2 BHK,1100.0,2.0,60.0,2,5454
Electronic City Phase II,2 BHK,1065.0,2.0,30.75,2,2887
TC Palaya,3 Bedroom,1500.0,2.0,120.0,3,8000
7th Phase JP Nagar,3 BHK,1575.0,2.0,120.0,3,7619
Hennur Road,3 BHK,1975.0,3.0,123.0,3,6227
other,2 BHK,820.0,2.0,24.0,2,2926
Gubbalala,3 BHK,1745.0,3.0,110.0,3,6303
Nagarbhavi,2 BHK,1223.0,2.0,60.0,2,4905
Ananth Nagar,3 BHK,1200.0,2.0,65.0,3,5416
Basaveshwara Nagar,2 BHK,1200.0,2.0,90.0,2,7500
Binny Pete,1 BHK,665.0,1.0,46.08,1,6929
Bharathi Nagar,2 BHK,1351.0,2.0,70.0,2,5181
Hebbal,3 BHK,1355.0,3.0,83.87,3,6189
Sector 7 HSR Layout,3 BHK,3900.0,3.0,300.0,3,7692
Thanisandra,3 BHK,1588.0,3.0,75.55,3,4757
Amruthahalli,2 BHK,900.0,2.0,60.0,2,6666
other,3 Bedroom,440.0,3.0,50.0,3,11363
Harlur,4 Bedroom,3385.0,4.0,260.0,4,7680
Panathur,3 BHK,1370.0,2.0,75.0,3,5474
Whitefield,3 Bedroom,1500.0,3.0,71.8,3,4786
other,3 BHK,2280.0,3.0,285.0,3,12500
Kanakpura Road,3 BHK,1100.0,3.0,53.0,3,4818
Hoodi,2 BHK,1247.0,2.0,53.0,2,4250
other,3 BHK,2000.0,3.0,200.0,3,10000
Balagere,2 BHK,1020.07,2.0,68.38,2,6703
other,3 BHK,1033.0,2.0,32.0,3,3097
other,3 BHK,1460.0,2.0,73.0,3,5000
Garudachar Palya,2 BHK,1150.0,2.0,52.8,2,4591
Sarjapur  Road,5 BHK,5965.0,5.0,385.0,5,6454
Babusapalaya,2 BHK,1141.0,2.0,46.0,2,4031
Yelahanka,4 BHK,2912.5,6.0,125.5,4,4309
Doddaballapur,4 Bedroom,4200.0,4.0,325.0,4,7738
Hosa Road,3 BHK,1665.0,2.0,125.0,3,7507
Kanakapura,2 BHK,1295.0,2.0,64.0,2,4942
Haralur Road,2 BHK,1230.0,2.0,78.0,2,6341
other,3 BHK,1080.0,2.0,75.0,3,6944
Kaggalipura,2 BHK,1000.0,2.0,60.0,2,6000
Kaval Byrasandra,3 BHK,2400.0,2.0,50.0,3,2083
Whitefield,1 BHK,630.5,1.0,32.79,1,5200
Rayasandra,3 BHK,1708.0,3.0,75.0,3,4391
Thanisandra,3 BHK,1697.0,3.0,115.0,3,6776
Whitefield,4 BHK,3537.0,5.0,210.0,4,5937
Hosa Road,1 BHK,615.0,1.0,28.29,1,4600
Banashankari,3 BHK,1350.0,2.0,83.0,3,6148
Jalahalli,2 BHK,1083.0,2.0,32.49,2,3000
Gottigere,3 BHK,1460.0,2.0,65.0,3,4452
Babusapalaya,2 BHK,1084.0,2.0,58.0,2,5350
Bommasandra,2 BHK,1035.0,2.0,41.43,2,4002
other,3 BHK,2400.0,2.0,150.0,3,6250
Hosur Road,1 BHK,460.0,1.0,21.0,1,4565
other,5 BHK,950.0,5.0,90.0,5,9473
Whitefield,2 BHK,1285.0,2.0,55.9,2,4350
other,19 BHK,2000.0,16.0,490.0,19,24500
Sarjapur,3 Bedroom,1200.0,3.0,78.0,3,6500
other,2 BHK,1070.0,2.0,35.0,2,3271
Hegde Nagar,3 BHK,1801.0,3.0,140.0,3,7773
Whitefield,3 BHK,1697.0,3.0,115.0,3,6776
AECS Layout,2 BHK,1199.0,2.0,65.0,2,5421
Iblur Village,3 BHK,2300.0,3.0,220.0,3,9565
Mysore Road,3 BHK,1340.0,2.0,64.31,3,4799
Marathahalli,4 BHK,3800.0,4.0,205.0,4,5394
Jigani,5 Bedroom,600.0,5.0,50.0,5,8333
other,2 BHK,1125.0,2.0,75.0,2,6666
Anandapura,2 Bedroom,1200.0,2.0,67.0,2,5583
Subramanyapura,2 BHK,1050.0,2.0,41.0,2,3904
Nagarbhavi,2 BHK,956.0,2.0,49.0,2,5125
Sarjapur,2 BHK,1000.0,2.0,30.0,2,3000
other,3 BHK,1596.0,3.0,95.0,3,5952
other,6 Bedroom,1200.0,5.0,150.0,6,12500
Kanakpura Road,2 BHK,1296.0,2.0,89.06,2,6871
other,6 Bedroom,1200.0,7.0,170.0,6,14166
other,3 BHK,1639.0,3.0,123.0,3,7504
Electronic City Phase II,2 BHK,1219.0,2.0,61.25,2,5024
Uttarahalli,3 BHK,1200.0,3.0,50.0,3,4166
Bannerghatta Road,2 BHK,1218.0,2.0,73.0,2,5993
other,2 BHK,920.0,2.0,37.0,2,4021
Yeshwanthpur,3 BHK,2559.0,3.0,141.0,3,5509
2nd Phase Judicial Layout,1 BHK,525.0,1.0,26.0,1,4952
Bannerghatta Road,3 BHK,1427.0,2.0,48.51,3,3399
Kothannur,3 BHK,1270.0,3.0,45.0,3,3543
Ramagondanahalli,2 BHK,920.0,2.0,65.0,2,7065
Dommasandra,4 Bedroom,2700.0,3.0,68.0,4,2518
other,3 BHK,1740.0,3.0,199.0,3,11436
Ramamurthy Nagar,3 BHK,1355.0,2.0,40.0,3,2952
Lingadheeranahalli,3 BHK,1530.0,2.0,90.0,3,5882
other,3 BHK,1372.0,2.0,90.0,3,6559
Nehru Nagar,4 BHK,2363.0,5.0,120.0,4,5078
Marathahalli,4 BHK,3895.0,4.0,220.0,4,5648
Jakkur,3 BHK,1950.0,3.0,131.0,3,6717
other,2 BHK,885.0,2.0,40.0,2,4519
Yelahanka,2 BHK,600.0,2.0,27.0,2,4500
Kumaraswami Layout,5 Bedroom,1270.0,4.0,128.0,5,10078
Vijayanagar,2 Bedroom,1380.0,2.0,225.0,2,16304
Kanakpura Road,2 BHK,1339.0,2.0,62.0,2,4630
Whitefield,2 BHK,1130.0,2.0,36.0,2,3185
Uttarahalli,2 BHK,1085.0,2.0,43.0,2,3963
Yeshwanthpur,3 BHK,1430.0,3.0,72.0,3,5034
Bhoganhalli,4 BHK,1451.5,4.0,121.0,4,8336
other,5 BHK,6200.0,5.0,720.0,5,11612
other,3 BHK,1490.0,2.0,59.6,3,4000
other,4 BHK,1650.0,4.0,99.0,4,6000
Bellandur,2 BHK,1286.0,2.0,57.0,2,4432
other,2 BHK,925.0,2.0,63.0,2,6810
Bellandur,4 BHK,3016.0,4.0,230.0,4,7625
Haralur Road,2 BHK,1000.0,2.0,44.0,2,4400
Whitefield,2 BHK,1215.0,2.0,39.0,2,3209
other,4 Bedroom,1200.0,3.0,95.0,4,7916
Lakshminarayana Pura,2 BHK,1336.0,2.0,100.0,2,7485
Yeshwanthpur,1 BHK,673.0,1.0,36.85,1,5475
Anandapura,2 Bedroom,1200.0,2.0,58.0,2,4833
other,6 Bedroom,600.0,7.0,63.0,6,10500
Malleshwaram,3 BHK,1600.0,3.0,160.0,3,10000
other,2 Bedroom,1200.0,2.0,79.0,2,6583
Indira Nagar,6 Bedroom,2480.0,4.0,750.0,6,30241
other,3 BHK,2500.0,3.0,270.0,3,10800
Devanahalli,5 Bedroom,2400.0,5.0,100.0,5,4166
Sector 2 HSR Layout,2 BHK,1143.0,2.0,65.0,2,5686
Kanakpura Road,2 BHK,1470.0,2.0,61.74,2,4200
other,4 Bedroom,1584.0,3.0,145.0,4,9154
Marathahalli,3 BHK,1785.0,3.0,94.0,3,5266
Koramangala,3 BHK,2292.0,3.0,286.0,3,12478
Kothanur,3 BHK,1460.0,3.0,71.0,3,4863
Kasturi Nagar,3 BHK,1500.0,2.0,80.0,3,5333
Yelahanka,5 Bedroom,1200.0,4.0,140.0,5,11666
Sanjay nagar,5 BHK,3300.0,3.0,250.0,5,7575
Chikkalasandra,2 BHK,1230.0,2.0,48.0,2,3902
Yelahanka New Town,1 BHK,650.0,1.0,20.0,1,3076
Benson Town,2 BHK,1480.0,2.0,120.0,2,8108
Thanisandra,2 BHK,1220.0,2.0,43.92,2,3600
other,3 BHK,1570.0,3.0,115.0,3,7324
other,3 BHK,3000.0,3.0,400.0,3,13333
Mallasandra,3 BHK,1665.0,3.0,86.91,3,5219
other,3 Bedroom,1350.0,3.0,380.0,3,28148
Sarjapur  Road,4 Bedroom,4000.0,5.0,600.0,4,15000
Ramamurthy Nagar,2 Bedroom,1100.0,2.0,68.0,2,6181
Kathriguppe,3 BHK,1350.0,2.0,87.01,3,6445
Kasavanhalli,3 BHK,1719.0,3.0,116.0,3,6748
Budigere,1 BHK,664.0,1.0,35.0,1,5271
Sonnenahalli,2 BHK,1011.0,2.0,50.53,2,4998
other,6 Bedroom,2400.0,6.0,130.0,6,5416
other,3 BHK,1680.0,3.0,85.0,3,5059
Yeshwanthpur,2 BHK,717.0,2.0,49.475,2,6900
other,4 Bedroom,700.0,4.0,110.0,4,15714
Whitefield,3 BHK,1520.0,2.0,150.0,3,9868
Uttarahalli,3 BHK,1350.0,2.0,53.0,3,3925
Kothanur,3 BHK,1760.0,3.0,150.0,3,8522
Pattandur Agrahara,2 BHK,875.0,2.0,33.0,2,3771
Hennur,2 BHK,1231.0,2.0,48.0,2,3899
Somasundara Palya,2 BHK,1174.0,2.0,74.0,2,6303
Munnekollal,3 BHK,1560.0,3.0,75.0,3,4807
other,2 BHK,1464.0,2.0,56.0,2,3825
Kanakpura Road,3 Bedroom,2200.0,3.0,130.0,3,5909
Shivaji Nagar,2 BHK,850.0,2.0,55.0,2,6470
other,3 BHK,1750.0,3.0,180.0,3,10285
1st Phase JP Nagar,3 BHK,2024.0,3.0,157.0,3,7756
other,2 Bedroom,1500.0,1.0,160.0,2,10666
other,6 Bedroom,2100.0,6.0,275.0,6,13095
Kodihalli,3 BHK,2408.0,4.0,260.0,3,10797
Devanahalli,2 BHK,1080.0,2.0,44.0,2,4074
Indira Nagar,3 BHK,2070.0,3.0,225.0,3,10869
Electronic City,2 BHK,1020.0,2.0,29.46,2,2888
Begur Road,3 BHK,1500.0,2.0,55.0,3,3666
Mahalakshmi Layout,4 Bedroom,1575.0,5.0,158.0,4,10031
Kodichikkanahalli,4 Bedroom,3250.0,4.0,180.0,4,5538
Uttarahalli,2 BHK,1175.0,2.0,47.0,2,4000
Channasandra,2 BHK,1030.0,2.0,34.5,2,3349
Hebbal,2 BHK,1440.0,2.0,115.0,2,7986
Hoodi,2 BHK,1500.0,2.0,84.0,2,5600
Thanisandra,1 Bedroom,1380.0,1.0,62.1,1,4500
other,2 BHK,1200.0,2.0,50.0,2,4166
other,1 BHK,705.0,1.0,36.5,1,5177
Harlur,3 BHK,2240.0,5.0,155.0,3,6919
Hegde Nagar,3 BHK,1584.01,3.0,103.0,3,6502
other,3 BHK,1340.0,3.0,77.0,3,5746
other,3 BHK,1800.0,3.0,70.0,3,3888
R.T. Nagar,2 BHK,1200.0,2.0,45.0,2,3750
Sarjapur  Road,1 BHK,685.0,1.0,40.0,1,5839
other,2 BHK,925.0,2.0,42.0,2,4540
other,2 BHK,1230.0,2.0,57.0,2,4634
other,5 Bedroom,1200.0,5.0,185.0,5,15416
Kodichikkanahalli,2 BHK,1125.0,2.0,52.0,2,4622
Hebbal,2 BHK,1100.0,2.0,54.0,2,4909
Jigani,3 Bedroom,2400.0,3.0,149.0,3,6208
Electronic City Phase II,3 BHK,1400.0,2.0,41.58,3,2970
Uttarahalli,7 Bedroom,1200.0,7.0,225.0,7,18750
Jakkur,3 BHK,1858.0,3.0,98.3,3,5290
other,1 Bedroom,1260.0,1.0,41.0,1,3253
Ambedkar Nagar,4 BHK,3530.0,4.0,290.0,4,8215
HSR Layout,2 BHK,1185.0,2.0,60.0,2,5063
other,2 Bedroom,1200.0,2.0,216.0,2,18000
Hennur,2 BHK,1255.0,2.0,58.0,2,4621
Rajaji Nagar,4 BHK,3436.0,6.0,475.0,4,13824
Kudlu Gate,3 BHK,1335.0,3.0,89.0,3,6666
Dodda Nekkundi,3 BHK,1999.0,3.0,132.0,3,6603
other,2 Bedroom,1200.0,2.0,125.0,2,10416
Kundalahalli,3 BHK,1397.0,3.0,110.0,3,7874
Jigani,10 Bedroom,1200.0,10.0,105.0,10,8750
Cooke Town,4 BHK,3950.0,4.0,450.0,4,11392
Yelahanka New Town,2 Bedroom,1200.0,2.0,130.0,2,10833
Chikka Tirupathi,3 Bedroom,1616.0,3.0,95.0,3,5878
Banashankari Stage III,8 Bedroom,1200.0,7.0,350.0,8,29166
other,4 Bedroom,1270.0,2.0,175.0,4,13779
Raja Rajeshwari Nagar,2 BHK,1140.0,2.0,38.55,2,3381
Kanakpura Road,1 BHK,525.0,1.0,30.0,1,5714
Varthur,3 Bedroom,1200.0,2.0,98.0,3,8166
other,1 Bedroom,700.0,1.0,162.0,1,23142
Balagere,2 BHK,1205.47,2.0,81.0,2,6719
other,3 Bedroom,1200.0,3.0,130.0,3,10833
Electronic City,3 BHK,1700.0,3.0,75.0,3,4411
Thanisandra,5 Bedroom,750.0,4.0,62.0,5,8266
Kanakpura Road,2 BHK,1090.0,2.0,54.54,2,5003
Yeshwanthpur,4 BHK,3600.0,5.0,220.0,4,6111
Kundalahalli,1 BHK,2400.0,1.0,650.0,1,27083
Thanisandra,1 BHK,923.0,1.0,60.84,1,6591
other,4 BHK,3595.0,6.0,360.0,4,10013
Green Glen Layout,2 BHK,1050.0,2.0,48.0,2,4571
Whitefield,2 BHK,1495.0,2.0,79.5,2,5317
Narayanapura,3 BHK,1485.0,3.0,85.0,3,5723
Dommasandra,3 BHK,1033.0,2.0,31.5,3,3049
Electronic City,3 BHK,1400.0,2.0,48.0,3,3428
Indira Nagar,1 Bedroom,500.0,1.0,70.0,1,14000
Sarjapur  Road,2 BHK,944.0,2.0,35.0,2,3707
Whitefield,5 Bedroom,3250.0,5.0,290.0,5,8923
Whitefield,2 BHK,1216.0,2.0,66.48,2,5467
other,2 BHK,1100.0,2.0,39.5,2,3590
Thanisandra,3 BHK,1743.0,3.0,93.25,3,5349
Yelahanka,2 BHK,1274.0,2.0,48.42,2,3800
other,3 BHK,2500.0,3.0,425.0,3,17000
Mysore Road,3 BHK,1512.0,2.0,110.0,3,7275
other,4 BHK,3100.0,5.0,385.0,4,12419
Cox Town,3 BHK,1650.0,3.0,100.0,3,6060
Doddathoguru,3 BHK,1875.0,3.0,55.0,3,2933
Whitefield,3 BHK,1350.0,3.0,54.0,3,4000
other,2 BHK,1219.0,2.0,40.0,2,3281
other,4 Bedroom,1200.0,4.0,150.0,4,12500
Sarjapura - Attibele Road,3 Bedroom,2400.0,3.0,85.0,3,3541
Hormavu,2 BHK,1141.0,2.0,44.4,2,3891
Anandapura,2 BHK,1151.0,2.0,43.1,2,3744
Bommenahalli,3 Bedroom,832.0,3.0,77.0,3,9254
Sarjapur  Road,2 BHK,915.0,2.0,38.0,2,4153
other,3 Bedroom,3259.0,3.0,270.0,3,8284
Bannerghatta Road,2 BHK,1115.0,2.0,65.12,2,5840
Thigalarapalya,3 BHK,2072.0,4.0,139.0,3,6708
Kaggadasapura,2 BHK,1200.0,2.0,53.0,2,4416
Singasandra,2 BHK,1300.0,2.0,65.0,2,5000
Bellandur,2 BHK,1457.0,2.0,85.0,2,5833
Kannamangala,2 BHK,1262.0,2.0,55.0,2,4358
BEML Layout,4 Bedroom,1196.0,3.0,150.0,4,12541
Ananth Nagar,2 BHK,900.0,2.0,22.5,2,2500
Attibele,1 BHK,420.0,1.0,17.0,1,4047
Kanakapura,2 BHK,1160.0,2.0,46.39,2,3999
Kasavanhalli,2 BHK,1115.0,2.0,50.0,2,4484
JP Nagar,3 BHK,1315.0,2.0,85.4,3,6494
Babusapalaya,2 BHK,1500.0,2.0,57.0,2,3800
HSR Layout,2 BHK,1127.0,2.0,68.0,2,6033
Kundalahalli,3 BHK,1397.0,3.0,104.0,3,7444
Whitefield,2 BHK,1173.0,2.0,60.0,2,5115
Sompura,2 BHK,1090.0,2.0,28.0,2,2568
Sarjapur  Road,3 BHK,2720.0,5.0,210.0,3,7720
Jakkur,2 BHK,1230.0,2.0,76.0,2,6178
other,4 BHK,3200.0,3.0,190.0,4,5937
Sarjapur  Road,2 BHK,858.0,2.0,31.57,2,3679
Talaghattapura,3 BHK,1868.0,3.0,121.0,3,6477
Uttarahalli,3 BHK,1689.28,3.0,91.2,3,5398
Vishwapriya Layout,4 BHK,600.0,3.0,73.0,4,12166
Banashankari,3 BHK,1425.0,3.0,49.88,3,3500
Subramanyapura,6 Bedroom,1200.0,6.0,170.0,6,14166
Tindlu,2 Bedroom,1350.0,2.0,100.0,2,7407
Channasandra,6 Bedroom,4200.0,6.0,180.0,6,4285
Uttarahalli,3 BHK,1310.0,2.0,85.0,3,6488
other,2 BHK,970.0,2.0,40.0,2,4123
Kundalahalli,3 BHK,1397.0,3.0,108.0,3,7730
Sarjapur,2 BHK,1157.0,2.0,40.0,2,3457
Kasavanhalli,2 BHK,1125.0,2.0,61.2,2,5440
Sonnenahalli,2 BHK,1157.0,2.0,42.0,2,3630
Sarjapur  Road,2 BHK,1241.0,2.0,82.0,2,6607
Binny Pete,1 BHK,660.0,1.0,62.0,1,9393
Marathahalli,4 BHK,4000.0,4.0,220.0,4,5500
Rajiv Nagar,4 BHK,2340.0,4.0,148.0,4,6324
other,1 BHK,596.0,1.0,42.0,1,7046
other,16 BHK,10000.0,16.0,550.0,16,5500
other,4 Bedroom,1200.0,4.0,110.0,4,9166
other,3 BHK,1910.0,3.0,190.0,3,9947
Brookefield,2 BHK,1139.0,2.0,80.0,2,7023
Thigalarapalya,4 BHK,3122.0,5.0,225.0,4,7206
Sarjapura - Attibele Road,2 BHK,1033.0,2.0,28.41,2,2750
Yeshwanthpur,2 BHK,1168.0,2.0,64.08,2,5486
Chandapura,2 BHK,877.0,2.0,42.0,2,4789
Electronics City Phase 1,2 BHK,1340.0,2.0,75.0,2,5597
other,3 BHK,1405.0,2.0,70.0,3,4982
other,3 BHK,1800.0,3.0,65.0,3,3611
other,3 BHK,1750.0,3.0,149.0,3,8514
other,3 BHK,1360.0,2.0,65.0,3,4779
Malleshwaram,3 Bedroom,675.0,2.0,148.0,3,21925
other,2 BHK,985.0,2.0,63.0,2,6395
other,2 BHK,1300.0,2.0,129.0,2,9923
7th Phase JP Nagar,3 BHK,1850.0,3.0,150.0,3,8108
other,3 BHK,1490.0,2.0,88.0,3,5906
other,2 BHK,620.0,1.0,25.0,2,4032
Hormavu,2 BHK,1130.0,2.0,42.3,2,3743
Yelachenahalli,2 BHK,1080.0,2.0,55.0,2,5092
Singasandra,4 BHK,3126.0,4.0,120.0,4,3838
Begur Road,3 BHK,1410.0,2.0,52.17,3,3700
Bommanahalli,3 BHK,1375.0,3.0,59.0,3,4290
Hennur Road,3 BHK,1290.0,2.0,94.5,3,7325
Hosa Road,3 BHK,1318.0,2.0,85.28,3,6470
Electronic City,2 BHK,1128.0,2.0,68.0,2,6028
other,4 Bedroom,4428.0,4.0,255.0,4,5758
Kanakpura Road,2 BHK,1051.0,2.0,56.74,2,5398
Kengeri,3 Bedroom,3500.0,5.0,275.0,3,7857
Kannamangala,4 Bedroom,2400.0,4.0,130.0,4,5416
Yelahanka,2 BHK,1327.0,2.0,85.0,2,6405
Thigalarapalya,4 BHK,2912.0,6.0,210.0,4,7211
other,2 BHK,1301.0,2.0,55.0,2,4227
other,1 BHK,650.0,1.0,30.0,1,4615
Varthur,3 Bedroom,1200.0,3.0,67.75,3,5645
other,3 BHK,1500.0,2.0,89.99,3,5999
other,4 Bedroom,2700.0,6.0,675.0,4,25000
other,4 Bedroom,1350.0,4.0,145.0,4,10740
Electronic City Phase II,2 BHK,1000.0,2.0,28.88,2,2888
Sarjapur  Road,2 BHK,984.0,2.0,45.91,2,4665
Jakkur,3 BHK,1819.18,3.0,160.0,3,8795
Nagarbhavi,1 Bedroom,900.0,1.0,175.0,1,19444
Hebbal,3 BHK,1355.0,3.0,83.87,3,6189
Sarakki Nagar,3 BHK,2145.0,4.0,270.0,3,12587
Malleshwaram,3 BHK,2475.0,4.0,300.0,3,12121
Kaggadasapura,2 BHK,1118.0,2.0,30.0,2,2683
other,1 BHK,910.0,1.0,42.0,1,4615
other,3 BHK,1250.0,3.0,39.0,3,3120
Hosakerehalli,2 BHK,1304.0,2.0,100.0,2,7668
Begur Road,3 BHK,1615.0,3.0,58.0,3,3591
Sarjapur  Road,2 BHK,1400.0,2.0,68.0,2,4857
CV Raman Nagar,2 BHK,1070.0,2.0,47.5,2,4439
Hebbal,3 BHK,3500.0,3.0,395.0,3,11285
Hoodi,2 BHK,1132.0,2.0,89.37,2,7894
other,3 BHK,2000.0,3.0,85.0,3,4250
Hebbal,7 Bedroom,2400.0,5.0,550.0,7,22916
Sector 2 HSR Layout,1 BHK,794.0,1.0,65.0,1,8186
Raja Rajeshwari Nagar,2 BHK,1090.0,2.0,43.59,2,3999
Gottigere,3 BHK,1500.0,3.0,70.0,3,4666
1st Phase JP Nagar,3 BHK,2059.0,3.0,225.0,3,10927
Kanakpura Road,1 BHK,525.0,1.0,30.0,1,5714
Bhoganhalli,2 BHK,1447.0,2.0,75.97,2,5250
Whitefield,2 BHK,1269.0,2.0,75.0,2,5910
other,4 Bedroom,4800.0,4.0,700.0,4,14583
Kanakpura Road,2 BHK,1070.0,2.0,37.45,2,3500
Talaghattapura,2 BHK,1150.0,2.0,28.0,2,2434
Marathahalli,2 BHK,1208.0,2.0,70.0,2,5794
Ramamurthy Nagar,2 BHK,1185.0,2.0,45.5,2,3839
Laggere,3 Bedroom,1200.0,3.0,130.0,3,10833
Anekal,2 BHK,888.0,2.0,34.0,2,3828
Chandapura,2 BHK,1015.0,2.0,25.88,2,2549
other,1 BHK,532.0,1.0,22.49,1,4227
other,3 BHK,1060.0,3.0,62.0,3,5849
Electronic City,1 BHK,700.0,1.0,35.0,1,5000
other,2 BHK,902.0,2.0,42.0,2,4656
other,6 Bedroom,600.0,3.0,95.0,6,15833
Uttarahalli,2 BHK,1175.0,2.0,47.0,2,4000
Binny Pete,2 BHK,1365.0,2.0,122.0,2,8937
Doddathoguru,2 BHK,984.0,2.0,51.0,2,5182
Hebbal,4 Bedroom,2700.0,2.0,100.0,4,3703
Jalahalli East,3 BHK,1475.0,3.0,96.0,3,6508
Haralur Road,2 BHK,1300.0,2.0,79.0,2,6076
5th Block Hbr Layout,5 Bedroom,1200.0,6.0,220.0,5,18333
Babusapalaya,3 BHK,1453.0,2.0,43.59,3,3000
other,2 BHK,1550.0,2.0,160.0,2,10322
Akshaya Nagar,3 BHK,1666.0,3.0,95.0,3,5702
Akshaya Nagar,4 Bedroom,1410.0,4.0,160.0,4,11347
Yelahanka,4 Bedroom,4025.0,5.0,800.0,4,19875
Murugeshpalya,3 BHK,1845.0,3.0,91.0,3,4932
Sarjapura - Attibele Road,3 BHK,1676.0,3.0,52.0,3,3102
Uttarahalli,2 BHK,1137.0,2.0,63.0,2,5540
Electronic City,3 BHK,1360.0,2.0,70.5,3,5183
other,3 BHK,1374.0,2.0,70.0,3,5094
Judicial Layout,3 Bedroom,1700.0,3.0,155.0,3,9117
Whitefield,5 Bedroom,3227.0,5.0,250.0,5,7747
other,4 BHK,2145.0,4.0,125.0,4,5827
Koramangala,4 Bedroom,2400.0,6.0,600.0,4,25000
Kasavanhalli,3 BHK,2170.0,3.0,151.0,3,6958
other,2 BHK,1080.0,2.0,55.0,2,5092
Electronic City Phase II,3 BHK,1790.0,3.0,100.0,3,5586
Doddathoguru,1 BHK,550.0,1.0,17.0,1,3090
Poorna Pragna Layout,3 BHK,1270.0,2.0,50.79,3,3999
Somasundara Palya,2 BHK,1033.0,2.0,48.0,2,4646
other,2 Bedroom,1000.0,2.0,80.0,2,8000
Binny Pete,4 BHK,2940.0,6.0,280.0,4,9523
other,2 BHK,1240.0,2.0,52.6,2,4241
Kasturi Nagar,5 Bedroom,1650.0,5.0,450.0,5,27272
Kambipura,2 BHK,883.0,2.0,39.0,2,4416
other,3 BHK,1500.0,2.0,97.0,3,6466
Marathahalli,3 BHK,1730.0,3.0,139.0,3,8034
Kasavanhalli,3 BHK,1915.0,3.0,80.0,3,4177
Whitefield,3 BHK,1545.0,3.0,79.0,3,5113
other,3 BHK,1865.0,4.0,140.0,3,7506
Hebbal Kempapura,2 BHK,1150.0,2.0,40.0,2,3478
other,2 BHK,1200.0,2.0,49.0,2,4083
Kengeri,2 BHK,963.0,2.0,40.0,2,4153
Marsur,3 Bedroom,1200.0,3.0,120.0,3,10000
Basavangudi,4 BHK,2600.0,4.0,260.0,4,10000
Kaikondrahalli,4 BHK,1700.0,4.0,108.0,4,6352
Lingadheeranahalli,3 BHK,1682.0,3.0,120.0,3,7134
Yelahanka,3 BHK,1371.0,2.0,65.0,3,4741
Hennur Road,2 BHK,901.0,2.0,49.5,2,5493
Vishwapriya Layout,7 Bedroom,600.0,7.0,73.0,7,12166
Ambalipura,3 BHK,1730.0,3.0,120.0,3,6936
Konanakunte,4 Bedroom,1200.0,4.0,125.0,4,10416
EPIP Zone,3 BHK,1500.0,3.0,102.0,3,6800
Sarjapur  Road,4 BHK,3785.0,6.0,280.0,4,7397
other,4 BHK,3500.0,4.0,245.0,4,7000
other,2 BHK,1203.0,2.0,59.86,2,4975
Horamavu Banaswadi,2 BHK,1357.0,2.0,54.0,2,3979
other,3 BHK,1600.0,3.0,76.0,3,4750
Yelahanka New Town,1 BHK,350.0,1.0,13.5,1,3857
Varthur,3 BHK,1665.0,3.0,53.0,3,3183
Ramamurthy Nagar,3 Bedroom,1400.0,3.0,200.0,3,14285
Sarjapur  Road,1 BHK,835.0,1.0,46.95,1,5622
other,2 Bedroom,815.0,3.0,160.0,2,19631
Haralur Road,3 BHK,1444.0,2.0,80.0,3,5540
Kanakapura,2 BHK,1277.0,2.0,55.0,2,4306
Lingadheeranahalli,3 BHK,1521.0,3.0,94.71,3,6226
other,2 BHK,1134.0,2.0,40.0,2,3527
Vasanthapura,2 BHK,995.0,2.0,34.82,2,3499
Old Madras Road,3 BHK,2760.0,5.0,157.0,3,5688
Marathahalli,2 BHK,1325.0,1.0,92.0,2,6943
Kothanur,6 Bedroom,4000.0,6.0,150.0,6,3750
Marathahalli,3 BHK,1525.0,3.0,90.0,3,5901
other,3 BHK,1320.0,2.0,73.0,3,5530
Electronics City Phase 1,1 BHK,650.0,1.0,23.0,1,3538
Old Airport Road,3 BHK,1858.0,2.0,167.0,3,8988
Electronic City,2 BHK,1150.0,2.0,38.0,2,3304
Electronic City Phase II,2 BHK,769.0,2.0,40.0,2,5201
Electronic City,3 BHK,1571.0,3.0,105.0,3,6683
Hennur Road,3 BHK,1445.0,3.0,86.56,3,5990
Hosur Road,3 BHK,1590.0,2.0,135.0,3,8490
Bommasandra,2 BHK,902.0,2.0,49.0,2,5432
Sarjapur,3 Bedroom,2238.0,3.0,140.0,3,6255
Electronic City,2 BHK,1090.0,2.0,31.49,2,2888
Yelahanka,3 BHK,1712.0,3.0,82.0,3,4789
Haralur Road,2 BHK,1088.0,2.0,68.0,2,6250
Somasundara Palya,3 BHK,1575.0,3.0,63.1,3,4006
other,6 Bedroom,2150.0,6.0,205.0,6,9534
Kundalahalli,3 BHK,1920.0,3.0,150.0,3,7812
Margondanahalli,2 Bedroom,1000.0,2.0,57.0,2,5700
other,2 BHK,1100.0,2.0,56.1,2,5100
Electronic City,2 BHK,1089.0,2.0,45.0,2,4132
Horamavu Agara,2 BHK,1106.0,2.0,43.09,2,3896
Kengeri,2 BHK,927.0,2.0,37.0,2,3991
Kanakpura Road,2 BHK,1339.0,2.0,58.0,2,4331
Thanisandra,1 BHK,923.0,1.0,45.69,1,4950
Sarjapur  Road,3 Bedroom,2900.0,3.0,285.0,3,9827
Lakshminarayana Pura,2 BHK,1175.0,2.0,75.0,2,6382
Thubarahalli,2 BHK,975.0,2.0,55.0,2,5641
Hebbal,3 BHK,2400.0,4.0,180.0,3,7500
Subramanyapura,3 BHK,1330.0,3.0,72.0,3,5413
Thanisandra,2 BHK,1183.0,2.0,59.74,2,5049
Harlur,3 BHK,1758.0,3.0,133.0,3,7565
Nagarbhavi,3 Bedroom,1200.0,3.0,265.0,3,22083
Hebbal,3 BHK,1255.0,2.0,77.68,3,6189
other,3 Bedroom,600.0,4.0,110.0,3,18333
Pai Layout,2 BHK,1100.0,2.0,50.0,2,4545
Ramamurthy Nagar,2 BHK,1170.0,2.0,46.8,2,4000
other,5 Bedroom,2400.0,5.0,120.0,5,5000
Raja Rajeshwari Nagar,3 BHK,1530.0,3.0,51.77,3,3383
Hebbal,2 BHK,1420.0,2.0,107.0,2,7535
other,4 Bedroom,1200.0,4.0,300.0,4,25000
7th Phase JP Nagar,6 Bedroom,1045.0,6.0,160.0,6,15311
Kanakpura Road,3 BHK,1591.0,3.0,115.0,3,7228
7th Phase JP Nagar,3 BHK,1680.0,3.0,120.0,3,7142
other,2 Bedroom,675.0,3.0,52.0,2,7703
Hoodi,2 BHK,1055.0,2.0,57.63,2,5462
Iblur Village,3 BHK,1918.0,3.0,129.0,3,6725
Sarjapur  Road,2 BHK,1280.0,2.0,76.0,2,5937
Nagarbhavi,3 BHK,1560.0,3.0,66.0,3,4230
Sarjapur  Road,2 BHK,1145.0,2.0,49.5,2,4323
Talaghattapura,3 BHK,2106.0,3.0,126.0,3,5982
other,3 BHK,1801.0,3.0,139.0,3,7717
Seegehalli,5 Bedroom,1200.0,5.0,120.0,5,10000
Rajaji Nagar,2 BHK,1718.0,3.0,288.0,2,16763
Whitefield,3 BHK,1306.0,3.0,54.65,3,4184
Hebbal,2 BHK,1320.0,2.0,92.0,2,6969
other,3 BHK,1350.0,2.0,65.0,3,4814
other,4 BHK,2990.0,4.0,324.0,4,10836
Kalyan nagar,4 BHK,3800.0,5.0,300.0,4,7894
Bannerghatta Road,3 BHK,1344.0,2.0,42.0,3,3125
Bannerghatta Road,3 BHK,1893.0,3.0,100.0,3,5282
Chandapura,3 BHK,1107.0,2.0,43.0,3,3884
R.T. Nagar,9 Bedroom,1200.0,10.0,180.0,9,15000
Lakshminarayana Pura,2 BHK,1190.0,2.0,75.0,2,6302
Whitefield,2 BHK,1000.0,2.0,35.0,2,3500
other,2 Bedroom,2400.0,2.0,160.0,2,6666
Kumaraswami Layout,2 BHK,1000.0,2.0,48.5,2,4850
Jalahalli,3 BHK,2113.0,3.0,155.0,3,7335
KR Puram,2 BHK,1100.0,2.0,46.0,2,4181
Sector 2 HSR Layout,5 Bedroom,1200.0,4.0,160.0,5,13333
Harlur,4 BHK,1884.0,4.0,120.0,4,6369
Electronic City,2 BHK,1110.0,2.0,40.0,2,3603
Kudlu,2 BHK,1027.0,2.0,43.0,2,4186
other,3 BHK,1250.0,3.0,40.0,3,3200
Sarjapur  Road,3 BHK,1470.0,3.0,62.0,3,4217
other,2 BHK,1150.0,2.0,75.0,2,6521
other,3 Bedroom,1200.0,3.0,165.0,3,13750
Kudlu Gate,1 BHK,703.0,1.0,52.0,1,7396
other,3 BHK,1400.0,2.0,62.0,3,4428
Bellandur,2 BHK,1195.0,2.0,51.0,2,4267
Narayanapura,2 BHK,1302.0,2.0,69.1,2,5307
Begur Road,3 BHK,1583.0,3.0,74.4,3,4699
other,3 BHK,1488.0,2.0,60.0,3,4032
other,4 Bedroom,600.0,5.0,85.0,4,14166
Kammasandra,3 BHK,1385.0,2.0,34.63,3,2500
Whitefield,4 BHK,3262.0,4.0,260.0,4,7970
Electronic City,3 BHK,1600.0,3.0,104.0,3,6500
Old Airport Road,2 BHK,1150.0,2.0,90.0,2,7826
other,9 Bedroom,4500.0,9.0,166.0,9,3688
Sarjapur  Road,2 BHK,1263.0,2.0,78.0,2,6175
KR Puram,7 Bedroom,1200.0,8.0,110.0,7,9166
Kanakpura Road,2 BHK,1299.0,2.0,85.0,2,6543
Banashankari,2 BHK,1330.0,2.0,78.0,2,5864
Devanahalli,3 BHK,1282.0,2.0,52.43,3,4089
other,3 BHK,1853.0,3.0,82.0,3,4425
other,2 BHK,861.0,2.0,35.0,2,4065
Bannerghatta Road,3 BHK,1600.0,3.0,95.0,3,5937
other,2 BHK,1256.0,2.0,62.8,2,5000
other,3 BHK,1390.0,2.0,80.0,3,5755
other,3 BHK,1550.0,2.0,90.0,3,5806
other,11 Bedroom,1200.0,6.0,150.0,11,12500
Hebbal,2 BHK,919.0,2.0,49.0,2,5331
Margondanahalli,3 Bedroom,1625.0,3.0,80.0,3,4923
Gunjur,2 BHK,1195.0,2.0,44.0,2,3682
Domlur,6 BHK,2400.0,4.0,600.0,6,25000
Yelenahalli,2 BHK,1159.0,2.0,60.0,2,5176
Whitefield,3 BHK,1386.0,3.0,57.46,3,4145
Harlur,2 BHK,1197.0,2.0,79.0,2,6599
Old Airport Road,4 BHK,2690.0,4.0,191.0,4,7100
Bommasandra,3 BHK,1295.0,2.0,50.28,3,3882
other,3 BHK,1310.0,2.0,37.83,3,2887
Whitefield,2 BHK,1132.0,2.0,79.28,2,7003
other,3 BHK,2350.0,3.0,400.0,3,17021
other,4 Bedroom,2400.0,4.0,500.0,4,20833
Hebbal,3 BHK,1645.0,3.0,135.0,3,8206
Whitefield,2 BHK,1200.0,2.0,45.84,2,3820
Whitefield,4 Bedroom,3000.0,4.0,330.0,4,11000
Marathahalli,3 BHK,1937.0,3.0,140.0,3,7227
Doddathoguru,2 BHK,1015.0,2.0,33.5,2,3300
other,2 BHK,1080.0,2.0,39.0,2,3611
Kathriguppe,3 BHK,1350.0,2.0,80.99,3,5999
Kengeri Satellite Town,3 BHK,1149.0,2.0,29.0,3,2523
Kothanur,2 BHK,1185.0,2.0,59.0,2,4978
Domlur,3 BHK,1720.0,3.0,135.0,3,7848
Hebbal,3 BHK,2080.0,3.0,175.0,3,8413
Tindlu,3 Bedroom,1500.0,3.0,87.0,3,5800
Yelahanka,3 BHK,1566.0,3.0,97.0,3,6194
Rajaji Nagar,3 BHK,2386.0,3.0,334.0,3,13998
other,4 Bedroom,1500.0,4.0,110.0,4,7333
KR Puram,2 BHK,1290.0,2.0,49.02,2,3800
other,3 BHK,1531.0,3.0,72.0,3,4702
other,3 BHK,1839.0,3.0,165.0,3,8972
other,4 Bedroom,4200.0,4.0,760.0,4,18095
Electronic City Phase II,3 BHK,1400.0,2.0,40.43,3,2887
Electronic City,2 BHK,550.0,1.0,15.0,2,2727
Whitefield,2 BHK,1140.0,2.0,39.0,2,3421
Tumkur Road,3 BHK,1343.5,3.0,57.76,3,4299
Banashankari Stage III,4 Bedroom,2000.0,4.0,100.0,4,5000
Kaval Byrasandra,2 BHK,1100.0,2.0,46.0,2,4181
other,2 BHK,1200.0,2.0,55.0,2,4583
Bannerghatta,3 BHK,1618.0,3.0,68.0,3,4202
Whitefield,4 BHK,2830.0,5.0,160.0,4,5653
Uttarahalli,3 BHK,1315.0,2.0,74.0,3,5627
Anandapura,2 BHK,1141.0,2.0,42.75,2,3746
AECS Layout,2 BHK,1080.0,2.0,55.0,2,5092
HBR Layout,2 BHK,1200.0,2.0,60.0,2,5000
Varthur,2 BHK,1111.0,2.0,41.88,2,3769
NGR Layout,2 BHK,1022.0,2.0,45.9,2,4491
Bannerghatta Road,2 BHK,1275.0,2.0,65.0,2,5098
other,2 BHK,900.0,2.0,70.0,2,7777
Rachenahalli,2 BHK,1224.0,2.0,39.2,2,3202
Battarahalli,2 BHK,1135.0,2.0,43.0,2,3788
Rachenahalli,2 BHK,1050.0,2.0,52.1,2,4961
Kudlu Gate,3 BHK,1535.0,3.0,86.0,3,5602
Nagavarapalya,2 BHK,1392.0,2.0,130.0,2,9339
Bellandur,2 BHK,1050.0,2.0,72.0,2,6857
Hosa Road,3 BHK,1513.0,3.0,103.0,3,6807
Gottigere,3 BHK,1621.0,3.0,72.0,3,4441
Bannerghatta Road,2 BHK,892.0,2.0,39.0,2,4372
Whitefield,2 BHK,1205.0,2.0,40.0,2,3319
Hebbal,3 BHK,1355.0,3.0,83.87,3,6189
other,5 Bedroom,1250.0,5.0,300.0,5,24000
other,7 Bedroom,1650.0,9.0,345.0,7,20909
other,4 BHK,1200.0,6.0,70.0,4,5833
Indira Nagar,2 BHK,1400.0,2.0,168.0,2,12000
Kengeri,2 BHK,726.0,2.0,35.0,2,4820
other,3 Bedroom,1500.0,3.0,149.0,3,9933
Hebbal,4 BHK,3895.0,4.0,405.0,4,10397
Horamavu Agara,2 BHK,1079.0,2.0,41.0,2,3799
Begur Road,2 BHK,1170.0,2.0,46.0,2,3931
Doddathoguru,2 BHK,1104.0,2.0,39.0,2,3532
Kanakpura Road,2 BHK,950.0,2.0,45.0,2,4736
Hormavu,4 Bedroom,4818.0,4.0,390.0,4,8094
other,3 BHK,1650.0,2.0,52.0,3,3151
Kogilu,3 BHK,1559.0,3.0,127.0,3,8146
other,2 BHK,1100.0,2.0,38.0,2,3454
Benson Town,3 BHK,1750.0,3.0,150.0,3,8571
Ramagondanahalli,2 BHK,1215.0,2.0,45.5,2,3744
Talaghattapura,3 BHK,1868.0,3.0,135.0,3,7226
Electronic City,2 BHK,1130.0,2.0,32.63,2,2887
Kanakpura Road,3 BHK,1570.0,3.0,64.5,3,4108
Sarjapur  Road,3 BHK,1691.0,3.0,98.0,3,5795
Ardendale,5 Bedroom,4355.0,3.0,350.0,5,8036
Hulimavu,2 BHK,1225.0,2.0,55.0,2,4489
JP Nagar,3 BHK,1690.0,3.0,112.0,3,6627
Thigalarapalya,2 BHK,1418.0,2.0,103.0,2,7263
Kasturi Nagar,4 BHK,1896.0,3.0,125.0,4,6592
Sonnenahalli,1 BHK,605.0,1.0,40.0,1,6611
Begur Road,5 Bedroom,800.0,5.0,95.0,5,11875
Electronic City Phase II,2 BHK,900.0,2.0,32.5,2,3611
other,1 Bedroom,750.0,2.0,50.0,1,6666
Hoodi,3 Bedroom,780.0,3.0,62.0,3,7948
Ananth Nagar,2 BHK,908.0,2.0,26.0,2,2863
Sarjapur  Road,4 Bedroom,3416.5,6.0,143.0,4,4185
Thanisandra,4 BHK,3216.0,4.0,226.0,4,7027
BEML Layout,5 Bedroom,2400.0,4.0,225.0,5,9375
Iblur Village,2 BHK,1400.0,2.0,68.0,2,4857
Kudlu,2 BHK,1092.0,2.0,44.0,2,4029
Hulimavu,2 BHK,1255.0,2.0,74.0,2,5896
Electronic City,3 BHK,1500.0,2.0,100.0,3,6666
Thanisandra,2 BHK,1296.0,2.0,72.0,2,5555
Horamavu Banaswadi,2 BHK,925.0,2.0,38.0,2,4108
Raja Rajeshwari Nagar,2 BHK,1023.0,2.0,42.0,2,4105
other,2 BHK,600.0,3.0,75.0,2,12500
Thanisandra,2 BHK,1020.0,2.0,44.5,2,4362
Choodasandra,3 BHK,1530.0,3.0,77.0,3,5032
Gubbalala,3 BHK,1650.0,3.0,71.0,3,4303
other,3 BHK,1605.0,2.0,85.0,3,5295
Hosakerehalli,2 BHK,1925.0,2.0,50.0,2,2597
Kogilu,2 BHK,1190.0,2.0,50.66,2,4257
Jakkur,3 BHK,1374.0,3.0,68.0,3,4949
6th Phase JP Nagar,2 BHK,1140.0,2.0,65.0,2,5701
Green Glen Layout,3 BHK,1725.0,3.0,130.0,3,7536
Bisuvanahalli,3 BHK,1075.0,2.0,50.0,3,4651
Kannamangala,3 BHK,1574.0,3.0,93.34,3,5930
Sanjay nagar,1 BHK,965.0,1.0,32.0,1,3316
Marathahalli,3 BHK,1550.0,3.0,83.0,3,5354
other,2 BHK,1205.0,2.0,60.0,2,4979
Kambipura,3 BHK,1082.0,2.0,55.0,3,5083
Uttarahalli,3 BHK,1215.0,2.0,52.85,3,4349
Hebbal,4 BHK,2483.0,5.0,230.0,4,9262
other,2 Bedroom,1200.0,3.0,46.14,2,3845
other,1 BHK,1500.0,1.0,19.5,1,1300
other,4 Bedroom,2360.0,4.0,600.0,4,25423
Ardendale,4 BHK,3198.0,4.0,250.0,4,7817
Kanakpura Road,3 BHK,1452.0,3.0,60.98,3,4199
Electronic City Phase II,3 BHK,1329.0,2.0,41.0,3,3085
other,2 BHK,1125.0,2.0,44.0,2,3911
other,2 Bedroom,1200.0,2.0,70.0,2,5833
Sarjapur  Road,3 BHK,1350.0,3.0,39.97,3,2960
Uttarahalli,3 BHK,1312.5,3.0,51.19,3,3900
Hebbal,3 BHK,2600.0,3.0,195.0,3,7500
other,2 Bedroom,1200.0,2.0,160.0,2,13333
Electronic City Phase II,3 BHK,1220.0,3.0,35.23,3,2887
Pai Layout,2 BHK,1255.0,2.0,80.0,2,6374
other,3 BHK,1904.0,3.0,155.0,3,8140
Somasundara Palya,2 BHK,1329.0,2.0,70.0,2,5267
Chandapura,2 Bedroom,1200.0,2.0,60.0,2,5000
Murugeshpalya,3 BHK,2135.0,3.0,160.0,3,7494
other,3 BHK,2770.0,4.0,315.0,3,11371
other,1 BHK,813.0,1.0,39.0,1,4797
other,4 Bedroom,3884.0,4.0,240.0,4,6179
Doddaballapur,3 Bedroom,3876.0,3.0,300.0,3,7739
Kalena Agrahara,2 BHK,800.0,2.0,30.0,2,3750
other,4 Bedroom,4750.0,2.0,120.0,4,2526
Hormavu,3 BHK,1500.0,3.0,70.0,3,4666
Bannerghatta Road,3 BHK,1880.0,3.0,95.0,3,5053
Electronic City,3 BHK,1440.0,2.0,55.0,3,3819
Akshaya Nagar,3 BHK,1575.0,3.0,90.0,3,5714
other,3 BHK,1630.0,2.0,97.78,3,5998
Yelahanka,3 BHK,1180.0,3.0,55.0,3,4661
other,2 BHK,1090.0,2.0,36.0,2,3302
NGR Layout,2 BHK,1020.0,2.0,48.45,2,4750
other,4 BHK,2425.0,4.0,173.0,4,7134
Varthur,2 BHK,1097.0,2.0,33.82,2,3082
Raja Rajeshwari Nagar,2 BHK,1070.0,2.0,50.0,2,4672
Kumaraswami Layout,2 BHK,1000.0,2.0,58.0,2,5800
Hoodi,8 Bedroom,1120.0,8.0,150.0,8,13392
other,2 BHK,1150.0,2.0,56.0,2,4869
Frazer Town,3 BHK,1900.0,4.0,145.0,3,7631
other,2 BHK,1141.0,2.0,49.0,2,4294
Jigani,2 BHK,937.0,2.0,44.0,2,4695
Green Glen Layout,2 BHK,1075.0,2.0,60.0,2,5581
Panathur,2 BHK,1000.0,2.0,65.3,2,6530
Kasavanhalli,3 BHK,1450.0,3.0,77.41,3,5338
Yelachenahalli,5 Bedroom,897.0,2.0,85.0,5,9476
other,2 BHK,936.0,2.0,43.0,2,4594
other,2 BHK,1280.0,2.0,75.0,2,5859
Babusapalaya,2 BHK,1050.0,2.0,45.0,2,4285
Marathahalli,2 BHK,1350.0,2.0,86.0,2,6370
Mallasandra,2 BHK,1325.0,2.0,70.0,2,5283
Devanahalli,2 BHK,1174.0,2.0,59.0,2,5025
Hosur Road,3 BHK,1890.0,3.0,162.0,3,8571
other,1 BHK,735.0,1.0,75.0,1,10204
other,1 BHK,500.0,1.0,24.0,1,4800
Gottigere,3 BHK,1304.0,3.0,80.0,3,6134
Yelahanka,3 BHK,1653.0,3.0,120.0,3,7259
Bisuvanahalli,3 BHK,1075.0,2.0,52.0,3,4837
7th Phase JP Nagar,2 BHK,1212.0,3.0,42.0,2,3465
other,2 BHK,1100.0,2.0,50.0,2,4545
Rajaji Nagar,4 BHK,3436.0,6.0,500.0,4,14551
other,3 Bedroom,1200.0,4.0,145.0,3,12083
Billekahalli,2 BHK,1140.0,2.0,50.0,2,4385
other,2 Bedroom,1200.0,2.0,100.0,2,8333
Kudlu Gate,3 BHK,1532.0,3.0,83.0,3,5417
Doddaballapur,3 Bedroom,2776.0,4.0,195.0,3,7024
Padmanabhanagar,3 BHK,1531.0,2.0,75.0,3,4898
Poorna Pragna Layout,3 BHK,1270.0,2.0,50.79,3,3999
Uttarahalli,2 BHK,1325.0,2.0,53.0,2,4000
other,3 Bedroom,1200.0,3.0,160.0,3,13333
Electronic City,3 BHK,1156.0,2.0,44.0,3,3806
R.T. Nagar,4 Bedroom,2000.0,4.0,320.0,4,16000
Bannerghatta Road,2 BHK,1260.0,2.0,70.0,2,5555
Vittasandra,2 BHK,1934.0,2.0,90.0,2,4653
Malleshpalya,2 BHK,1060.0,2.0,40.0,2,3773
Kasavanhalli,3 BHK,1476.0,3.0,95.0,3,6436
Murugeshpalya,2 BHK,1185.0,2.0,85.0,2,7172
other,3 BHK,1265.0,2.0,82.0,3,6482
other,4 BHK,3200.0,5.0,150.0,4,4687
other,2 BHK,1196.0,2.0,56.81,2,4750
EPIP Zone,2 BHK,1280.0,2.0,69.0,2,5390
Marathahalli,4 Bedroom,800.0,4.0,86.0,4,10750
Hulimavu,2 BHK,1175.0,2.0,45.0,2,3829
other,3 BHK,1850.0,3.0,98.0,3,5297
Indira Nagar,4 Bedroom,2400.0,4.0,525.0,4,21875
Whitefield,2 BHK,1195.0,2.0,70.0,2,5857
Ulsoor,3 BHK,2000.0,3.0,185.0,3,9250
other,7 BHK,600.0,4.0,100.0,7,16666
Hebbal Kempapura,4 Bedroom,800.0,5.0,155.0,4,19375
Hosur Road,2 BHK,1180.0,2.0,36.5,2,3093
Jakkur,2 BHK,1050.0,2.0,49.87,2,4749
Electronic City,3 BHK,1400.0,2.0,40.44,3,2888
other,4 Bedroom,2360.0,4.0,150.0,4,6355
other,2 BHK,1030.0,2.0,77.25,2,7500
other,2 BHK,1512.0,1.0,65.01,2,4299
Subramanyapura,2 BHK,929.0,2.0,48.0,2,5166
Jigani,3 BHK,1252.0,3.0,61.0,3,4872
Green Glen Layout,3 BHK,1485.0,3.0,115.0,3,7744
Electronics City Phase 1,3 BHK,1124.0,2.0,55.0,3,4893
TC Palaya,6 Bedroom,2400.0,6.0,370.0,6,15416
Indira Nagar,3 BHK,1407.0,3.0,120.0,3,8528
Hennur Road,3 BHK,1482.0,2.0,97.0,3,6545
Sector 7 HSR Layout,2 BHK,1163.0,2.0,98.86,2,8500
Jalahalli,3 BHK,1570.0,3.0,140.0,3,8917
Whitefield,3 BHK,3480.0,4.0,284.0,3,8160
Chandapura,1 BHK,520.0,1.0,14.04,1,2700
other,3 BHK,2000.0,3.0,160.0,3,8000
Prithvi Layout,2 BHK,1352.0,2.0,75.0,2,5547
Electronic City,2 BHK,1152.0,2.0,65.0,2,5642
other,3 BHK,1575.0,3.0,93.0,3,5904
Hegde Nagar,3 BHK,1965.0,4.0,125.0,3,6361
other,3 BHK,1300.0,2.0,70.0,3,5384
Sarjapur  Road,4 Bedroom,1.0,4.0,120.0,4,12000000
Marathahalli,2 BHK,957.0,2.0,47.0,2,4911
Hoskote,3 BHK,1740.0,3.0,60.0,3,3448
Kanakpura Road,5 Bedroom,1080.0,6.0,65.0,5,6018
Neeladri Nagar,2 BHK,1060.0,2.0,35.0,2,3301
other,2 BHK,975.0,2.0,50.0,2,5128
Ramamurthy Nagar,2 BHK,1150.0,2.0,51.0,2,4434
Whitefield,3 BHK,1738.0,2.0,79.0,3,4545
Bommanahalli,3 BHK,1250.0,3.0,55.0,3,4400
Bannerghatta Road,3 BHK,1788.0,3.0,90.0,3,5033
Kanakapura,2 BHK,1151.0,2.0,59.0,2,5125
Uttarahalli,3 BHK,1300.0,2.0,60.0,3,4615
Garudachar Palya,2 BHK,1150.0,2.0,52.8,2,4591
other,2 BHK,1380.0,2.0,75.0,2,5434
Hebbal,3 BHK,1645.0,3.0,117.0,3,7112
Babusapalaya,2 BHK,1141.0,2.0,45.9,2,4022
7th Phase JP Nagar,2 BHK,1175.0,2.0,82.5,2,7021
Banashankari Stage V,3 BHK,1355.0,2.0,61.0,3,4501
Dasarahalli,2 BHK,1150.0,2.0,70.0,2,6086
other,5 BHK,5800.0,5.0,80.0,5,1379
other,2 BHK,1000.0,2.0,45.0,2,4500
Balagere,2 BHK,1007.0,2.0,65.0,2,6454
other,4 BHK,3000.0,5.0,100.0,4,3333
Kothanur,3 BHK,1436.0,3.0,70.0,3,4874
Marathahalli,4 BHK,4000.0,4.0,220.0,4,5500
Somasundara Palya,3 BHK,2372.0,3.0,140.0,3,5902
R.T. Nagar,6 Bedroom,1500.0,6.0,240.0,6,16000
Rajaji Nagar,2 Bedroom,1200.0,2.0,150.0,2,12500
Rayasandra,2 BHK,1016.0,2.0,58.0,2,5708
Dodda Nekkundi,2 BHK,1100.0,2.0,48.0,2,4363
other,2 BHK,1080.0,2.0,45.0,2,4166
Electronic City,2 BHK,1070.0,2.0,60.0,2,5607
Kadubeesanahalli,3 BHK,1532.0,3.0,115.0,3,7506
Hennur,2 BHK,1040.0,2.0,42.12,2,4050
other,3 Bedroom,3100.0,3.0,250.0,3,8064
Tumkur Road,2 BHK,1137.5,2.0,48.905,2,4299
Thanisandra,3 BHK,1430.0,2.0,51.6,3,3608
Rajaji Nagar,2 Bedroom,1160.0,1.0,163.0,2,14051
ITPL,4 BHK,3262.0,5.0,205.0,4,6284
other,7 Bedroom,2100.0,7.0,410.0,7,19523
other,2 BHK,1190.0,2.0,100.0,2,8403
other,3 BHK,1827.0,3.0,100.0,3,5473
HRBR Layout,2 BHK,1210.0,2.0,75.0,2,6198
other,3 BHK,1500.0,3.0,95.0,3,6333
Rachenahalli,2 BHK,1050.0,2.0,53.5,2,5095
Hebbal,2 BHK,1251.0,2.0,88.73,2,7092
HSR Layout,3 BHK,1600.0,2.0,115.0,3,7187
Electronic City,2 BHK,1128.0,2.0,65.5,2,5806
Budigere,3 BHK,1820.0,3.0,93.0,3,5109
Thigalarapalya,3 BHK,2072.0,4.0,159.0,3,7673
Raja Rajeshwari Nagar,3 BHK,1510.0,2.0,59.0,3,3907
Hoodi,3 BHK,1925.0,2.0,114.0,3,5922
Hennur Road,3 Bedroom,2651.0,2.0,225.0,3,8487
Thigalarapalya,3 BHK,2072.0,4.0,147.0,3,7094
Kumaraswami Layout,2 BHK,1000.0,2.0,36.0,2,3600
KR Puram,3 Bedroom,1752.0,3.0,145.0,3,8276
Sahakara Nagar,2 BHK,960.0,2.0,56.0,2,5833
other,2 BHK,1180.0,2.0,63.0,2,5338
Benson Town,3 BHK,2270.0,3.0,180.0,3,7929
NRI Layout,2 Bedroom,650.0,2.0,59.0,2,9076
Lakshminarayana Pura,2 BHK,1200.0,2.0,75.0,2,6250
Bisuvanahalli,3 BHK,1180.0,2.0,46.0,3,3898
other,3 BHK,1300.0,3.0,47.0,3,3615
Bannerghatta Road,2 BHK,1157.0,2.0,40.0,2,3457
Kanakpura Road,3 BHK,1300.0,3.0,69.0,3,5307
Banashankari Stage III,3 BHK,1305.0,2.0,59.0,3,4521
Hegde Nagar,3 BHK,2006.0,4.0,196.0,3,9770
Jakkur,2 BHK,1290.0,2.0,80.0,2,6201
Hoskote,2 BHK,1170.0,2.0,45.0,2,3846
other,4 Bedroom,971.0,2.0,135.0,4,13903
Yelachenahalli,2 BHK,1400.0,2.0,78.0,2,5571
Lakshminarayana Pura,2 BHK,1200.0,2.0,75.0,2,6250
other,3 BHK,1155.0,2.0,60.0,3,5194
Hormavu,3 BHK,1725.0,3.0,85.0,3,4927
Kanakpura Road,2 BHK,700.0,2.0,40.0,2,5714
other,4 Bedroom,7000.0,5.0,2050.0,4,29285
Hoodi,2 BHK,1112.0,2.0,88.0,2,7913
Somasundara Palya,2 BHK,1178.0,2.0,78.0,2,6621
Badavala Nagar,2 BHK,1274.0,2.0,81.0,2,6357
Doddathoguru,2 BHK,915.0,2.0,32.0,2,3497
6th Phase JP Nagar,8 Bedroom,1650.0,6.0,135.0,8,8181
Hosur Road,1 BHK,760.0,1.0,21.28,1,2800
other,3 BHK,1540.0,3.0,68.0,3,4415
other,2 BHK,1206.0,2.0,38.5,2,3192
other,3 BHK,1645.0,3.0,95.0,3,5775
other,4 Bedroom,1200.0,4.0,170.0,4,14166
other,4 Bedroom,1200.0,4.0,140.0,4,11666
Nehru Nagar,4 BHK,2342.0,3.0,115.0,4,4910
other,3 BHK,1200.0,2.0,55.0,3,4583
Yelenahalli,2 BHK,1056.0,2.0,33.0,2,3125
Raja Rajeshwari Nagar,8 Bedroom,4800.0,8.0,225.0,8,4687
Kanakpura Road,3 BHK,1450.0,3.0,54.3,3,3744
other,2 BHK,1161.0,2.0,36.69,2,3160
other,2 BHK,1097.0,2.0,65.0,2,5925
Uttarahalli,2 BHK,850.0,2.0,35.0,2,4117
Green Glen Layout,3 BHK,1715.0,3.0,115.0,3,6705
Basavangudi,3 BHK,2300.0,3.0,317.0,3,13782
Electronic City,2 BHK,1200.0,2.0,59.0,2,4916
8th Phase JP Nagar,2 BHK,1080.0,2.0,38.0,2,3518
Kasavanhalli,2 BHK,1200.0,2.0,50.3,2,4191
other,4 Bedroom,2400.0,4.0,550.0,4,22916
Banashankari,3 BHK,1400.0,2.0,78.0,3,5571
Whitefield,4 Bedroom,1500.0,4.0,200.0,4,13333
Kanakpura Road,1 BHK,458.0,1.0,19.695,1,4300
Marathahalli,2 BHK,1230.0,2.0,57.0,2,4634
Kalyan nagar,3 BHK,1640.0,3.0,88.0,3,5365
Banashankari,3 BHK,1700.0,3.0,125.0,3,7352
Bommasandra,3 BHK,1478.0,2.0,44.0,3,2976
other,4 BHK,6600.0,4.0,986.0,4,14939
Banashankari,2 BHK,1020.0,2.0,40.79,2,3999
Nagarbhavi,2 Bedroom,1200.0,2.0,150.0,2,12500
Yelahanka,3 BHK,1275.0,2.0,95.0,3,7450
other,4 Bedroom,1600.0,3.0,165.0,4,10312
Kanakapura,2 BHK,1110.0,2.0,38.85,2,3500
Uttarahalli,3 BHK,1627.86,3.0,88.0,3,5405
other,2 BHK,620.0,2.0,22.0,2,3548
Whitefield,4 Bedroom,3004.0,4.0,285.0,4,9487
Hosa Road,2 BHK,1133.0,2.0,52.0,2,4589
Hennur Road,2 BHK,1182.0,2.0,82.5,2,6979
Tumkur Road,3 BHK,1500.0,3.0,100.0,3,6666
Hebbal Kempapura,4 BHK,3900.0,5.0,360.0,4,9230
Marathahalli,2 BHK,1200.0,2.0,58.0,2,4833
Raja Rajeshwari Nagar,2 BHK,1128.0,2.0,48.79,2,4325
other,2 BHK,900.0,2.0,110.0,2,12222
Amruthahalli,2 BHK,1100.0,2.0,45.0,2,4090
Kengeri Satellite Town,2 BHK,635.0,1.0,22.0,2,3464
Ardendale,4 BHK,2062.0,3.0,140.0,4,6789
Thanisandra,3 BHK,1874.0,4.0,130.0,3,6937
Vidyaranyapura,4 Bedroom,600.0,3.0,78.0,4,13000
Old Madras Road,2 BHK,1171.0,2.0,75.0,2,6404
Sarjapur,3 BHK,2020.0,3.0,85.0,3,4207
8th Phase JP Nagar,2 BHK,1059.0,2.0,34.5,2,3257
Panathur,3 BHK,1315.0,2.0,59.1,3,4494
Jakkur,3 BHK,1760.0,3.0,110.0,3,6250
other,2 Bedroom,1200.0,2.0,125.0,2,10416
Hennur Road,3 BHK,1192.0,2.0,62.0,3,5201
Hebbal,3 BHK,1645.0,3.0,121.0,3,7355
Whitefield,2 BHK,1250.0,2.0,72.0,2,5760
Whitefield,1 BHK,709.0,1.0,34.385,1,4849
Whitefield,2 BHK,1340.0,2.0,77.0,2,5746
Cooke Town,3 BHK,2388.0,4.0,239.0,3,10008
Hosa Road,2 BHK,1369.1,2.0,104.0,2,7596
other,2 BHK,1355.0,2.0,75.0,2,5535
Bannerghatta Road,2 BHK,1150.0,2.0,65.0,2,5652
Kanakapura,3 BHK,1938.0,3.0,113.0,3,5830
Begur Road,4 BHK,2462.0,4.0,124.0,4,5036
Budigere,3 BHK,1636.0,3.0,85.0,3,5195
Akshaya Nagar,3 BHK,1430.0,2.0,75.0,3,5244
other,3 BHK,2000.0,3.0,85.0,3,4250
Electronic City Phase II,3 BHK,993.0,2.0,50.0,3,5035
Laggere,4 Bedroom,1260.0,4.0,150.0,4,11904
Kanakpura Road,3 BHK,1700.0,3.0,88.0,3,5176
Yelahanka,2 BHK,1270.0,2.0,78.0,2,6141
Anjanapura,3 Bedroom,1500.0,3.0,121.0,3,8066
Koramangala,3 BHK,2300.0,3.0,290.0,3,12608
other,4 Bedroom,7500.0,4.0,900.0,4,12000
Marathahalli,3 BHK,1650.0,3.0,86.45,3,5239
Panathur,2 BHK,1109.0,2.0,63.0,2,5680
other,2 BHK,1280.0,2.0,49.75,2,3886
Sarjapur,4 Bedroom,2885.0,3.0,185.0,4,6412
Banashankari,9 Bedroom,1200.0,9.0,145.0,9,12083
Devanahalli,3 Bedroom,1600.0,3.0,180.0,3,11250
Kothanur,3 BHK,1462.0,3.0,71.0,3,4856
Marathahalli,1 BHK,607.0,1.0,35.0,1,5766
Kammasandra,2 BHK,930.0,2.0,25.0,2,2688
Varthur,4 Bedroom,4500.0,5.0,300.0,4,6666
Sahakara Nagar,5 Bedroom,1200.0,5.0,160.0,5,13333
Malleshwaram,3 BHK,2600.0,3.0,237.0,3,9115
JP Nagar,4 BHK,3000.0,4.0,140.0,4,4666
Electronic City Phase II,3 BHK,1418.0,3.0,52.47,3,3700
other,2 Bedroom,4200.0,1.0,160.0,2,3809
EPIP Zone,2 BHK,1125.0,2.0,65.0,2,5777
CV Raman Nagar,3 BHK,1825.0,3.0,126.0,3,6904
Whitefield,2 BHK,1320.0,2.0,79.5,2,6022
JP Nagar,4 Bedroom,3000.0,4.0,140.0,4,4666
HAL 2nd Stage,3 BHK,1490.0,2.0,300.0,3,20134
other,4 Bedroom,1650.0,3.0,300.0,4,18181
Whitefield,2 BHK,750.0,2.0,52.7,2,7026
Yelahanka,3 BHK,1400.0,2.0,80.54,3,5752
other,4 BHK,4104.0,4.0,360.0,4,8771
Rajaji Nagar,4 BHK,5000.0,4.0,375.0,4,7500
Mahadevpura,5 Bedroom,3295.0,4.0,260.0,5,7890
other,4 Bedroom,1150.0,3.0,110.0,4,9565
Whitefield,3 Bedroom,1200.0,3.0,61.96,3,5163
Poorna Pragna Layout,2 BHK,965.0,2.0,48.0,2,4974
Gottigere,1 Bedroom,812.0,1.0,26.0,1,3201
KR Puram,2 BHK,1100.0,2.0,43.0,2,3909
Whitefield,2 BHK,1180.0,2.0,41.0,2,3474
HSR Layout,2 BHK,1145.0,2.0,48.0,2,4192
Haralur Road,2 BHK,1027.0,2.0,44.0,2,4284
other,2 BHK,1215.0,2.0,49.85,2,4102
Hegde Nagar,2 Bedroom,1200.0,2.0,75.0,2,6250
Kenchenahalli,2 BHK,1015.0,2.0,58.0,2,5714
Kaggalipura,3 Bedroom,2400.0,3.0,165.0,3,6875
Whitefield,4 Bedroom,3000.0,4.0,306.0,4,10200
7th Phase JP Nagar,3 Bedroom,3300.0,4.0,160.0,3,4848
other,2 BHK,1076.0,2.0,52.0,2,4832
JP Nagar,2 BHK,1300.0,2.0,109.0,2,8384
Hebbal Kempapura,2 BHK,1300.0,2.0,102.0,2,7846
Hosa Road,3 BHK,1512.0,3.0,123.0,3,8134
Bhoganhalli,3 BHK,1053.4,3.0,88.91,3,8440
Yeshwanthpur,1 BHK,605.0,1.0,41.745,1,6899
Hormavu,2 BHK,1153.0,2.0,65.0,2,5637
other,2 Bedroom,2080.0,2.0,150.0,2,7211
Hulimavu,2 BHK,1255.0,2.0,69.0,2,5498
Marathahalli,3 BHK,1583.0,3.0,109.0,3,6885
other,2 BHK,1505.0,2.0,90.0,2,5980
Kathriguppe,3 BHK,1365.0,2.0,75.08,3,5500
other,3 Bedroom,1200.0,3.0,95.75,3,7979
other,3 BHK,1810.0,4.0,148.0,3,8176
Magadi Road,1 Bedroom,440.0,1.0,35.0,1,7954
Subramanyapura,2 BHK,975.0,2.0,63.0,2,6461
other,2 BHK,1250.0,2.0,39.0,2,3120
other,3 BHK,1530.0,2.0,97.0,3,6339
Kereguddadahalli,3 BHK,1300.0,2.0,39.0,3,3000
Channasandra,2 BHK,1115.0,2.0,40.0,2,3587
Sarjapur  Road,2 BHK,1308.0,2.0,83.0,2,6345
Parappana Agrahara,2 BHK,1194.0,2.0,47.0,2,3936
Uttarahalli,3 BHK,1490.0,2.0,59.6,3,4000
Bannerghatta Road,4 Bedroom,10000.0,4.0,450.0,4,4500
Sarjapur  Road,3 BHK,1592.0,3.0,91.0,3,5716
Rajaji Nagar,2 BHK,1222.0,2.0,112.0,2,9165
Raja Rajeshwari Nagar,2 BHK,1267.0,2.0,42.85,2,3382
Balagere,1 BHK,656.0,1.0,53.0,1,8079
Dodda Nekkundi,4 Bedroom,3400.0,4.0,530.0,4,15588
Garudachar Palya,4 BHK,4500.0,5.0,400.0,4,8888
Chandapura,2 Bedroom,1200.0,1.0,36.0,2,3000
Thanisandra,3 BHK,1698.0,3.0,102.0,3,6007
other,3 BHK,1500.0,3.0,70.0,3,4666
other,4 Bedroom,2000.0,5.0,80.0,4,4000
Electronic City,2 BHK,1152.0,2.0,65.75,2,5707
Raja Rajeshwari Nagar,1 BHK,550.0,1.0,21.0,1,3818
Pattandur Agrahara,2 BHK,900.0,2.0,42.0,2,4666
other,3 BHK,1410.0,2.0,45.12,3,3200
Akshaya Nagar,3 BHK,1419.0,2.0,73.0,3,5144
Amruthahalli,2 BHK,1200.0,2.0,55.0,2,4583
AECS Layout,3 BHK,1800.0,3.0,65.0,3,3611
Gottigere,2 BHK,990.0,2.0,40.0,2,4040
Sarjapur  Road,2 BHK,1016.0,2.0,45.0,2,4429
Bannerghatta Road,2 BHK,1100.0,2.0,39.0,2,3545
Ramamurthy Nagar,4 BHK,850.0,2.0,65.0,4,7647
Whitefield,2 BHK,1095.0,2.0,35.0,2,3196
other,3 BHK,2000.0,3.0,85.0,3,4250
Budigere,2 BHK,1444.0,2.0,70.0,2,4847
Whitefield,2 BHK,1415.0,2.0,67.0,2,4734
other,3 Bedroom,1500.0,2.0,130.0,3,8666
other,3 BHK,1600.0,3.0,72.0,3,4500
Whitefield,2 BHK,1105.0,2.0,41.36,2,3742
Electronic City,3 BHK,1225.0,3.0,45.03,3,3675
Frazer Town,3 BHK,2560.0,3.0,265.0,3,10351
Kanakpura Road,2 BHK,1080.0,2.0,37.8,2,3499
Dasarahalli,7 Bedroom,2400.0,3.0,150.0,7,6250
Talaghattapura,3 BHK,1856.0,3.0,135.0,3,7273
Haralur Road,3 BHK,1464.0,3.0,56.0,3,3825
other,6 Bedroom,2100.0,6.0,71.0,6,3380
Sarjapur  Road,2 BHK,1150.0,2.0,77.0,2,6695
Abbigere,5 Bedroom,1200.0,4.0,98.0,5,8166
5th Phase JP Nagar,3 BHK,1485.0,2.0,67.0,3,4511
other,3 BHK,2500.0,3.0,75.0,3,3000
Whitefield,2 Bedroom,1200.0,2.0,45.0,2,3750
Hennur Road,2 BHK,1036.0,2.0,59.46,2,5739
Whitefield,4 Bedroom,3445.0,5.0,270.0,4,7837
other,2 BHK,1400.0,2.0,80.0,2,5714
Bommanahalli,2 BHK,1110.0,2.0,33.0,2,2972
Kodichikkanahalli,2 BHK,1150.0,2.0,48.0,2,4173
Kenchenahalli,1 BHK,715.0,1.0,45.0,1,6293
Magadi Road,4 Bedroom,1200.0,4.0,130.0,4,10833
Rajaji Nagar,5 Bedroom,1200.0,3.0,260.0,5,21666
Jakkur,4 BHK,2283.0,4.0,130.0,4,5694
other,6 Bedroom,1790.0,5.0,250.0,6,13966
Hormavu,3 BHK,1453.0,2.0,43.58,3,2999
Nagarbhavi,3 BHK,1400.0,2.0,85.0,3,6071
Padmanabhanagar,3 BHK,1400.0,3.0,65.0,3,4642
Whitefield,3 BHK,1700.0,3.0,140.0,3,8235
Kanakpura Road,3 BHK,1591.0,3.0,120.0,3,7542
Nagarbhavi,3 BHK,1515.0,2.0,79.0,3,5214
Marathahalli,5 Bedroom,2500.0,3.0,200.0,5,8000
Chandapura,2 BHK,975.0,2.0,24.86,2,2549
Yeshwanthpur,3 BHK,1382.0,2.0,76.18,3,5512
Nagasandra,7 Bedroom,1750.0,5.0,350.0,7,20000
other,3 BHK,1533.0,3.0,150.0,3,9784
Cunningham Road,3 BHK,3815.0,3.0,763.0,3,20000
other,2 BHK,1100.0,2.0,38.0,2,3454
other,4 BHK,3730.0,6.0,450.0,4,12064
Banashankari Stage II,3 BHK,1500.0,2.0,120.0,3,8000
Yelahanka,3 BHK,1391.0,2.0,64.5,3,4636
Begur Road,2 BHK,1160.0,2.0,42.92,2,3700
Chandapura,1 BHK,620.0,1.0,27.0,1,4354
other,8 Bedroom,2400.0,9.0,350.0,8,14583
Vidyaranyapura,8 Bedroom,1200.0,7.0,130.0,8,10833
Whitefield,4 Bedroom,4827.0,4.0,400.0,4,8286
Kanakpura Road,3 BHK,1482.0,3.0,60.98,3,4114
Whitefield,2 BHK,1405.0,2.0,84.5,2,6014
Gunjur,2 BHK,1190.0,2.0,40.0,2,3361
Harlur,2 BHK,1174.0,2.0,78.0,2,6643
Seegehalli,2 Bedroom,1155.0,2.0,72.0,2,6233
Thanisandra,3 BHK,1411.0,3.0,93.04,3,6593
other,2 BHK,745.0,2.0,50.0,2,6711
Kereguddadahalli,2 BHK,1000.0,2.0,30.0,2,3000
Jigani,2 BHK,923.0,2.0,50.0,2,5417
Yeshwanthpur,2 BHK,1000.0,2.0,32.0,2,3200
Battarahalli,2 BHK,1276.0,2.0,45.0,2,3526
other,3 BHK,1460.0,3.0,70.0,3,4794
other,2 BHK,1355.0,2.0,95.0,2,7011
other,1 BHK,833.0,1.0,49.0,1,5882
other,8 Bedroom,1200.0,8.0,275.0,8,22916
other,5 Bedroom,1750.0,5.0,450.0,5,25714
Ramamurthy Nagar,2 BHK,1200.0,2.0,48.0,2,4000
Yeshwanthpur,1 BHK,669.0,1.0,36.85,1,5508
7th Phase JP Nagar,3 Bedroom,2200.0,4.0,105.0,3,4772
Koramangala,3 BHK,1594.0,3.0,125.0,3,7841
Kothanur,3 BHK,1790.0,3.0,105.0,3,5865
NGR Layout,2 BHK,1020.0,2.0,45.0,2,4411
other,4 Bedroom,3250.0,5.0,850.0,4,26153
other,2 BHK,1037.0,2.0,36.28,2,3498
Whitefield,3 BHK,1345.0,2.0,57.0,3,4237
Kodichikkanahalli,2 BHK,1070.0,2.0,43.0,2,4018
other,4 BHK,3000.0,5.0,270.0,4,9000
Yeshwanthpur,3 BHK,1385.0,2.0,76.18,3,5500
Kanakpura Road,2 BHK,1207.0,2.0,79.9,2,6619
Varthur,2 BHK,1140.0,2.0,55.5,2,4868
other,2 BHK,1220.0,2.0,68.0,2,5573
other,4 Bedroom,3811.0,3.0,475.0,4,12463
Iblur Village,4 BHK,3596.0,5.0,252.0,4,7007
Sarjapur  Road,3 BHK,1965.0,3.0,120.0,3,6106
Koramangala,3 BHK,1760.0,2.0,115.0,3,6534
Haralur Road,2 BHK,1225.0,2.0,70.0,2,5714
Marathahalli,2 BHK,1075.0,2.0,52.43,2,4877
Panathur,2 BHK,1000.0,2.0,68.0,2,6800
Hennur Road,3 BHK,1891.0,3.0,109.0,3,5764
Yelahanka,2 BHK,1322.0,2.0,86.0,2,6505
other,3 Bedroom,1095.0,3.0,215.0,3,19634
Bellandur,4 Bedroom,3600.0,4.0,260.0,4,7222
AECS Layout,2 BHK,1045.0,2.0,36.58,2,3500
other,3 BHK,1545.0,3.0,85.0,3,5501
Thanisandra,3 BHK,1996.0,3.0,135.0,3,6763
Brookefield,2 BHK,1125.0,2.0,70.0,2,6222
Mysore Road,3 BHK,1560.0,3.0,115.0,3,7371
BEML Layout,3 BHK,2000.0,2.0,85.0,3,4250
Murugeshpalya,4 BHK,3100.0,4.0,310.0,4,10000
other,2 BHK,780.0,2.0,35.0,2,4487
other,2 Bedroom,1350.0,2.0,110.0,2,8148
Sarjapur  Road,3 BHK,2275.0,4.0,180.0,3,7912
other,2 BHK,1350.0,2.0,41.0,2,3037
Haralur Road,3 BHK,1464.0,3.0,56.0,3,3825
Kanakpura Road,3 BHK,1570.0,3.0,64.5,3,4108
Dasanapura,1 BHK,708.0,1.0,28.32,1,4000
Sarjapur  Road,3 BHK,1680.0,3.0,160.0,3,9523
Hoodi,2 BHK,1400.0,2.0,82.0,2,5857
Bellandur,2 BHK,1310.0,2.0,72.0,2,5496
BTM Layout,6 Bedroom,3300.0,6.0,165.0,6,5000
Electronics City Phase 1,2 BHK,1175.0,2.0,57.0,2,4851
Chikkalasandra,3 BHK,1428.0,2.0,80.0,3,5602
Akshaya Nagar,4 Bedroom,750.0,4.0,62.0,4,8266
Whitefield,4 BHK,3252.0,4.0,280.0,4,8610
Haralur Road,2 BHK,1194.0,2.0,46.0,2,3852
other,8 Bedroom,750.0,8.0,200.0,8,26666
8th Phase JP Nagar,2 BHK,1080.0,2.0,35.0,2,3240
other,3 BHK,1200.0,2.0,68.0,3,5666
Whitefield,2 Bedroom,1200.0,2.0,45.0,2,3750
other,5 Bedroom,1200.0,4.0,120.0,5,10000
Bommasandra Industrial Area,3 BHK,1320.0,2.0,38.13,3,2888
Badavala Nagar,3 BHK,1842.0,3.0,115.0,3,6243
Sector 2 HSR Layout,2 BHK,1146.0,2.0,55.0,2,4799
Rachenahalli,1 BHK,440.0,1.0,28.0,1,6363
Sahakara Nagar,2 BHK,1048.0,2.0,50.0,2,4770
Thanisandra,3 BHK,1801.0,3.0,115.0,3,6385
Hennur Road,2 BHK,1232.0,2.0,69.61,2,5650
Whitefield,2 BHK,1150.0,2.0,90.0,2,7826
Talaghattapura,3 BHK,1868.0,3.0,120.0,3,6423
Begur,3 BHK,1445.0,2.0,47.1,3,3259
Hebbal Kempapura,3 BHK,1466.0,2.0,140.0,3,9549
Harlur,3 BHK,1730.0,3.0,100.0,3,5780
JP Nagar,3 BHK,3860.0,3.0,402.0,3,10414
Somasundara Palya,2 BHK,1185.0,2.0,70.0,2,5907
Yeshwanthpur,3 BHK,3027.0,4.0,260.0,3,8589
Whitefield,4 BHK,2400.0,4.0,147.0,4,6125
Kaval Byrasandra,2 BHK,1185.0,2.0,49.0,2,4135
other,2 BHK,1150.0,2.0,49.0,2,4260
other,5 Bedroom,2000.0,4.0,180.0,5,9000
other,6 Bedroom,875.0,3.0,98.0,6,11200
Pai Layout,2 BHK,1075.0,2.0,35.0,2,3255
Hebbal,3 BHK,1760.0,3.0,123.0,3,6988
Banashankari,2 BHK,1600.0,2.0,63.98,2,3998
other,4 Bedroom,910.0,4.0,140.0,4,15384
Haralur Road,2 BHK,1000.0,2.0,78.0,2,7800
7th Phase JP Nagar,2 BHK,1215.0,2.0,90.0,2,7407
EPIP Zone,3 BHK,1734.0,3.0,125.0,3,7208
Akshaya Nagar,2 BHK,1280.0,2.0,60.0,2,4687
Rachenahalli,2 BHK,1113.0,2.0,55.0,2,4941
Horamavu Banaswadi,2 BHK,1272.0,2.0,48.0,2,3773
Kereguddadahalli,3 BHK,1240.0,3.0,41.0,3,3306
Varthur,8 Bedroom,704.0,8.0,92.0,8,13068
Yelahanka,2 BHK,1100.0,2.0,70.0,2,6363
Ambalipura,2 BHK,896.0,2.0,30.13,2,3362
Rachenahalli,4 BHK,3657.0,5.0,220.0,4,6015
BTM 2nd Stage,2 BHK,1200.0,2.0,35.0,2,2916
Ramagondanahalli,4 BHK,2800.0,4.0,155.0,4,5535
Hebbal,4 BHK,2790.0,4.0,198.0,4,7096
Ambalipura,3 BHK,3500.0,3.0,198.0,3,5657
Ramamurthy Nagar,3 Bedroom,3600.0,3.0,150.0,3,4166
Sarjapur,2 BHK,1500.0,2.0,30.0,2,2000
other,8 Bedroom,1200.0,7.0,220.0,8,18333
other,2 BHK,1096.0,2.0,48.0,2,4379
other,2 Bedroom,580.0,2.0,95.0,2,16379
other,2 BHK,1094.0,2.0,44.0,2,4021
other,2 BHK,1100.0,2.0,49.0,2,4454
Iblur Village,4 BHK,2987.5,5.0,191.0,4,6393
Koramangala,4 BHK,2503.0,6.0,325.0,4,12984
Nagarbhavi,2 BHK,1145.0,2.0,40.0,2,3493
Rachenahalli,3 BHK,1550.0,3.0,74.5,3,4806
Seegehalli,3 BHK,1830.0,4.0,82.0,3,4480
Hoodi,3 BHK,1746.0,3.0,84.95,3,4865
other,4 Bedroom,1280.0,4.0,115.0,4,8984
Kengeri,2 BHK,918.0,2.0,27.54,2,3000
Chandapura,1 BHK,645.0,1.0,16.45,1,2550
Banashankari Stage III,3 BHK,1305.0,2.0,60.0,3,4597
Thigalarapalya,2 BHK,1207.0,2.0,62.0,2,5136
Thanisandra,3 BHK,1678.0,3.0,125.0,3,7449
Kaggalipura,2 BHK,950.0,2.0,50.0,2,5263
Munnekollal,10 Bedroom,1200.0,7.0,240.0,10,20000
Electronic City,2 BHK,1070.0,2.0,55.0,2,5140
Ardendale,3 BHK,1732.46,3.0,92.0,3,5310
Attibele,3 BHK,1639.0,2.0,40.98,3,2500
Marathahalli,2 BHK,1360.0,2.0,103.0,2,7573
Malleshpalya,4 Bedroom,2592.0,4.0,250.0,4,9645
other,3 Bedroom,4470.0,3.0,700.0,3,15659
other,5 Bedroom,720.0,4.0,62.0,5,8611
Lakshminarayana Pura,3 BHK,1656.0,3.0,150.0,3,9057
Sarjapur  Road,2 BHK,985.0,2.0,49.0,2,4974
Dodda Nekkundi,3 BHK,1999.0,3.0,135.0,3,6753
7th Phase JP Nagar,2 BHK,1050.0,2.0,75.0,2,7142
Whitefield,3 BHK,1440.0,2.0,64.0,3,4444
Basavangudi,3 BHK,1600.0,2.0,160.0,3,10000
Ramagondanahalli,2 BHK,1253.0,2.0,52.66,2,4202
other,3 Bedroom,1200.0,3.0,85.0,3,7083
Padmanabhanagar,3 BHK,1360.0,2.0,75.0,3,5514
other,2 BHK,1086.0,2.0,47.5,2,4373
other,3 BHK,1655.0,3.0,155.0,3,9365
Old Madras Road,4 Bedroom,2600.0,4.0,260.0,4,10000
Sarjapur  Road,2 BHK,1418.0,2.0,78.0,2,5500
Kalena Agrahara,2 BHK,1270.0,2.0,100.0,2,7874
Begur Road,3 BHK,1410.0,2.0,52.17,3,3700
other,2 BHK,1150.0,2.0,74.0,2,6434
Ramamurthy Nagar,2 Bedroom,600.0,3.0,45.0,2,7500
other,3 BHK,1440.0,3.0,65.8,3,4569
other,3 Bedroom,1200.0,2.0,108.0,3,9000
other,2 BHK,1200.0,2.0,75.0,2,6250
Begur Road,3 BHK,1410.0,2.0,50.06,3,3550
Chandapura,4 Bedroom,1200.0,5.0,92.0,4,7666
Jalahalli,3 Bedroom,600.0,3.0,60.0,3,10000
Whitefield,1 BHK,624.0,1.0,35.0,1,5608
Electronic City,2 BHK,1060.0,2.0,58.0,2,5471
Hebbal,4 BHK,3960.0,5.0,386.0,4,9747
Whitefield,2 Bedroom,1200.0,2.0,45.0,2,3750
Vidyaranyapura,4 Bedroom,1200.0,4.0,175.0,4,14583
Sarjapur,4 BHK,1917.0,3.0,60.0,4,3129
KR Puram,2 BHK,950.0,2.0,40.0,2,4210
Channasandra,2 Bedroom,3040.0,2.0,48.0,2,1578
BTM 2nd Stage,3 BHK,1250.0,3.0,50.0,3,4000
Vishwapriya Layout,2 BHK,1235.0,2.0,60.0,2,4858
Budigere,1 BHK,722.0,1.0,46.0,1,6371
Kanakpura Road,4 Bedroom,2775.0,3.0,165.0,4,5945
Sarjapur  Road,2 BHK,1200.0,2.0,93.0,2,7750
Bannerghatta Road,3 BHK,1400.0,3.0,68.0,3,4857
other,3 BHK,1225.0,3.0,60.0,3,4897
Marathahalli,2 BHK,1220.0,2.0,58.5,2,4795
Nagavarapalya,1 BHK,705.0,1.0,52.5,1,7446
Battarahalli,3 BHK,1880.0,3.0,94.36,3,5019
Yelahanka,2 Bedroom,1200.0,2.0,70.0,2,5833
other,3 BHK,1550.0,2.0,60.0,3,3870
Lingadheeranahalli,3 BHK,1893.0,3.0,130.0,3,6867
other,3 BHK,1650.0,3.0,75.0,3,4545
other,2 BHK,1113.0,2.0,46.0,2,4132
other,2 Bedroom,1500.0,2.0,203.0,2,13533
other,9 Bedroom,1800.0,9.0,180.0,9,10000
Sarjapur  Road,3 BHK,1586.0,3.0,70.0,3,4413
Seegehalli,3 Bedroom,4000.0,3.0,235.0,3,5875
Yelahanka New Town,3 BHK,1541.0,3.0,80.0,3,5191
Haralur Road,3 BHK,1830.0,3.0,101.0,3,5519
Kodigehalli,2 BHK,1200.0,2.0,48.0,2,4000
other,3 Bedroom,1800.0,2.0,72.0,3,4000
Bommasandra,2 BHK,1015.0,2.0,40.0,2,3940
Raja Rajeshwari Nagar,2 BHK,1120.0,2.0,37.95,2,3388
Hoodi,1 BHK,706.0,1.0,48.24,1,6832
other,4 Bedroom,1350.0,4.0,85.0,4,6296
Kumaraswami Layout,6 Bedroom,610.0,4.0,95.0,6,15573
Kanakpura Road,3 Bedroom,1900.0,3.0,99.0,3,5210
Harlur,4 Bedroom,1200.0,4.0,243.0,4,20250
Seegehalli,4 BHK,3000.0,5.0,150.0,4,5000
Bisuvanahalli,3 BHK,1075.0,2.0,42.0,3,3906
Kanakpura Road,2 BHK,1290.0,2.0,80.0,2,6201
Akshaya Nagar,2 BHK,1200.0,2.0,50.0,2,4166
Hormavu,2 BHK,1150.0,2.0,53.11,2,4618
Uttarahalli,3 BHK,1360.0,2.0,61.2,3,4500
other,9 BHK,4500.0,9.0,500.0,9,11111
other,2 BHK,935.0,2.0,32.73,2,3500
Cox Town,3 BHK,1730.0,3.0,140.0,3,8092
Sarjapur  Road,2 BHK,1260.0,2.0,53.0,2,4206
other,6 Bedroom,675.0,6.0,125.0,6,18518
Marathahalli,4 Bedroom,4000.0,4.0,650.0,4,16250
Marathahalli,2 BHK,950.0,2.0,45.68,2,4808
other,2 Bedroom,600.0,3.0,85.0,2,14166
other,5 Bedroom,2500.0,5.0,120.0,5,4800
Hegde Nagar,4 BHK,3216.0,5.0,300.0,4,9328
other,6 Bedroom,800.0,3.0,120.0,6,15000
Budigere,3 BHK,1991.0,4.0,88.0,3,4419
Sarjapur  Road,2 BHK,1115.0,2.0,60.0,2,5381
Koramangala,3 BHK,1900.0,3.0,150.0,3,7894
Whitefield,2 BHK,1109.0,2.0,41.49,2,3741
Rajiv Nagar,4 BHK,2340.0,5.0,129.0,4,5512
Judicial Layout,6 Bedroom,1200.0,6.0,300.0,6,25000
Whitefield,4 Bedroom,4356.0,5.0,850.0,4,19513
9th Phase JP Nagar,2 BHK,1005.0,2.0,43.0,2,4278
Kengeri,1 BHK,416.0,1.0,19.5,1,4687
1st Phase JP Nagar,2 BHK,1394.0,2.0,100.0,2,7173
Harlur,3 BHK,2138.0,3.0,110.0,3,5144
other,3 BHK,1550.0,3.0,160.0,3,10322
Nagavarapalya,3 BHK,1300.0,3.0,80.0,3,6153
Thanisandra,2 BHK,1265.0,2.0,82.0,2,6482
other,2 BHK,1140.0,2.0,52.0,2,4561
other,2 BHK,1050.0,2.0,40.0,2,3809
Thanisandra,2 BHK,1100.0,2.0,48.0,2,4363
Varthur,2 BHK,1012.0,2.0,59.0,2,5830
Varthur,4 Bedroom,1600.0,4.0,112.0,4,7000
Yelahanka,2 BHK,1326.0,2.0,57.0,2,4298
Hebbal Kempapura,3 BHK,1654.0,2.0,85.0,3,5139
other,3 Bedroom,1705.0,2.0,290.0,3,17008
Chikkalasandra,2 BHK,1070.0,2.0,42.8,2,4000
Whitefield,3 BHK,1700.0,3.0,95.0,3,5588
Bhoganhalli,2 BHK,910.2,2.0,80.64,2,8859
Kogilu,2 BHK,1140.0,2.0,50.66,2,4443
Whitefield,4 Bedroom,4960.0,4.0,400.0,4,8064
Talaghattapura,3 BHK,2254.0,3.0,170.0,3,7542
Kanakapura,2 BHK,945.0,2.0,47.0,2,4973
Begur Road,4 BHK,2462.0,6.0,150.0,4,6092
Giri Nagar,3 Bedroom,1350.0,3.0,240.0,3,17777
Marathahalli,1 BHK,615.0,1.0,44.0,1,7154
Ramamurthy Nagar,2 Bedroom,1110.0,2.0,65.0,2,5855
Haralur Road,1 BHK,575.0,1.0,88.0,1,15304
Electronic City,2 BHK,1025.0,2.0,54.1,2,5278
other,3 BHK,1758.0,3.0,95.0,3,5403
other,4 BHK,3179.0,5.0,351.0,4,11041
Kaikondrahalli,3 BHK,1342.0,3.0,92.0,3,6855
Kalyan nagar,3 BHK,1800.0,3.0,85.0,3,4722
Whitefield,4 BHK,2928.0,4.0,198.0,4,6762
Doddathoguru,2 BHK,1140.0,2.0,32.0,2,2807
Raja Rajeshwari Nagar,2 BHK,1608.0,2.0,85.0,2,5286
Kadugodi,3 BHK,1645.0,2.0,82.0,3,4984
Ramamurthy Nagar,3 BHK,1410.0,2.0,56.4,3,4000
Harlur,2 BHK,1200.0,2.0,52.0,2,4333
8th Phase JP Nagar,3 BHK,1275.0,2.0,41.5,3,3254
Dodda Nekkundi,4 Bedroom,3800.0,4.0,600.0,4,15789
other,3 BHK,909.0,2.0,40.0,3,4400
other,2 Bedroom,1200.0,3.0,85.0,2,7083
Kalyan nagar,2 BHK,1080.0,2.0,73.0,2,6759
other,2 Bedroom,1200.0,2.0,34.0,2,2833
Akshaya Nagar,2 BHK,1092.0,2.0,60.0,2,5494
Jalahalli,3 BHK,1932.0,4.0,135.0,3,6987
KR Puram,2 BHK,1156.0,2.0,72.0,2,6228
Bellandur,2 BHK,924.0,2.0,35.11,2,3799
Thubarahalli,2 BHK,1330.0,2.0,75.0,2,5639
Thanisandra,3 BHK,1917.0,4.0,113.0,3,5894
Chandapura,2 BHK,985.0,2.0,25.12,2,2550
other,6 Bedroom,2400.0,6.0,408.0,6,17000
other,4 Bedroom,1800.0,5.0,500.0,4,27777
other,2 Bedroom,1290.0,1.0,225.0,2,17441
Kodigehalli,1 BHK,655.0,1.0,53.0,1,8091
other,3 BHK,1800.0,3.0,140.0,3,7777
Margondanahalli,4 Bedroom,1360.0,2.0,45.0,4,3308
Bellandur,3 BHK,1785.0,3.0,118.0,3,6610
Kadugodi,7 BHK,4000.0,7.0,130.0,7,3250
Hebbal,3 BHK,1645.0,3.0,120.0,3,7294
Gottigere,2 BHK,1235.0,2.0,63.0,2,5101
Anandapura,2 Bedroom,900.0,2.0,48.0,2,5333
Thanisandra,1 BHK,662.0,1.0,42.0,1,6344
TC Palaya,5 Bedroom,1440.0,5.0,100.0,5,6944
Whitefield,2 BHK,1215.0,2.0,62.0,2,5102
Bisuvanahalli,2 BHK,845.0,2.0,37.0,2,4378
Sarjapur  Road,3 BHK,1525.0,2.0,87.0,3,5704
other,2 BHK,1000.0,2.0,48.0,2,4800
Hebbal,3 BHK,1255.0,2.0,90.0,3,7171
Mysore Road,3 BHK,1560.0,3.0,120.0,3,7692
Kasavanhalli,3 Bedroom,2100.0,3.0,110.0,3,5238
Kodichikkanahalli,3 BHK,1620.0,2.0,85.0,3,5246
Marathahalli,3 BHK,1595.0,3.0,90.0,3,5642
other,2 BHK,1095.0,2.0,38.52,2,3517
9th Phase JP Nagar,2 BHK,835.0,2.0,37.0,2,4431
Hennur Road,3 BHK,2182.0,3.0,72.04,3,3301
Thigalarapalya,2 BHK,1418.0,2.0,106.0,2,7475
Sanjay nagar,3 BHK,1900.0,2.0,180.0,3,9473
Whitefield,3 BHK,2700.0,5.0,175.0,3,6481
BTM 2nd Stage,2 BHK,1070.0,2.0,48.0,2,4485
Ramamurthy Nagar,9 Bedroom,2000.0,9.0,150.0,9,7500
Munnekollal,43 Bedroom,2400.0,40.0,660.0,43,27500
Electronics City Phase 1,1 BHK,755.0,1.0,42.41,1,5617
Hebbal,3 BHK,1645.0,3.0,117.0,3,7112
Sarjapur,3 Bedroom,2400.0,3.0,190.0,3,7916
Chandapura,1 BHK,645.0,1.0,16.45,1,2550
other,4 BHK,3900.0,4.0,333.0,4,8538
other,3 Bedroom,2400.0,4.0,200.0,3,8333
other,10 Bedroom,2416.0,10.0,600.0,10,24834
Electronics City Phase 1,2 BHK,919.0,2.0,28.0,2,3046
Bannerghatta Road,3 BHK,1725.0,3.0,110.0,3,6376
Sarjapur  Road,2 BHK,1346.0,2.0,71.32,2,5298
Electronic City Phase II,2 BHK,1070.0,2.0,45.0,2,4205
Begur Road,2 BHK,1225.0,2.0,35.52,2,2899
Sarakki Nagar,6 Bedroom,880.0,6.0,75.0,6,8522
Neeladri Nagar,1 BHK,640.0,1.0,20.5,1,3203
Chikkalasandra,3 BHK,1310.0,2.0,52.4,3,4000
Kanakapura,2 BHK,1190.0,2.0,39.0,2,3277
other,3 BHK,1645.0,3.0,87.5,3,5319
Marathahalli,2 BHK,1350.0,2.0,89.0,2,6592
Vittasandra,2 BHK,1404.0,2.0,68.8,2,4900
Electronic City,3 BHK,1105.0,2.0,35.0,3,3167
Kanakpura Road,2 BHK,900.0,2.0,41.0,2,4555
other,4 Bedroom,1650.0,4.0,320.0,4,19393
Dodda Nekkundi,2 BHK,1100.0,2.0,41.15,2,3740
Thanisandra,4 Bedroom,2100.0,4.0,95.0,4,4523
Thanisandra,3 BHK,1948.0,3.0,130.0,3,6673
Singasandra,3 BHK,1464.0,3.0,56.0,3,3825
Raja Rajeshwari Nagar,2 BHK,1210.0,2.0,45.0,2,3719
Yelahanka,5 Bedroom,4000.0,5.0,220.0,5,5500
other,2 BHK,1252.0,2.0,55.0,2,4392
Nagarbhavi,5 Bedroom,1200.0,4.0,240.0,5,20000
Sarjapur  Road,5 BHK,3905.0,7.0,290.0,5,7426
other,3 BHK,2600.0,3.0,140.0,3,5384
7th Phase JP Nagar,3 BHK,1300.0,3.0,89.13,3,6856
7th Phase JP Nagar,2 BHK,1130.0,2.0,73.0,2,6460
Budigere,4 Bedroom,1200.0,3.0,135.0,4,11250
Thanisandra,3 BHK,1690.0,3.0,125.0,3,7396
Padmanabhanagar,3 Bedroom,4500.0,4.0,360.0,3,8000
other,2 BHK,950.0,2.0,60.0,2,6315
Marathahalli,2 BHK,1020.0,2.0,57.0,2,5588
other,4 BHK,2107.0,4.0,95.0,4,4508
Marathahalli,3 BHK,1735.0,3.0,72.0,3,4149
Sarjapur  Road,2 BHK,915.0,2.0,37.0,2,4043
other,3 BHK,1450.0,3.0,68.0,3,4689
Thigalarapalya,3 BHK,2215.0,4.0,156.0,3,7042
other,3 BHK,2431.0,3.0,100.0,3,4113
Hennur Road,3 BHK,1804.0,3.0,102.0,3,5654
Marathahalli,2 BHK,1600.0,3.0,88.0,2,5500
Uttarahalli,2 BHK,1280.0,2.0,75.0,2,5859
Hebbal,3 BHK,1255.0,3.0,77.68,3,6189
Koramangala,2 BHK,1320.0,2.0,150.0,2,11363
Hoodi,2 BHK,1257.0,2.0,69.0,2,5489
Raja Rajeshwari Nagar,2 BHK,1240.0,2.0,53.27,2,4295
other,4 Bedroom,1200.0,4.0,240.0,4,20000
Bannerghatta Road,6 Bedroom,1850.0,6.0,150.0,6,8108
Hulimavu,2 BHK,1255.0,2.0,66.0,2,5258
Ananth Nagar,2 BHK,870.0,2.0,36.0,2,4137
Tumkur Road,3 BHK,1789.0,3.0,98.0,3,5477
Basaveshwara Nagar,6 Bedroom,800.0,6.0,110.0,6,13750
Electronic City,2 BHK,660.0,1.0,18.0,2,2727
other,3 Bedroom,600.0,3.0,54.0,3,9000
other,2 BHK,1155.0,2.0,56.0,2,4848
R.T. Nagar,2 BHK,1235.0,2.0,65.0,2,5263
Yelahanka,4 Bedroom,2675.0,4.0,140.0,4,5233
CV Raman Nagar,2 BHK,1025.0,2.0,50.0,2,4878
Raja Rajeshwari Nagar,3 BHK,1395.0,2.0,72.0,3,5161
Rachenahalli,3 BHK,1530.0,2.0,74.4,3,4862
Dodda Nekkundi,3 BHK,1252.0,2.0,71.0,3,5670
Hennur Road,2 BHK,1081.0,2.0,30.27,2,2800
Haralur Road,3 BHK,1255.0,3.0,130.0,3,10358
Ardendale,4 BHK,2062.0,3.0,150.0,4,7274
Choodasandra,3 BHK,1220.0,3.0,56.0,3,4590
Jigani,3 Bedroom,1408.0,2.0,55.0,3,3906
Bisuvanahalli,3 BHK,1180.0,2.0,60.0,3,5084
Sarjapur  Road,1 BHK,715.0,1.0,29.0,1,4055
Sarjapur  Road,2 BHK,1230.0,2.0,94.5,2,7682
Kaggadasapura,2 BHK,1035.0,2.0,45.5,2,4396
EPIP Zone,4 BHK,3360.0,5.0,225.0,4,6696
Kathriguppe,3 BHK,1300.0,3.0,77.99,3,5999
Kanakpura Road,3 BHK,1100.0,3.0,52.97,3,4815
Whitefield,2 BHK,1216.0,2.0,84.0,2,6907
other,2 BHK,1450.0,2.0,62.63,2,4319
Nagavarapalya,1 BHK,705.0,1.0,57.0,1,8085
Gottigere,3 BHK,1385.0,3.0,64.0,3,4620
Anekal,2 Bedroom,1200.0,2.0,36.1,2,3008
Electronic City Phase II,3 BHK,1310.0,2.0,32.75,3,2500
Sarjapur  Road,2 BHK,1043.0,2.0,39.5,2,3787
Koramangala,2 BHK,1320.0,2.0,110.0,2,8333
other,3 BHK,1900.0,3.0,205.0,3,10789
Varthur,2 BHK,1140.0,2.0,39.9,2,3500
Kadugodi,2 BHK,1100.0,2.0,45.0,2,4090
Hebbal,3 BHK,1586.0,3.0,90.0,3,5674
Chandapura,2 BHK,975.0,2.0,24.86,2,2549
other,2 BHK,1365.0,3.0,60.0,2,4395
Rajaji Nagar,2 BHK,1263.0,2.0,107.0,2,8471
Begur Road,3 BHK,1565.0,2.0,59.76,3,3818
Karuna Nagar,3 Bedroom,1163.0,3.0,180.0,3,15477
Whitefield,4 BHK,2882.0,5.0,204.0,4,7078
Hennur Road,3 BHK,2320.0,3.0,170.0,3,7327
Sarjapur  Road,3 BHK,1750.0,3.0,99.0,3,5657
Kudlu,2 BHK,1143.0,2.0,55.0,2,4811
Hennur Road,3 BHK,1936.0,3.0,118.0,3,6095
Electronic City,2 BHK,660.0,1.0,15.0,2,2272
Chandapura,2 BHK,700.0,1.0,19.0,2,2714
other,3 BHK,1402.0,2.0,85.0,3,6062
Hebbal,4 BHK,4640.0,4.0,600.0,4,12931
Electronic City,2 BHK,1000.0,2.0,41.0,2,4100
Gottigere,2 BHK,1000.0,2.0,35.0,2,3500
Kengeri Satellite Town,2 BHK,818.0,2.0,26.0,2,3178
Bommenahalli,4 Bedroom,1680.0,3.0,135.0,4,8035
Kalyan nagar,4 BHK,2422.0,3.0,150.0,4,6193
Indira Nagar,2 BHK,1145.0,2.0,100.0,2,8733
Chandapura,3 BHK,1190.0,2.0,30.35,3,2550
Sarjapur  Road,2 BHK,1145.0,2.0,62.0,2,5414
Somasundara Palya,3 BHK,1575.0,3.0,68.0,3,4317
Bannerghatta Road,3 Bedroom,9000.0,4.0,390.0,3,4333
Thanisandra,2 BHK,1200.0,2.0,46.0,2,3833
Yelahanka,3 BHK,1500.0,2.0,63.0,3,4200
Ramamurthy Nagar,4 BHK,2560.0,4.0,75.0,4,2929
Kothanur,3 BHK,1583.0,3.0,76.1,3,4807
Kanakpura Road,2 BHK,700.0,2.0,34.98,2,4997
Bannerghatta Road,2 BHK,950.0,2.0,38.0,2,4000
Banashankari,2 BHK,1041.0,2.0,36.44,2,3500
Thubarahalli,2 BHK,1085.0,2.0,70.0,2,6451
Anjanapura,2 BHK,1070.0,2.0,37.0,2,3457
Raja Rajeshwari Nagar,2 BHK,1165.0,2.0,45.0,2,3862
Bommasandra,2 BHK,1000.0,2.0,35.0,2,3500
Horamavu Agara,2 BHK,1169.0,2.0,64.0,2,5474
other,4 Bedroom,1150.0,4.0,110.0,4,9565
Whitefield,3 BHK,1840.0,3.0,140.0,3,7608
other,3 BHK,1550.0,3.0,75.0,3,4838
Kaval Byrasandra,2 BHK,997.0,2.0,53.0,2,5315
Gunjur,2 BHK,1457.0,2.0,60.0,2,4118
ISRO Layout,2 BHK,1000.0,2.0,60.0,2,6000
KR Puram,2 Bedroom,1000.0,2.0,60.0,2,6000
Sarjapur  Road,3 Bedroom,1500.0,3.0,88.0,3,5866
other,5 Bedroom,958.0,3.0,210.0,5,21920
Lakshminarayana Pura,2 BHK,1165.0,2.0,75.0,2,6437
other,7 Bedroom,1200.0,7.0,169.0,7,14083
Kasavanhalli,3 BHK,1787.0,3.0,123.0,3,6883
Raja Rajeshwari Nagar,3 BHK,1288.0,2.0,70.0,3,5434
Thanisandra,3 BHK,1430.0,2.0,54.11,3,3783
Hulimavu,3 BHK,3035.0,5.0,271.0,3,8929
Doddathoguru,1 BHK,750.0,1.0,25.5,1,3400
Yelachenahalli,4 Bedroom,900.0,2.0,115.0,4,12777
Channasandra,3 Bedroom,1200.0,3.0,67.77,3,5647
Electronic City,3 BHK,1500.0,2.0,78.0,3,5200
Ambalipura,2 BHK,1230.0,2.0,66.25,2,5386
Marathahalli,1 BHK,780.0,1.0,55.0,1,7051
Bannerghatta Road,2 BHK,1220.0,2.0,80.0,2,6557
Kanakpura Road,2 BHK,1110.0,2.0,43.0,2,3873
other,3 BHK,2100.0,3.0,240.0,3,11428
Hosakerehalli,9 Bedroom,1380.0,9.0,150.0,9,10869
Akshaya Nagar,3 BHK,1690.0,3.0,85.0,3,5029
Bannerghatta Road,2 BHK,1130.0,2.0,78.0,2,6902
Electronic City,2 BHK,1213.0,2.0,59.32,2,4890
Kengeri,2 BHK,900.0,2.0,45.0,2,5000
Panathur,2 BHK,1199.0,2.0,85.0,2,7089
other,4 Bedroom,5000.0,4.0,290.0,4,5800
Bellandur,7 Bedroom,700.0,8.0,100.0,7,14285
Uttarahalli,1 BHK,661.0,1.0,36.0,1,5446
other,2 BHK,1804.0,2.0,285.0,2,15798
Marathahalli,3 BHK,1710.0,3.0,100.0,3,5847
Kengeri Satellite Town,2 BHK,1030.0,2.0,50.0,2,4854
Choodasandra,3 BHK,1220.0,3.0,56.0,3,4590
Marathahalli,3 BHK,1530.0,3.0,73.5,3,4803
Ramamurthy Nagar,2 Bedroom,1200.0,2.0,78.0,2,6500
other,3 BHK,2000.0,3.0,85.0,3,4250
Nagarbhavi,1 BHK,300.0,1.0,20.0,1,6666
Green Glen Layout,3 BHK,1751.0,2.0,115.0,3,6567
R.T. Nagar,2 Bedroom,800.0,1.0,170.0,2,21250
Panathur,2 BHK,1255.0,2.0,79.0,2,6294
Whitefield,3 BHK,1778.0,3.0,122.0,3,6861
other,2 BHK,1100.0,2.0,45.0,2,4090
Hennur Road,2 BHK,1480.0,2.0,65.0,2,4391
Marathahalli,2 BHK,1358.0,2.0,81.0,2,5964
Electronics City Phase 1,1 BHK,580.0,1.0,27.0,1,4655
Choodasandra,2 BHK,1197.0,2.0,63.0,2,5263
HRBR Layout,4 Bedroom,1200.0,2.0,170.0,4,14166
Lingadheeranahalli,3 BHK,1705.0,3.0,110.0,3,6451
Electronic City Phase II,3 BHK,1310.0,2.0,37.83,3,2887
Hoodi,2 BHK,1132.0,2.0,67.9,2,5998
other,3 Bedroom,1200.0,3.0,140.0,3,11666
other,2 BHK,814.0,2.0,55.0,2,6756
Electronic City,3 BHK,1575.0,3.0,90.0,3,5714
Pattandur Agrahara,2 BHK,1302.0,2.0,66.0,2,5069
Sultan Palaya,6 Bedroom,1800.0,8.0,175.0,6,9722
Hegde Nagar,3 BHK,2006.8,4.0,196.0,3,9766
other,2 BHK,1100.0,2.0,44.0,2,4000
other,5 Bedroom,2400.0,5.0,400.0,5,16666
other,2 BHK,830.0,2.0,37.0,2,4457
Electronic City,1 RK,435.0,1.0,19.5,1,4482
other,3 BHK,1590.0,3.0,120.0,3,7547
Electronic City Phase II,2 BHK,1140.0,2.0,33.06,2,2900
Kasavanhalli,2 BHK,1090.0,2.0,59.8,2,5486
Thanisandra,3 BHK,1922.0,3.0,106.0,3,5515
Thanisandra,2 BHK,965.0,2.0,67.0,2,6943
Mysore Road,2 BHK,883.0,2.0,40.95,2,4637
8th Phase JP Nagar,2 BHK,1062.0,2.0,42.47,2,3999
Electronic City Phase II,2 BHK,1089.0,2.0,32.67,2,3000
Babusapalaya,3 BHK,1358.0,2.0,43.1,3,3173
Hoodi,2 BHK,1108.0,2.0,87.0,2,7851
Harlur,2 BHK,1197.0,2.0,79.0,2,6599
Sarjapur  Road,2 BHK,1190.0,2.0,50.0,2,4201
other,4 BHK,3754.0,6.0,430.0,4,11454
Vijayanagar,3 BHK,1749.0,3.0,117.0,3,6689
Sarjapur  Road,4 Bedroom,1750.0,3.0,215.0,4,12285
Kasavanhalli,3 BHK,1691.0,3.0,104.0,3,6150
Electronic City,3 BHK,1449.0,3.0,100.0,3,6901
Whitefield,2 BHK,1400.0,2.0,80.0,2,5714
Electronic City,3 BHK,1310.0,2.0,37.83,3,2887
other,2 BHK,1100.0,2.0,48.0,2,4363
other,3 BHK,1230.0,2.0,53.51,3,4350
Green Glen Layout,3 BHK,1750.0,3.0,130.0,3,7428
7th Phase JP Nagar,3 BHK,1450.0,3.0,100.0,3,6896
Haralur Road,3 BHK,1810.0,3.0,126.0,3,6961
Vittasandra,2 BHK,1246.0,2.0,62.3,2,5000
Brookefield,3 BHK,1420.0,3.0,85.0,3,5985
Old Airport Road,4 BHK,2658.0,5.0,187.0,4,7035
other,3 BHK,1450.0,2.0,89.6,3,6179
other,4 BHK,1200.0,4.0,230.0,4,19166
other,2 BHK,1367.0,2.0,62.0,2,4535
Talaghattapura,3 BHK,2254.0,3.0,170.0,3,7542
Begur Road,2 BHK,1160.0,2.0,42.0,2,3620
Sanjay nagar,5 Bedroom,1200.0,4.0,185.0,5,15416
Kadugodi,3 BHK,1711.0,3.0,75.0,3,4383
other,9 Bedroom,750.0,6.0,112.0,9,14933
Raja Rajeshwari Nagar,2 BHK,1120.0,2.0,37.97,2,3390
Kaggalipura,2 BHK,950.0,2.0,47.0,2,4947
Basavangudi,2 BHK,1560.0,2.0,145.0,2,9294
other,14 BHK,1250.0,15.0,125.0,14,10000
Kanakpura Road,2 BHK,1225.0,2.0,60.0,2,4897
TC Palaya,3 Bedroom,1200.0,2.0,66.0,3,5500
other,2 BHK,1340.0,2.0,77.0,2,5746
Sarjapur  Road,3 BHK,1846.0,3.0,145.0,3,7854
Dodda Nekkundi,2 BHK,1155.0,2.0,46.0,2,3982
Balagere,2 BHK,1205.0,2.0,70.0,2,5809
Thanisandra,2 BHK,1140.0,2.0,36.0,2,3157
Raja Rajeshwari Nagar,2 BHK,1140.0,2.0,38.65,2,3390
Jalahalli East,3 BHK,1300.0,2.0,63.0,3,4846
Bellandur,2 BHK,1051.0,2.0,60.0,2,5708
Uttarahalli,2 BHK,1134.0,2.0,45.36,2,4000
Kodihalli,3 BHK,2500.0,4.0,260.0,3,10400
other,8 Bedroom,12000.0,9.0,1900.0,8,15833
other,2 BHK,1135.0,2.0,32.91,2,2899
other,3 BHK,1745.0,3.0,77.0,3,4412
Banashankari,5 Bedroom,500.0,5.0,92.0,5,18400
other,3 BHK,1665.0,3.0,105.0,3,6306
other,2 BHK,1220.0,2.0,48.79,2,3999
Kanakpura Road,3 BHK,1868.0,3.0,113.0,3,6049
Anandapura,3 Bedroom,640.0,3.0,45.0,3,7031
other,4 Bedroom,2300.0,4.0,200.0,4,8695
other,2 BHK,1210.0,2.0,76.0,2,6280
Ambalipura,2 BHK,1351.0,2.0,77.0,2,5699
Hebbal,3 BHK,1636.0,3.0,119.0,3,7273
R.T. Nagar,5 Bedroom,2400.0,6.0,450.0,5,18750
Haralur Road,2 BHK,1464.0,2.0,56.0,2,3825
Attibele,1 BHK,418.0,1.0,12.0,1,2870
Kanakpura Road,2 BHK,1339.0,2.0,62.0,2,4630
Bannerghatta,3 BHK,2370.0,4.0,195.0,3,8227
Rachenahalli,3 BHK,1530.0,2.0,74.5,3,4869
other,2 BHK,1100.0,2.0,46.0,2,4181
7th Phase JP Nagar,3 BHK,1350.0,3.0,58.0,3,4296
Marathahalli,2 BHK,1290.0,2.0,52.0,2,4031
Bommasandra Industrial Area,3 BHK,1220.0,2.0,35.23,3,2887
Sarjapur  Road,3 BHK,1846.0,3.0,155.0,3,8396
Hegde Nagar,3 BHK,1847.0,3.0,135.0,3,7309
Whitefield,4 Bedroom,3100.0,4.0,350.0,4,11290
Hoodi,2 BHK,1430.0,2.0,110.0,2,7692
Old Madras Road,2 BHK,935.0,2.0,45.0,2,4812
Attibele,3 Bedroom,1350.0,3.0,85.0,3,6296
Kammanahalli,3 Bedroom,540.0,3.0,60.0,3,11111
Hennur Road,3 BHK,1590.0,3.0,96.5,3,6069
other,3 BHK,1286.0,3.0,67.0,3,5209
other,2 BHK,1141.0,2.0,40.0,2,3505
Shivaji Nagar,2 BHK,1554.0,3.0,130.0,2,8365
Hebbal,3 BHK,3520.0,3.0,430.0,3,12215
Malleshwaram,5 BHK,7500.0,7.0,1700.0,5,22666
Harlur,3 BHK,1749.0,3.0,112.0,3,6403
other,5 Bedroom,2250.0,5.0,360.0,5,16000
other,2 BHK,1140.0,2.0,45.0,2,3947
Bommenahalli,4 Bedroom,1632.0,3.0,128.0,4,7843
Ramamurthy Nagar,3 BHK,1000.0,2.0,41.0,3,4100
other,2 BHK,1065.0,2.0,58.0,2,5446
Haralur Road,3 BHK,1520.0,2.0,96.0,3,6315
Bisuvanahalli,3 BHK,1075.0,2.0,43.0,3,4000
other,7 BHK,5.0,7.0,115.0,7,2300000
Gubbalala,3 BHK,1539.0,2.0,52.36,3,3402
Sector 7 HSR Layout,2 BHK,1342.0,2.0,115.0,2,8569
HBR Layout,2 BHK,1004.0,2.0,49.0,2,4880
Ramagondanahalli,2 BHK,1295.0,2.0,95.0,2,7335
Thanisandra,2 BHK,1098.0,2.0,74.0,2,6739
Kothanur,2 BHK,1070.0,2.0,45.5,2,4252
Hosa Road,2 BHK,1016.0,2.0,40.0,2,3937
Bellandur,3 BHK,1450.0,3.0,68.0,3,4689
Yelahanka,1 BHK,654.0,1.0,37.0,1,5657
Koramangala,4 BHK,5985.0,4.0,775.0,4,12949
Yelachenahalli,2 BHK,1080.0,2.0,38.0,2,3518
Vijayanagar,2 BHK,1046.0,2.0,75.0,2,7170
Domlur,3 BHK,1950.0,3.0,165.0,3,8461
Bhoganhalli,3 BHK,1610.0,3.0,84.53,3,5250
Doddakallasandra,3 BHK,2493.0,3.0,115.0,3,4612
Ramagondanahalli,2 BHK,1151.0,2.0,50.5,2,4387
Electronic City,2 BHK,1125.0,2.0,58.0,2,5155
Kammasandra,2 BHK,992.0,2.0,25.79,2,2599
1st Block Jayanagar,2 BHK,1000.0,3.0,60.0,2,6000
Banashankari Stage II,5 Bedroom,2400.0,3.0,510.0,5,21250
Bommanahalli,3 BHK,1850.0,3.0,90.0,3,4864
Harlur,2 BHK,1174.0,2.0,75.0,2,6388
Sultan Palaya,2 BHK,1100.0,2.0,45.0,2,4090
Raja Rajeshwari Nagar,2 BHK,1095.0,2.0,38.33,2,3500
Banashankari,3 BHK,3580.0,3.0,411.0,3,11480
Choodasandra,3 Bedroom,900.0,4.0,155.0,3,17222
other,3 BHK,1082.0,2.0,42.0,3,3881
Uttarahalli,3 BHK,1350.0,2.0,47.24,3,3499
Kadugodi,2 BHK,1314.0,2.0,78.0,2,5936
other,2 BHK,1290.0,2.0,40.63,2,3149
Kaggadasapura,2 BHK,1010.0,2.0,55.0,2,5445
Shivaji Nagar,2 Bedroom,2200.0,2.0,110.0,2,5000
other,3 BHK,1636.0,3.0,97.0,3,5929
9th Phase JP Nagar,2 BHK,1050.0,2.0,39.0,2,3714
7th Phase JP Nagar,2 Bedroom,1200.0,2.0,200.0,2,16666
Haralur Road,3 BHK,1520.0,2.0,125.0,3,8223
5th Phase JP Nagar,5 BHK,2500.0,5.0,110.0,5,4400
Sarjapur  Road,2 BHK,1067.0,2.0,60.0,2,5623
NRI Layout,2 BHK,1060.0,2.0,35.0,2,3301
Bommasandra,2 BHK,955.0,2.0,35.28,2,3694
Thigalarapalya,2 BHK,1418.0,2.0,104.0,2,7334
8th Phase JP Nagar,9 Bedroom,1200.0,8.0,225.0,9,18750
Anekal,2 BHK,700.0,1.0,19.4,2,2771
Kanakpura Road,2 BHK,1366.0,2.0,83.0,2,6076
Electronic City,2 BHK,1090.0,2.0,35.03,2,3213
Jigani,3 BHK,1250.0,3.0,58.0,3,4640
other,4 BHK,3600.0,4.0,450.0,4,12500
Electronics City Phase 1,2 BHK,950.0,2.0,33.0,2,3473
Indira Nagar,2 BHK,1470.0,2.0,170.0,2,11564
Yelahanka,2 BHK,1250.0,2.0,40.0,2,3200
HBR Layout,4 Bedroom,800.0,4.0,180.0,4,22500
Badavala Nagar,3 BHK,1494.0,2.0,94.55,3,6328
Hennur Road,3 BHK,1445.0,2.0,105.0,3,7266
Kundalahalli,3 BHK,1397.0,3.0,105.0,3,7516
Vijayanagar,1 BHK,606.0,1.0,40.0,1,6600
Gollarapalya Hosahalli,2 BHK,861.0,2.0,34.5,2,4006
Devarachikkanahalli,2 BHK,1170.0,2.0,40.0,2,3418
Koramangala,4 BHK,2461.0,6.0,353.0,4,14343
Marathahalli,2 BHK,1205.0,2.0,75.0,2,6224
other,8 Bedroom,1224.0,4.0,200.0,8,16339
Rajaji Nagar,1 BHK,660.0,1.0,75.0,1,11363
Dommasandra,2 BHK,1153.0,2.0,43.5,2,3772
OMBR Layout,5 Bedroom,2400.0,5.0,275.0,5,11458
Old Madras Road,5 BHK,4500.0,7.0,294.0,5,6533
Vijayanagar,2 BHK,1200.0,2.0,62.0,2,5166
other,3 BHK,1450.0,3.0,42.0,3,2896
Budigere,2 BHK,1139.0,2.0,60.0,2,5267
Green Glen Layout,3 BHK,1530.0,3.0,105.0,3,6862
other,6 Bedroom,825.0,6.0,400.0,6,48484
Electronic City,3 BHK,1575.0,3.0,101.0,3,6412
Whitefield,1 BHK,650.0,1.0,25.0,1,3846
other,2 BHK,1120.0,2.0,40.0,2,3571
Akshaya Nagar,3 BHK,1700.0,3.0,80.0,3,4705
Varthur,2 BHK,1090.0,2.0,48.0,2,4403
Akshaya Nagar,3 BHK,1897.0,4.0,93.0,3,4902
other,3 Bedroom,1200.0,3.0,72.0,3,6000
Kanakpura Road,2 BHK,700.0,2.0,40.0,2,5714
Thanisandra,2 BHK,1437.0,2.0,71.13,2,4949
Ambedkar Nagar,3 BHK,1856.0,3.0,120.0,3,6465
other,3 Bedroom,1500.0,2.0,65.0,3,4333
Hennur Road,2 BHK,1036.0,2.0,59.45,2,5738
other,6 BHK,3000.0,8.0,90.0,6,3000
Mico Layout,2 BHK,1125.0,2.0,40.0,2,3555
Uttarahalli,3 BHK,1320.0,2.0,55.44,3,4200
Malleshwaram,3 BHK,2215.0,4.0,310.0,3,13995
Electronic City,2 BHK,660.0,1.0,16.0,2,2424
Marathahalli,3 BHK,1730.0,3.0,110.0,3,6358
other,4 Bedroom,750.0,4.0,85.0,4,11333
Talaghattapura,3 BHK,1804.0,3.0,115.0,3,6374
Thanisandra,3 BHK,1795.0,3.0,140.0,3,7799
7th Phase JP Nagar,2 BHK,1050.0,2.0,50.0,2,4761
Dasarahalli,3 BHK,2100.0,3.0,120.0,3,5714
Rayasandra,3 BHK,1199.0,2.0,48.5,3,4045
other,3 Bedroom,1350.0,2.0,280.0,3,20740
other,2 BHK,1100.0,2.0,47.0,2,4272
other,2 Bedroom,1000.0,1.0,75.0,2,7500
Kanakapura,3 BHK,1254.0,2.0,70.0,3,5582
other,8 Bedroom,600.0,8.0,70.0,8,11666
Electronics City Phase 1,2 BHK,940.0,2.0,49.0,2,5212
BTM 2nd Stage,2 BHK,1280.0,2.0,80.0,2,6250
Whitefield,4 BHK,2155.0,3.0,125.0,4,5800
Sarjapur  Road,5 Bedroom,3200.0,5.0,140.0,5,4375
Kasavanhalli,4 BHK,4260.0,4.0,272.0,4,6384
Hebbal,3 BHK,1255.0,2.0,95.0,3,7569
other,3 BHK,1388.0,3.0,69.99,3,5042
Whitefield,1 RK,905.0,1.0,52.0,1,5745
other,2 BHK,1176.0,2.0,44.0,2,3741
Uttarahalli,3 BHK,1390.0,2.0,61.16,3,4400
Giri Nagar,4 Bedroom,2400.0,4.0,400.0,4,16666
Sahakara Nagar,2 BHK,1160.0,2.0,55.0,2,4741
Seegehalli,4 BHK,2800.0,4.0,140.0,4,5000
5th Block Hbr Layout,4 Bedroom,1200.0,4.0,205.0,4,17083
Marathahalli,3 BHK,1583.0,3.0,115.0,3,7264
Jigani,3 BHK,1252.0,3.0,63.0,3,5031
other,2 BHK,1170.0,2.0,86.0,2,7350
Bannerghatta Road,3 BHK,1350.0,2.0,51.0,3,3777
Gottigere,2 BHK,1075.0,2.0,30.0,2,2790
Hosa Road,2 BHK,972.0,2.0,40.0,2,4115
Bannerghatta Road,3 BHK,1486.0,3.0,83.22,3,5600
Margondanahalli,2 Bedroom,555.0,1.0,37.0,2,6666
other,4 Bedroom,4830.0,5.0,390.0,4,8074
other,2 BHK,850.0,2.0,75.0,2,8823
Binny Pete,3 BHK,1282.0,3.0,178.0,3,13884
Hebbal,4 BHK,2790.0,4.0,198.0,4,7096
Sarjapur  Road,4 BHK,2383.0,5.0,215.0,4,9022
other,2 BHK,1103.0,2.0,55.0,2,4986
Rayasandra,3 BHK,1577.0,3.0,67.0,3,4248
Yelahanka,3 BHK,1465.0,3.0,80.01,3,5461
other,3 BHK,1600.0,3.0,64.0,3,4000
Uttarahalli,2 BHK,1090.0,2.0,45.0,2,4128
Yelahanka,3 BHK,1721.0,3.0,80.86,3,4698
other,2 BHK,1100.0,2.0,55.0,2,5000
Varthur,3 BHK,1450.0,3.0,71.0,3,4896
Battarahalli,2 BHK,1071.0,2.0,62.0,2,5788
Old Madras Road,3 BHK,2266.0,3.0,169.0,3,7458
other,3 BHK,2065.0,3.0,130.0,3,6295
Kanakpura Road,3 BHK,1800.0,3.0,90.0,3,5000
other,2 BHK,1150.0,2.0,125.0,2,10869
other,8 Bedroom,1200.0,8.0,220.0,8,18333
Hormavu,2 BHK,1123.0,2.0,56.0,2,4986
Akshaya Nagar,3 BHK,1884.0,3.0,98.0,3,5201
1st Block Jayanagar,7 Bedroom,930.0,4.0,85.0,7,9139
Lakshminarayana Pura,2 BHK,1145.0,2.0,75.0,2,6550
Hebbal,3 BHK,1328.0,2.0,60.0,3,4518
Subramanyapura,2 BHK,950.0,2.0,55.0,2,5789
other,2 BHK,1250.0,2.0,59.0,2,4720
other,2 BHK,1115.0,2.0,46.0,2,4125
Gollarapalya Hosahalli,3 BHK,1318.0,3.0,54.0,3,4097
JP Nagar,3 BHK,3300.0,4.0,160.0,3,4848
Kalyan nagar,2 BHK,1280.0,2.0,48.0,2,3750
Whitefield,2 BHK,1230.0,2.0,59.0,2,4796
Raja Rajeshwari Nagar,3 BHK,1420.0,2.0,60.0,3,4225
Kasavanhalli,3 Bedroom,2111.0,4.0,120.0,3,5684
Vidyaranyapura,2 BHK,1100.0,2.0,54.0,2,4909
Harlur,3 BHK,1750.0,3.0,136.0,3,7771
other,3 BHK,2550.0,4.0,290.0,3,11372
Shampura,2 BHK,700.0,1.0,35.0,2,5000
Uttarahalli,2 BHK,1050.0,2.0,42.0,2,4000
Uttarahalli,2 BHK,1235.0,2.0,90.0,2,7287
Yeshwanthpur,2 Bedroom,1200.0,2.0,63.0,2,5250
Kengeri,2 BHK,900.0,2.0,30.0,2,3333
Thanisandra,2 BHK,1146.0,2.0,38.0,2,3315
Yeshwanthpur,4 Bedroom,1200.0,4.0,110.0,4,9166
Banashankari,1 BHK,2400.0,1.0,200.0,1,8333
Marathahalli,6 Bedroom,1200.0,5.0,108.0,6,9000
Devarachikkanahalli,1 BHK,615.0,1.0,24.0,1,3902
Mysore Road,2 BHK,1070.0,2.0,49.65,2,4640
Rajaji Nagar,5 Bedroom,1538.0,5.0,300.0,5,19505
Poorna Pragna Layout,3 BHK,1355.0,2.0,58.25,3,4298
other,4 BHK,3080.0,2.0,285.0,4,9253
Kundalahalli,3 BHK,1800.0,3.0,80.0,3,4444
Begur Road,3 BHK,1400.0,2.0,50.06,3,3575
Uttarahalli,2 BHK,1123.0,2.0,48.0,2,4274
Konanakunte,4 Bedroom,1200.0,2.0,130.0,4,10833
Kengeri,2 BHK,787.0,2.0,31.0,2,3939
Kambipura,3 BHK,1082.0,2.0,45.0,3,4158
Seegehalli,2 BHK,1096.0,2.0,41.0,2,3740
Vidyaranyapura,7 Bedroom,1200.0,7.0,140.0,7,11666
R.T. Nagar,3 Bedroom,1140.0,3.0,130.0,3,11403
other,2 BHK,1130.0,2.0,52.0,2,4601
Electronic City,1 BHK,550.0,1.0,12.0,1,2181
Mahadevpura,2 BHK,1058.0,2.0,85.0,2,8034
Hebbal,3 BHK,1269.0,2.0,73.0,3,5752
Prithvi Layout,3 Bedroom,946.0,3.0,140.0,3,14799
other,4 BHK,3400.0,6.0,420.0,4,12352
Kanakpura Road,4 BHK,2689.0,6.0,245.0,4,9111
Hennur Road,3 BHK,1976.0,3.0,65.21,3,3300
Kothanur,3 BHK,1820.0,3.0,77.0,3,4230
Uttarahalli,3 BHK,1329.0,2.0,56.0,3,4213
Kathriguppe,3 BHK,1350.0,2.0,80.99,3,5999
Nagarbhavi,4 Bedroom,1250.0,3.0,175.0,4,14000
Whitefield,2 BHK,1205.0,2.0,40.0,2,3319
other,6 Bedroom,800.0,6.0,98.0,6,12250
Kanakpura Road,2 BHK,1339.0,2.0,71.5,2,5339
Uttarahalli,2 BHK,938.0,2.0,35.0,2,3731
other,1 BHK,450.0,1.0,25.0,1,5555
BTM 2nd Stage,2 BHK,1100.0,2.0,70.0,2,6363
2nd Stage Nagarbhavi,4 Bedroom,600.0,3.0,95.0,4,15833
Yelahanka,2 BHK,1060.0,2.0,40.0,2,3773
Whitefield,2 BHK,1125.0,2.0,40.0,2,3555
Whitefield,1 BHK,810.0,1.0,21.0,1,2592
Bellandur,2 BHK,1060.0,2.0,65.0,2,6132
Anandapura,4 Bedroom,1749.0,4.0,90.0,4,5145
Sahakara Nagar,3 BHK,1200.0,2.0,75.0,3,6250
other,2 BHK,1100.0,2.0,85.0,2,7727
Rachenahalli,1 BHK,680.0,1.0,32.64,1,4800
Jalahalli East,4 Bedroom,720.0,4.0,48.0,4,6666
Gunjur,5 Bedroom,6613.0,7.0,950.0,5,14365
Hoodi,3 BHK,1925.0,3.0,110.0,3,5714
Kengeri,1 BHK,502.0,1.0,25.0,1,4980
Bommenahalli,3 Bedroom,3339.0,3.0,250.0,3,7487
Begur,3 BHK,1475.0,2.0,74.0,3,5016
other,3 BHK,1550.0,3.0,258.0,3,16645
other,2 BHK,966.0,2.0,49.0,2,5072
other,6 Bedroom,1350.0,6.0,160.0,6,11851
Haralur Road,2 BHK,1300.0,2.0,75.0,2,5769
other,5 Bedroom,4400.0,4.0,350.0,5,7954
Kogilu,3 BHK,1559.0,3.0,120.0,3,7697
Vittasandra,2 BHK,1246.0,2.0,63.0,2,5056
Rajaji Nagar,3 Bedroom,2790.0,3.0,950.0,3,34050
Begur Road,3 BHK,1410.0,2.0,53.58,3,3800
Varthur Road,4 Bedroom,1300.0,3.0,75.0,4,5769
Kanakpura Road,2 BHK,1339.0,2.0,58.0,2,4331
other,4 Bedroom,900.0,4.0,75.0,4,8333
Channasandra,2 BHK,1104.0,2.0,35.5,2,3215
Sultan Palaya,2 BHK,900.0,2.0,45.0,2,5000
other,3 BHK,1550.0,2.0,79.0,3,5096
Kanakpura Road,1 BHK,670.0,1.0,35.0,1,5223
Jakkur,2 BHK,1483.0,2.0,98.0,2,6608
other,3 BHK,1540.0,2.0,70.0,3,4545
other,8 Bedroom,1200.0,7.0,100.0,8,8333
Hennur,2 BHK,1255.0,2.0,54.5,2,4342
OMBR Layout,3 Bedroom,600.0,3.0,100.0,3,16666
other,2 BHK,1036.0,2.0,110.0,2,10617
Bannerghatta Road,3 BHK,1460.0,2.0,75.0,3,5136
Devarachikkanahalli,2 BHK,1230.0,2.0,58.0,2,4715
Kanakpura Road,2 BHK,1340.0,2.0,60.0,2,4477
Yelenahalli,3 BHK,1600.0,3.0,61.37,3,3835
Hegde Nagar,3 BHK,2162.0,3.0,200.0,3,9250
Electronic City,2 BHK,919.0,2.0,40.0,2,4352
other,2 BHK,1070.0,2.0,38.9,2,3635
Hosur Road,3 BHK,1733.0,2.0,85.0,3,4904
Jigani,3 BHK,1221.0,3.0,72.0,3,5896
other,2 BHK,1041.0,2.0,36.44,2,3500
Ambalipura,4 BHK,2550.0,4.0,149.0,4,5843
Shampura,3 BHK,1400.0,3.0,70.0,3,5000
Anjanapura,1 Bedroom,600.0,2.0,55.0,1,9166
Nagarbhavi,1 BHK,710.0,2.0,40.0,1,5633
Electronic City,3 BHK,1500.0,2.0,78.25,3,5216
Thigalarapalya,4 BHK,3122.0,6.0,237.0,4,7591
Akshaya Nagar,3 BHK,2061.0,3.0,175.0,3,8491
Brookefield,5 Bedroom,2400.0,5.0,325.0,5,13541
Hebbal,3 BHK,1255.0,2.0,95.0,3,7569
Malleshwaram,7 BHK,2425.0,7.0,140.0,7,5773
Electronic City Phase II,2 BHK,1031.0,2.0,55.0,2,5334
Balagere,2 BHK,1210.0,2.0,74.0,2,6115
Anjanapura,2 BHK,950.0,2.0,40.0,2,4210
other,4 Bedroom,1376.0,4.0,156.0,4,11337
Hosa Road,2 BHK,1161.0,2.0,55.15,2,4750
Raja Rajeshwari Nagar,2 BHK,1303.0,2.0,44.17,2,3389
Thigalarapalya,2 BHK,1418.0,2.0,105.0,2,7404
Kengeri Satellite Town,2 Bedroom,1200.0,2.0,65.0,2,5416
Basavangudi,3 BHK,1825.0,3.0,175.0,3,9589
Bannerghatta Road,2 BHK,1015.0,2.0,48.0,2,4729
other,3 BHK,1150.0,2.0,60.0,3,5217
Singasandra,2 BHK,1010.0,2.0,40.0,2,3960
other,3 BHK,1508.0,3.0,75.0,3,4973
7th Phase JP Nagar,2 BHK,1275.0,2.0,87.0,2,6823
7th Phase JP Nagar,3 BHK,1460.0,3.0,70.0,3,4794
Old Airport Road,2 BHK,946.0,2.0,58.0,2,6131
other,2 BHK,1110.0,2.0,58.0,2,5225
Hennur Road,3 BHK,1832.0,3.0,125.0,3,6823
Hebbal,2 BHK,1088.0,2.0,61.45,2,5647
Thanisandra,3 BHK,1564.0,3.0,104.0,3,6649
Kadugodi,2 BHK,1030.0,2.0,35.02,2,3400
9th Phase JP Nagar,2 BHK,660.0,2.0,27.0,2,4090
Sarjapur  Road,3 BHK,1700.0,3.0,95.0,3,5588
other,4 Bedroom,1100.0,5.0,121.0,4,11000
Panathur,2 BHK,1205.0,2.0,62.9,2,5219
other,3 BHK,1950.0,3.0,90.0,3,4615
other,2 BHK,1080.0,2.0,62.0,2,5740
Ardendale,4 BHK,2800.0,4.0,140.0,4,5000
other,2 BHK,999.0,2.0,43.66,2,4370
other,4 Bedroom,900.0,4.0,175.0,4,19444
Kanakpura Road,2 BHK,1300.0,2.0,89.76,2,6904
HRBR Layout,2 BHK,1265.0,2.0,78.0,2,6166
Attibele,2 BHK,1200.0,2.0,42.0,2,3500
Bhoganhalli,4 BHK,1974.0,4.0,106.0,4,5369
other,2 BHK,1180.0,2.0,40.0,2,3389
Amruthahalli,3 BHK,1350.0,2.0,63.0,3,4666
Kodichikkanahalli,3 BHK,1560.0,2.0,65.0,3,4166
other,2 Bedroom,2800.0,2.0,650.0,2,23214
Hosakerehalli,2 BHK,1085.0,2.0,41.23,2,3800
Vijayanagar,3 BHK,1300.0,2.0,75.0,3,5769
Marathahalli,3 BHK,1600.0,3.0,56.0,3,3500
other,2 BHK,1127.0,2.0,55.0,2,4880
Kengeri Satellite Town,2 BHK,795.0,2.0,32.0,2,4025
Nagarbhavi,3 BHK,1850.0,3.0,88.0,3,4756
Bommasandra Industrial Area,2 BHK,1020.0,2.0,29.45,2,2887
other,3 BHK,1475.0,2.0,200.0,3,13559
other,3 Bedroom,1200.0,3.0,96.25,3,8020
Uttarahalli,3 BHK,1345.0,2.0,47.0,3,3494
Kaikondrahalli,2 BHK,849.0,2.0,25.4,2,2991
Kanakpura Road,3 BHK,1706.0,3.0,135.0,3,7913
Yelahanka,4 BHK,2100.0,4.0,180.0,4,8571
other,3 BHK,1250.0,2.0,65.0,3,5200
Kanakpura Road,3 BHK,1260.0,3.0,63.58,3,5046
Horamavu Agara,3 BHK,1265.0,2.0,65.0,3,5138
Abbigere,3 BHK,1260.0,2.0,45.0,3,3571
Jalahalli,2 BHK,1045.0,2.0,58.0,2,5550
Hebbal Kempapura,4 BHK,3895.0,4.0,495.0,4,12708
Rachenahalli,1 RK,412.5,1.0,19.8,1,4800
Bellandur,2 BHK,1100.0,2.0,62.0,2,5636
Kudlu Gate,2 BHK,1113.0,2.0,50.0,2,4492
Old Airport Road,4 BHK,2774.0,4.0,208.0,4,7498
Varthur,2 BHK,1210.0,2.0,60.0,2,4958
Kogilu,1 BHK,700.0,1.0,30.84,1,4405
Kanakpura Road,2 BHK,1155.0,2.0,55.5,2,4805
Hoodi,2 BHK,1132.0,2.0,89.0,2,7862
Hennur Road,2 BHK,1359.0,2.0,104.0,2,7652
other,4 BHK,2631.0,3.0,200.0,4,7601
Kothanur,2 BHK,1145.0,2.0,56.0,2,4890
5th Phase JP Nagar,1 BHK,600.0,1.0,30.0,1,5000
Laggere,1 Bedroom,1200.0,1.0,125.0,1,10416
other,2 BHK,830.0,2.0,60.0,2,7228
HSR Layout,2 BHK,1009.0,2.0,56.0,2,5550
Basavangudi,3 BHK,1800.0,3.0,195.0,3,10833
Yelahanka,2 BHK,1226.0,2.0,52.71,2,4299
other,2 BHK,1500.0,2.0,88.0,2,5866
Electronic City,2 BHK,1363.0,2.0,79.0,2,5796
Poorna Pragna Layout,3 Bedroom,1400.0,2.0,240.0,3,17142
Bommasandra Industrial Area,2 BHK,1165.0,2.0,33.64,2,2887
HSR Layout,4 Bedroom,2792.0,4.0,220.0,4,7879
Kammanahalli,1 Bedroom,700.0,1.0,59.0,1,8428
Kaval Byrasandra,2 BHK,1060.0,2.0,42.0,2,3962
Marathahalli,2 BHK,1200.0,2.0,67.0,2,5583
other,3 BHK,1518.0,3.0,76.0,3,5006
other,3 BHK,1495.0,2.0,75.0,3,5016
Sanjay nagar,2 BHK,1575.0,2.0,90.0,2,5714
Rajaji Nagar,2 BHK,1440.0,2.0,170.0,2,11805
Kasavanhalli,3 BHK,1545.0,3.0,69.0,3,4466
Kengeri,3 BHK,1470.0,3.0,50.0,3,3401
Cooke Town,3 BHK,1500.0,3.0,95.0,3,6333
BTM Layout,3 BHK,1450.0,2.0,81.0,3,5586
Hosur Road,2 BHK,1179.0,2.0,49.0,2,4156
7th Phase JP Nagar,3 BHK,1650.0,3.0,110.0,3,6666
Chandapura,3 BHK,1095.0,2.0,32.0,3,2922
other,2 BHK,856.0,2.0,48.0,2,5607
other,2 BHK,1143.0,2.0,62.0,2,5424
Hebbal,2 BHK,1349.0,2.0,97.8,2,7249
Anjanapura,3 BHK,1280.0,2.0,45.0,3,3515
Malleshwaram,1 BHK,480.0,1.0,60.0,1,12500
Whitefield,2 BHK,1216.0,2.0,69.06,2,5679
Doddakallasandra,3 BHK,2493.0,3.0,115.0,3,4612
Malleshwaram,2 BHK,1290.0,2.0,120.0,2,9302
Banjara Layout,5 Bedroom,1200.0,5.0,96.0,5,8000
Electronic City,2 BHK,1156.0,2.0,42.0,2,3633
Thanisandra,2 BHK,1230.0,2.0,44.0,2,3577
other,2 BHK,1070.0,2.0,33.15,2,3098
Ramagondanahalli,2 BHK,1235.0,2.0,49.3,2,3991
other,2 BHK,1117.0,2.0,110.0,2,9847
Kundalahalli,3 BHK,1724.0,3.0,124.0,3,7192
Jalahalli,4 Bedroom,1200.0,4.0,90.0,4,7500
other,1 BHK,600.0,1.0,43.0,1,7166
KR Puram,3 BHK,1930.0,2.0,57.9,3,3000
Whitefield,2 BHK,1216.0,2.0,75.0,2,6167
KR Puram,3 BHK,1452.0,3.0,60.0,3,4132
Whitefield,2 BHK,1355.0,2.0,78.0,2,5756
other,4 Bedroom,7500.0,4.0,425.0,4,5666
Sarjapur  Road,3 BHK,2145.0,3.0,190.0,3,8857
Bommanahalli,2 BHK,957.0,2.0,53.0,2,5538
other,2 BHK,925.0,2.0,48.0,2,5189
Bellandur,2 BHK,1550.0,2.0,59.0,2,3806
Yelahanka New Town,4 Bedroom,4000.0,2.0,899.0,4,22475
Kanakpura Road,3 Bedroom,2100.0,3.0,140.0,3,6666
Bannerghatta Road,2 BHK,1115.0,2.0,61.0,2,5470
Yelahanka,3 BHK,1692.0,3.0,83.0,3,4905
other,2 BHK,1100.0,2.0,52.0,2,4727
Hormavu,3 BHK,1385.0,2.0,69.0,3,4981
Hosa Road,1 BHK,760.0,1.0,24.9,1,3276
Electronics City Phase 1,2 BHK,1335.0,2.0,67.0,2,5018
Bommanahalli,2 BHK,910.0,2.0,45.0,2,4945
Kanakpura Road,3 BHK,1610.0,2.0,55.0,3,3416
Hennur Road,2 BHK,1341.0,2.0,75.76,2,5649
other,3 Bedroom,1200.0,3.0,70.0,3,5833
Uttarahalli,3 BHK,1285.0,2.0,44.98,3,3500
HAL 2nd Stage,3 BHK,2100.0,3.0,252.0,3,12000
Yeshwanthpur,3 BHK,1679.0,3.0,92.13,3,5487
other,4 Bedroom,2675.0,4.0,290.0,4,10841
other,2 BHK,1016.0,2.0,45.9,2,4517
Banashankari Stage II,4 Bedroom,1500.0,2.0,650.0,4,43333
other,3 Bedroom,660.0,2.0,100.0,3,15151
Rajaji Nagar,3 BHK,2448.0,3.0,400.0,3,16339
Bellandur,2 BHK,924.0,2.0,35.11,2,3799
8th Phase JP Nagar,1 BHK,500.0,1.0,31.0,1,6200
other,3 BHK,1800.0,3.0,88.0,3,4888
Whitefield,2 BHK,1255.0,2.0,77.0,2,6135
Hennur Road,2 BHK,1040.0,2.0,50.0,2,4807
other,2 BHK,1062.0,2.0,47.0,2,4425
Varthur Road,2 BHK,805.5,2.0,19.33,2,2399
Hosur Road,2 BHK,950.0,2.0,60.0,2,6315
Whitefield,2 BHK,1216.0,2.0,73.0,2,6003
HSR Layout,2 BHK,1300.0,2.0,98.0,2,7538
5th Phase JP Nagar,2 BHK,1390.0,2.0,65.0,2,4676
Yelahanka,2 BHK,1033.0,2.0,52.58,2,5090
Yelahanka,2 BHK,1390.0,2.0,65.31,2,4698
Jalahalli,2 BHK,1020.0,2.0,52.0,2,5098
Whitefield,1 BHK,709.0,1.0,34.735,1,4899
Whitefield,2 BHK,1140.0,2.0,45.0,2,3947
Yeshwanthpur,1 BHK,670.0,1.0,36.85,1,5500
Bhoganhalli,4 BHK,2171.0,4.0,111.0,4,5112
Bannerghatta Road,2 BHK,1136.0,2.0,48.0,2,4225
other,3 BHK,1477.0,2.0,80.0,3,5416
KR Puram,2 BHK,1001.0,2.0,30.0,2,2997
Yelahanka,2 BHK,1040.0,2.0,48.86,2,4698
Garudachar Palya,2 BHK,1060.0,2.0,48.5,2,4575
Sarjapur,2 BHK,1240.0,2.0,44.3,2,3572
other,4 BHK,3500.0,5.0,450.0,4,12857
other,9 BHK,42000.0,8.0,175.0,9,416
Varthur,2 BHK,1070.0,2.0,46.0,2,4299
Electronic City Phase II,2 BHK,1253.0,2.0,65.8,2,5251
Whitefield,2 BHK,1190.0,2.0,60.0,2,5042
Whitefield,2 BHK,1312.0,2.0,55.0,2,4192
KR Puram,2 BHK,750.0,2.0,40.0,2,5333
Koramangala,3 BHK,1615.0,3.0,115.0,3,7120
Marathahalli,2 BHK,1095.0,2.0,60.0,2,5479
Whitefield,3 BHK,1745.0,3.0,100.0,3,5730
Rachenahalli,2 BHK,1050.0,2.0,52.5,2,5000
other,4 Bedroom,2080.0,4.0,440.0,4,21153
Thanisandra,2 BHK,1220.0,2.0,44.6,2,3655
other,3 BHK,1793.0,3.0,98.0,3,5465
Uttarahalli,2 BHK,1100.0,1.0,45.0,2,4090
Yelahanka New Town,2 BHK,1385.0,2.0,65.0,2,4693
Begur Road,2 BHK,1200.0,2.0,45.0,2,3750
Haralur Road,3 BHK,2017.0,2.0,125.0,3,6197
Attibele,1 BHK,400.0,1.0,10.0,1,2500
Bellandur,3 BHK,1400.0,2.0,44.81,3,3200
other,2 BHK,960.0,2.0,33.0,2,3437
Uttarahalli,2 BHK,1125.0,2.0,47.0,2,4177
Akshaya Nagar,3 BHK,1551.0,2.0,60.0,3,3868
BTM Layout,4 Bedroom,600.0,4.0,122.0,4,20333
Vittasandra,2 BHK,1246.0,2.0,67.4,2,5409
Whitefield,4 BHK,3200.0,4.0,224.0,4,7000
Marathahalli,2 BHK,1060.0,2.0,48.0,2,4528
Bharathi Nagar,2 BHK,1384.0,2.0,59.0,2,4263
other,2 BHK,620.0,2.0,23.0,2,3709
Bommasandra,2 BHK,1005.0,2.0,39.77,2,3957
Sarakki Nagar,3 Bedroom,1200.0,3.0,128.0,3,10666
other,3 BHK,1400.0,3.0,35.0,3,2500
HSR Layout,3 BHK,2400.0,3.0,88.0,3,3666
Whitefield,2 BHK,1096.0,2.0,39.0,2,3558
Kaggadasapura,2 BHK,1180.0,1.0,50.0,2,4237
Uttarahalli,3 BHK,1300.0,2.0,47.0,3,3615
other,2 Bedroom,1020.0,2.0,62.0,2,6078
Margondanahalli,2 Bedroom,600.0,2.0,35.0,2,5833
other,4 BHK,3563.0,6.0,310.0,4,8700
other,2 BHK,1165.0,2.0,47.0,2,4034
Bellandur,3 BHK,1685.0,3.0,100.0,3,5934
Raja Rajeshwari Nagar,3 BHK,1580.0,3.0,53.56,3,3389
Balagere,2 BHK,1210.0,2.0,83.0,2,6859
Kammasandra,2 BHK,1030.0,1.0,25.75,2,2500
other,2 BHK,1100.0,2.0,43.5,2,3954
Hebbal,3 BHK,1255.0,3.0,77.68,3,6189
Jakkur,3 BHK,1482.0,3.0,86.0,3,5802
Hebbal,2 BHK,1344.0,2.0,108.0,2,8035
Kothannur,3 BHK,1275.0,2.0,39.8,3,3121
Vidyaranyapura,3 Bedroom,462.0,3.0,38.0,3,8225
other,4 BHK,3300.0,5.0,318.0,4,9636
Panathur,2 BHK,1210.0,2.0,85.0,2,7024
Hebbal,3 BHK,1320.0,3.0,226.0,3,17121
Hosa Road,3 BHK,1532.0,3.0,98.13,3,6405
JP Nagar,2 BHK,1245.0,2.0,110.0,2,8835
Nagarbhavi,2 BHK,1212.0,2.0,50.0,2,4125
Sarjapur  Road,2 BHK,1230.0,2.0,65.0,2,5284
Indira Nagar,5 Bedroom,1000.0,5.0,180.0,5,18000
Ramagondanahalli,3 BHK,2257.0,3.0,159.0,3,7044
Vittasandra,2 BHK,1246.0,2.0,65.4,2,5248
Gunjur,2 BHK,1080.0,2.0,41.0,2,3796
Varthur,3 BHK,1360.0,2.0,85.0,3,6250
Yelenahalli,2 BHK,1260.0,2.0,47.88,2,3800
Kengeri Satellite Town,4 Bedroom,1200.0,3.0,110.0,4,9166
Thigalarapalya,2 BHK,1418.0,2.0,104.0,2,7334
other,3 Bedroom,1200.0,3.0,165.0,3,13750
Kasavanhalli,2 BHK,950.0,2.0,39.0,2,4105
Nagarbhavi,2 BHK,1020.0,2.0,55.0,2,5392
Electronic City Phase II,3 BHK,1400.0,2.0,40.44,3,2888
other,4 Bedroom,1440.0,4.0,88.0,4,6111
other,6 BHK,800.0,6.0,75.0,6,9375
Electronics City Phase 1,2 BHK,1175.0,2.0,69.0,2,5872
Raja Rajeshwari Nagar,2 BHK,1185.0,2.0,40.17,2,3389
other,3 BHK,2250.0,3.0,113.0,3,5022
Sarjapur  Road,2 BHK,1178.0,2.0,40.43,2,3432
Ulsoor,4 BHK,36000.0,4.0,450.0,4,1250
Whitefield,1 BHK,825.0,1.0,44.9,1,5442
other,3 BHK,1250.0,3.0,43.0,3,3440
Electronics City Phase 1,2 BHK,1113.12,2.0,55.0,2,4941
other,3 BHK,1718.0,3.0,223.0,3,12980
Gottigere,2 BHK,1010.0,2.0,35.0,2,3465
Kanakpura Road,2 BHK,900.0,2.0,41.0,2,4555
other,4 Bedroom,1500.0,4.0,240.0,4,16000
Rayasandra,1 Bedroom,540.0,1.0,16.5,1,3055
Electronic City,2 BHK,1020.0,2.0,29.46,2,2888
Hosur Road,2 BHK,1194.0,2.0,71.64,2,6000
Vijayanagar,8 Bedroom,1470.0,6.0,300.0,8,20408
2nd Stage Nagarbhavi,3 Bedroom,600.0,4.0,100.0,3,16666
Electronic City,2 BHK,1108.0,2.0,67.15,2,6060
Hebbal,4 BHK,3895.0,2.0,451.0,4,11578
Kanakpura Road,2 BHK,700.0,2.0,34.99,2,4998
Somasundara Palya,3 BHK,1600.0,3.0,68.0,3,4250
Nagarbhavi,3 BHK,1850.0,3.0,90.0,3,4864
Tumkur Road,2 BHK,1250.0,2.0,33.0,2,2640
Ramamurthy Nagar,1 Bedroom,600.0,1.0,43.0,1,7166
Raja Rajeshwari Nagar,2 BHK,1140.0,2.0,49.27,2,4321
other,3 BHK,1200.0,2.0,75.0,3,6250
other,2 BHK,985.0,2.0,55.0,2,5583
Bellandur,2 BHK,982.0,2.0,55.0,2,5600
9th Phase JP Nagar,1 BHK,600.0,1.0,20.0,1,3333
Old Madras Road,3 BHK,2640.0,5.0,142.0,3,5378
Akshaya Nagar,2 BHK,1314.0,2.0,68.8,2,5235
other,3 BHK,1700.0,2.0,68.5,3,4029
1st Phase JP Nagar,2 BHK,1077.0,2.0,93.0,2,8635
other,5 BHK,2000.0,4.0,99.0,5,4950
other,3 BHK,1643.0,3.0,75.0,3,4564
Kengeri,3 Bedroom,1200.0,3.0,67.0,3,5583
Whitefield,2 BHK,1116.0,2.0,51.91,2,4651
Abbigere,2 BHK,1020.0,2.0,41.82,2,4100
Vittasandra,7 Bedroom,621.0,7.0,64.0,7,10305
other,3 BHK,1740.0,3.0,55.0,3,3160
GM Palaya,2 BHK,1225.0,2.0,47.0,2,3836
Thanisandra,3 BHK,1241.0,2.0,61.0,3,4915
Hennur,2 BHK,1041.0,2.0,40.6,2,3900
Bommanahalli,2 BHK,1050.0,2.0,46.0,2,4380
other,3 BHK,4400.0,3.0,901.0,3,20477
Kanakpura Road,2 BHK,1309.0,2.0,64.39,2,4919
Dodda Nekkundi,2 BHK,1128.0,2.0,45.0,2,3989
Raja Rajeshwari Nagar,3 BHK,1430.0,2.0,50.0,3,3496
Old Madras Road,2 BHK,1065.0,2.0,47.86,2,4493
OMBR Layout,2 BHK,1085.0,2.0,64.0,2,5898
Vishveshwarya Layout,1 Bedroom,750.0,1.0,30.0,1,4000
Bellandur,2 BHK,981.0,2.0,37.28,2,3800
Bhoganhalli,4 BHK,1548.3,4.0,129.0,4,8331
Hebbal Kempapura,4 BHK,2302.0,4.0,300.0,4,13032
Kalyan nagar,3 BHK,1410.0,2.0,60.0,3,4255
Hosa Road,3 BHK,1525.84,3.0,117.0,3,7667
Begur Road,2 BHK,1200.0,2.0,42.6,2,3550
other,2 BHK,1030.0,2.0,80.0,2,7766
other,3 BHK,1852.0,2.0,95.0,3,5129
Electronic City,2 BHK,1160.0,2.0,33.5,2,2887
Jigani,2 BHK,918.0,2.0,49.0,2,5337
Hebbal,3 BHK,1255.0,2.0,90.0,3,7171
HRBR Layout,6 Bedroom,1200.0,6.0,250.0,6,20833
JP Nagar,3 BHK,1150.0,2.0,49.9,3,4339
Indira Nagar,3 BHK,2700.0,3.0,324.0,3,12000
other,2 BHK,1150.0,2.0,69.0,2,6000
Whitefield,2 BHK,1227.0,2.0,53.49,2,4359
Hebbal,3 BHK,1255.0,3.0,77.68,3,6189
other,6 Bedroom,840.0,8.0,120.0,6,14285
Panathur,2 BHK,1198.0,2.0,82.0,2,6844
Devarachikkanahalli,3 Bedroom,1200.0,3.0,160.0,3,13333
other,4 Bedroom,2600.0,3.0,95.0,4,3653
Thigalarapalya,3 BHK,2072.0,4.0,139.0,3,6708
Pattandur Agrahara,2 BHK,797.0,2.0,32.0,2,4015
other,3 BHK,1350.0,2.0,68.0,3,5037
Sahakara Nagar,4 BHK,2500.0,4.0,200.0,4,8000
Thigalarapalya,3 BHK,2215.0,4.0,160.0,3,7223
Kanakpura Road,1 BHK,525.0,1.0,30.0,1,5714
Bannerghatta Road,4 BHK,2700.0,4.0,260.0,4,9629
other,3 BHK,1500.0,2.0,90.0,3,6000
other,2 Bedroom,900.0,2.0,55.0,2,6111
Poorna Pragna Layout,3 BHK,1355.0,2.0,58.27,3,4300
Kudlu Gate,3 BHK,1465.0,2.0,49.81,3,3400
Ramamurthy Nagar,2 Bedroom,1006.0,2.0,57.0,2,5666
Vishveshwarya Layout,4 BHK,2000.0,4.0,75.0,4,3750
Electronic City,2 BHK,1128.0,2.0,65.7,2,5824
other,2 BHK,850.0,2.0,35.0,2,4117
ISRO Layout,3 BHK,1310.0,2.0,68.0,3,5190
Banashankari,2 BHK,1390.0,2.0,86.0,2,6187
Kathriguppe,3 BHK,1390.0,2.0,76.45,3,5500
Marathahalli,4 Bedroom,2396.0,4.0,325.0,4,13564
Electronic City,2 BHK,1170.0,2.0,33.0,2,2820
Gottigere,3 BHK,1385.0,2.0,50.0,3,3610
Old Madras Road,2 BHK,1160.0,2.0,42.0,2,3620
Electronic City Phase II,2 BHK,1160.0,2.0,33.5,2,2887
Bannerghatta Road,3 BHK,1200.0,2.0,69.0,3,5750
Mahadevpura,2 BHK,1475.0,2.0,55.0,2,3728
Sarjapur  Road,3 Bedroom,1939.0,3.0,98.0,3,5054
7th Phase JP Nagar,2 BHK,850.0,2.0,42.0,2,4941
Balagere,1 BHK,675.0,1.0,45.0,1,6666
Electronic City,2 BHK,1060.0,2.0,51.0,2,4811
Whitefield,2 BHK,1235.0,2.0,47.0,2,3805
other,2 Bedroom,600.0,4.0,110.0,2,18333
other,2 BHK,1250.0,2.0,60.0,2,4800
Banashankari,2 BHK,1460.0,2.0,70.0,2,4794
Akshaya Nagar,3 BHK,1690.0,3.0,85.0,3,5029
Indira Nagar,1 BHK,850.0,1.0,57.0,1,6705
Kanakpura Road,3 BHK,1843.0,3.0,82.0,3,4449
7th Phase JP Nagar,3 BHK,1680.0,3.0,117.0,3,6964
Kundalahalli,2 BHK,1047.0,2.0,84.0,2,8022
8th Phase JP Nagar,2 BHK,1100.0,2.0,35.0,2,3181
Indira Nagar,8 Bedroom,3250.0,8.0,600.0,8,18461
Nagarbhavi,2 BHK,1120.0,2.0,53.0,2,4732
Raja Rajeshwari Nagar,2 BHK,1095.0,2.0,38.33,2,3500
other,2 BHK,970.0,2.0,35.0,2,3608
Bommanahalli,5 Bedroom,1200.0,7.0,125.0,5,10416
Hosa Road,3 BHK,1100.0,2.0,85.0,3,7727
Uttarahalli,3 BHK,1685.0,3.0,85.0,3,5044
other,2 BHK,1148.0,2.0,59.0,2,5139
Marathahalli,2 BHK,1050.0,2.0,65.0,2,6190
Bannerghatta Road,3 BHK,1465.0,2.0,76.0,3,5187
Kothanur,4 Bedroom,3401.0,5.0,230.0,4,6762
Electronic City,1 BHK,630.0,1.0,60.0,1,9523
7th Phase JP Nagar,3 BHK,1850.0,3.0,150.0,3,8108
Vidyaranyapura,9 Bedroom,1200.0,9.0,175.0,9,14583
other,4 Bedroom,1750.0,2.0,225.0,4,12857
other,3 BHK,1350.0,3.0,60.0,3,4444
Sector 7 HSR Layout,3 BHK,1760.0,3.0,160.0,3,9090
Rajaji Nagar,2 BHK,1440.0,2.0,185.0,2,12847
Banashankari Stage III,2 BHK,1200.0,2.0,42.0,2,3500
other,2 BHK,1100.0,2.0,78.0,2,7090
Kaggadasapura,2 BHK,1140.0,2.0,46.0,2,4035
Hormavu,3 BHK,1617.5,3.0,73.595,3,4549
Gottigere,2 BHK,1410.0,2.0,45.0,2,3191
Mahadevpura,3 BHK,1434.0,2.0,67.39,3,4699
other,2 Bedroom,1180.0,2.0,74.0,2,6271
Begur Road,2 BHK,1160.0,2.0,42.0,2,3620
EPIP Zone,2 BHK,1343.0,2.0,86.0,2,6403
LB Shastri Nagar,2 BHK,1184.0,2.0,69.0,2,5827
Koramangala,3 BHK,2246.0,3.0,300.0,3,13357
other,3 Bedroom,4510.0,4.0,200.0,3,4434
Banashankari Stage III,3 BHK,1305.0,2.0,59.0,3,4521
Sarjapur  Road,4 BHK,3785.0,5.0,280.0,4,7397
other,3 BHK,1540.0,3.0,85.0,3,5519
Kanakpura Road,3 BHK,1550.0,3.0,65.11,3,4200
Yelahanka,3 BHK,1345.0,2.0,57.0,3,4237
Electronic City Phase II,3 BHK,975.0,2.0,45.0,3,4615
HSR Layout,2 BHK,1140.0,2.0,46.0,2,4035
JP Nagar,2 BHK,1152.0,2.0,34.56,2,3000
Old Airport Road,4 Bedroom,3200.0,4.0,700.0,4,21875
other,3 BHK,1033.0,2.0,31.5,3,3049
other,3 BHK,1800.0,3.0,230.0,3,12777
Marathahalli,2 BHK,1102.0,2.0,53.67,2,4870
Subramanyapura,3 BHK,1223.0,2.0,42.81,3,3500
Kadugodi,3 BHK,1762.0,3.0,112.0,3,6356
Kengeri Satellite Town,2 BHK,560.0,2.0,16.6,2,2964
KR Puram,6 Bedroom,2000.0,6.0,85.0,6,4250
Electronic City,3 BHK,1410.0,2.0,38.0,3,2695
Thanisandra,3 BHK,1853.0,3.0,64.86,3,3500
Kothanur,2 BHK,1455.0,2.0,69.0,2,4742
Hebbal,2 BHK,1138.0,2.0,110.0,2,9666
Sarjapur  Road,3 Bedroom,2556.0,3.0,169.0,3,6611
TC Palaya,3 BHK,1200.0,2.0,66.5,3,5541
other,2 BHK,1200.0,2.0,37.0,2,3083
Anekal,2 Bedroom,1200.0,1.0,36.0,2,3000
Whitefield,3 BHK,1870.0,3.0,138.0,3,7379
Panathur,2 BHK,1199.0,2.0,86.0,2,7172
other,6 Bedroom,800.0,8.0,150.0,6,18750
Doddakallasandra,3 BHK,1360.0,2.0,54.4,3,4000
Babusapalaya,3 BHK,1230.0,3.0,53.0,3,4308
other,3 BHK,2000.0,3.0,120.0,3,6000
Kasturi Nagar,4 Bedroom,1200.0,4.0,250.0,4,20833
Kundalahalli,2 BHK,1047.0,2.0,83.0,2,7927
Electronic City Phase II,4 Bedroom,6000.0,4.0,260.0,4,4333
Ramamurthy Nagar,2 Bedroom,1200.0,2.0,76.0,2,6333
Kengeri Satellite Town,2 BHK,1302.5,2.0,44.285,2,3400
Kengeri,2 BHK,725.0,2.0,30.0,2,4137
Jakkur,3 BHK,1645.0,3.0,100.0,3,6079
Sultan Palaya,2 BHK,870.0,2.0,65.0,2,7471
Marathahalli,2 BHK,1026.0,2.0,49.95,2,4868
other,3 BHK,2000.0,3.0,145.0,3,7250
Cooke Town,2 BHK,1710.0,2.0,200.0,2,11695
Hosur Road,3 BHK,1766.0,3.0,130.0,3,7361
other,1 BHK,500.0,1.0,15.0,1,3000
JP Nagar,2 BHK,1100.0,1.0,15.0,2,1363
Jigani,3 BHK,1221.0,3.0,52.0,3,4258
Rachenahalli,1 BHK,680.0,1.0,46.0,1,6764
Sarjapur  Road,2 BHK,1278.0,2.0,90.0,2,7042
Kodichikkanahalli,2 BHK,1025.0,2.0,66.0,2,6439
Whitefield,2 BHK,1175.0,2.0,39.0,2,3319
Sarjapur  Road,2 BHK,975.0,2.0,28.275,2,2900
6th Phase JP Nagar,1 Bedroom,600.0,1.0,75.0,1,12500
other,2 BHK,850.0,2.0,42.0,2,4941
Anjanapura,2 BHK,710.0,2.0,25.0,2,3521
other,2 BHK,1165.0,2.0,33.64,2,2887
Sarjapur,1 BHK,625.0,1.0,19.0,1,3040
Kaggadasapura,3 BHK,1340.0,3.0,66.0,3,4925
Ananth Nagar,2 BHK,937.0,2.0,35.0,2,3735
other,7 Bedroom,1500.0,6.0,210.0,7,14000
JP Nagar,1 BHK,750.0,1.0,52.5,1,7000
Shampura,3 BHK,1475.0,3.0,78.0,3,5288
other,4 Bedroom,2230.0,4.0,130.0,4,5829
other,4 Bedroom,2200.0,5.0,100.0,4,4545
Mysore Road,2 Bedroom,900.0,2.0,65.0,2,7222
Electronic City Phase II,2 BHK,1116.0,2.0,37.0,2,3315
Hoodi,2 BHK,948.0,2.0,75.0,2,7911
R.T. Nagar,2 BHK,970.0,2.0,55.0,2,5670
other,3 BHK,1519.0,3.0,67.23,3,4425
Thanisandra,3 BHK,1200.0,2.0,65.0,3,5416
Bhoganhalli,3 BHK,1760.0,3.0,127.0,3,7215
Whitefield,4 Bedroom,4400.0,5.0,525.0,4,11931
Sahakara Nagar,3 BHK,1914.0,3.0,179.0,3,9352
Budigere,3 BHK,1820.0,3.0,85.5,3,4697
Budigere,3 BHK,1820.0,3.0,93.0,3,5109
Kalena Agrahara,3 BHK,1565.0,3.0,78.0,3,4984
Begur Road,2 BHK,1334.0,2.0,42.69,2,3200
Sarakki Nagar,2 BHK,924.0,2.0,76.23,2,8250
Jigani,2 BHK,914.0,2.0,49.0,2,5361
Kengeri,1 BHK,500.0,1.0,17.0,1,3400
Yelahanka,3 BHK,2145.0,3.0,125.0,3,5827
Yelahanka,5 Bedroom,1200.0,5.0,162.0,5,13500
Electronic City,3 BHK,1650.0,3.0,100.0,3,6060
Marathahalli,4 BHK,3940.0,4.0,220.0,4,5583
Electronic City,2 BHK,1020.0,2.0,29.45,2,2887
Basaveshwara Nagar,2 BHK,1180.0,2.0,58.0,2,4915
Ulsoor,2 BHK,1255.0,2.0,120.0,2,9561
other,3 BHK,2777.29,5.0,650.0,3,23404
Bommasandra,2 BHK,1014.0,2.0,35.0,2,3451
Shampura,4 Bedroom,920.0,2.0,90.0,4,9782
Attibele,2 BHK,695.0,1.0,25.0,2,3597
Prithvi Layout,3 Bedroom,2273.0,4.0,220.0,3,9678
GM Palaya,2 BHK,1140.0,2.0,48.95,2,4293
Giri Nagar,6 Bedroom,3000.0,4.0,375.0,6,12500
Whitefield,2 BHK,1180.0,2.0,41.0,2,3474
Begur Road,4 BHK,2400.0,4.0,152.0,4,6333
Attibele,2 BHK,850.0,1.0,25.0,2,2941
Banashankari,2 BHK,1200.0,2.0,60.0,2,5000
other,4 BHK,3160.0,4.0,206.0,4,6518
Hoodi,3 BHK,1769.0,3.0,96.87,3,5475
Bhoganhalli,4 BHK,1554.3,4.0,131.0,4,8428
Haralur Road,2 BHK,1243.0,2.0,46.0,2,3700
other,4 Bedroom,4920.0,5.0,1150.0,4,23373
Mallasandra,2 BHK,905.0,2.0,40.0,2,4419
Kothanur,1 Bedroom,627.0,1.0,30.0,1,4784
Domlur,3 BHK,2180.0,3.0,285.0,3,13073
Frazer Town,3 BHK,2400.0,3.0,260.0,3,10833
Marathahalli,3 BHK,1435.0,3.0,73.0,3,5087
Jakkur,4 BHK,3405.1,6.0,400.0,4,11747
Kodihalli,3 BHK,1620.0,3.0,130.0,3,8024
Whitefield,3 BHK,1862.0,3.0,125.0,3,6713
Kaikondrahalli,2 BHK,674.0,1.0,19.9,2,2952
Marathahalli,2 BHK,1220.0,2.0,59.0,2,4836
Haralur Road,2 BHK,1027.0,2.0,44.0,2,4284
Electronic City,2 BHK,1025.0,2.0,49.0,2,4780
Thanisandra,3 BHK,1430.0,2.0,54.5,3,3811
Panathur,2 BHK,1235.0,2.0,65.0,2,5263
Electronics City Phase 1,3 BHK,1291.0,2.0,32.0,3,2478
Kanakpura Road,3 BHK,1498.0,3.0,95.0,3,6341
Sarjapur  Road,3 BHK,2180.0,3.0,240.0,3,11009
Electronic City,2 BHK,1000.0,2.0,29.7,2,2970
other,2 Bedroom,922.0,2.0,70.0,2,7592
other,2 BHK,850.0,2.0,72.0,2,8470
other,3 Bedroom,750.0,3.0,155.0,3,20666
AECS Layout,2 BHK,1123.0,2.0,64.0,2,5699
Raja Rajeshwari Nagar,2 BHK,1235.0,2.0,75.0,2,6072
Kogilu,2 BHK,1250.0,2.0,55.55,2,4444
Whitefield,4 Bedroom,2403.0,5.0,360.0,4,14981
Sarjapur  Road,2 BHK,984.0,2.0,45.91,2,4665
Sarjapur  Road,2 BHK,1305.0,2.0,99.0,2,7586
other,2 Bedroom,1200.0,1.0,65.0,2,5416
other,3 Bedroom,600.0,3.0,90.0,3,15000
Hebbal,4 BHK,2790.0,5.0,204.0,4,7311
Old Madras Road,4 BHK,2010.0,3.0,125.0,4,6218
Electronics City Phase 1,2 BHK,995.0,2.0,42.0,2,4221
Varthur Road,2 BHK,1277.0,2.0,59.0,2,4620
7th Phase JP Nagar,2 BHK,1236.0,2.0,61.33,2,4961
Akshaya Nagar,2 BHK,1265.0,2.0,67.0,2,5296
Gubbalala,5 BHK,2570.0,4.0,141.0,5,5486
Hennur Road,4 Bedroom,2400.0,5.0,390.0,4,16250
other,3 BHK,1442.0,3.0,56.24,3,3900
Ardendale,2 BHK,1224.0,2.0,70.0,2,5718
Bellandur,2 BHK,1205.0,2.0,59.0,2,4896
other,2 BHK,1233.0,2.0,100.0,2,8110
5th Block Hbr Layout,2 BHK,1312.0,2.0,69.0,2,5259
Electronic City,3 BHK,1449.0,3.0,90.0,3,6211
Yelenahalli,2 BHK,1090.0,2.0,54.0,2,4954
Kaggadasapura,5 Bedroom,2100.0,5.0,150.0,5,7142
Uttarahalli,3 BHK,1390.0,2.0,62.55,3,4500
JP Nagar,2 BHK,1200.0,2.0,78.0,2,6500
Mallasandra,4 Bedroom,600.0,2.0,50.0,4,8333
BTM 2nd Stage,2 BHK,1000.0,2.0,58.0,2,5800
Devanahalli,4 Bedroom,2400.0,5.0,190.0,4,7916
Thanisandra,3 BHK,1732.0,3.0,85.73,3,4949
Banashankari,2 BHK,1290.0,2.0,80.0,2,6201
other,3 BHK,1840.0,3.0,85.0,3,4619
Thanisandra,3 BHK,1917.0,4.0,100.0,3,5216
other,2 BHK,1100.0,2.0,57.0,2,5181
Kudlu,3 BHK,1455.0,2.0,60.0,3,4123
Old Airport Road,4 BHK,2774.0,4.0,208.0,4,7498
Haralur Road,5 Bedroom,1200.0,5.0,255.0,5,21250
Devarachikkanahalli,3 BHK,1425.0,2.0,65.0,3,4561
Whitefield,3 BHK,1655.0,3.0,113.0,3,6827
Haralur Road,4 BHK,2805.0,5.0,160.0,4,5704
R.T. Nagar,7 Bedroom,600.0,7.0,93.0,7,15500
other,2 BHK,1040.0,2.0,46.79,2,4499
Akshaya Nagar,2 BHK,1126.0,2.0,55.0,2,4884
Kodichikkanahalli,6 Bedroom,1600.0,4.0,200.0,6,12500
Hebbal,2 BHK,1250.0,2.0,55.55,2,4444
Yelahanka,4 Bedroom,1800.0,4.0,180.0,4,10000
other,2 Bedroom,1200.0,2.0,53.0,2,4416
other,4 BHK,2050.0,3.0,100.0,4,4878
Whitefield,1 BHK,530.0,1.0,29.44,1,5554
Sarjapur,4 Bedroom,2585.5,4.0,139.5,4,5395
Kambipura,2 BHK,883.0,2.0,37.0,2,4190
Uttarahalli,3 BHK,1490.0,2.0,59.6,3,4000
Ramagondanahalli,2 BHK,1295.0,2.0,95.0,2,7335
Sompura,3 BHK,1275.0,3.0,59.0,3,4627
other,2 BHK,800.0,2.0,30.0,2,3750
Devanahalli,1 BHK,1200.0,1.0,95.0,1,7916
Bellandur,2 BHK,1063.0,2.0,58.0,2,5456
Balagere,1 BHK,656.0,1.0,43.0,1,6554
other,2 BHK,1125.0,2.0,36.0,2,3200
Marathahalli,2 BHK,881.0,2.0,48.06,2,5455
other,9 Bedroom,1350.0,8.0,200.0,9,14814
Bisuvanahalli,2 BHK,700.0,1.0,32.0,2,4571
other,3 Bedroom,1936.0,3.0,110.0,3,5681
Hosa Road,3 BHK,1680.0,3.0,101.0,3,6011
Rayasandra,2 BHK,1253.0,2.0,57.0,2,4549
Rajaji Nagar,3 BHK,1640.0,3.0,230.0,3,14024
Mysore Road,3 BHK,1082.0,3.0,50.0,3,4621
Electronic City,2 BHK,1065.0,2.0,30.75,2,2887
Iblur Village,3 BHK,2100.0,4.0,145.0,3,6904
Electronic City,2 BHK,1070.0,2.0,58.0,2,5420
Raja Rajeshwari Nagar,3 BHK,1280.0,3.0,57.6,3,4500
other,5 Bedroom,1200.0,5.0,300.0,5,25000
Electronic City Phase II,2 BHK,1125.0,2.0,29.25,2,2600
Sarjapur  Road,3 BHK,1755.0,3.0,122.0,3,6951
Electronic City,2 BHK,1210.0,2.0,54.0,2,4462
Talaghattapura,2 BHK,951.0,2.0,30.43,2,3199
Electronic City,3 BHK,1500.0,2.0,75.0,3,5000
other,3 BHK,1652.5,3.0,90.0,3,5446
Yeshwanthpur,3 BHK,1381.0,2.0,76.18,3,5516
Whitefield,2 BHK,1085.0,2.0,57.0,2,5253
Whitefield,1 BHK,905.0,1.0,50.0,1,5524
Hennur Road,3 BHK,1685.0,3.0,95.0,3,5637
Somasundara Palya,2 BHK,1200.0,2.0,75.0,2,6250
Anandapura,2 BHK,1151.0,2.0,42.9,2,3727
Brookefield,4 Bedroom,3532.0,3.0,170.0,4,4813
Kanakpura Road,3 BHK,1665.0,3.0,86.91,3,5219
Kengeri Satellite Town,2 BHK,800.0,2.0,40.0,2,5000
Chamrajpet,2 Bedroom,1050.0,2.0,162.0,2,15428
Rajiv Nagar,2 BHK,1030.0,2.0,46.5,2,4514
Hoodi,2 BHK,1257.0,2.0,71.03,2,5650
Whitefield,3 BHK,1600.0,3.0,82.0,3,5125
Choodasandra,3 BHK,1465.0,3.0,75.0,3,5119
Jakkur,3 BHK,1650.0,3.0,111.0,3,6727
Haralur Road,2 BHK,1243.0,2.0,46.0,2,3700
Yelahanka New Town,2 Bedroom,680.0,1.0,80.0,2,11764
Whitefield,4 Bedroom,1230.0,4.0,80.0,4,6504
Koramangala,2 BHK,1300.0,2.0,82.0,2,6307
KR Puram,2 BHK,1000.0,2.0,44.5,2,4450
Mahadevpura,2 BHK,1146.0,2.0,53.8,2,4694
other,6 BHK,1200.0,6.0,120.0,6,10000
Indira Nagar,3 BHK,2601.0,4.0,312.0,3,11995
other,3 BHK,2159.0,3.0,110.0,3,5094
Whitefield,2 BHK,1255.0,2.0,40.0,2,3187
Hebbal Kempapura,2 BHK,1130.0,2.0,60.0,2,5309
Padmanabhanagar,2 BHK,1200.0,2.0,60.0,2,5000
Sarjapur  Road,2 BHK,1040.0,2.0,37.0,2,3557
TC Palaya,8 Bedroom,1500.0,8.0,150.0,8,10000
other,4 Bedroom,906.0,3.0,55.0,4,6070
Uttarahalli,2 BHK,1180.0,2.0,41.3,2,3499
other,7 Bedroom,1200.0,7.0,169.0,7,14083
other,6 Bedroom,850.0,7.0,100.0,6,11764
KR Puram,2 BHK,1200.0,2.0,47.0,2,3916
Sarjapur  Road,3 BHK,1374.0,3.0,51.0,3,3711
Vidyaranyapura,9 BHK,4700.0,10.0,130.0,9,2765
Whitefield,1 BHK,630.5,1.0,32.15,1,5099
other,3 Bedroom,2400.0,2.0,480.0,3,20000
Haralur Road,2 BHK,1027.0,2.0,44.0,2,4284
other,5 Bedroom,1200.0,4.0,130.0,5,10833
Jalahalli,3 BHK,2384.0,4.0,165.0,3,6921
Electronic City,3 BHK,1033.0,2.0,28.41,3,2750
Yelahanka,2 BHK,1200.0,2.0,54.0,2,4500
Haralur Road,3 BHK,2017.0,3.0,105.0,3,5205
other,3 Bedroom,1500.0,3.0,138.0,3,9200
other,2 BHK,700.0,2.0,43.0,2,6142
Shampura,6 Bedroom,1150.0,6.0,75.0,6,6521
Kanakpura Road,3 BHK,1495.0,2.0,52.33,3,3500
Kothanur,2 BHK,1302.0,2.0,68.5,2,5261
Whitefield,1 BHK,905.0,1.0,52.0,1,5745
Raja Rajeshwari Nagar,2 BHK,1303.0,2.0,55.79,2,4281
Electronics City Phase 1,3 BHK,1350.0,3.0,71.0,3,5259
Gottigere,3 BHK,1380.0,2.0,95.0,3,6884
Harlur,3 BHK,1620.0,3.0,110.0,3,6790
Chandapura,4 Bedroom,4800.0,4.0,250.0,4,5208
Sarjapur  Road,2 BHK,1000.0,2.0,38.5,2,3850
Ramamurthy Nagar,3 Bedroom,800.0,4.0,88.0,3,11000
Thanisandra,2 BHK,1250.0,1.0,54.0,2,4320
other,3 BHK,1750.0,3.0,170.0,3,9714
Anekal,3 BHK,997.0,3.0,55.0,3,5516
Kothanur,3 BHK,1760.0,3.0,111.0,3,6306
Yelahanka,2 BHK,1322.0,2.0,75.0,2,5673
other,3 Bedroom,1365.0,3.0,99.0,3,7252
Sahakara Nagar,3 BHK,1500.0,3.0,125.0,3,8333
Kaggadasapura,3 BHK,1256.0,2.0,50.0,3,3980
Brookefield,2 BHK,941.0,2.0,48.0,2,5100
Devanahalli,4 Bedroom,5000.0,4.0,465.0,4,9300
Kanakpura Road,3 BHK,1665.0,3.0,86.58,3,5200
Hennur Road,2 BHK,1414.0,2.0,84.18,2,5953
Chikkalasandra,3 BHK,1270.0,2.0,55.25,3,4350
other,3 Bedroom,800.0,2.0,80.0,3,10000
Kengeri Satellite Town,4 Bedroom,4100.0,3.0,300.0,4,7317
other,6 Bedroom,1800.0,3.0,105.0,6,5833
other,2 BHK,850.0,2.0,35.0,2,4117
Kanakpura Road,1 BHK,712.0,1.0,38.44,1,5398
Rajaji Nagar,4 BHK,2648.0,5.0,251.5,4,9497
other,3 Bedroom,900.0,4.0,180.0,3,20000
TC Palaya,2 Bedroom,900.0,2.0,55.0,2,6111
Hebbal,2 BHK,1200.0,2.0,49.0,2,4083
Kengeri Satellite Town,1 BHK,654.0,1.0,22.0,1,3363
other,2 BHK,708.0,1.0,55.0,2,7768
other,4 Bedroom,600.0,5.0,75.0,4,12500
other,3 BHK,1680.0,3.0,135.0,3,8035
other,2 BHK,990.0,2.0,58.0,2,5858
Uttarahalli,2 BHK,1380.0,2.0,42.0,2,3043
Hebbal,2 BHK,1088.0,2.0,62.0,2,5698
Harlur,3 BHK,1414.0,3.0,77.5,3,5480
other,2 BHK,1240.0,2.0,100.0,2,8064
other,3 BHK,1810.0,3.0,118.0,3,6519
other,3 BHK,2005.0,3.0,240.0,3,11970
Kothanur,2 BHK,1185.0,2.0,58.0,2,4894
Whitefield,3 Bedroom,1200.0,3.0,57.15,3,4762
Electronic City,2 BHK,1070.0,2.0,52.0,2,4859
other,2 BHK,1265.0,2.0,58.0,2,4584
Sahakara Nagar,2 BHK,1100.0,2.0,55.0,2,5000
other,2 BHK,1035.0,2.0,44.0,2,4251
Sonnenahalli,1 BHK,614.5,1.0,30.715,1,4998
other,2 BHK,1000.0,2.0,97.0,2,9700
Hegde Nagar,2 BHK,1179.0,2.0,49.35,2,4185
Vittasandra,3 BHK,1650.0,3.0,84.0,3,5090
other,6 Bedroom,2800.0,4.0,100.0,6,3571
other,8 Bedroom,3500.0,6.0,120.0,8,3428
other,3 BHK,1430.0,3.0,78.65,3,5500
Marathahalli,3 BHK,1130.0,2.0,58.0,3,5132
JP Nagar,3 BHK,1850.0,2.0,110.0,3,5945
Hennur,3 BHK,1340.0,3.0,54.24,3,4047
other,2 Bedroom,600.0,2.0,30.0,2,5000
Banashankari,3 BHK,2582.0,5.0,250.0,3,9682
Channasandra,2 BHK,1160.0,2.0,55.0,2,4741
other,1 Bedroom,1200.0,2.0,45.0,1,3750
other,2 BHK,1116.0,2.0,37.6,2,3369
other,3 Bedroom,600.0,2.0,85.0,3,14166
Sarakki Nagar,3 BHK,2663.0,4.0,338.0,3,12692
other,2 BHK,1090.0,2.0,45.0,2,4128
JP Nagar,2 BHK,1100.0,2.0,45.0,2,4090
Kammasandra,3 BHK,1365.0,2.0,34.13,3,2500
Vijayanagar,3 BHK,1484.0,3.0,79.19,3,5336
Green Glen Layout,3 BHK,1630.0,3.0,68.0,3,4171
Whitefield,3 BHK,1451.0,2.0,52.43,3,3613
Kanakpura Road,2 BHK,1330.0,2.0,107.0,2,8045
Thanisandra,3 BHK,2127.0,3.0,135.0,3,6346
other,5 Bedroom,875.0,5.0,120.0,5,13714
other,4 Bedroom,1260.0,4.0,65.0,4,5158
Sarjapur  Road,3 BHK,1173.0,2.0,45.75,3,3900
Electronics City Phase 1,3 BHK,1445.0,2.0,56.0,3,3875
other,4 BHK,2250.0,4.0,400.0,4,17777
Bannerghatta Road,3 Bedroom,1500.0,4.0,239.0,3,15933
Thanisandra,3 BHK,2336.5,3.0,115.89,3,4959
JP Nagar,2 BHK,1197.0,2.0,47.88,2,4000
2nd Stage Nagarbhavi,4 Bedroom,1200.0,4.0,225.0,4,18750
Sarjapur  Road,5 BHK,3930.0,7.0,329.0,5,8371
Kanakpura Road,3 BHK,1450.0,3.0,62.4,3,4303
Whitefield,2 BHK,1187.0,2.0,37.85,2,3188
other,6 Bedroom,4000.0,6.0,850.0,6,21250
Channasandra,2 BHK,1050.0,2.0,41.0,2,3904
Panathur,3 BHK,1370.0,2.0,54.8,3,4000
Yelahanka,2 BHK,1036.0,2.0,39.37,2,3800
JP Nagar,3 BHK,1275.0,3.0,41.2,3,3231
Old Madras Road,4 BHK,2010.0,3.0,115.0,4,5721
Thigalarapalya,2 BHK,1418.0,2.0,106.0,2,7475
Thanisandra,2 BHK,1265.0,2.0,46.6,2,3683
Anandapura,3 BHK,1576.0,3.0,58.0,3,3680
Dodda Nekkundi,2 BHK,1148.0,2.0,56.0,2,4878
other,2 BHK,1000.0,2.0,36.0,2,3600
Kudlu,3 BHK,1293.0,2.0,85.0,3,6573
TC Palaya,3 Bedroom,1200.0,2.0,66.0,3,5500
Munnekollal,2 BHK,1210.0,2.0,75.0,2,6198
HSR Layout,2 BHK,1126.0,2.0,69.0,2,6127
Sarakki Nagar,3 BHK,1840.0,4.0,175.0,3,9510
Vidyaranyapura,3 BHK,1440.0,2.0,70.0,3,4861
Ramamurthy Nagar,3 BHK,1300.0,2.0,52.0,3,4000
Thanisandra,2 BHK,1276.0,2.0,75.0,2,5877
other,6 Bedroom,600.0,4.0,105.0,6,17500
Whitefield,2 BHK,1216.0,2.0,83.0,2,6825
other,1 BHK,840.0,1.0,75.0,1,8928
other,5 Bedroom,1240.0,5.0,300.0,5,24193
Sompura,2 BHK,1126.0,2.0,39.0,2,3463
Mysore Road,1 Bedroom,45.0,1.0,23.0,1,51111
Raja Rajeshwari Nagar,2 BHK,1162.0,2.0,65.0,2,5593
Whitefield,2 BHK,1301.0,2.0,65.0,2,4996
Devanahalli,1 BHK,658.0,1.0,34.0,1,5167
Sahakara Nagar,3 Bedroom,1350.0,3.0,250.0,3,18518
Whitefield,3 BHK,2500.0,3.0,280.0,3,11200
Chandapura,3 BHK,1208.51,3.0,42.0,3,3475
Raja Rajeshwari Nagar,2 BHK,1133.0,2.0,38.41,2,3390
other,3 BHK,1760.0,2.0,116.0,3,6590
Bisuvanahalli,3 BHK,1075.0,2.0,52.0,3,4837
Sarjapur,2 BHK,940.0,2.0,25.38,2,2700
Badavala Nagar,2 BHK,1145.0,2.0,57.0,2,4978
Bannerghatta Road,2 BHK,1302.5,2.0,67.73,2,5200
other,4 Bedroom,3900.0,5.0,550.0,4,14102
other,4 BHK,3317.5,5.0,268.5,4,8093
Dasarahalli,2 BHK,1300.0,2.0,50.0,2,3846
Malleshwaram,3 BHK,2475.0,4.0,326.0,3,13171
6th Phase JP Nagar,3 BHK,2040.0,3.0,200.0,3,9803
Hosakerehalli,1 Bedroom,1200.0,1.0,150.0,1,12500
Jigani,4 Bedroom,2420.0,4.0,190.0,4,7851
Thubarahalli,2 BHK,1216.0,2.0,80.0,2,6578
Tumkur Road,3 BHK,1459.0,2.0,110.0,3,7539
Jakkur,3 BHK,1622.0,2.0,80.0,3,4932
Kanakpura Road,3 BHK,1843.0,3.0,85.0,3,4612
Badavala Nagar,2 BHK,1274.0,2.0,81.0,2,6357
other,2 Bedroom,600.0,1.0,52.0,2,8666
Malleshpalya,3 BHK,1430.0,2.0,69.5,3,4860
Kammanahalli,4 Bedroom,1800.0,2.0,280.0,4,15555
Begur Road,3 BHK,1565.0,2.0,57.91,3,3700
other,3 BHK,1670.0,3.0,160.0,3,9580
Lakshminarayana Pura,3 BHK,1695.0,3.0,150.0,3,8849
other,2 BHK,1410.0,2.0,75.0,2,5319
Kengeri Satellite Town,2 BHK,1060.0,2.0,42.0,2,3962
Sarjapur  Road,2 BHK,1120.0,2.0,50.0,2,4464
Hosa Road,2 Bedroom,880.0,2.0,50.0,2,5681
Jalahalli,2 BHK,1083.0,2.0,32.49,2,3000
Dodda Nekkundi,3 BHK,1435.0,3.0,57.2,3,3986
Doddakallasandra,3 BHK,1425.0,2.0,56.99,3,3999
Arekere,2 BHK,900.0,2.0,50.0,2,5555
other,5 BHK,2400.0,5.0,93.0,5,3875
Thanisandra,1 BHK,693.0,1.0,34.3,1,4949
Kengeri,3 Bedroom,600.0,3.0,90.0,3,15000
Electronic City,4 BHK,2435.0,4.0,119.0,4,4887
Iblur Village,3 BHK,1920.0,3.0,150.0,3,7812
Electronic City,1 BHK,630.0,1.0,47.0,1,7460
Bisuvanahalli,3 BHK,1075.0,2.0,50.0,3,4651
Hulimavu,5 Bedroom,1200.0,5.0,101.0,5,8416
Nagavarapalya,1 BHK,705.0,1.0,55.0,1,7801
other,3 BHK,1200.0,2.0,75.6,3,6299
Mahadevpura,2 BHK,1532.0,2.0,62.0,2,4046
Rajiv Nagar,2 BHK,1330.0,2.0,93.0,2,6992
other,4 Bedroom,1200.0,3.0,210.0,4,17500
Sarjapur  Road,3 BHK,1530.0,2.0,60.0,3,3921
Kasavanhalli,2 BHK,1105.0,2.0,42.0,2,3800
1st Phase JP Nagar,2 Bedroom,1566.0,2.0,180.0,2,11494
Harlur,3 BHK,1630.0,3.0,95.1,3,5834
5th Phase JP Nagar,2 BHK,1256.0,2.0,62.8,2,5000
Banashankari,6 BHK,1200.0,6.0,180.0,6,15000
Brookefield,3 BHK,1595.0,2.0,75.0,3,4702
7th Phase JP Nagar,2 BHK,1000.0,2.0,53.0,2,5300
Poorna Pragna Layout,3 BHK,1475.0,2.0,58.99,3,3999
other,5 Bedroom,2400.0,5.0,285.0,5,11875
Electronic City Phase II,2 BHK,1000.0,2.0,28.88,2,2888
other,4 Bedroom,2360.0,4.0,650.0,4,27542
Uttarahalli,3 BHK,1737.0,2.0,95.0,3,5469
Electronic City Phase II,2 BHK,1000.0,2.0,28.88,2,2888
other,4 BHK,2100.0,4.0,150.0,4,7142
Hosakerehalli,8 Bedroom,3600.0,6.0,145.0,8,4027
Mysore Road,2 BHK,883.0,2.0,50.0,2,5662
Haralur Road,2 BHK,1243.0,2.0,46.0,2,3700
BTM Layout,3 BHK,1776.0,2.0,75.0,3,4222
Anjanapura,3 BHK,1347.0,2.0,37.72,3,2800
Whitefield,3 BHK,1330.0,2.0,55.0,3,4135
other,2 Bedroom,600.0,2.0,120.0,2,20000
Rajaji Nagar,4 BHK,3730.0,4.0,370.0,4,9919
Harlur,3 BHK,1459.0,3.0,135.0,3,9252
Thanisandra,4 BHK,2259.0,5.0,135.0,4,5976
JP Nagar,7 Bedroom,2700.0,4.0,160.0,7,5925
Gubbalala,2 BHK,1008.0,2.0,32.25,2,3199
other,2 BHK,1200.0,2.0,52.5,2,4375
other,4 BHK,2328.0,4.0,528.0,4,22680
other,7 Bedroom,600.0,7.0,110.0,7,18333
Electronic City,2 BHK,660.0,1.0,15.5,2,2348
Rajaji Nagar,3 BHK,1800.0,2.0,250.0,3,13888
Rajaji Nagar,4 BHK,3100.0,4.0,550.0,4,17741
Yelahanka,2 BHK,1405.0,2.0,50.58,2,3600
Yelahanka,1 BHK,602.0,1.0,26.0,1,4318
Marathahalli,3 BHK,1710.0,3.0,100.0,3,5847
JP Nagar,3 BHK,1500.0,2.0,82.0,3,5466
other,3 BHK,1539.0,3.0,60.0,3,3898
other,6 Bedroom,806.0,3.0,128.0,6,15880
Judicial Layout,3 BHK,1100.0,2.0,53.0,3,4818
Raja Rajeshwari Nagar,2 BHK,1151.0,2.0,49.0,2,4257
other,2 BHK,920.0,2.0,46.0,2,5000
Kathriguppe,3 BHK,1350.0,3.0,78.3,3,5800
Bannerghatta Road,2 BHK,1200.0,2.0,65.0,2,5416
Thanisandra,2 BHK,1265.0,2.0,78.0,2,6166
other,8 Bedroom,600.0,6.0,60.0,8,10000
Sarjapur  Road,3 BHK,1700.0,3.0,93.5,3,5500
Ramamurthy Nagar,3 Bedroom,1600.0,3.0,75.0,3,4687
other,4 Bedroom,1550.0,4.0,120.0,4,7741
R.T. Nagar,3 BHK,1560.0,3.0,125.0,3,8012
Horamavu Agara,6 Bedroom,1000.0,5.0,98.0,6,9800
Haralur Road,2 BHK,1243.0,2.0,46.0,2,3700
Raja Rajeshwari Nagar,4 Bedroom,1200.0,4.0,160.0,4,13333
Sarjapur  Road,3 BHK,1403.0,3.0,56.0,3,3991
Doddakallasandra,2 BHK,1010.0,2.0,40.39,2,3999
Tumkur Road,3 BHK,1416.0,3.0,94.0,3,6638
Kanakpura Road,2 BHK,1628.0,2.0,68.0,2,4176
Chamrajpet,4 Bedroom,1660.4,4.0,211.0,4,12707
7th Phase JP Nagar,3 BHK,1680.0,3.0,125.0,3,7440
Doddathoguru,3 BHK,1595.0,3.0,62.0,3,3887
Kanakpura Road,3 BHK,1507.0,2.0,95.0,3,6303
Frazer Town,3 BHK,2560.0,3.0,288.0,3,11250
Hennur Road,3 BHK,1949.0,3.0,133.0,3,6824
BTM 2nd Stage,2 Bedroom,800.0,2.0,175.0,2,21875
Electronic City Phase II,2 BHK,1000.0,2.0,28.88,2,2888
other,2 BHK,905.0,2.0,50.0,2,5524
Hosakerehalli,3 BHK,2480.0,3.0,245.0,3,9879
other,3 BHK,1495.0,3.0,200.0,3,13377
Kaikondrahalli,3 BHK,1605.0,3.0,105.0,3,6542
Bommanahalli,2 BHK,1290.0,2.0,73.0,2,5658
Yelahanka,2 BHK,1100.0,2.0,60.0,2,5454
Vishveshwarya Layout,6 BHK,2100.0,6.0,92.0,6,4380
Electronics City Phase 1,2 BHK,995.0,2.0,49.8,2,5005
Banashankari Stage II,2 Bedroom,500.0,2.0,55.0,2,11000
Rajaji Nagar,4 BHK,3526.0,4.0,598.0,4,16959
Gottigere,3 BHK,1425.0,2.0,47.0,3,3298
other,3 BHK,1455.0,2.0,55.0,3,3780
other,1 BHK,550.0,1.0,23.0,1,4181
Uttarahalli,3 BHK,1385.0,2.0,48.48,3,3500
Vishveshwarya Layout,4 Bedroom,7000.0,4.0,210.0,4,3000
other,5 BHK,4000.0,3.0,130.0,5,3250
other,2 BHK,1060.0,2.0,53.0,2,5000
other,3 BHK,1080.0,2.0,75.0,3,6944
HBR Layout,3 BHK,1783.0,3.0,125.0,3,7010
Haralur Road,3 BHK,1520.0,2.0,90.0,3,5921
Sarjapur  Road,2 BHK,1060.0,2.0,55.0,2,5188
Raja Rajeshwari Nagar,2 BHK,1267.0,2.0,54.35,2,4289
Whitefield,2 BHK,1155.0,2.0,55.0,2,4761
Karuna Nagar,3 Bedroom,2500.0,3.0,180.0,3,7200
Bannerghatta Road,2 BHK,1240.0,2.0,65.0,2,5241
other,3 BHK,1570.0,3.0,78.0,3,4968
NGR Layout,2 BHK,1019.0,2.0,45.9,2,4504
Whitefield,2 BHK,1355.0,3.0,76.0,2,5608
other,3 BHK,1537.0,3.0,110.0,3,7156
Whitefield,4 BHK,4075.0,5.0,226.0,4,5546
Raja Rajeshwari Nagar,3 BHK,1550.0,3.0,52.45,3,3383
Raja Rajeshwari Nagar,2 BHK,1100.0,2.0,50.0,2,4545
other,3 BHK,1750.0,3.0,150.0,3,8571
9th Phase JP Nagar,2 BHK,1466.0,2.0,71.0,2,4843
Rachenahalli,3 BHK,1550.0,3.0,72.5,3,4677
Akshaya Nagar,3 BHK,2300.0,4.0,102.0,3,4434
Kengeri,1 BHK,540.0,1.0,22.5,1,4166
8th Phase JP Nagar,3 Bedroom,950.0,3.0,130.0,3,13684
other,7 Bedroom,2800.0,7.0,250.0,7,8928
other,2 BHK,1020.0,2.0,48.0,2,4705
Cunningham Road,3 BHK,2880.0,3.0,501.0,3,17395
Hebbal,3 BHK,1987.0,4.0,165.0,3,8303
Hoodi,2 BHK,1305.0,2.0,75.0,2,5747
Sarjapur,4 Bedroom,2100.0,3.0,125.0,4,5952
Electronic City,2 BHK,1125.0,2.0,32.5,2,2888
Horamavu Banaswadi,2 BHK,1225.0,2.0,49.5,2,4040
Hennur,2 BHK,1255.0,2.0,55.5,2,4422
other,2 BHK,1191.0,2.0,90.0,2,7556
Uttarahalli,6 Bedroom,3600.0,6.0,303.0,6,8416
Electronic City,2 BHK,1165.0,2.0,33.64,2,2887
other,2 BHK,865.0,2.0,35.11,2,4058
Nagarbhavi,3 BHK,1850.0,2.0,88.0,3,4756
Marsur,2 BHK,497.0,1.0,17.0,2,3420
Hennur Road,2 BHK,1232.0,2.0,89.0,2,7224
Thanisandra,3 BHK,1732.0,3.0,85.73,3,4949
Yelahanka,3 BHK,1614.0,2.0,100.0,3,6195
Magadi Road,2 BHK,600.0,1.0,22.0,2,3666
Thigalarapalya,3 BHK,2072.0,4.0,155.0,3,7480
Electronic City,2 BHK,919.0,2.0,34.0,2,3699
other,4 Bedroom,2100.0,4.0,90.0,4,4285
Shampura,4 BHK,672.0,3.0,54.0,4,8035
other,8 BHK,800.0,6.0,110.0,8,13750
Marsur,3 Bedroom,1200.0,3.0,150.0,3,12500
other,4 BHK,2849.0,5.0,250.0,4,8775
other,2 BHK,1000.0,2.0,42.0,2,4200
Whitefield,2 BHK,1216.0,2.0,75.0,2,6167
Lingadheeranahalli,4 BHK,2245.0,4.0,154.0,4,6859
Bannerghatta Road,3 BHK,1476.0,3.0,125.0,3,8468
Ambedkar Nagar,3 BHK,1852.0,3.0,130.0,3,7019
other,2 BHK,1225.0,2.0,47.0,2,3836
Panathur,3 BHK,1597.0,3.0,67.12,3,4202
Jakkur,3 BHK,2423.0,4.0,160.0,3,6603
Marathahalli,2 BHK,1152.0,2.0,90.0,2,7812
Harlur,3 BHK,1762.0,3.0,133.0,3,7548
Kammasandra,3 Bedroom,1400.0,3.0,92.0,3,6571
Rachenahalli,3 BHK,1530.0,2.0,74.4,3,4862
other,6 Bedroom,2400.0,6.0,400.0,6,16666
Raja Rajeshwari Nagar,2 BHK,1185.0,2.0,40.17,2,3389
Yelahanka,3 BHK,1890.0,3.0,108.0,3,5714
Varthur Road,3 BHK,1247.0,2.0,58.0,3,4651
1st Block Jayanagar,8 Bedroom,700.0,4.0,104.0,8,14857
HRBR Layout,6 Bedroom,1000.0,6.0,275.0,6,27500
Bellandur,2 BHK,1201.0,2.0,59.99,2,4995
other,2 BHK,830.0,2.0,40.0,2,4819
Electronic City Phase II,2 BHK,1165.0,2.0,30.29,2,2600
Bommanahalli,2 BHK,1100.0,1.0,55.0,2,5000
9th Phase JP Nagar,3 BHK,1890.0,2.0,93.0,3,4920
other,3 BHK,1685.0,3.0,60.0,3,3560
Sarakki Nagar,4 BHK,3131.0,4.0,349.0,4,11146
Battarahalli,2 BHK,1165.0,2.0,43.0,2,3690
other,2 BHK,827.0,2.0,45.0,2,5441
other,3 Bedroom,396.0,2.0,48.0,3,12121
Uttarahalli,4 Bedroom,1200.0,4.0,179.0,4,14916
Electronics City Phase 1,1 BHK,635.0,1.0,26.0,1,4094
Whitefield,4 Bedroom,4000.0,4.0,380.0,4,9500
other,2 BHK,1050.0,2.0,50.0,2,4761
other,5 Bedroom,600.0,4.0,76.0,5,12666
Jakkur,3 BHK,1819.18,3.0,168.0,3,9234
Kanakapura,3 Bedroom,1500.0,3.0,125.0,3,8333
Yelenahalli,3 BHK,1500.0,2.0,57.0,3,3800
Benson Town,3 BHK,2850.0,4.0,470.0,3,16491
Raja Rajeshwari Nagar,3 BHK,1751.0,3.0,72.0,3,4111
other,4 BHK,4750.0,6.0,1102.0,4,23200
Bommasandra,2 BHK,1020.0,2.0,40.89,2,4008
Thanisandra,2 BHK,1148.0,2.0,43.5,2,3789
Talaghattapura,3 BHK,1856.0,3.0,135.0,3,7273
Basavangudi,6 Bedroom,1754.0,6.0,650.0,6,37058
other,3 Bedroom,4273.0,3.0,1100.0,3,25743
other,2 BHK,1060.0,2.0,49.27,2,4648
Haralur Road,2 BHK,1300.0,2.0,60.0,2,4615
Hormavu,4 Bedroom,3000.0,4.0,100.0,4,3333
Bellandur,2 BHK,1325.0,2.0,65.0,2,4905
other,2 BHK,1261.0,2.0,61.0,2,4837
other,2 BHK,1100.0,2.0,46.0,2,4181
Ramamurthy Nagar,2 BHK,700.0,2.0,33.0,2,4714
7th Phase JP Nagar,3 BHK,1675.0,2.0,135.0,3,8059
other,4 Bedroom,3400.0,4.0,190.0,4,5588
Kodihalli,4 BHK,3260.0,5.0,388.0,4,11901
other,2 BHK,1180.0,2.0,56.0,2,4745
Uttarahalli,2 BHK,1160.0,2.0,45.0,2,3879
Kasavanhalli,4 BHK,1863.0,3.0,105.0,4,5636
Sahakara Nagar,3 BHK,1914.0,3.0,149.0,3,7784
BTM Layout,3 BHK,1458.0,3.0,79.0,3,5418
other,5 Bedroom,1200.0,4.0,115.0,5,9583
Babusapalaya,2 BHK,1105.0,2.0,40.0,2,3619
Hulimavu,3 BHK,3035.0,5.0,220.0,3,7248
Bannerghatta Road,6 Bedroom,1200.0,6.0,160.0,6,13333
Akshaya Nagar,3 BHK,1662.0,3.0,90.0,3,5415
Yelahanka,3 BHK,1556.0,3.0,86.0,3,5526
Jigani,2 BHK,918.0,2.0,50.0,2,5446
Sarjapur  Road,3 BHK,1455.0,2.0,98.0,3,6735
other,2 BHK,1205.0,2.0,70.0,2,5809
BEML Layout,2 BHK,1194.0,2.0,65.0,2,5443
Laggere,2 Bedroom,1200.0,2.0,75.0,2,6250
Bannerghatta Road,3 BHK,1100.0,2.0,52.0,3,4727
Sarjapur  Road,3 BHK,1405.0,3.0,48.87,3,3478
Subramanyapura,2 BHK,929.0,2.0,56.0,2,6027
Frazer Town,4 BHK,3436.0,5.0,341.0,4,9924
7th Phase JP Nagar,2 BHK,1035.0,2.0,39.33,2,3800
8th Phase JP Nagar,4 BHK,2100.0,4.0,92.0,4,4380
Channasandra,2 BHK,1065.0,2.0,40.47,2,3800
Bannerghatta Road,3 BHK,1532.5,3.0,84.29,3,5500
Kengeri,6 Bedroom,1500.0,6.0,150.0,6,10000
other,5 Bedroom,1200.0,5.0,99.0,5,8250
Hormavu,2 BHK,1093.0,2.0,32.78,2,2999
Anandapura,2 BHK,1250.0,2.0,45.0,2,3600
Marathahalli,3 BHK,1680.0,3.0,105.0,3,6250
Nagarbhavi,2 BHK,1146.0,2.0,50.0,2,4363
other,2 BHK,1250.0,2.0,58.0,2,4640
Ramagondanahalli,3 BHK,1451.0,3.0,60.44,3,4165
other,3 BHK,1424.0,2.0,55.0,3,3862
Gottigere,3 BHK,1435.0,2.0,62.0,3,4320
Devanahalli,1 BHK,658.0,1.0,26.91,1,4089
Bommanahalli,3 BHK,1300.0,3.0,45.0,3,3461
Whitefield,2 BHK,1242.0,2.0,65.0,2,5233
Marathahalli,3 BHK,1910.0,3.0,119.0,3,6230
Yelahanka,3 BHK,1610.0,4.0,85.0,3,5279
Hulimavu,2 BHK,1127.0,2.0,65.0,2,5767
Yelahanka,3 BHK,1075.0,2.0,42.0,3,3906
Mysore Road,2 BHK,1155.0,2.0,51.0,2,4415
other,5 Bedroom,1280.0,3.0,250.0,5,19531
other,3 BHK,1250.0,3.0,55.0,3,4400
other,3 Bedroom,2200.0,4.0,160.0,3,7272
Billekahalli,3 Bedroom,2400.0,3.0,150.0,3,6250
Chandapura,2 BHK,876.0,2.0,29.5,2,3367
Nagarbhavi,3 BHK,1850.0,2.0,89.0,3,4810
Kadugodi,2 BHK,1314.0,2.0,83.0,2,6316
Rachenahalli,1 BHK,690.0,1.0,39.8,1,5768
other,2 BHK,1230.0,2.0,55.0,2,4471
Bommasandra,2 BHK,920.0,2.0,37.2,2,4043
Sarjapur  Road,3 BHK,1850.0,3.0,140.0,3,7567
Electronic City,2 BHK,1073.0,2.0,60.0,2,5591
other,3 BHK,1540.0,3.0,76.0,3,4935
Kanakpura Road,3 BHK,1550.0,3.0,63.5,3,4096
Kalyan nagar,2 BHK,1198.0,2.0,65.0,2,5425
HBR Layout,3 BHK,1656.0,3.0,90.0,3,5434
other,3 BHK,1900.0,3.0,155.0,3,8157
Ramagondanahalli,3 BHK,1910.0,3.0,142.0,3,7434
Mysore Road,2 BHK,1003.0,2.0,43.0,2,4287
Whitefield,3 BHK,1495.0,2.0,67.0,3,4481
Thanisandra,3 BHK,2172.0,3.0,76.02,3,3500
Sarakki Nagar,4 Bedroom,2400.0,4.0,358.0,4,14916
other,2 Bedroom,1200.0,1.0,95.0,2,7916
Whitefield,3 BHK,1548.0,3.0,65.0,3,4198
other,7 Bedroom,5400.0,5.0,972.0,7,18000
Whitefield,4 BHK,2856.0,5.0,154.5,4,5409
Hosur Road,3 BHK,1689.0,3.0,103.0,3,6098
Electronic City Phase II,2 BHK,1031.0,2.0,53.0,2,5140
Bhoganhalli,4 BHK,2119.0,4.0,111.0,4,5238
Whitefield,2 Bedroom,1200.0,2.0,46.13,2,3844
other,2 BHK,1060.0,2.0,37.0,2,3490
other,2 BHK,1100.0,2.0,61.0,2,5545
other,4 Bedroom,1350.0,4.0,225.0,4,16666
other,5 Bedroom,1050.0,4.0,80.0,5,7619
Hennur Road,3 BHK,1250.0,2.0,77.88,3,6230
other,2 BHK,1100.0,2.0,110.0,2,10000
Whitefield,2 Bedroom,1200.0,2.0,46.13,2,3844
Seegehalli,3 BHK,1420.0,2.0,46.0,3,3239
Electronic City,3 BHK,1500.0,2.0,78.0,3,5200
other,3 BHK,1942.0,3.0,155.0,3,7981
Haralur Road,3 BHK,1985.0,3.0,130.0,3,6549
Sarjapur  Road,1 BHK,950.0,1.0,39.9,1,4200
other,3 Bedroom,1200.0,4.0,80.0,3,6666
Green Glen Layout,3 BHK,1752.0,3.0,105.0,3,5993
Abbigere,4 Bedroom,1200.0,3.0,120.0,4,10000
Mahalakshmi Layout,5 Bedroom,1200.0,5.0,200.0,5,16666
Old Madras Road,3 BHK,1480.0,2.0,84.0,3,5675
other,3 Bedroom,600.0,4.0,81.0,3,13500
Sarjapur  Road,3 BHK,1826.0,3.0,130.0,3,7119
8th Phase JP Nagar,2 BHK,1035.0,2.0,39.33,2,3800
Sarjapur  Road,4 Bedroom,2600.0,4.0,98.0,4,3769
Kasavanhalli,3 BHK,1667.0,3.0,92.0,3,5518
Thanisandra,2 BHK,934.0,2.0,55.0,2,5888
Mahadevpura,3 BHK,1620.0,2.0,76.0,3,4691
Electronic City,2 BHK,1070.0,2.0,47.0,2,4392
Hosur Road,5 Bedroom,3600.0,5.0,180.0,5,5000
Uttarahalli,3 BHK,1280.0,2.0,60.0,3,4687
other,2 BHK,1400.0,2.0,60.0,2,4285
Vishveshwarya Layout,7 Bedroom,1200.0,8.0,180.0,7,15000
other,2 BHK,1330.0,2.0,64.0,2,4812
Balagere,2 BHK,1012.0,2.0,54.59,2,5394
Electronic City Phase II,3 BHK,1611.0,2.0,48.33,3,3000
Whitefield,3 BHK,1405.0,3.0,58.0,3,4128
other,3 BHK,1515.0,3.0,90.0,3,5940
Thigalarapalya,3 BHK,1830.0,4.0,150.0,3,8196
Kaggadasapura,3 BHK,1710.0,3.0,60.0,3,3508
BTM Layout,2 BHK,935.0,2.0,51.5,2,5508
Jakkur,3 BHK,2197.0,3.0,136.0,3,6190
TC Palaya,4 Bedroom,600.0,4.0,65.0,4,10833
Yeshwanthpur,3 BHK,1678.0,3.0,92.13,3,5490
Brookefield,3 BHK,1410.0,2.0,80.0,3,5673
Jakkur,2 BHK,1125.0,2.0,65.0,2,5777
Magadi Road,3 BHK,1191.0,2.0,55.38,3,4649
Thubarahalli,2 BHK,1200.0,2.0,79.0,2,6583
CV Raman Nagar,3 BHK,1525.0,2.0,67.0,3,4393
other,3 BHK,1464.0,3.0,56.0,3,3825
Seegehalli,2 BHK,920.0,2.0,27.0,2,2934
Yelahanka,1 BHK,651.0,1.0,17.5,1,2688
Chikkabanavar,2 BHK,600.0,1.0,25.0,2,4166
Chandapura,1 BHK,645.0,1.0,16.45,1,2550
Thanisandra,2 BHK,1185.0,2.0,44.0,2,3713
Electronics City Phase 1,2 BHK,1190.0,2.0,68.0,2,5714
Whitefield,4 BHK,2856.0,5.0,158.5,4,5549
Whitefield,2 BHK,1355.0,2.0,73.0,2,5387
KR Puram,2 Bedroom,1200.0,1.0,65.0,2,5416
Kundalahalli,4 Bedroom,7000.0,4.0,550.0,4,7857
Ramamurthy Nagar,6 Bedroom,1200.0,4.0,110.0,6,9166
Yelahanka New Town,1 BHK,500.0,2.0,24.0,1,4800
Bommasandra Industrial Area,3 BHK,1250.0,3.0,50.0,3,4000
Uttarahalli,3 BHK,1480.0,2.0,58.0,3,3918
Ambalipura,4 BHK,3300.0,4.0,329.0,4,9969
other,2 BHK,1206.0,2.0,53.0,2,4394
Rajaji Nagar,2 BHK,1357.0,2.0,130.0,2,9579
Hosakerehalli,2 BHK,925.0,2.0,46.5,2,5027
8th Phase JP Nagar,2 BHK,909.0,2.0,40.9,2,4499
Seegehalli,3 Bedroom,3000.0,5.0,335.0,3,11166
Brookefield,2 BHK,1100.0,2.0,55.0,2,5000
Malleshwaram,4 Bedroom,3000.0,5.0,815.0,4,27166
Vidyaranyapura,2 BHK,1100.0,2.0,54.0,2,4909
Subramanyapura,2 BHK,975.0,2.0,65.0,2,6666
Raja Rajeshwari Nagar,2 BHK,1306.0,2.0,44.17,2,3382
other,4 Bedroom,1200.0,4.0,95.0,4,7916
Kalena Agrahara,1 BHK,610.0,1.0,39.0,1,6393
Yelahanka,3 BHK,1484.0,2.0,56.4,3,3800
Ramamurthy Nagar,2 BHK,1125.0,2.0,47.0,2,4177
Bhoganhalli,2 BHK,1447.0,2.0,75.97,2,5250
Whitefield,4 BHK,2882.0,5.0,158.0,4,5482
Vidyaranyapura,4 Bedroom,750.0,4.0,88.0,4,11733
Kundalahalli,3 BHK,2075.0,3.0,115.0,3,5542
other,2 BHK,1250.0,2.0,58.0,2,4640
Horamavu Banaswadi,2 BHK,1272.0,2.0,51.5,2,4048
Banashankari Stage II,4 Bedroom,1050.0,3.0,125.0,4,11904
other,2 BHK,625.0,2.0,24.0,2,3840
1st Phase JP Nagar,1 BHK,840.0,2.0,50.0,1,5952
Hennur Road,4 BHK,2502.0,4.0,180.0,4,7194
Rayasandra,1 BHK,583.0,1.0,26.9,1,4614
other,2 BHK,900.0,2.0,27.0,2,3000
Sarjapur  Road,2 BHK,1115.0,2.0,50.0,2,4484
other,2 BHK,1464.0,2.0,56.0,2,3825
Kanakapura,3 Bedroom,2200.0,3.0,121.0,3,5500
Whitefield,1 BHK,750.0,1.0,51.0,1,6800
Marathahalli,3 BHK,1600.0,3.0,58.0,3,3625
Chikkalasandra,3 BHK,1365.0,2.0,49.0,3,3589
other,3 BHK,1300.0,3.0,64.0,3,4923
other,1 Bedroom,1200.0,1.0,69.0,1,5750
Hormavu,2 BHK,1260.0,2.0,35.0,2,2777
Electronics City Phase 1,2 BHK,1190.0,2.0,33.92,2,2850
Hosakerehalli,3 BHK,2480.0,4.0,260.0,3,10483
Vishwapriya Layout,3 BHK,1250.0,3.0,38.0,3,3040
Kaggadasapura,3 BHK,1500.0,3.0,75.0,3,5000
Banashankari Stage V,3 BHK,1510.0,3.0,47.57,3,3150
7th Phase JP Nagar,2 BHK,1187.0,2.0,59.0,2,4970
Kundalahalli,2 BHK,1120.0,2.0,53.76,2,4800
Whitefield,2 BHK,1115.0,2.0,34.555,2,3099
other,3 BHK,1717.0,3.0,99.0,3,5765
other,3 BHK,2099.0,3.0,95.0,3,4525
Kasturi Nagar,2 BHK,1100.0,2.0,58.0,2,5272
other,1 Bedroom,1200.0,1.0,48.0,1,4000
Bannerghatta Road,3 BHK,1465.0,2.0,45.87,3,3131
Kothanur,1 Bedroom,1500.0,1.0,75.0,1,5000
other,5 Bedroom,4050.0,4.0,600.0,5,14814
Haralur Road,3 BHK,1255.0,2.0,90.0,3,7171
5th Phase JP Nagar,2 BHK,1220.0,2.0,88.0,2,7213
Electronic City Phase II,2 BHK,1065.0,2.0,30.88,2,2899
Sultan Palaya,3 BHK,1765.0,3.0,80.0,3,4532
Bellandur,2 BHK,1281.0,2.0,87.0,2,6791
Electronic City,2 BHK,1210.0,2.0,25.0,2,2066
Kaggadasapura,2 BHK,1215.0,2.0,43.0,2,3539
Ananth Nagar,4 Bedroom,1200.0,3.0,87.0,4,7250
Mysore Road,2 BHK,1237.0,2.0,73.0,2,5901
Lakshminarayana Pura,2 BHK,1300.0,2.0,70.0,2,5384
Whitefield,2 BHK,1105.0,2.0,39.99,2,3619
Chandapura,3 BHK,1505.0,2.0,42.0,3,2790
Hebbal Kempapura,4 BHK,3900.0,5.0,310.0,4,7948
Banashankari,2 BHK,1186.0,2.0,65.0,2,5480
other,4 Bedroom,900.0,3.0,51.0,4,5666
Kengeri,2 BHK,750.0,2.0,35.5,2,4733
Varthur,2 BHK,1083.0,2.0,28.0,2,2585
other,3 BHK,1430.0,2.0,78.65,3,5500
Thanisandra,2 BHK,1056.0,2.0,68.0,2,6439
other,3 BHK,2108.0,3.0,85.0,3,4032
Banashankari,3 BHK,1650.0,3.0,101.0,3,6121
Electronic City Phase II,3 BHK,1220.0,3.0,36.6,3,3000
Rayasandra,2 BHK,1198.0,2.0,54.0,2,4507
other,2 BHK,1190.0,2.0,70.0,2,5882
other,2 BHK,900.0,2.0,41.0,2,4555
other,4 BHK,2690.0,4.0,185.0,4,6877
Kanakapura,2 BHK,1340.0,2.0,65.0,2,4850
Bommenahalli,4 Bedroom,2940.0,3.0,2250.0,4,76530
Giri Nagar,6 Bedroom,2400.0,6.0,400.0,6,16666
Yelahanka,2 BHK,1260.0,2.0,59.2,2,4698
Talaghattapura,3 BHK,2254.0,3.0,153.0,3,6787
Old Madras Road,3 BHK,2990.0,5.0,210.0,3,7023
other,4 Bedroom,1800.0,5.0,250.0,4,13888
JP Nagar,2 BHK,820.0,2.0,45.0,2,5487
Electronic City,2 BHK,1296.0,2.0,65.0,2,5015
Uttarahalli,2 BHK,1190.0,2.0,53.55,2,4500
Whitefield,2 BHK,1173.0,2.0,77.0,2,6564
Sarjapura - Attibele Road,3 Bedroom,1800.0,3.0,90.0,3,5000
Yelahanka,2 BHK,1102.0,2.0,52.0,2,4718
Sarjapur,2 BHK,913.0,2.0,36.0,2,3943
Hormavu,3 BHK,1176.0,2.0,35.27,3,2999
Prithvi Layout,3 BHK,1620.0,3.0,65.0,3,4012
Billekahalli,3 BHK,2968.0,3.0,225.0,3,7580
Yelahanka,2 BHK,1519.0,2.0,65.0,2,4279
Hosur Road,2 BHK,1345.0,2.0,106.0,2,7881
other,3 BHK,1594.0,3.0,99.0,3,6210
other,2 BHK,1250.0,2.0,68.0,2,5440
Haralur Road,2 BHK,1056.0,2.0,60.0,2,5681
Whitefield,3 Bedroom,1200.0,3.0,68.4,3,5700
other,2 Bedroom,1290.0,1.0,77.5,2,6007
other,2 BHK,1150.0,2.0,48.0,2,4173
other,2 BHK,1464.0,2.0,145.0,2,9904
Anekal,4 Bedroom,1200.0,2.0,36.0,4,3000
Gollarapalya Hosahalli,2 BHK,861.0,2.0,36.5,2,4239
Kundalahalli,2 BHK,1047.0,2.0,82.62,2,7891
other,2 Bedroom,1200.0,2.0,75.0,2,6250
other,10 Bedroom,1200.0,10.0,190.0,10,15833
other,2 BHK,1065.0,2.0,45.0,2,4225
Rajaji Nagar,3 BHK,2300.0,3.0,369.0,3,16043
Ramamurthy Nagar,2 BHK,1000.0,2.0,62.0,2,6200
Lingadheeranahalli,3 BHK,1682.0,3.0,113.0,3,6718
Harlur,3 BHK,1756.0,3.0,131.0,3,7460
other,5 Bedroom,4000.0,4.0,1000.0,5,25000
other,4 BHK,1050.0,4.0,90.0,4,8571
Vittasandra,2 BHK,1238.0,2.0,67.32,2,5437
Dasanapura,2 BHK,814.0,2.0,42.0,2,5159
Dodda Nekkundi,2 BHK,1145.0,2.0,46.5,2,4061
Varthur,2 BHK,965.0,2.0,34.0,2,3523
other,2 BHK,750.0,2.0,36.0,2,4800
Banashankari,3 BHK,1300.0,2.0,55.0,3,4230
1st Phase JP Nagar,3 BHK,1590.0,3.0,131.0,3,8238
Begur Road,1 BHK,644.0,1.0,40.0,1,6211
Whitefield,2 BHK,982.0,2.0,37.0,2,3767
Indira Nagar,2 BHK,1260.0,2.0,100.0,2,7936
Rayasandra,3 BHK,1555.0,3.0,75.17,3,4834
other,4 Bedroom,2500.0,4.0,450.0,4,18000
Kaggadasapura,2 BHK,1225.0,2.0,48.0,2,3918
Electronics City Phase 1,2 BHK,1085.0,2.0,46.0,2,4239
Rajaji Nagar,3 BHK,1640.0,3.0,220.0,3,13414
Thanisandra,3 BHK,1595.0,3.0,110.0,3,6896
Raja Rajeshwari Nagar,2 BHK,1196.0,2.0,40.44,2,3381
Yelahanka,3 BHK,1859.0,3.0,108.0,3,5809
9th Phase JP Nagar,2 BHK,1079.0,2.0,60.0,2,5560
Shivaji Nagar,2 BHK,550.0,1.0,27.0,2,4909
Whitefield,2 BHK,1118.0,2.0,42.0,2,3756
CV Raman Nagar,2 BHK,1198.0,2.0,60.0,2,5008
Thigalarapalya,3 BHK,1830.0,4.0,135.0,3,7377
Banashankari Stage VI,2 BHK,1160.0,2.0,59.0,2,5086
other,3 BHK,1485.0,2.0,156.0,3,10505
other,3 BHK,2070.0,3.0,135.0,3,6521
Basaveshwara Nagar,6 Bedroom,750.0,4.0,150.0,6,20000
Kannamangala,2 BHK,957.0,2.0,52.5,2,5485
Basaveshwara Nagar,5 Bedroom,2400.0,4.0,310.0,5,12916
Bannerghatta Road,2 BHK,1268.0,2.0,69.0,2,5441
Bannerghatta Road,3 BHK,1450.0,2.0,87.0,3,6000
other,3 BHK,1600.0,2.0,75.0,3,4687
Uttarahalli,3 BHK,1330.0,2.0,57.0,3,4285
Jakkur,3 BHK,1355.0,2.0,65.0,3,4797
Horamavu Banaswadi,2 BHK,1272.0,2.0,51.5,2,4048
other,2 BHK,1100.0,2.0,38.49,2,3499
Sarjapur  Road,3 BHK,1220.0,3.0,58.0,3,4754
Chikkalasandra,2 BHK,1075.0,2.0,46.76,2,4349
other,2 BHK,800.0,2.0,55.0,2,6875
Vasanthapura,2 BHK,995.0,2.0,34.82,2,3499
other,2 BHK,900.0,1.0,44.0,2,4888
Haralur Road,2 BHK,1140.0,2.0,42.0,2,3684
other,3 BHK,1480.0,2.0,125.0,3,8445
Padmanabhanagar,3 BHK,1710.0,2.0,90.63,3,5300
Talaghattapura,3 BHK,2106.0,3.0,148.0,3,7027
Yeshwanthpur,9 Bedroom,2400.0,6.0,270.0,9,11250
other,3 Bedroom,3800.0,2.0,130.0,3,3421
other,3 Bedroom,800.0,3.0,120.0,3,15000
Marathahalli,2 BHK,1602.0,2.0,104.0,2,6491
Sarjapur  Road,3 BHK,1660.0,2.0,116.0,3,6987
Electronic City Phase II,3 BHK,1549.0,3.0,80.0,3,5164
Marathahalli,1 BHK,700.0,1.0,38.0,1,5428
Balagere,2 BHK,1012.0,2.0,64.0,2,6324
Marathahalli,3 BHK,1469.0,3.0,89.0,3,6058
Whitefield,2 BHK,1216.0,2.0,69.0,2,5674
Electronic City Phase II,2 BHK,1065.0,2.0,56.13,2,5270
Thanisandra,3 BHK,1411.0,3.0,93.15,3,6601
Pai Layout,2 BHK,1400.0,2.0,57.5,2,4107
Hoskote,3 BHK,1250.0,3.0,29.0,3,2320
Haralur Road,2 BHK,1245.0,2.0,62.0,2,4979
Choodasandra,2 BHK,1115.0,2.0,50.0,2,4484
Vidyaranyapura,2 BHK,1000.0,2.0,35.0,2,3500
Panathur,3 BHK,1580.0,3.0,71.89,3,4550
Subramanyapura,3 BHK,1800.0,3.0,90.0,3,5000
other,2 BHK,1500.0,2.0,130.0,2,8666
Chandapura,1 BHK,645.0,1.0,16.45,1,2550
Kothannur,3 BHK,1275.0,3.0,44.0,3,3450
Kasavanhalli,3 BHK,1494.0,3.0,82.0,3,5488
Bhoganhalli,3 BHK,1949.0,3.0,129.0,3,6618
BTM Layout,2 BHK,1020.0,2.0,46.5,2,4558
Mysore Road,2 BHK,883.0,2.0,45.0,2,5096
Old Airport Road,4 BHK,3200.0,4.0,280.0,4,8750
Uttarahalli,6 Bedroom,1680.0,6.0,125.0,6,7440
Mysore Road,12 Bedroom,2232.0,6.0,300.0,12,13440
other,3 BHK,2010.0,3.0,125.0,3,6218
Yeshwanthpur,2 BHK,1240.0,2.0,80.0,2,6451
6th Phase JP Nagar,2 BHK,1213.0,2.0,63.0,2,5193
Hegde Nagar,3 BHK,1835.0,3.0,88.0,3,4795
Hoskote,9 Bedroom,1800.0,10.0,185.0,9,10277
Bannerghatta Road,4 BHK,2932.0,5.0,195.0,4,6650
Whitefield,4 Bedroom,3425.0,5.0,250.0,4,7299
Gottigere,2 BHK,945.0,2.0,45.0,2,4761
other,3 BHK,1504.0,2.0,100.0,3,6648
Lakshminarayana Pura,3 BHK,1569.0,3.0,150.0,3,9560
Harlur,3 BHK,1755.0,3.0,130.0,3,7407
Vijayanagar,3 BHK,1760.0,3.0,140.0,3,7954
Hoodi,2 BHK,1240.0,2.0,43.94,2,3543
Electronics City Phase 1,3 BHK,1450.0,3.0,72.54,3,5002
5th Phase JP Nagar,2 BHK,1000.0,2.0,48.0,2,4800
Shivaji Nagar,1 Bedroom,3820.0,1.0,306.0,1,8010
other,5 BHK,2000.0,4.0,145.0,5,7250
Choodasandra,2 BHK,1215.0,2.0,59.0,2,4855
Yelachenahalli,3 BHK,1330.0,3.0,73.5,3,5526
Raja Rajeshwari Nagar,2 BHK,1178.0,2.0,75.0,2,6366
Hulimavu,4 Bedroom,1500.0,4.0,192.0,4,12800
other,2 BHK,2040.0,2.0,59.0,2,2892
Bellandur,3 BHK,2000.0,3.0,85.0,3,4250
other,3 BHK,1602.0,2.0,165.0,3,10299
other,4 Bedroom,1200.0,4.0,130.0,4,10833
Parappana Agrahara,2 BHK,1194.0,2.0,46.0,2,3852
other,3 BHK,1800.0,3.0,220.0,3,12222
Dasarahalli,3 Bedroom,2400.0,2.0,152.0,3,6333
Sarjapur  Road,3 BHK,1700.0,3.0,108.0,3,6352
Jigani,3 BHK,1200.0,3.0,65.0,3,5416
other,3 Bedroom,1500.0,3.0,72.0,3,4800
Karuna Nagar,3 Bedroom,1240.0,3.0,165.0,3,13306
Whitefield,3 Bedroom,3000.0,3.0,400.0,3,13333
Kanakpura Road,3 BHK,1737.0,3.0,85.0,3,4893
TC Palaya,2 Bedroom,900.0,2.0,55.0,2,6111
other,3 BHK,1280.0,2.0,48.0,3,3750
CV Raman Nagar,2 BHK,1560.0,2.0,85.0,2,5448
Kambipura,2 BHK,883.0,2.0,44.0,2,4983
Kengeri Satellite Town,2 BHK,750.0,2.0,33.0,2,4400
Brookefield,4 BHK,2400.0,3.0,140.0,4,5833
other,2 BHK,900.0,2.0,45.0,2,5000
Hormavu,2 BHK,1030.0,2.0,49.35,2,4791
7th Phase JP Nagar,3 BHK,1850.0,3.0,155.0,3,8378
Hennur Road,3 BHK,1831.0,3.0,110.0,3,6007
other,2 BHK,1095.0,2.0,38.33,2,3500
Electronic City Phase II,3 BHK,1320.0,2.0,38.13,3,2888
Sarjapur  Road,3 BHK,1272.5,2.0,40.72,3,3200
7th Phase JP Nagar,2 BHK,1000.0,2.0,61.0,2,6100
Budigere,3 BHK,1820.0,3.0,85.4,3,4692
Bisuvanahalli,3 BHK,1075.0,2.0,41.0,3,3813
TC Palaya,3 Bedroom,1500.0,3.0,83.0,3,5533
Electronics City Phase 1,1 RK,360.0,1.0,16.9,1,4694
Kadugodi,1 BHK,925.0,1.0,40.7,1,4400
Electronic City,2 BHK,1140.0,2.0,70.93,2,6221
other,2 BHK,1145.0,2.0,58.4,2,5100
other,3 BHK,1862.0,3.0,120.0,3,6444
Devarachikkanahalli,2 BHK,947.0,2.0,43.0,2,4540
other,3 Bedroom,700.0,3.0,195.0,3,27857
Kalena Agrahara,2 BHK,1120.0,2.0,40.0,2,3571
other,3 BHK,2700.0,3.0,200.0,3,7407
Kaggadasapura,3 BHK,1805.0,2.0,75.0,3,4155
Basavangudi,2 BHK,1100.0,2.0,93.0,2,8454
Bommenahalli,4 Bedroom,1355.0,3.0,135.0,4,9963
other,3 BHK,1677.0,3.0,98.0,3,5843
other,8 Bedroom,2500.0,8.0,95.0,8,3800
Uttarahalli,3 BHK,1560.0,3.0,74.0,3,4743
Arekere,3 BHK,2060.0,3.0,150.0,3,7281
Koramangala,3 BHK,1716.0,2.0,115.0,3,6701
Harlur,3 BHK,1864.0,3.0,140.0,3,7510
Electronics City Phase 1,1 BHK,638.0,1.0,32.0,1,5015
other,9 Bedroom,1200.0,7.0,95.0,9,7916
Kodigehalli,8 BHK,1150.0,9.0,170.0,8,14782
Chandapura,2 BHK,937.0,2.0,30.0,2,3201
other,3 BHK,1490.0,2.0,115.0,3,7718
Frazer Town,4 BHK,4856.0,5.0,410.0,4,8443
Whitefield,4 Bedroom,4003.0,6.0,525.0,4,13115
Whitefield,2 BHK,1100.0,2.0,37.0,2,3363
Thigalarapalya,2 BHK,1000.0,2.0,48.53,2,4853
Horamavu Agara,2 BHK,1060.0,2.0,37.0,2,3490
Kundalahalli,2 BHK,1047.0,2.0,91.0,2,8691
Basaveshwara Nagar,2 BHK,1200.0,2.0,80.0,2,6666
Akshaya Nagar,3 BHK,1420.0,2.0,75.0,3,5281
Sarjapur  Road,3 BHK,1539.0,3.0,85.0,3,5523
other,3 BHK,1550.0,3.0,59.0,3,3806
Kasavanhalli,3 BHK,1555.0,3.0,82.0,3,5273
other,7 Bedroom,600.0,6.0,138.0,7,23000
Seegehalli,3 BHK,1553.0,3.0,68.0,3,4378
Electronic City Phase II,2 BHK,920.0,2.0,26.0,2,2826
other,3 BHK,1740.0,3.0,120.0,3,6896
OMBR Layout,3 BHK,1855.0,3.0,145.0,3,7816
Kundalahalli,3 BHK,1724.0,3.0,146.0,3,8468
other,2 BHK,1020.0,2.0,60.0,2,5882
Banaswadi,6 Bedroom,1000.0,4.0,150.0,6,15000
Uttarahalli,3 BHK,1250.0,2.0,50.0,3,4000
Sanjay nagar,3 BHK,1450.0,3.0,75.0,3,5172
Thubarahalli,2 BHK,1200.0,2.0,80.0,2,6666
Haralur Road,2 BHK,1140.0,2.0,58.0,2,5087
Hoskote,5 Bedroom,900.0,5.0,110.0,5,12222
other,1 BHK,550.0,1.0,16.0,1,2909
Bannerghatta Road,3 BHK,1450.0,3.0,44.0,3,3034
Kanakpura Road,2 BHK,700.0,2.0,35.07,2,5010
Whitefield,3 Bedroom,1200.0,3.0,56.0,3,4666
Yelahanka,2 BHK,1050.0,2.0,45.0,2,4285
Somasundara Palya,3 BHK,1600.0,3.0,69.0,3,4312
Bannerghatta Road,3 BHK,1625.0,2.0,75.0,3,4615
other,2 BHK,1492.0,2.0,65.0,2,4356
Whitefield,4 BHK,2856.0,5.0,157.5,4,5514
Sarjapur  Road,3 BHK,1489.0,2.0,82.0,3,5507
Kanakapura,3 BHK,1560.0,2.0,62.38,3,3998
Haralur Road,2 BHK,1140.0,2.0,42.0,2,3684
other,3 BHK,1431.0,3.0,65.0,3,4542
Lingadheeranahalli,3 BHK,1682.0,3.0,114.0,3,6777
Panathur,2 BHK,1198.0,2.0,86.26,2,7200
other,3 Bedroom,3664.0,3.0,325.0,3,8870
Sarjapur  Road,3 BHK,1550.0,2.0,69.75,3,4500
Kodichikkanahalli,2 BHK,1026.0,2.0,45.0,2,4385
Hennur,3 BHK,1260.0,2.0,52.0,3,4126
other,2 Bedroom,690.0,2.0,42.0,2,6086
Sarjapur  Road,1 BHK,539.0,1.0,45.0,1,8348
Sarakki Nagar,4 Bedroom,1200.0,4.0,85.0,4,7083
Attibele,3 Bedroom,2400.0,3.0,120.0,3,5000
Thubarahalli,2 BHK,1200.0,2.0,79.55,2,6629
Kundalahalli,4 Bedroom,2500.0,5.0,350.0,4,14000
Sarjapur  Road,3 BHK,1186.0,2.0,40.0,3,3372
Thanisandra,2 BHK,1039.0,2.0,39.5,2,3801
other,4 Bedroom,1500.0,5.0,233.0,4,15533
Gottigere,5 Bedroom,1500.0,4.0,100.0,5,6666
Malleshwaram,3 BHK,2520.0,3.0,150.0,3,5952
Brookefield,2 BHK,1225.0,2.0,66.5,2,5428
Bommanahalli,2 BHK,1350.0,2.0,72.0,2,5333
other,2 BHK,1030.0,2.0,300.0,2,29126
Whitefield,3 BHK,1550.0,2.0,49.6,3,3200
Hoodi,3 BHK,1490.0,3.0,79.0,3,5302
other,3 BHK,1520.0,3.0,111.0,3,7302
Whitefield,4 Bedroom,4100.0,4.0,550.0,4,13414
other,3 BHK,1391.0,2.0,64.5,3,4636
Rachenahalli,2 BHK,1200.0,2.0,50.0,2,4166
other,4 Bedroom,1200.0,4.0,86.0,4,7166
other,5 Bedroom,2500.0,6.0,500.0,5,20000
Yelahanka,2 BHK,1322.0,2.0,80.0,2,6051
Whitefield,3 BHK,2173.0,3.0,128.0,3,5890
Sarjapur  Road,2 BHK,1200.0,2.0,75.0,2,6250
Thubarahalli,2 BHK,1200.0,2.0,47.0,2,3916
Mysore Road,2 BHK,1170.0,2.0,65.0,2,5555
NRI Layout,2 BHK,1125.0,2.0,45.56,2,4049
Devanahalli,1 BHK,698.5,1.0,28.57,1,4090
other,2 BHK,1250.0,2.0,90.0,2,7200
Chandapura,2 BHK,1025.0,2.0,27.68,2,2700
Yelahanka,3 BHK,1075.0,2.0,36.0,3,3348
other,3 BHK,1900.0,2.0,190.0,3,10000
Whitefield,3 BHK,1435.0,3.0,45.0,3,3135
Electronics City Phase 1,2 BHK,1175.0,2.0,64.98,2,5530
other,3 BHK,1464.0,3.0,56.0,3,3825
other,2 BHK,900.0,2.0,33.0,2,3666
Electronics City Phase 1,1 BHK,585.0,1.0,21.0,1,3589
Jigani,4 Bedroom,3170.0,5.0,230.0,4,7255
other,3 BHK,1602.0,2.0,170.0,3,10611
Thanisandra,2 BHK,1220.0,2.0,42.0,2,3442
Rajaji Nagar,3 BHK,2500.0,3.0,350.0,3,14000
Sarjapur  Road,3 BHK,2145.0,3.0,180.0,3,8391
Bannerghatta Road,2 BHK,960.0,2.0,58.0,2,6041
NGR Layout,1 BHK,907.0,1.0,38.0,1,4189
Old Madras Road,2 BHK,1210.0,2.0,74.0,2,6115
Kammanahalli,4 Bedroom,900.0,4.0,150.0,4,16666
other,2 BHK,1230.0,2.0,70.0,2,5691
5th Phase JP Nagar,3 BHK,1350.0,2.0,75.0,3,5555
Kenchenahalli,2 BHK,1150.0,2.0,58.0,2,5043
Sarjapur,3 BHK,1404.0,2.0,53.35,3,3799
OMBR Layout,2 BHK,1050.0,2.0,48.0,2,4571
Kothanur,8 Bedroom,1020.0,10.0,155.0,8,15196
Yelahanka,2 BHK,1285.0,2.0,56.0,2,4357
Hormavu,2 BHK,1018.0,2.0,50.8,2,4990
Hoskote,3 Bedroom,1200.0,3.0,57.0,3,4750
Sarjapur  Road,2 BHK,1115.0,2.0,50.0,2,4484
other,3 BHK,1600.0,3.0,95.0,3,5937
9th Phase JP Nagar,2 BHK,1480.0,2.0,80.0,2,5405
Kodichikkanahalli,5 Bedroom,2500.0,7.0,125.0,5,5000
Kaval Byrasandra,1 BHK,480.0,1.0,25.0,1,5208
Akshaya Nagar,3 BHK,1435.0,3.0,53.0,3,3693
other,1 BHK,450.0,1.0,60.0,1,13333
other,1 BHK,600.0,2.0,23.0,1,3833
CV Raman Nagar,3 BHK,1590.0,2.0,85.0,3,5345
Nagavara,2 BHK,936.0,2.0,40.2,2,4294
Banashankari,2 BHK,1181.0,2.0,75.0,2,6350
Kengeri,3 BHK,1741.0,3.0,80.0,3,4595
JP Nagar,3 BHK,1750.0,3.0,123.0,3,7028
Varthur Road,2 BHK,1050.0,2.0,42.7,2,4066
Tumkur Road,2 BHK,1027.0,2.0,60.0,2,5842
other,2 BHK,596.0,2.0,22.0,2,3691
other,4 BHK,3463.0,6.0,310.0,4,8951
Sector 2 HSR Layout,3 BHK,1600.0,3.0,70.0,3,4375
Hebbal,4 BHK,2790.0,5.0,198.0,4,7096
Sahakara Nagar,2 BHK,1000.0,2.0,60.0,2,6000
Varthur,2 BHK,1120.0,2.0,44.24,2,3950
other,2 Bedroom,1200.0,2.0,85.0,2,7083
other,2 BHK,1200.0,2.0,35.0,2,2916
Kasturi Nagar,3 BHK,1570.0,3.0,110.0,3,7006
Koramangala,3 BHK,1500.0,3.0,135.0,3,9000
Frazer Town,2 BHK,1315.0,2.0,140.0,2,10646
Electronic City Phase II,2 BHK,1140.0,2.0,33.84,2,2968
Banashankari,2 BHK,1105.0,2.0,90.0,2,8144
Electronic City,2 BHK,1200.0,2.0,34.65,2,2887
Whitefield,1 BHK,905.0,1.0,62.0,1,6850
HSR Layout,4 Bedroom,2400.0,4.0,350.0,4,14583
Kudlu Gate,2 BHK,1185.0,2.0,50.0,2,4219
Koramangala,2 BHK,1083.0,2.0,100.0,2,9233
Rayasandra,3 BHK,1458.0,3.0,60.0,3,4115
Hormavu,3 BHK,1555.0,3.0,75.0,3,4823
other,2 BHK,1200.0,2.0,45.0,2,3750
Sarjapur  Road,2 BHK,1308.0,2.0,58.86,2,4500
Kudlu,2 BHK,1024.0,2.0,44.0,2,4296
Parappana Agrahara,2 BHK,1194.0,2.0,47.0,2,3936
Kanakapura,3 BHK,1450.0,3.0,75.0,3,5172
Seegehalli,3 BHK,1639.0,3.0,120.0,3,7321
Whitefield,3 BHK,1870.0,3.0,138.0,3,7379
Hennur Road,2 BHK,1232.0,2.0,87.0,2,7061
Kasavanhalli,3 BHK,1719.0,3.0,184.0,3,10703
Chandapura,2 BHK,975.0,2.0,24.86,2,2549
other,3 BHK,2000.0,4.0,130.0,3,6500
Jakkur,2 BHK,1230.0,2.0,79.0,2,6422
other,4 Bedroom,1350.0,4.0,140.0,4,10370
KR Puram,2 BHK,1076.0,2.0,67.0,2,6226
Kumaraswami Layout,7 Bedroom,2400.0,7.0,125.0,7,5208
other,3 BHK,995.0,2.0,34.82,3,3499
Kodigehaali,3 BHK,2120.0,3.0,177.0,3,8349
other,6 Bedroom,2400.0,7.0,460.0,6,19166
other,6 Bedroom,850.0,6.0,78.0,6,9176
Electronic City,2 BHK,1096.0,2.0,40.55,2,3699
Horamavu Agara,2 BHK,1107.83,2.0,41.51,2,3746
Chamrajpet,1 BHK,505.0,1.0,85.0,1,16831
Sarjapur  Road,2 BHK,1320.0,2.0,115.0,2,8712
8th Phase JP Nagar,3 BHK,1431.5,3.0,72.02,3,5031
Whitefield,2 BHK,1320.0,2.0,99.26,2,7519
Whitefield,3 BHK,2901.0,4.0,190.0,3,6549
GM Palaya,3 BHK,1315.0,2.0,65.0,3,4942
other,2 BHK,1000.0,2.0,44.0,2,4400
Hosakerehalli,3 BHK,1596.0,3.0,79.8,3,5000
Bannerghatta Road,3 BHK,1750.0,3.0,130.0,3,7428
Whitefield,4 Bedroom,2403.0,4.0,270.0,4,11235
Old Madras Road,3 BHK,1350.0,3.0,47.25,3,3500
other,2 BHK,1175.0,2.0,79.0,2,6723
Talaghattapura,2 BHK,921.0,2.0,29.47,2,3199
Malleshwaram,4 Bedroom,4000.0,4.0,1100.0,4,27500
Sultan Palaya,3 BHK,1713.0,3.0,125.0,3,7297
other,2 BHK,1200.0,2.0,43.0,2,3583
8th Phase JP Nagar,3 Bedroom,1900.0,4.0,90.0,3,4736
Kudlu Gate,3 BHK,1664.0,3.0,68.0,3,4086
Binny Pete,3 BHK,1795.0,3.0,139.0,3,7743
Kereguddadahalli,1 BHK,600.0,1.0,19.5,1,3250
Hoodi,2 BHK,1240.0,2.0,65.0,2,5241
Hegde Nagar,3 BHK,1965.0,4.0,132.0,3,6717
Vishveshwarya Layout,3 Bedroom,600.0,4.0,99.0,3,16500
Sarjapur,4 Bedroom,4111.0,4.0,370.0,4,9000
Electronics City Phase 1,2 BHK,1200.0,2.0,59.76,2,4980
other,2 BHK,1112.0,2.0,68.0,2,6115
Raja Rajeshwari Nagar,3 BHK,3000.0,3.0,60.0,3,2000
Mahadevpura,2 BHK,1136.0,2.0,64.5,2,5677
Jigani,2 BHK,914.0,2.0,55.0,2,6017
Poorna Pragna Layout,3 BHK,1270.0,2.0,50.79,3,3999
Rachenahalli,2 BHK,1050.0,2.0,47.25,2,4500
Marathahalli,3 BHK,1385.0,2.0,68.5,3,4945
other,2 BHK,990.0,2.0,35.0,2,3535
Yelahanka,2 BHK,845.0,2.0,43.0,2,5088
Whitefield,2 BHK,1485.0,2.0,96.51,2,6498
Kanakapura,3 BHK,1460.0,2.0,43.79,3,2999
Thanisandra,2 BHK,1450.0,2.0,57.0,2,3931
HSR Layout,3 BHK,1360.0,2.0,53.0,3,3897
Singasandra,2 BHK,1375.0,2.0,58.0,2,4218
Uttarahalli,2 BHK,1125.0,2.0,48.0,2,4266
Banashankari,1 BHK,720.0,1.0,60.0,1,8333
7th Phase JP Nagar,3 BHK,1460.0,2.0,100.0,3,6849
other,2 BHK,1107.0,2.0,40.0,2,3613
other,2 BHK,1158.0,2.0,75.0,2,6476
Yelachenahalli,2 BHK,800.0,1.0,30.0,2,3750
Sector 2 HSR Layout,9 Bedroom,600.0,9.0,90.0,9,15000
Bhoganhalli,3 BHK,1760.0,3.0,128.0,3,7272
other,4 Bedroom,3200.0,4.0,200.0,4,6250
HAL 2nd Stage,2 Bedroom,600.0,3.0,145.0,2,24166
TC Palaya,3 Bedroom,1350.0,3.0,60.0,3,4444
Giri Nagar,6 Bedroom,600.0,6.0,100.0,6,16666
other,2 BHK,1125.0,2.0,32.49,2,2888
Frazer Town,3 BHK,1706.0,3.0,154.0,3,9026
BTM Layout,3 BHK,1600.0,3.0,112.0,3,7000
other,2 BHK,1200.0,2.0,90.0,2,7500
Whitefield,3 BHK,1755.0,3.0,101.0,3,5754
Hulimavu,2 BHK,1200.0,2.0,50.0,2,4166
BEML Layout,2 BHK,1200.0,2.0,67.0,2,5583
other,2 BHK,1266.67,2.0,69.0,2,5447
other,3 BHK,1800.0,3.0,90.0,3,5000
Sector 7 HSR Layout,3 Bedroom,1200.0,4.0,275.0,3,22916
2nd Phase Judicial Layout,3 BHK,1300.0,3.0,64.0,3,4923
other,3 Bedroom,800.0,3.0,60.0,3,7500
Whitefield,4 Bedroom,4800.0,5.0,550.0,4,11458
Mysore Road,2 BHK,1029.5,2.0,50.855,2,4939
Raja Rajeshwari Nagar,2 BHK,1128.0,2.0,40.0,2,3546
other,2 Bedroom,2000.0,2.0,160.0,2,8000
Panathur,2 BHK,1210.0,2.0,80.0,2,6611
Arekere,4 BHK,2710.0,6.0,142.0,4,5239
Raja Rajeshwari Nagar,2 BHK,1151.0,2.0,39.0,2,3388
Rajaji Nagar,5 BHK,7500.0,8.0,1700.0,5,22666
other,3 BHK,1464.0,3.0,56.0,3,3825
Electronic City,2 BHK,1017.0,2.0,27.0,2,2654
Sarjapur  Road,2 BHK,920.0,2.0,40.0,2,4347
HRBR Layout,4 Bedroom,1200.0,3.0,180.0,4,15000
other,5 Bedroom,1000.0,5.0,140.0,5,14000
Munnekollal,3 BHK,1750.0,3.0,86.5,3,4942
Dasanapura,2 BHK,1163.0,2.0,65.0,2,5588
7th Phase JP Nagar,3 BHK,1650.0,3.0,110.0,3,6666
other,3 BHK,1800.0,3.0,95.0,3,5277
CV Raman Nagar,2 BHK,1100.0,2.0,58.0,2,5272
R.T. Nagar,2 BHK,1200.0,2.0,60.0,2,5000
other,3 BHK,1410.0,3.0,65.0,3,4609
Begur Road,3 BHK,1933.0,3.0,98.0,3,5069
Begur Road,2 BHK,1200.0,2.0,45.6,2,3800
Whitefield,3 BHK,1608.0,2.0,63.0,3,3917
other,4 Bedroom,2679.0,5.0,280.0,4,10451
Horamavu Banaswadi,2 BHK,1156.0,2.0,46.9,2,4057
other,2 Bedroom,1230.0,2.0,75.0,2,6097
Kasavanhalli,1 BHK,770.0,1.0,43.82,1,5690
Ardendale,3 BHK,1777.26,3.0,105.0,3,5907
Kanakpura Road,2 BHK,1339.0,2.0,85.0,2,6348
Sarjapur  Road,3 Bedroom,3044.0,3.0,183.0,3,6011
other,2 BHK,1080.0,2.0,44.8,2,4148
other,8 Bedroom,3150.0,5.0,145.0,8,4603
other,2 Bedroom,2400.0,2.0,135.0,2,5625
Hulimavu,3 BHK,1450.0,3.0,72.0,3,4965
Kothanur,2 BHK,1075.0,2.0,53.0,2,4930
Sarjapur  Road,3 BHK,1163.0,2.0,62.0,3,5331
TC Palaya,2 Bedroom,1200.0,2.0,65.0,2,5416
other,8 Bedroom,1200.0,8.0,135.0,8,11250
Marathahalli,4 Bedroom,3090.0,4.0,325.0,4,10517
Hebbal,2 BHK,1299.0,2.0,100.0,2,7698
Electronic City,2 BHK,1130.0,2.0,32.63,2,2887
HRBR Layout,2 BHK,1170.0,2.0,75.0,2,6410
Yelahanka,4 BHK,2600.0,4.0,175.0,4,6730
KR Puram,2 BHK,1140.0,2.0,43.32,2,3800
Kanakpura Road,3 BHK,1900.0,3.0,125.0,3,6578
Bommasandra Industrial Area,2 BHK,1090.0,2.0,31.48,2,2888
Electronic City,2 BHK,1110.0,2.0,39.95,2,3599
Nagavarapalya,1 BHK,705.0,1.0,57.0,1,8085
HBR Layout,2 BHK,1089.0,2.0,60.0,2,5509
other,2 Bedroom,880.0,2.0,48.0,2,5454
Kereguddadahalli,2 BHK,1080.0,2.0,32.0,2,2962
Thanisandra,2 BHK,1093.0,2.0,68.1,2,6230
Banashankari Stage III,3 BHK,1480.0,3.0,75.48,3,5100
Hegde Nagar,3 BHK,1703.0,3.0,125.0,3,7339
Hebbal,2 BHK,1200.0,2.0,60.0,2,5000
Marathahalli,3 BHK,1690.0,3.0,116.0,3,6863
Yelahanka,2 BHK,1305.0,2.0,78.0,2,5977
Talaghattapura,3 BHK,1257.0,2.0,40.22,3,3199
other,2 BHK,1150.0,2.0,45.0,2,3913
other,2 BHK,1194.0,2.0,55.0,2,4606
other,3 Bedroom,2000.0,3.0,90.0,3,4500
Anekal,3 Bedroom,1500.0,4.0,99.0,3,6600
other,2 BHK,1025.0,2.0,42.5,2,4146
Horamavu Agara,2 BHK,980.0,2.0,38.19,2,3896
Hosakerehalli,3 Bedroom,1200.0,2.0,100.0,3,8333
Whitefield,3 BHK,2225.0,3.0,149.0,3,6696
other,2 BHK,1022.0,2.0,57.0,2,5577
other,4 Bedroom,882.0,3.0,72.0,4,8163
Narayanapura,2 BHK,1153.0,2.0,45.0,2,3902
Pai Layout,2 BHK,1175.0,2.0,60.0,2,5106
Vidyaranyapura,3 BHK,1560.0,3.0,68.0,3,4358
Yelahanka New Town,1 BHK,650.0,1.0,17.0,1,2615
other,3 Bedroom,600.0,3.0,65.0,3,10833
Billekahalli,2 BHK,1350.0,3.0,55.0,2,4074
other,2 Bedroom,1200.0,2.0,76.0,2,6333
other,2 BHK,1200.0,1.0,54.0,2,4500
Chandapura,2 BHK,800.0,2.0,25.0,2,3125
Electronic City,4 BHK,2093.0,4.0,104.0,4,4968
Uttarahalli,3 BHK,1320.0,2.0,55.44,3,4200
Bannerghatta Road,2 BHK,1275.0,2.0,73.0,2,5725
Begur Road,2 BHK,1215.0,2.0,43.13,2,3549
Kanakapura,2 BHK,1130.0,2.0,45.2,2,4000
Begur Road,3 BHK,1444.0,2.0,71.5,3,4951
other,2 BHK,950.0,2.0,38.0,2,4000
Padmanabhanagar,2 BHK,1000.0,1.0,90.0,2,9000
Banashankari Stage II,3 BHK,1240.0,2.0,59.52,3,4800
2nd Stage Nagarbhavi,4 Bedroom,600.0,4.0,125.0,4,20833
Hennur,2 BHK,1231.0,2.0,50.0,2,4061
Benson Town,2 BHK,1300.0,2.0,120.0,2,9230
Yeshwanthpur,3 BHK,1713.0,3.0,110.0,3,6421
Kogilu,2 BHK,1200.0,2.0,53.33,2,4444
Sarjapur  Road,3 Bedroom,3004.0,4.0,158.0,3,5259
Sonnenahalli,3 BHK,1310.0,2.0,46.0,3,3511
Jalahalli,2 BHK,790.0,2.0,49.5,2,6265
Koramangala,3 BHK,1744.0,3.0,200.0,3,11467
other,3 BHK,1605.0,3.0,100.0,3,6230
Whitefield,2 BHK,1250.0,2.0,82.0,2,6560
Malleshwaram,3 BHK,2476.0,3.0,337.0,3,13610
Subramanyapura,3 BHK,1260.0,2.0,76.0,3,6031
Rajaji Nagar,3 BHK,1615.0,3.0,175.0,3,10835
Yelahanka,2 BHK,1180.0,2.0,56.0,2,4745
Uttarahalli,3 BHK,1255.0,2.0,50.19,3,3999
Electronic City,2 BHK,1025.0,2.0,53.9,2,5258
Sarjapur  Road,3 BHK,1877.0,4.0,150.0,3,7991
Lakshminarayana Pura,3 BHK,1649.0,3.0,150.0,3,9096
5th Block Hbr Layout,9 Bedroom,2600.0,12.0,675.0,9,25961
Hennur,5 Bedroom,2500.0,5.0,125.0,5,5000
other,3 BHK,1542.0,3.0,98.0,3,6355
Thigalarapalya,2 BHK,1418.0,2.0,105.0,2,7404
other,2 BHK,1012.0,2.0,42.0,2,4150
Chandapura,2 Bedroom,1200.0,2.0,65.0,2,5416
Sarjapur  Road,2 BHK,1115.0,2.0,43.0,2,3856
Green Glen Layout,3 BHK,1680.0,4.0,95.0,3,5654
Old Madras Road,2 BHK,1171.0,2.0,72.0,2,6148
Tumkur Road,2 BHK,1027.0,2.0,68.62,2,6681
Shampura,3 BHK,1700.0,3.0,75.0,3,4411
Kanakapura,2 BHK,1090.0,2.0,38.15,2,3500
Hennur Road,2 BHK,1232.0,2.0,74.0,2,6006
Anekal,2 BHK,766.0,2.0,28.0,2,3655
Banashankari Stage III,5 Bedroom,1200.0,5.0,180.0,5,15000
Kanakpura Road,1 BHK,525.0,1.0,27.0,1,5142
JP Nagar,3 BHK,1452.0,2.0,41.0,3,2823
Choodasandra,3 BHK,1254.0,2.0,65.0,3,5183
other,5 Bedroom,3600.0,5.0,110.0,5,3055
Kanakpura Road,3 BHK,1593.0,3.0,118.0,3,7407
other,2 BHK,1333.0,2.0,77.27,2,5796
other,3 Bedroom,1100.0,3.0,110.0,3,10000
Raja Rajeshwari Nagar,3 BHK,1375.0,2.0,54.99,3,3999
Ardendale,3 BHK,1650.0,3.0,82.0,3,4969
Thanisandra,3 BHK,1430.0,2.0,51.48,3,3600
Electronic City,2 BHK,940.0,2.0,54.0,2,5744
Electronic City,1 BHK,435.0,1.0,21.0,1,4827
Begur Road,2 BHK,1160.0,2.0,36.54,2,3150
Electronic City,3 BHK,1500.0,2.0,77.5,3,5166
Koramangala,3 BHK,1890.0,2.0,100.0,3,5291
other,2 BHK,1050.0,2.0,70.0,2,6666
Singasandra,6 Bedroom,1200.0,6.0,220.0,6,18333
Haralur Road,2 BHK,1027.0,2.0,43.0,2,4186
Seegehalli,3 Bedroom,3000.0,4.0,235.0,3,7833
Tumkur Road,3 BHK,1459.0,2.0,95.0,3,6511
Yelahanka,3 BHK,1711.0,3.0,85.61,3,5003
Hoodi,2 BHK,1250.0,2.0,98.0,2,7840
Chandapura,1 BHK,645.0,1.0,16.45,1,2550
Anandapura,2 Bedroom,875.0,2.0,57.0,2,6514
Ramamurthy Nagar,2 BHK,1334.0,2.0,67.0,2,5022
other,2 BHK,1140.0,2.0,32.92,2,2887
Sarjapur  Road,2 BHK,1115.0,2.0,50.0,2,4484
Amruthahalli,3 Bedroom,1200.0,5.0,200.0,3,16666
Rajaji Nagar,3 BHK,1776.0,3.0,190.0,3,10698
Thigalarapalya,3 BHK,2072.0,4.0,160.0,3,7722
Kundalahalli,3 Bedroom,1600.0,3.0,170.0,3,10625
other,6 Bedroom,1200.0,4.0,115.0,6,9583
Kengeri Satellite Town,2 BHK,920.0,2.0,39.0,2,4239
Chandapura,2 BHK,778.0,2.0,25.29,2,3250
Nagasandra,4 Bedroom,1310.0,3.0,95.0,4,7251
Electronic City,3 BHK,1631.0,3.0,80.0,3,4904
Kanakpura Road,1 BHK,525.0,1.0,27.5,1,5238
Jalahalli,3 BHK,1395.0,3.0,82.0,3,5878
other,3 BHK,2000.0,3.0,105.0,3,5250
Hennur Road,2 BHK,1232.0,2.0,66.0,2,5357
other,2 BHK,1075.0,2.0,55.0,2,5116
Gubbalala,2 BHK,1205.0,2.0,70.0,2,5809
other,4 Bedroom,1200.0,4.0,165.0,4,13750
Rajiv Nagar,4 BHK,2340.0,5.0,129.0,4,5512
Rayasandra,3 BHK,1179.0,2.0,55.0,3,4664
Kodigehaali,2 BHK,1013.0,2.0,51.0,2,5034
Uttarahalli,2 BHK,1160.0,2.0,40.6,2,3500
7th Phase JP Nagar,2 BHK,1200.0,2.0,62.5,2,5208
other,7 Bedroom,1200.0,5.0,80.0,7,6666
Uttarahalli,3 BHK,1255.0,2.0,50.19,3,3999
other,3 BHK,1100.0,2.0,145.0,3,13181
other,2 BHK,1069.0,2.0,55.0,2,5144
other,3 Bedroom,1200.0,3.0,99.0,3,8250
Kudlu Gate,1 BHK,1400.0,1.0,285.0,1,20357
Marathahalli,2 BHK,1065.0,2.0,55.0,2,5164
Lakshminarayana Pura,2 BHK,1395.0,2.0,75.0,2,5376
Hebbal Kempapura,3 BHK,1725.0,2.0,165.0,3,9565
other,2 BHK,1007.0,2.0,43.0,2,4270
Kengeri,3 BHK,1436.0,2.0,55.0,3,3830
other,6 BHK,4250.0,6.0,150.0,6,3529
Hebbal,3 BHK,1645.0,3.0,117.0,3,7112
Kengeri,2 Bedroom,1200.0,2.0,58.0,2,4833
Mysore Road,2 BHK,940.0,2.0,43.61,2,4639
KR Puram,6 Bedroom,4300.0,6.0,175.0,6,4069
Hebbal,3 BHK,2850.0,5.0,343.0,3,12035
Indira Nagar,3 BHK,1740.0,3.0,191.0,3,10977
Haralur Road,2 BHK,1194.0,2.0,47.0,2,3936
other,3 BHK,3761.0,3.0,650.0,3,17282
Dodda Nekkundi,3 BHK,1829.0,3.0,125.0,3,6834
Whitefield,4 Bedroom,1800.0,4.0,178.0,4,9888
other,2 BHK,1000.0,2.0,60.0,2,6000
other,3 Bedroom,5720.0,3.0,350.0,3,6118
other,5 Bedroom,800.0,4.0,98.0,5,12250
other,2 BHK,675.0,1.0,15.0,2,2222
Hebbal,2 BHK,1390.0,2.0,95.0,2,6834
Lakshminarayana Pura,3 BHK,1500.0,3.0,104.0,3,6933
Vijayanagar,3 BHK,2047.0,3.0,136.0,3,6643
Jakkur,2 BHK,1202.0,2.0,79.0,2,6572
other,2 BHK,1012.0,2.0,59.0,2,5830
Gunjur,3 BHK,2132.0,3.0,80.0,3,3752
Raja Rajeshwari Nagar,4 Bedroom,1350.0,4.0,340.0,4,25185
Whitefield,1 BHK,605.0,1.0,40.0,1,6611
Harlur,2 BHK,1331.0,2.0,88.0,2,6611
Margondanahalli,2 Bedroom,1200.0,2.0,69.0,2,5750
other,2 BHK,1150.0,2.0,250.0,2,21739
Sahakara Nagar,2 Bedroom,1200.0,2.0,136.0,2,11333
other,2 Bedroom,1800.0,2.0,300.0,2,16666
Kammanahalli,3 Bedroom,600.0,2.0,50.0,3,8333
other,3 BHK,1280.0,2.0,49.0,3,3828
Balagere,2 BHK,1007.0,2.0,62.0,2,6156
Thanisandra,3 BHK,1370.0,2.0,61.0,3,4452
other,2 BHK,925.0,2.0,35.0,2,3783
TC Palaya,2 BHK,1500.0,2.0,75.0,2,5000
Attibele,1 BHK,450.0,1.0,12.5,1,2777
other,4 Bedroom,1200.0,4.0,140.0,4,11666
Hennur Road,2 BHK,1341.0,2.0,95.0,2,7084
2nd Phase Judicial Layout,3 BHK,1350.0,2.0,47.25,3,3500
Tumkur Road,2 BHK,992.0,2.0,60.0,2,6048
Uttarahalli,2 BHK,1163.0,2.0,68.0,2,5846
Sarjapur  Road,2 BHK,1229.0,2.0,70.0,2,5695
other,2 Bedroom,600.0,2.0,39.0,2,6500
Electronic City Phase II,2 BHK,1219.0,2.0,61.26,2,5025
Raja Rajeshwari Nagar,3 BHK,1525.0,2.0,99.13,3,6500
Kasavanhalli,2 BHK,1147.0,2.0,60.0,2,5231
Akshaya Nagar,3 BHK,1412.0,2.0,88.0,3,6232
Munnekollal,9 Bedroom,1200.0,9.0,215.0,9,17916
Sarjapur  Road,2 BHK,1044.0,2.0,34.77,2,3330
Nagarbhavi,2 BHK,1190.0,2.0,63.0,2,5294
other,3 BHK,1240.0,2.0,65.0,3,5241
Jigani,3 BHK,1352.0,3.0,65.0,3,4807
other,2 BHK,1100.0,2.0,55.0,2,5000
Uttarahalli,3 BHK,1330.0,2.0,39.9,3,3000
other,3 BHK,1700.0,3.0,111.0,3,6529
Electronic City,2 BHK,1355.0,2.0,67.0,2,4944
Ramamurthy Nagar,2 BHK,1200.0,2.0,46.0,2,3833
Kengeri,1 BHK,400.0,2.0,25.0,1,6250
Sarjapur  Road,2 BHK,1342.0,2.0,70.0,2,5216
Begur Road,3 BHK,1615.0,3.0,50.87,3,3149
other,2 BHK,1340.0,2.0,65.0,2,4850
other,2 BHK,1200.0,2.0,54.0,2,4500
Yeshwanthpur,1 Bedroom,400.0,1.0,55.0,1,13750
Devanahalli,2 BHK,1080.0,2.0,52.0,2,4814
other,2 BHK,1600.0,2.0,31.0,2,1937
Electronic City,2 BHK,1200.0,2.0,34.0,2,2833
Hennur Road,3 Bedroom,2700.0,3.0,150.0,3,5555
other,3 Bedroom,1780.0,3.0,110.0,3,6179
Amruthahalli,3 BHK,1605.0,3.0,65.0,3,4049
Ulsoor,4 BHK,7200.0,5.0,1584.0,4,22000
other,3 Bedroom,646.0,2.0,65.0,3,10061
Kammanahalli,2 BHK,1200.0,2.0,80.0,2,6666
Margondanahalli,2 Bedroom,900.0,2.0,49.0,2,5444
Sarjapur,3 Bedroom,1830.0,3.0,88.0,3,4808
Vishwapriya Layout,5 Bedroom,690.0,4.0,83.0,5,12028
other,1 BHK,650.0,1.0,500.0,1,76923
other,5 Bedroom,1800.0,4.0,68.0,5,3777
Chandapura,3 BHK,715.0,2.0,32.0,3,4475
Chandapura,3 BHK,1230.0,2.0,31.37,3,2550
Uttarahalli,2 BHK,1065.0,2.0,42.59,2,3999
other,8 Bedroom,1350.0,8.0,185.0,8,13703
other,3 Bedroom,1080.0,3.0,149.0,3,13796
other,3 BHK,900.0,2.0,65.0,3,7222
other,2 BHK,946.0,2.0,45.0,2,4756
Laggere,1 BHK,1500.0,1.0,60.0,1,4000
Sarjapur  Road,3 BHK,1711.0,3.0,110.0,3,6428
other,4 Bedroom,1200.0,3.0,160.0,4,13333
Chandapura,2 BHK,950.0,2.0,25.65,2,2700
Kudlu Gate,3 BHK,1656.0,3.0,80.0,3,4830
Uttarahalli,2 BHK,1075.0,2.0,46.76,2,4349
Harlur,3 BHK,1752.12,3.0,133.0,3,7590
Hebbal,8 Bedroom,6000.0,8.0,220.0,8,3666
other,3 BHK,1540.0,3.0,85.0,3,5519
6th Phase JP Nagar,4 Bedroom,600.0,4.0,125.0,4,20833
LB Shastri Nagar,1 BHK,665.0,1.0,32.0,1,4812
Sector 2 HSR Layout,3 BHK,1665.0,3.0,110.0,3,6606
Electronics City Phase 1,2 BHK,1145.0,2.0,54.0,2,4716
Whitefield,2 BHK,1185.0,2.0,41.0,2,3459
other,3 Bedroom,1200.0,3.0,190.0,3,15833
Marathahalli,3 BHK,1693.0,3.0,125.0,3,7383
other,3 BHK,2456.0,3.0,220.0,3,8957
Rajaji Nagar,3 BHK,2559.0,3.0,403.0,3,15748
Uttarahalli,2 BHK,1155.0,2.0,40.6,2,3515
other,3 BHK,1610.0,2.0,64.4,3,4000
Whitefield,3 Bedroom,1200.0,3.0,56.58,3,4715
other,2 Bedroom,1530.0,2.0,185.0,2,12091
EPIP Zone,3 BHK,1709.0,3.0,80.0,3,4681
Amruthahalli,4 Bedroom,600.0,4.0,60.0,4,10000
Brookefield,8 Bedroom,2700.0,8.0,290.0,8,10740
other,1 Bedroom,840.0,1.0,150.0,1,17857
Gottigere,3 BHK,1400.0,2.0,50.0,3,3571
KR Puram,2 BHK,1200.0,2.0,42.0,2,3500
other,3 BHK,1439.0,3.0,57.23,3,3977
Kasavanhalli,2 BHK,1375.0,2.0,72.0,2,5236
BTM 2nd Stage,5 Bedroom,2990.0,5.0,416.0,5,13913
Vijayanagar,3 BHK,1200.0,2.0,75.0,3,6250
Yelahanka,2 BHK,1315.0,2.0,75.0,2,5703
Dommasandra,2 Bedroom,1500.0,2.0,40.0,2,2666
Yelahanka,6 Bedroom,3600.0,6.0,200.0,6,5555
Vijayanagar,2 BHK,1178.0,2.0,82.0,2,6960
Bannerghatta Road,3 BHK,1430.0,3.0,59.0,3,4125
Mahadevpura,3 BHK,1400.0,2.0,65.8,3,4700
Thanisandra,2 BHK,1093.0,2.0,72.0,2,6587
Horamavu Banaswadi,2 BHK,1200.0,2.0,45.0,2,3750
Magadi Road,5 Bedroom,5000.0,6.0,360.0,5,7200
1st Block Jayanagar,3 BHK,1200.0,2.0,130.0,3,10833
Whitefield,3 BHK,2210.0,3.0,156.0,3,7058
Uttarahalli,2 BHK,1150.0,2.0,70.0,2,6086
other,3 BHK,1435.0,2.0,60.0,3,4181
other,4 Bedroom,600.0,3.0,100.0,4,16666
Whitefield,2 BHK,1173.0,2.0,78.2,2,6666
Thanisandra,2 BHK,1140.0,2.0,39.0,2,3421
other,3 BHK,1717.0,3.0,120.0,3,6988
Mallasandra,2 BHK,1325.0,2.0,65.0,2,4905
Hoodi,3 BHK,1370.0,3.0,67.0,3,4890
Electronics City Phase 1,2 BHK,1025.0,2.0,90.0,2,8780
other,3 BHK,1409.0,2.0,62.0,3,4400
Bellandur,2 BHK,1200.0,2.0,52.0,2,4333
Thanisandra,2 BHK,1200.0,2.0,53.5,2,4458
Whitefield,3 BHK,1740.0,4.0,130.0,3,7471
Hennur Road,2 BHK,1232.0,2.0,88.0,2,7142
Chandapura,2 Bedroom,1200.0,2.0,36.0,2,3000
Whitefield,4 BHK,5924.0,4.0,625.0,4,10550
other,2 BHK,1250.0,2.0,50.0,2,4000
other,2 BHK,1475.0,2.0,171.0,2,11593
Bommanahalli,4 BHK,2100.0,4.0,85.0,4,4047
5th Phase JP Nagar,1 BHK,552.0,1.0,23.5,1,4257
Uttarahalli,3 BHK,1065.0,2.0,42.59,3,3999
Hennur,3 Bedroom,1000.0,3.0,130.0,3,13000
other,2 Bedroom,1225.0,2.0,65.0,2,5306
other,1 BHK,650.0,1.0,16.0,1,2461
Padmanabhanagar,2 BHK,1067.0,2.0,52.0,2,4873
Sarjapur,3 BHK,1364.0,2.0,56.0,3,4105
other,3 BHK,1750.0,3.0,75.0,3,4285
Bannerghatta Road,2 BHK,1240.0,2.0,71.0,2,5725
Hulimavu,2 Bedroom,1200.0,2.0,75.0,2,6250
Kaggadasapura,3 BHK,1540.0,3.0,75.0,3,4870
other,3 BHK,1504.0,2.0,102.0,3,6781
other,2 BHK,1116.0,2.0,49.8,2,4462
Hennur Road,3 BHK,2365.0,4.0,175.0,3,7399
Sarjapura - Attibele Road,2 BHK,1308.0,2.0,37.0,2,2828
Vijayanagar,2 BHK,864.0,2.0,45.0,2,5208
Vidyaranyapura,5 Bedroom,1200.0,5.0,139.0,5,11583
Hoskote,3 BHK,1069.0,2.0,38.0,3,3554
Hebbal,2 BHK,1333.0,2.0,100.0,2,7501
other,4 BHK,1800.0,4.0,145.0,4,8055
Devanahalli,4 Bedroom,6136.0,4.0,550.0,4,8963
Thanisandra,3 BHK,1698.0,3.0,107.0,3,6301
JP Nagar,3 BHK,1275.0,2.0,43.6,3,3419
Raja Rajeshwari Nagar,2 BHK,1060.0,2.0,48.0,2,4528
Hennur,2 BHK,1160.0,2.0,49.0,2,4224
Sarjapur  Road,3 BHK,1484.0,2.0,65.0,3,4380
Hosa Road,3 Bedroom,675.0,3.0,70.0,3,10370
BEML Layout,3 Bedroom,1200.0,5.0,325.0,3,27083
Hennur Road,2 BHK,1310.0,2.0,66.0,2,5038
Vijayanagar,6 Bedroom,1806.0,6.0,370.0,6,20487
other,4 BHK,3000.0,4.0,240.0,4,8000
Varthur,3 BHK,2145.0,3.0,165.0,3,7692
Marathahalli,3 BHK,1693.0,3.0,125.0,3,7383
Jigani,3 BHK,1230.0,3.0,60.0,3,4878
Devanahalli,3 Bedroom,1200.0,3.0,145.0,3,12083
Bommasandra Industrial Area,3 BHK,1310.0,2.0,37.83,3,2887
other,2 BHK,1150.0,2.0,60.0,2,5217
Munnekollal,3 BHK,1385.0,2.0,66.5,3,4801
Tumkur Road,2 BHK,1066.0,2.0,52.5,2,4924
Thanisandra,3 BHK,2293.0,3.0,80.26,3,3500
Tindlu,3 Bedroom,720.0,3.0,81.0,3,11250
other,3 BHK,1625.0,3.0,92.0,3,5661
Vittasandra,2 BHK,1246.0,2.0,67.4,2,5409
Vijayanagar,4 Bedroom,2100.0,6.0,252.0,4,12000
other,4 BHK,2400.0,4.0,250.0,4,10416
Raja Rajeshwari Nagar,6 Bedroom,3900.0,6.0,195.0,6,5000
7th Phase JP Nagar,2 BHK,1000.0,2.0,70.65,2,7065
Nagavarapalya,2 BHK,907.0,2.0,38.0,2,4189
Ulsoor,4 Bedroom,3500.0,4.0,138.0,4,3942
Varthur,3 BHK,1395.0,2.0,79.0,3,5663
other,2 BHK,1180.0,2.0,50.0,2,4237
Banaswadi,5 Bedroom,1903.0,6.0,140.0,5,7356
Electronic City,2 BHK,1175.0,2.0,47.0,2,4000
other,2 BHK,1272.0,2.0,51.5,2,4048
other,3 BHK,1490.0,2.0,92.0,3,6174
Yeshwanthpur,2 BHK,1144.0,2.0,67.0,2,5856
other,4 Bedroom,2400.0,5.0,140.0,4,5833
Hormavu,2 BHK,1143.0,2.0,53.0,2,4636
Karuna Nagar,3 Bedroom,2500.0,3.0,190.0,3,7600
Whitefield,2 BHK,1216.0,2.0,75.0,2,6167
other,2 BHK,1044.0,2.0,60.0,2,5747
Prithvi Layout,2 BHK,1352.0,3.0,93.0,2,6878
Hennur Road,3 BHK,1552.0,3.0,75.0,3,4832
Bellandur,3 BHK,1785.0,3.0,120.0,3,6722
Pai Layout,2 BHK,810.0,2.0,34.5,2,4259
Banashankari Stage V,3 BHK,1650.0,3.0,51.98,3,3150
other,3 BHK,1595.0,3.0,70.39,3,4413
Whitefield,2 BHK,1060.0,2.0,35.0,2,3301
Babusapalaya,3 BHK,1410.0,2.0,65.0,3,4609
Nagarbhavi,3 BHK,1523.0,2.0,56.0,3,3676
Kanakpura Road,3 BHK,1591.0,3.0,122.0,3,7668
other,3 BHK,1640.0,3.0,84.0,3,5121
Yelahanka New Town,1 BHK,450.0,1.0,16.0,1,3555
Bannerghatta Road,3 BHK,1305.0,2.0,60.0,3,4597
Yelahanka,3 BHK,1590.0,3.0,54.0,3,3396
Kaggadasapura,2 BHK,1240.0,2.0,45.0,2,3629
other,2 BHK,1200.0,2.0,54.0,2,4500
Dodda Nekkundi,2 BHK,1100.0,2.0,41.16,2,3741
Koramangala,2 BHK,1180.0,2.0,128.0,2,10847
Sarjapur  Road,2 BHK,1115.0,2.0,50.0,2,4484
Bisuvanahalli,2 BHK,845.0,2.0,33.0,2,3905
other,3 BHK,1567.2,2.0,190.0,3,12123
Channasandra,3 BHK,1310.0,2.0,45.0,3,3435
Yelahanka,1 Bedroom,26136.0,1.0,150.0,1,573
other,2 BHK,959.0,2.0,42.5,2,4431
8th Phase JP Nagar,4 BHK,1200.0,4.0,140.0,4,11666
Kambipura,2 BHK,883.0,2.0,37.0,2,4190
Rachenahalli,2 BHK,1050.0,2.0,55.5,2,5285
Yelahanka,2 BHK,1390.0,2.0,70.21,2,5051
Kenchenahalli,3 BHK,1280.0,2.0,69.0,3,5390
other,7 Bedroom,1280.0,5.0,150.0,7,11718
Yelahanka,3 BHK,1614.0,3.0,95.0,3,5885
Kasturi Nagar,2 BHK,1000.0,2.0,58.0,2,5800
Marathahalli,2 BHK,1196.0,2.0,57.9,2,4841
Sarjapur  Road,3 BHK,1220.0,3.0,57.0,3,4672
Ambalipura,2 BHK,950.0,2.0,31.95,2,3363
other,9 BHK,2400.0,8.0,325.0,9,13541
other,2 BHK,1141.0,2.0,55.0,2,4820
5th Phase JP Nagar,2 BHK,1200.0,2.0,51.0,2,4250
Kasavanhalli,3 BHK,1380.0,2.0,55.0,3,3985
Doddathoguru,2 BHK,855.0,2.0,32.0,2,3742
Chandapura,1 BHK,645.0,1.0,16.45,1,2550
Chandapura,2 BHK,922.0,2.0,36.0,2,3904
other,2 BHK,1200.0,2.0,43.2,2,3600
other,3 BHK,1240.0,2.0,65.0,3,5241
7th Phase JP Nagar,2 BHK,1070.0,2.0,42.0,2,3925
other,2 BHK,1293.0,2.0,75.0,2,5800
R.T. Nagar,3 BHK,1380.0,2.0,50.0,3,3623
Yelahanka New Town,3 BHK,1599.0,3.0,80.0,3,5003
Nehru Nagar,2 BHK,967.0,2.0,41.1,2,4250
other,3 BHK,1885.0,3.0,90.0,3,4774
other,2 BHK,860.0,2.0,65.5,2,7616
Yelahanka,1 Bedroom,1075.0,1.0,19.5,1,1813
Sarjapur  Road,2 BHK,1050.0,2.0,63.5,2,6047
Marathahalli,3 BHK,1595.0,3.0,88.0,3,5517
Sarjapur  Road,2 BHK,1350.0,2.0,102.0,2,7555
Hebbal,2 BHK,1204.0,2.0,45.0,2,3737
Sarjapur  Road,4 BHK,2990.0,4.0,240.0,4,8026
Sarjapur  Road,2 BHK,1035.0,2.0,49.0,2,4734
ITPL,2 BHK,1000.0,2.0,29.95,2,2995
Sarjapur  Road,3 BHK,1457.0,3.0,85.0,3,5833
Haralur Road,2 BHK,1194.0,2.0,47.0,2,3936
Bannerghatta,2 BHK,1100.0,2.0,66.0,2,6000
Whitefield,3 BHK,1655.0,3.0,120.0,3,7250
Kanakpura Road,3 BHK,1691.2,3.0,115.0,3,6799
Ramagondanahalli,2 BHK,1215.0,2.0,46.1,2,3794
Kanakpura Road,2 BHK,1155.0,2.0,50.125,2,4339
Hoodi,3 BHK,1639.0,3.0,129.0,3,7870
Sarjapur  Road,3 BHK,1685.0,3.0,135.0,3,8011
other,5 Bedroom,1008.0,4.0,180.0,5,17857
Hormavu,2 BHK,1100.0,2.0,38.0,2,3454
Marathahalli,3 BHK,2122.0,3.0,125.0,3,5890
Vasanthapura,2 BHK,1050.0,2.0,47.0,2,4476
Kengeri,2 BHK,1230.0,2.0,45.0,2,3658
5th Phase JP Nagar,3 BHK,1240.0,3.0,60.0,3,4838
Electronic City Phase II,3 BHK,1252.0,2.0,65.0,3,5191
other,3 BHK,1374.0,2.0,88.0,3,6404
8th Phase JP Nagar,1 BHK,500.0,1.0,30.0,1,6000
Arekere,8 Bedroom,1200.0,8.0,225.0,8,18750
Hebbal,3 BHK,1355.0,3.0,83.87,3,6189
Banashankari,2 BHK,1040.0,2.0,45.0,2,4326
Sarjapur  Road,2 BHK,1350.0,2.0,68.0,2,5037
Whitefield,4 BHK,4003.0,6.0,300.0,4,7494
Hennur,3 BHK,2400.0,3.0,135.0,3,5625
Haralur Road,3 BHK,1560.0,3.0,90.0,3,5769
Anandapura,2 BHK,1151.0,2.0,43.16,2,3749
Uttarahalli,3 BHK,1360.0,2.0,47.59,3,3499
other,2 BHK,1250.0,2.0,60.0,2,4800
Chandapura,3 BHK,1110.0,2.0,29.97,3,2700
Hennur Road,2 BHK,1015.0,2.0,50.0,2,4926
Kadugodi,3 BHK,1351.0,2.0,78.0,3,5773
Yelahanka New Town,2 BHK,960.0,2.0,50.0,2,5208
other,2 BHK,1110.0,2.0,52.0,2,4684
R.T. Nagar,3 BHK,1680.0,3.0,72.0,3,4285
Arekere,3 Bedroom,2400.0,3.0,180.0,3,7500
other,4 Bedroom,11000.0,5.0,2000.0,4,18181
other,3 BHK,1350.0,3.0,48.6,3,3600
Horamavu Banaswadi,2 BHK,1254.0,2.0,47.0,2,3748
Haralur Road,3 BHK,1255.0,3.0,115.0,3,9163
CV Raman Nagar,2 BHK,1051.0,2.0,61.0,2,5803
other,3 Bedroom,1200.0,3.0,205.0,3,17083
Uttarahalli,2 BHK,1099.0,2.0,65.0,2,5914
Hebbal,3 BHK,2650.0,4.0,199.0,3,7509
other,4 Bedroom,2000.0,2.0,160.0,4,8000
Yeshwanthpur,2 BHK,1541.0,2.0,130.0,2,8436
Sarjapur  Road,3 Bedroom,1200.0,3.0,70.0,3,5833
Whitefield,2 BHK,1125.0,2.0,45.0,2,4000
Vittasandra,3 BHK,1650.0,3.0,85.0,3,5151
other,1 BHK,834.0,1.0,62.0,1,7434
Gollarapalya Hosahalli,3 BHK,1320.0,3.0,60.0,3,4545
Malleshpalya,3 Bedroom,1200.0,3.0,149.0,3,12416
other,8 Bedroom,1000.0,7.0,160.0,8,16000
Mysore Road,3 BHK,1500.0,3.0,74.0,3,4933
Whitefield,4 Bedroom,4400.0,5.0,500.0,4,11363
other,1 Bedroom,600.0,1.0,52.0,1,8666
Whitefield,2 BHK,1180.0,2.0,70.0,2,5932
Kadugodi,2 BHK,1010.0,2.0,50.0,2,4950
Channasandra,2 BHK,1070.0,2.0,38.0,2,3551
Choodasandra,3 BHK,1580.0,3.0,105.0,3,6645
other,4 Bedroom,1200.0,5.0,325.0,4,27083
Green Glen Layout,3 BHK,2000.0,3.0,165.0,3,8250
Iblur Village,3 BHK,1920.0,3.0,130.0,3,6770
Sanjay nagar,3 BHK,1200.0,2.0,76.0,3,6333
Kalena Agrahara,2 BHK,1040.0,2.0,45.0,2,4326
Hulimavu,3 BHK,1430.0,2.0,90.0,3,6293
Uttarahalli,3 BHK,1215.0,2.0,52.85,3,4349
Domlur,1 BHK,650.0,1.0,70.0,1,10769
Thanisandra,3 BHK,1564.0,3.0,105.0,3,6713
other,3 Bedroom,800.0,2.0,52.0,3,6500
Hormavu,2 Bedroom,750.0,2.0,58.0,2,7733
Kasavanhalli,2 BHK,1200.0,2.0,60.0,2,5000
Raja Rajeshwari Nagar,2 BHK,1128.0,2.0,48.79,2,4325
Begur Road,2 BHK,1200.0,2.0,44.4,2,3700
other,4 BHK,2000.0,4.0,135.0,4,6750
other,3 BHK,2750.0,3.0,260.0,3,9454
other,4 Bedroom,2400.0,4.0,480.0,4,20000
Yelahanka,3 BHK,1650.0,3.0,105.0,3,6363
1st Block Jayanagar,2 BHK,1235.0,2.0,148.0,2,11983
Yelahanka,2 BHK,1200.0,2.0,55.0,2,4583
other,6 BHK,900.0,5.0,100.0,6,11111
Kengeri Satellite Town,2 BHK,985.0,2.0,42.0,2,4263
Ramagondanahalli,3 BHK,1475.0,2.0,63.0,3,4271
Hebbal,3 BHK,3450.0,5.0,348.0,3,10086
Sarjapur  Road,3 BHK,1157.0,2.0,53.99,3,4666
other,2 BHK,900.0,2.0,35.0,2,3888
Gottigere,4 Bedroom,2500.0,3.0,120.0,4,4800
GM Palaya,2 BHK,1030.0,2.0,60.0,2,5825
other,2 BHK,1000.0,2.0,46.0,2,4600
Sarjapur  Road,4 Bedroom,3800.0,3.0,325.0,4,8552
Kasavanhalli,4 Bedroom,4260.0,4.0,333.0,4,7816
Brookefield,5 Bedroom,1950.0,6.0,175.0,5,8974
Kudlu,3 BHK,1570.0,2.0,65.94,3,4200
Kanakapura,2 BHK,1283.0,2.0,68.0,2,5300
Electronic City Phase II,3 BHK,875.0,2.0,40.0,3,4571
other,2 BHK,1200.0,2.0,52.0,2,4333
HAL 2nd Stage,3 Bedroom,2700.0,3.0,500.0,3,18518
Konanakunte,3 Bedroom,2400.0,2.0,180.0,3,7500
Kathriguppe,3 BHK,1250.0,2.0,68.75,3,5500
Old Madras Road,4 BHK,3630.0,6.0,196.0,4,5399
Yelahanka,2 BHK,990.0,2.0,37.62,2,3800
Panathur,2 BHK,1199.0,2.0,85.0,2,7089
other,2 BHK,1280.0,2.0,90.0,2,7031
1st Block Jayanagar,4 BHK,2750.0,4.0,413.0,4,15018
other,2 BHK,1190.0,2.0,55.0,2,4621
Yelachenahalli,2 BHK,1100.0,2.0,60.0,2,5454
other,3 BHK,2075.0,3.0,175.0,3,8433
Harlur,2 BHK,1133.0,2.0,55.0,2,4854
Somasundara Palya,4 BHK,2400.0,3.0,140.0,4,5833
Whitefield,2 BHK,1280.0,2.0,69.0,2,5390
Horamavu Agara,3 BHK,1250.0,3.0,65.0,3,5200
Begur Road,3 BHK,1565.0,2.0,49.3,3,3150
other,3 BHK,1900.0,3.0,65.0,3,3421
Munnekollal,2 BHK,1080.0,2.0,45.0,2,4166
Battarahalli,1 BHK,600.0,1.0,47.0,1,7833
other,3 Bedroom,1240.0,4.0,180.0,3,14516
HRBR Layout,3 BHK,1335.0,2.0,87.0,3,6516
Vittasandra,2 BHK,1246.0,2.0,67.3,2,5401
Sarjapur  Road,3 BHK,1385.0,3.0,55.0,3,3971
other,3 BHK,1578.0,3.0,73.9,3,4683
HSR Layout,2 BHK,1140.0,2.0,46.0,2,4035
Sarjapur  Road,3 BHK,1700.0,3.0,79.9,3,4700
other,4 Bedroom,1050.0,4.0,90.0,4,8571
Vidyaranyapura,2 BHK,1120.0,2.0,70.0,2,6250
Electronic City,1 BHK,435.0,1.0,22.0,1,5057
Magadi Road,2 BHK,1116.0,2.0,51.89,2,4649
Old Madras Road,3 BHK,1480.0,3.0,95.0,3,6418
Hormavu,2 BHK,1000.0,2.0,50.0,2,5000
HSR Layout,2 BHK,1145.0,2.0,46.0,2,4017
Bommasandra,2 BHK,800.0,1.0,32.0,2,4000
Thanisandra,2 BHK,1265.0,2.0,83.0,2,6561
Devanahalli,3 BHK,1700.0,3.0,88.2,3,5188
other,2 BHK,1475.0,2.0,70.0,2,4745
BTM 2nd Stage,2 BHK,1265.0,2.0,55.0,2,4347
Anandapura,2 Bedroom,1200.0,2.0,58.0,2,4833
1st Phase JP Nagar,3 BHK,2180.0,3.0,210.0,3,9633
Brookefield,2 BHK,1080.0,2.0,50.0,2,4629
Whitefield,3 BHK,2023.0,4.0,130.0,3,6426
other,2 BHK,2400.0,2.0,125.0,2,5208
Kadubeesanahalli,3 BHK,1545.0,2.0,66.0,3,4271
Kalena Agrahara,3 BHK,1804.0,3.0,166.0,3,9201
other,2 Bedroom,1200.0,2.0,200.0,2,16666
6th Phase JP Nagar,3 BHK,1350.0,2.0,58.0,3,4296
Hoodi,2 BHK,1108.0,2.0,74.23,2,6699
Marathahalli,3 BHK,1730.0,3.0,115.0,3,6647
other,3 BHK,1427.0,2.0,70.0,3,4905
Yelahanka,2 BHK,935.0,2.0,35.53,2,3800
Sarjapura - Attibele Road,2 BHK,1073.0,2.0,31.77,2,2960
Ananth Nagar,2 BHK,960.0,2.0,37.0,2,3854
Thigalarapalya,3 BHK,2215.0,4.0,150.0,3,6772
Electronic City Phase II,2 BHK,1125.0,2.0,32.49,2,2888
Brookefield,2 BHK,1230.0,2.0,55.0,2,4471
Mahadevpura,2 BHK,1438.0,2.0,89.0,2,6189
7th Phase JP Nagar,2 BHK,1080.0,2.0,75.0,2,6944
Hebbal Kempapura,3 Bedroom,1200.0,3.0,125.0,3,10416
Marathahalli,2 Bedroom,1200.0,1.0,103.0,2,8583
Uttarahalli,3 BHK,1590.0,3.0,58.0,3,3647
other,3 BHK,1350.0,3.0,68.0,3,5037
Raja Rajeshwari Nagar,3 BHK,1400.0,2.0,48.0,3,3428
Kaggadasapura,3 BHK,1560.0,2.0,72.0,3,4615
Mallasandra,4 BHK,1810.0,3.0,57.0,4,3149
other,3 BHK,2274.24,4.0,160.0,3,7035
Sonnenahalli,3 BHK,1484.0,3.0,74.18,3,4998
Indira Nagar,3 BHK,1650.0,3.0,220.0,3,13333
other,3 BHK,1455.0,2.0,55.0,3,3780
Thanisandra,3 BHK,2280.0,4.0,184.0,3,8070
other,5 BHK,4900.0,5.0,980.0,5,20000
Frazer Town,3 BHK,2079.0,3.0,220.0,3,10582
Bharathi Nagar,2 BHK,1351.0,2.0,74.0,2,5477
Kanakpura Road,3 BHK,1223.0,2.0,42.81,3,3500
KR Puram,3 BHK,1430.0,2.0,61.0,3,4265
Whitefield,4 Bedroom,3589.0,5.0,400.0,4,11145
Electronic City Phase II,3 BHK,1310.0,2.0,37.83,3,2887
Channasandra,4 Bedroom,1800.0,4.0,300.0,4,16666
Giri Nagar,4 Bedroom,1120.0,4.0,175.0,4,15625
Sarjapur  Road,2 BHK,984.0,2.0,55.0,2,5589
Jalahalli,3 BHK,1770.0,3.0,100.0,3,5649
other,2 Bedroom,900.0,3.0,105.0,2,11666
Ramamurthy Nagar,5 Bedroom,1350.0,5.0,120.0,5,8888
Bhoganhalli,2 BHK,1447.0,2.0,75.97,2,5250
Margondanahalli,2 Bedroom,1140.0,2.0,60.0,2,5263
Bommanahalli,2 BHK,1170.0,2.0,34.0,2,2905
Poorna Pragna Layout,2 BHK,1160.0,2.0,46.39,2,3999
Kanakpura Road,2 BHK,1296.0,2.0,89.0,2,6867
other,3 BHK,1570.0,3.0,80.0,3,5095
other,3 BHK,1495.0,2.0,65.0,3,4347
Hoodi,3 BHK,1530.0,2.0,67.0,3,4379
Rajaji Nagar,5 Bedroom,900.0,4.0,140.0,5,15555
Gunjur,3 BHK,1800.0,3.0,70.0,3,3888
Bellandur,2 BHK,1088.0,2.0,48.0,2,4411
Hosa Road,3 BHK,1533.0,3.0,98.18,3,6404
Rachenahalli,2 BHK,1100.0,2.0,68.0,2,6181
Thubarahalli,3 BHK,1540.0,3.0,90.0,3,5844
Hebbal,2 BHK,1244.0,3.0,48.0,2,3858
Hosa Road,2 BHK,1040.0,2.0,68.59,2,6595
other,6 BHK,1034.0,6.0,101.0,6,9767
Begur Road,2 BHK,1260.0,2.0,39.69,2,3150
Uttarahalli,2 BHK,1180.0,2.0,41.3,2,3499
Thigalarapalya,2 BHK,1245.0,2.0,100.0,2,8032
other,3 BHK,1350.0,3.0,75.0,3,5555
other,3 BHK,1464.0,3.0,56.0,3,3825
other,1 BHK,470.0,2.0,10.0,1,2127
Vijayanagar,2 BHK,920.0,2.0,60.0,2,6521
other,6 Bedroom,2300.0,7.0,160.0,6,6956
Sarjapur,2 BHK,1124.0,2.0,30.35,2,2700
Yelahanka New Town,4 BHK,2095.0,3.0,109.0,4,5202
Begur Road,2 BHK,1100.0,2.0,71.42,2,6492
Vijayanagar,2 BHK,1019.0,2.0,58.0,2,5691
Harlur,3 BHK,1756.0,3.0,128.0,3,7289
Bommasandra,3 BHK,1280.0,3.0,50.0,3,3906
Kenchenahalli,4 Bedroom,1200.0,3.0,125.0,4,10416
Indira Nagar,3 BHK,1700.0,3.0,150.0,3,8823
Whitefield,3 BHK,2134.0,4.0,118.0,3,5529
Hosa Road,2 BHK,1369.1,2.0,98.0,2,7157
Subramanyapura,3 Bedroom,600.0,3.0,69.0,3,11500
other,3 Bedroom,1313.0,4.0,150.0,3,11424
other,3 BHK,2515.0,4.0,155.0,3,6163
Thigalarapalya,3 BHK,2215.0,4.0,154.0,3,6952
other,3 BHK,1440.0,3.0,72.0,3,5000
Hennur Road,3 BHK,1186.0,2.0,55.33,3,4665
Uttarahalli,3 BHK,1350.0,2.0,47.24,3,3499
Hebbal,3 BHK,1800.0,3.0,140.0,3,7777
Kudlu,3 BHK,1600.0,3.0,79.0,3,4937
GM Palaya,2 BHK,1000.0,2.0,35.0,2,3500
Cunningham Road,3 BHK,2275.0,3.0,285.0,3,12527
Banashankari Stage III,3 BHK,1305.0,2.0,59.0,3,4521
Binny Pete,3 BHK,1740.0,3.0,150.0,3,8620
Malleshwaram,2 BHK,900.0,2.0,90.0,2,10000
Benson Town,2 BHK,1567.0,3.0,140.0,2,8934
Kambipura,3 BHK,1082.0,2.0,45.5,3,4205
Nagarbhavi,2 BHK,1050.0,2.0,80.0,2,7619
other,2 BHK,1200.0,2.0,56.0,2,4666
Begur,3 Bedroom,3500.0,4.0,185.0,3,5285
Dasarahalli,2 BHK,1220.0,2.0,52.0,2,4262
Binny Pete,3 BHK,1970.0,3.0,164.0,3,8324
other,3 BHK,2665.0,3.0,350.0,3,13133
Kanakapura,4 BHK,2130.0,4.0,120.0,4,5633
Harlur,3 BHK,1749.0,3.0,120.0,3,6861
Subramanyapura,2 BHK,1260.0,2.0,108.0,2,8571
6th Phase JP Nagar,2 BHK,1170.0,2.0,120.0,2,10256
other,3 BHK,1670.0,3.0,75.0,3,4491
Hennur Road,3 BHK,2264.0,3.0,168.0,3,7420
other,4 BHK,3500.0,5.0,450.0,4,12857
Begur,2 BHK,1100.0,2.0,55.0,2,5000
Chandapura,2 BHK,674.0,1.0,19.9,2,2952
Koramangala,2 BHK,1320.0,2.0,160.0,2,12121
Vishveshwarya Layout,4 Bedroom,600.0,4.0,85.0,4,14166
Electronics City Phase 1,2 BHK,1125.0,2.0,65.0,2,5777
Nagavara,2 BHK,1125.0,2.0,44.5,2,3955
other,2 BHK,700.0,2.0,29.0,2,4142
BTM Layout,3 BHK,1300.0,2.0,75.0,3,5769
JP Nagar,3 BHK,1600.0,3.0,83.0,3,5187
Magadi Road,3 BHK,1052.0,2.0,60.0,3,5703
other,4 BHK,1140.0,3.0,145.0,4,12719
Devarachikkanahalli,2 BHK,1130.0,2.0,36.0,2,3185
other,4 Bedroom,1140.0,3.0,225.0,4,19736
BTM 2nd Stage,3 BHK,2200.0,3.0,220.0,3,10000
Electronic City,3 BHK,1500.0,2.0,63.0,3,4200
Old Madras Road,3 BHK,1350.0,3.0,47.25,3,3500
Gottigere,3 BHK,1230.0,2.0,52.0,3,4227
Haralur Road,3 BHK,1520.0,2.0,85.0,3,5592
Kanakpura Road,1 BHK,525.0,1.0,26.25,1,5000
Hosa Road,2 BHK,1040.0,2.0,61.0,2,5865
other,3 BHK,1300.0,2.0,67.0,3,5153
Old Airport Road,4 BHK,2690.0,4.0,201.0,4,7472
Domlur,1 BHK,780.0,1.0,70.0,1,8974
Panathur,2 BHK,1140.0,2.0,51.2,2,4491
Chandapura,3 BHK,1225.0,3.0,33.08,3,2700
Nagavarapalya,1 BHK,646.0,1.0,25.84,1,4000
Cox Town,3 BHK,1600.0,3.0,150.0,3,9375
Kadugodi,3 BHK,1890.0,4.0,125.0,3,6613
Kadugodi,2 BHK,1050.0,2.0,34.0,2,3238
other,6 Bedroom,1200.0,4.0,100.0,6,8333
Whitefield,3 BHK,1458.0,2.0,62.0,3,4252
other,5 Bedroom,1200.0,6.0,86.0,5,7166
Green Glen Layout,3 BHK,1970.0,3.0,160.0,3,8121
Whitefield,3 Bedroom,1808.0,4.0,80.0,3,4424
Balagere,2 BHK,1007.0,2.0,67.0,2,6653
Doddathoguru,2 BHK,1105.0,2.0,50.0,2,4524
Lakshminarayana Pura,3 BHK,1700.0,3.0,150.0,3,8823
Raja Rajeshwari Nagar,2 BHK,1170.0,2.0,50.0,2,4273
Chandapura,2 BHK,1080.0,2.0,45.0,2,4166
Kaggalipura,3 BHK,1150.0,2.0,62.0,3,5391
Ramamurthy Nagar,2 BHK,1200.0,2.0,57.9,2,4825
other,2 BHK,1200.0,2.0,54.0,2,4500
Kudlu Gate,3 BHK,1656.0,3.0,75.0,3,4528
other,4 Bedroom,600.0,4.0,87.0,4,14500
KR Puram,2 BHK,1040.0,2.0,50.0,2,4807
Hosur Road,2 BHK,1085.0,2.0,33.7,2,3105
other,3 BHK,1215.0,2.0,49.75,3,4094
NRI Layout,4 Bedroom,2150.0,4.0,115.0,4,5348
Hoodi,4 Bedroom,1100.0,4.0,120.0,4,10909
Panathur,2 BHK,1210.0,2.0,77.0,2,6363
other,2 BHK,1050.0,2.0,55.0,2,5238
other,3 Bedroom,600.0,3.0,70.0,3,11666
Iblur Village,3 BHK,1920.0,3.0,150.0,3,7812
Uttarahalli,3 BHK,1250.0,2.0,50.0,3,4000
Mysore Road,2 BHK,947.55,2.0,80.0,2,8442
Malleshpalya,2 BHK,1421.0,2.0,90.0,2,6333
other,6 Bedroom,1050.0,4.0,155.0,6,14761
Old Madras Road,3 BHK,1425.0,2.0,74.31,3,5214
Budigere,2 BHK,1162.0,2.0,58.0,2,4991
Whitefield,1 BHK,905.0,1.0,60.0,1,6629
Sarjapur  Road,2 BHK,914.0,2.0,32.0,2,3501
other,2 BHK,1258.0,2.0,58.0,2,4610
CV Raman Nagar,4 Bedroom,1100.0,4.0,220.0,4,20000
Hormavu,4 Bedroom,3500.0,4.0,289.0,4,8257
Poorna Pragna Layout,7 Bedroom,2400.0,6.0,450.0,7,18750
Ramagondanahalli,4 Bedroom,4200.0,4.0,800.0,4,19047
Whitefield,3 BHK,1537.0,2.0,70.0,3,4554
Chikkalasandra,3 BHK,1230.0,2.0,53.51,3,4350
other,6 Bedroom,2400.0,5.0,300.0,6,12500
Malleshwaram,3 BHK,2250.0,3.0,200.0,3,8888
Hosur Road,4 Bedroom,2000.0,4.0,69.0,4,3450
other,8 BHK,3000.0,4.0,130.0,8,4333
Kumaraswami Layout,4 Bedroom,2600.0,4.0,325.0,4,12500
Indira Nagar,4 Bedroom,3200.0,3.0,250.0,4,7812
Anjanapura,2 BHK,950.0,2.0,32.0,2,3368
Marathahalli,4 BHK,4000.0,4.0,200.0,4,5000
Yelahanka,3 BHK,1282.0,2.0,48.72,3,3800
Banashankari Stage III,2 BHK,1304.0,2.0,111.0,2,8512
7th Phase JP Nagar,3 BHK,2100.0,3.0,200.0,3,9523
Jakkur,3 BHK,1220.0,2.0,49.0,3,4016
Budigere,3 BHK,1636.0,3.0,85.0,3,5195
7th Phase JP Nagar,3 BHK,1415.0,3.0,65.0,3,4593
Green Glen Layout,3 BHK,1728.0,3.0,125.0,3,7233
7th Phase JP Nagar,3 BHK,1680.0,3.0,125.0,3,7440
other,2 BHK,1235.0,2.0,140.0,2,11336
Yelachenahalli,4 Bedroom,900.0,2.0,115.0,4,12777
Yelahanka,4 Bedroom,3990.0,4.0,260.0,4,6516
Kasavanhalli,2 BHK,1495.0,2.0,83.0,2,5551
Hennur Road,5 Bedroom,1500.0,5.0,175.0,5,11666
Yeshwanthpur,3 BHK,2501.0,3.0,138.0,3,5517
Kanakpura Road,1 BHK,381.0,1.0,28.0,1,7349
other,4 BHK,3301.8,5.0,570.0,4,17263
Hennur Road,3 BHK,2002.0,3.0,120.0,3,5994
other,3 Bedroom,3200.0,3.0,260.0,3,8125
Banjara Layout,2 Bedroom,1200.0,2.0,70.0,2,5833
Sarjapur  Road,4 Bedroom,1500.0,3.0,235.0,4,15666
Bommasandra,3 Bedroom,1942.0,3.0,90.0,3,4634
other,1 Bedroom,660.0,1.0,95.0,1,14393
Banashankari Stage III,4 Bedroom,1200.0,6.0,170.0,4,14166
other,3 BHK,1450.0,3.0,90.0,3,6206
Electronic City,2 BHK,1039.0,2.0,50.0,2,4812
Old Madras Road,2 BHK,1210.0,2.0,80.0,2,6611
Attibele,2 BHK,995.0,1.0,24.88,2,2500
Yelahanka,3 BHK,1620.0,3.0,90.0,3,5555
Choodasandra,3 Bedroom,3204.0,4.0,395.0,3,12328
Hebbal Kempapura,3 BHK,3408.0,3.0,260.0,3,7629
Marathahalli,2 BHK,1314.0,2.0,55.0,2,4185
Bisuvanahalli,2 BHK,945.0,2.0,33.0,2,3492
Nehru Nagar,3 BHK,1374.0,3.0,72.0,3,5240
Kanakpura Road,3 BHK,1603.0,3.0,113.0,3,7049
Kengeri,3 BHK,1230.0,2.0,52.0,3,4227
Basavangudi,1 RK,670.0,1.0,50.0,1,7462
Sarjapur  Road,4 BHK,1864.0,3.0,105.0,4,5633
Hosa Road,3 Bedroom,2500.0,3.0,100.0,3,4000
Raja Rajeshwari Nagar,3 BHK,1625.0,2.0,54.0,3,3323
Whitefield,2 BHK,1186.0,2.0,55.0,2,4637
KR Puram,8 Bedroom,1500.0,7.0,382.0,8,25466
Electronic City,2 BHK,770.0,1.0,39.9,2,5181
Varthur,2 BHK,1085.0,2.0,45.0,2,4147
other,4 BHK,3754.0,6.0,480.0,4,12786
other,3 BHK,1893.0,3.0,115.0,3,6075
other,4 Bedroom,600.0,3.0,65.0,4,10833
Kundalahalli,3 BHK,1496.0,2.0,78.0,3,5213
Chandapura,3 BHK,1305.0,3.0,33.28,3,2550
Electronics City Phase 1,3 BHK,1550.0,2.0,82.0,3,5290
Marathahalli,3 BHK,1200.0,2.0,58.17,3,4847
Badavala Nagar,3 BHK,1494.0,2.0,94.55,3,6328
Electronic City Phase II,2 BHK,1286.0,2.0,69.0,2,5365
other,3 Bedroom,600.0,4.0,100.0,3,16666
Hebbal,3 BHK,1860.0,3.0,95.0,3,5107
other,1 BHK,425.0,1.0,750.0,1,176470
Haralur Road,2 BHK,1243.0,2.0,45.0,2,3620
Brookefield,2 BHK,1200.0,2.0,75.0,2,6250
Sarjapur,2 BHK,1044.0,2.0,32.0,2,3065
7th Phase JP Nagar,2 BHK,1040.0,2.0,41.59,2,3999
8th Phase JP Nagar,2 BHK,1125.0,2.0,36.94,2,3283
Cunningham Road,3 BHK,3489.0,3.0,662.0,3,18973
Subramanyapura,3 BHK,1278.0,2.0,59.0,3,4616
other,6 BHK,2400.0,6.0,98.0,6,4083
other,2 BHK,1250.0,2.0,43.0,2,3440
Vijayanagar,5 Bedroom,2400.0,6.0,125.0,5,5208
KR Puram,5 Bedroom,4166.0,5.0,560.0,5,13442
Lakshminarayana Pura,2 BHK,1210.0,2.0,75.0,2,6198
8th Phase JP Nagar,4 BHK,1200.0,3.0,110.0,4,9166
Sarjapur  Road,3 BHK,1157.0,2.0,57.84,3,4999
Raja Rajeshwari Nagar,2 BHK,1185.0,2.0,51.15,2,4316
Kalena Agrahara,3 BHK,1450.0,3.0,100.0,3,6896
Old Madras Road,3 BHK,1859.0,3.0,121.0,3,6508
Rajaji Nagar,2 BHK,1763.0,3.0,262.0,2,14861
Attibele,2 BHK,900.0,1.0,25.0,2,2777
Ramamurthy Nagar,3 BHK,1305.0,3.0,60.0,3,4597
other,3 BHK,1200.0,2.0,100.0,3,8333
Thanisandra,1 BHK,784.0,1.0,35.285,1,4500
Sarjapur  Road,2 BHK,1150.0,2.0,36.8,2,3199
Whitefield,3 BHK,1806.0,3.0,85.0,3,4706
Kudlu,2 BHK,1027.0,2.0,42.0,2,4089
Uttarahalli,2 BHK,1002.0,2.0,40.08,2,4000
Electronic City Phase II,3 BHK,1220.0,2.0,36.21,3,2968
Whitefield,2 BHK,1109.0,2.0,35.22,2,3175
Uttarahalli,2 BHK,1002.0,2.0,40.08,2,4000
Narayanapura,2 BHK,1308.0,2.0,89.04,2,6807
Hennur Road,2 BHK,1232.0,2.0,69.6,2,5649
Haralur Road,2 BHK,1027.0,2.0,44.0,2,4284
Kalyan nagar,2 BHK,1250.0,2.0,65.0,2,5200
other,6 Bedroom,900.0,6.0,170.0,6,18888
KR Puram,4 Bedroom,960.0,4.0,80.0,4,8333
Raja Rajeshwari Nagar,2 BHK,1255.0,2.0,53.87,2,4292
Akshaya Nagar,3 BHK,1434.0,3.0,84.0,3,5857
Doddathoguru,2 BHK,925.0,2.0,30.0,2,3243
Jakkur,3 BHK,1858.0,3.0,123.0,3,6620
other,2 BHK,1205.0,2.0,70.0,2,5809
other,3 Bedroom,1500.0,3.0,90.0,3,6000
other,3 Bedroom,1200.0,3.0,200.0,3,16666
Mysore Road,2 BHK,1020.0,2.0,48.95,2,4799
Thanisandra,2 BHK,965.0,2.0,56.0,2,5803
7th Phase JP Nagar,5 Bedroom,1800.0,6.0,270.0,5,15000
other,6 Bedroom,1200.0,4.0,90.0,6,7500
Battarahalli,2 BHK,1636.0,2.0,82.89,2,5066
Kundalahalli,3 BHK,1724.0,3.0,125.0,3,7250
Whitefield,2 BHK,1216.0,2.0,67.0,2,5509
Yelahanka,1 BHK,654.0,1.0,36.25,1,5542
Chikkabanavar,4 Bedroom,2460.0,7.0,80.0,4,3252
other,2 BHK,1190.0,2.0,41.0,2,3445
other,7 Bedroom,1200.0,5.0,180.0,7,15000
Budigere,3 BHK,1636.0,3.0,92.0,3,5623
Hennur Road,2 BHK,1195.0,2.0,73.0,2,6108
Nagavara,3 BHK,2400.0,3.0,251.0,3,10458
other,2 BHK,1250.0,2.0,67.0,2,5360
Whitefield,3 BHK,1452.0,3.0,90.0,3,6198
Thanisandra,3 BHK,1546.0,3.0,95.0,3,6144
Sarjapur  Road,4 Bedroom,3385.5,6.0,142.0,4,4194
Indira Nagar,3 Bedroom,1440.0,3.0,275.0,3,19097
other,2 Bedroom,2260.0,3.0,80.0,2,3539
other,4 Bedroom,2700.0,3.0,90.0,4,3333
Indira Nagar,2 BHK,1475.0,2.0,171.0,2,11593
JP Nagar,4 BHK,3000.0,3.0,150.0,4,5000
other,1 Bedroom,375.0,1.0,26.0,1,6933
Balagere,2 BHK,1007.0,2.0,66.0,2,6554
other,4 BHK,5422.0,6.0,1900.0,4,35042
Hennur Road,2 BHK,1020.0,2.0,47.0,2,4607
other,2 BHK,1000.0,2.0,150.0,2,15000
other,2 Bedroom,800.0,3.0,59.5,2,7437
other,5 Bedroom,1200.0,3.0,95.0,5,7916
Marathahalli,2 BHK,1125.0,2.0,55.0,2,4888
Kudlu Gate,9 Bedroom,1150.0,9.0,140.0,9,12173
other,3 BHK,909.0,2.0,43.0,3,4730
Whitefield,4 Bedroom,2850.0,5.0,240.0,4,8421
other,2 BHK,1400.0,2.0,85.0,2,6071
Kadugodi,2 BHK,1152.0,2.0,61.0,2,5295
Subramanyapura,2 BHK,958.0,2.0,34.49,2,3600
Whitefield,2 BHK,1150.0,2.0,55.0,2,4782
Old Madras Road,3 BHK,2990.0,5.0,173.0,3,5785
Bellandur,2 BHK,1310.0,2.0,85.0,2,6488
Hosur Road,3 BHK,1590.0,2.0,126.0,3,7924
Green Glen Layout,3 BHK,1645.0,3.0,110.0,3,6686
other,3 Bedroom,1800.0,2.0,100.0,3,5555
Thanisandra,3 BHK,1795.0,3.0,150.0,3,8356
8th Phase JP Nagar,2 BHK,1100.0,2.0,35.0,2,3181
Thanisandra,1 BHK,777.0,1.0,38.535,1,4959
other,2 BHK,925.0,2.0,70.0,2,7567
Basavangudi,3 BHK,1542.14,3.0,120.0,3,7781
Konanakunte,4 Bedroom,3746.0,6.0,375.0,4,10010
Thanisandra,3 BHK,1430.0,2.0,54.0,3,3776
Bommanahalli,2 BHK,1020.0,2.0,55.0,2,5392
other,4 Bedroom,750.0,2.0,42.0,4,5600
Electronic City Phase II,2 BHK,1140.0,2.0,32.92,2,2887
Anekal,3 BHK,967.0,2.0,45.0,3,4653
Hebbal,4 BHK,2470.0,4.0,203.0,4,8218
other,2 BHK,850.0,1.0,35.5,2,4176
other,2 BHK,1100.0,2.0,32.0,2,2909
Thanisandra,3 BHK,1697.0,3.0,110.0,3,6482
Varthur,2 BHK,1250.0,2.0,47.0,2,3760
Gottigere,2 BHK,1120.0,2.0,50.0,2,4464
NRI Layout,4 Bedroom,1200.0,4.0,95.0,4,7916
Hennur Road,5 Bedroom,5100.0,6.0,375.0,5,7352
other,4 Bedroom,2400.0,3.0,480.0,4,20000
Hebbal,2 BHK,1200.0,2.0,52.0,2,4333
7th Phase JP Nagar,2 BHK,1100.0,2.0,44.0,2,4000
Devarachikkanahalli,3 BHK,1417.0,2.0,76.0,3,5363
Thanisandra,3 BHK,1492.0,3.0,88.0,3,5898
Devanahalli,3 BHK,1498.0,3.0,79.15,3,5283
other,4 BHK,3500.0,2.0,350.0,4,10000
Kundalahalli,2 BHK,1010.0,2.0,48.27,2,4779
Hennur,2 BHK,1255.0,2.0,53.5,2,4262
Channasandra,2 BHK,1050.0,2.0,35.0,2,3333
Hegde Nagar,3 BHK,1847.0,3.0,125.0,3,6767
other,2 BHK,1200.0,2.0,55.0,2,4583
Bellandur,2 BHK,982.0,2.0,37.28,2,3796
Hegde Nagar,3 BHK,2144.6,4.0,145.0,3,6761
Kengeri,2 BHK,750.0,2.0,36.5,2,4866
Electronic City,2 BHK,919.0,2.0,35.0,2,3808
Budigere,2 BHK,1087.0,2.0,54.3,2,4995
other,3 BHK,1480.0,3.0,74.0,3,5000
Tindlu,2 BHK,1180.0,2.0,52.0,2,4406
other,3 BHK,2100.0,4.0,255.0,3,12142
Jigani,3 BHK,1221.0,3.0,65.0,3,5323
Haralur Road,2 BHK,1194.0,2.0,46.0,2,3852
Thanisandra,3 BHK,2072.0,3.0,122.0,3,5888
Whitefield,3 BHK,1600.0,3.0,69.0,3,4312
Sahakara Nagar,3 BHK,2200.0,3.0,98.0,3,4454
Rajaji Nagar,3 BHK,1640.0,3.0,220.0,3,13414
Kaikondrahalli,2 BHK,1253.0,2.0,81.5,2,6504
Hennur Road,2 BHK,1061.0,2.0,54.92,2,5176
Marathahalli,3 BHK,1950.0,3.0,115.0,3,5897
Marathahalli,2 BHK,1250.0,2.0,68.0,2,5440
Hennur Road,3 BHK,1585.0,2.0,80.0,3,5047
Kanakpura Road,3 BHK,1452.0,3.0,60.98,3,4199
Kammasandra,2 BHK,1015.0,2.0,37.0,2,3645
Begur Road,4 BHK,2464.5,6.0,118.0,4,4787
Marathahalli,2 BHK,1125.0,2.0,56.8,2,5048
Electronic City Phase II,2 BHK,1116.0,2.0,40.0,2,3584
other,1 Bedroom,400.0,1.0,35.0,1,8750
Kasavanhalli,4 BHK,4000.0,4.0,325.0,4,8125
Hebbal,4 BHK,3900.0,4.0,410.0,4,10512
Old Madras Road,2 BHK,1300.0,2.0,100.0,2,7692
Raja Rajeshwari Nagar,2 BHK,1419.0,2.0,60.42,2,4257
other,3 BHK,1310.0,2.0,55.0,3,4198
Thigalarapalya,3 BHK,2072.0,4.0,152.0,3,7335
Kaggadasapura,2 BHK,1240.0,2.0,58.58,2,4724
Chamrajpet,3 BHK,1565.0,3.0,98.91,3,6320
Budigere,3 BHK,1820.0,3.0,89.0,3,4890
Gubbalala,3 Bedroom,2500.0,3.0,88.0,3,3520
KR Puram,5 Bedroom,3657.0,5.0,310.0,5,8476
Hosa Road,2 BHK,1245.0,2.0,62.0,2,4979
GM Palaya,3 BHK,1315.0,3.0,65.0,3,4942
Kenchenahalli,3 BHK,1410.0,2.0,73.0,3,5177
Dasarahalli,2 BHK,1375.0,2.0,60.0,2,4363
Old Madras Road,2 BHK,1299.0,2.0,95.0,2,7313
other,2 BHK,993.0,2.0,45.5,2,4582
Bommasandra Industrial Area,2 Bedroom,1200.0,3.0,45.0,2,3750
other,2 BHK,985.0,2.0,156.0,2,15837
other,4 Bedroom,2400.0,5.0,180.0,4,7500
Hebbal,3 BHK,1255.0,2.0,95.0,3,7569
Hoodi,3 BHK,1655.0,3.0,131.0,3,7915
Whitefield,4 BHK,2830.0,5.0,190.0,4,6713
other,1 BHK,6000.0,1.0,276.0,1,4600
Electronic City,2 BHK,550.0,1.0,15.0,2,2727
other,1 BHK,825.0,2.0,40.0,1,4848
Begur Road,3 BHK,1634.0,3.0,111.0,3,6793
Kogilu,8 Bedroom,1200.0,6.0,105.0,8,8750
Banaswadi,2 BHK,1222.0,2.0,80.0,2,6546
Gottigere,3 BHK,1500.0,3.0,63.0,3,4200
Akshaya Nagar,3 BHK,1890.0,4.0,100.0,3,5291
Hegde Nagar,4 Bedroom,3750.0,6.0,375.0,4,10000
Hulimavu,2 BHK,850.0,1.0,39.5,2,4647
Whitefield,2 BHK,1295.0,2.0,78.5,2,6061
Sarjapur  Road,2 BHK,1320.0,2.0,100.0,2,7575
ISRO Layout,4 Bedroom,950.0,3.0,179.0,4,18842
Hebbal,4 Bedroom,1200.0,4.0,225.0,4,18750
2nd Stage Nagarbhavi,4 Bedroom,3000.0,3.0,170.0,4,5666
Harlur,2 BHK,1197.0,2.0,82.0,2,6850
other,3 BHK,1320.0,2.0,60.0,3,4545
Billekahalli,5 Bedroom,1180.0,5.0,98.5,5,8347
Whitefield,1 BHK,840.0,1.0,57.6,1,6857
Whitefield,1 BHK,640.0,1.0,19.83,1,3098
Raja Rajeshwari Nagar,3 BHK,1693.0,3.0,71.38,3,4216
Kodihalli,2 BHK,1000.0,2.0,62.0,2,6200
Babusapalaya,2 BHK,1305.0,2.0,70.0,2,5363
Kodichikkanahalli,1 BHK,450.0,1.0,18.0,1,4000
CV Raman Nagar,1 BHK,705.0,1.0,57.0,1,8085
other,4 BHK,750.0,6.0,75.0,4,10000
Tumkur Road,5 Bedroom,4800.0,5.0,150.0,5,3125
other,4 Bedroom,5000.0,5.0,600.0,4,12000
Gubbalala,4 Bedroom,5000.0,4.0,425.0,4,8500
Whitefield,3 BHK,1991.0,3.0,104.0,3,5223
Battarahalli,3 BHK,1779.0,3.0,90.0,3,5059
Whitefield,4 Bedroom,10200.0,4.0,1250.0,4,12254
Banaswadi,2 BHK,870.0,2.0,38.0,2,4367
Thanisandra,3 BHK,1595.0,3.0,120.0,3,7523
Kanakpura Road,1 BHK,525.0,1.0,26.0,1,4952
Ramamurthy Nagar,4 Bedroom,945.0,5.0,120.0,4,12698
other,6 Bedroom,2400.0,4.0,420.0,6,17500
Iblur Village,3 BHK,2000.0,4.0,138.0,3,6900
Yelahanka,3 BHK,1614.0,3.0,93.0,3,5762
Gottigere,2 BHK,1200.0,2.0,40.0,2,3333
Electronic City Phase II,3 BHK,1150.0,3.0,46.0,3,4000
Indira Nagar,4 BHK,3200.0,4.0,440.0,4,13750
Sonnenahalli,3 BHK,1610.0,3.0,80.48,3,4998
Whitefield,2 BHK,1190.0,2.0,41.0,2,3445
other,2 BHK,1050.0,2.0,50.0,2,4761
other,4 Bedroom,4800.0,5.0,420.0,4,8750
7th Phase JP Nagar,3 BHK,1460.0,2.0,97.0,3,6643
other,3 BHK,3400.0,3.0,400.0,3,11764
other,1 Bedroom,600.0,1.0,33.0,1,5500
other,3 BHK,1569.0,3.0,70.0,3,4461
other,3 Bedroom,2600.0,3.0,147.0,3,5653
Electronics City Phase 1,1 BHK,645.0,1.0,42.5,1,6589
other,4 Bedroom,1200.0,3.0,180.0,4,15000
Nagarbhavi,3 Bedroom,800.0,3.0,95.0,3,11875
Mico Layout,2 Bedroom,800.0,2.0,80.0,2,10000
Marathahalli,2 BHK,1350.0,2.0,90.0,2,6666
BTM 2nd Stage,3 BHK,2780.0,4.0,325.0,3,11690
other,3 BHK,1410.0,2.0,45.12,3,3200
Whitefield,2 BHK,1109.0,2.0,35.47,2,3198
other,4 BHK,2000.0,3.0,1063.0,4,53150
other,1 BHK,800.0,1.0,72.0,1,9000
Bannerghatta Road,2 BHK,1215.0,2.0,68.0,2,5596
Basaveshwara Nagar,8 Bedroom,1230.0,6.0,250.0,8,20325
Prithvi Layout,3 Bedroom,2273.0,4.0,192.0,3,8446
Varthur Road,2 BHK,1255.0,2.0,52.76,2,4203
other,2 BHK,1020.0,2.0,49.0,2,4803
other,3 BHK,1225.0,2.0,82.0,3,6693
Singasandra,2 BHK,1120.0,2.0,60.0,2,5357
other,5 Bedroom,2400.0,5.0,625.0,5,26041
Kasavanhalli,3 BHK,1225.0,2.0,69.0,3,5632
Whitefield,4 Bedroom,5200.0,5.0,242.0,4,4653
Indira Nagar,4 BHK,4000.0,4.0,700.0,4,17500
Sarjapur  Road,3 BHK,1984.0,4.0,145.0,3,7308
Old Madras Road,3 BHK,2425.0,3.0,197.0,3,8123
ITPL,3 Bedroom,1200.0,3.0,56.12,3,4676
Bommasandra,2 BHK,1127.0,2.0,45.0,2,3992
Whitefield,3 BHK,1655.0,3.0,97.0,3,5861
Ananth Nagar,1 BHK,500.0,1.0,14.0,1,2800
Varthur,3 Bedroom,1200.0,3.0,68.38,3,5698
Kaggadasapura,2 BHK,1000.0,2.0,43.0,2,4300
CV Raman Nagar,2 BHK,1225.0,2.0,47.88,2,3908
Hormavu,2 BHK,1350.0,2.0,45.0,2,3333
Nagarbhavi,4 Bedroom,600.0,5.0,95.0,4,15833
Sahakara Nagar,2 BHK,1175.0,2.0,49.0,2,4170
other,1 BHK,655.0,1.0,27.0,1,4122
Hosa Road,3 BHK,1311.0,2.0,84.86,3,6472
Banashankari Stage VI,3 Bedroom,600.0,4.0,90.0,3,15000
CV Raman Nagar,2 BHK,1000.0,2.0,38.5,2,3850
other,2 BHK,1196.0,2.0,96.0,2,8026
Kanakpura Road,3 BHK,1050.0,2.0,59.0,3,5619
Somasundara Palya,3 BHK,1275.0,2.0,52.0,3,4078
other,2 BHK,927.0,2.0,50.0,2,5393
Varthur,3 BHK,1520.0,2.0,61.0,3,4013
other,4 Bedroom,2400.0,5.0,395.0,4,16458
Hebbal,2 BHK,1333.0,2.0,104.0,2,7801
other,1 BHK,600.0,1.0,18.0,1,3000
Abbigere,3 BHK,1326.0,2.0,34.8,3,2624
Koramangala,2 BHK,1370.0,2.0,125.0,2,9124
Koramangala,2 BHK,1100.0,2.0,85.0,2,7727
Haralur Road,3 BHK,1650.0,3.0,100.0,3,6060
Kasavanhalli,3 BHK,1917.0,3.0,132.0,3,6885
other,3 BHK,780.0,3.0,75.0,3,9615
Billekahalli,2 BHK,950.0,2.0,58.11,2,6116
other,2 BHK,1175.0,2.0,48.72,2,4146
8th Phase JP Nagar,2 BHK,1089.0,2.0,43.55,2,3999
Kalena Agrahara,3 BHK,1804.0,3.0,175.0,3,9700
Hoodi,2 BHK,1257.0,2.0,68.5,2,5449
Sarjapur,1 BHK,650.0,1.0,19.87,1,3056
Whitefield,3 BHK,1850.0,3.0,150.0,3,8108
other,4 Bedroom,2800.0,4.0,200.0,4,7142
Vidyaranyapura,2 BHK,945.0,2.0,40.0,2,4232
Koramangala,2 BHK,1161.0,2.0,63.85,2,5499
Banashankari Stage V,3 BHK,1630.0,3.0,51.35,3,3150
Jalahalli,3 BHK,1310.0,2.0,60.0,3,4580
other,2 BHK,750.0,2.0,28.0,2,3733
Whitefield,3 Bedroom,1200.0,3.0,67.77,3,5647
other,2 BHK,1000.0,2.0,45.0,2,4500
Old Madras Road,4 Bedroom,4000.0,4.0,225.0,4,5625
Nagavara,2 BHK,1315.0,2.0,70.0,2,5323
Sahakara Nagar,3 BHK,1370.0,2.0,90.0,3,6569
Ulsoor,5 Bedroom,1200.0,4.0,150.0,5,12500
Bellandur,2 BHK,1465.0,3.0,90.0,2,6143
JP Nagar,3 BHK,20000.0,3.0,175.0,3,875
Electronic City,2 BHK,1128.0,2.0,64.0,2,5673
Padmanabhanagar,3 BHK,1250.0,2.0,70.0,3,5600
Hebbal,3 BHK,1645.0,3.0,117.0,3,7112
Horamavu Agara,3 BHK,1284.0,2.0,75.0,3,5841
Hebbal,3 BHK,2600.0,3.0,199.0,3,7653
Kodigehalli,3 BHK,1820.0,3.0,150.0,3,8241
Haralur Road,3 BHK,1580.0,3.0,73.0,3,4620
Akshaya Nagar,3 Bedroom,2650.0,3.0,125.0,3,4716
Bannerghatta Road,2 BHK,1215.0,2.0,68.0,2,5596
Uttarahalli,2 BHK,1075.0,2.0,40.0,2,3720
R.T. Nagar,3 BHK,1400.0,2.0,78.0,3,5571
Subramanyapura,3 BHK,1400.0,2.0,55.0,3,3928
Bisuvanahalli,2 BHK,845.0,2.0,30.0,2,3550
Hegde Nagar,4 BHK,2600.0,4.0,40.0,4,1538
Sarjapur,4 Bedroom,3854.5,6.0,385.5,4,10001
other,2 BHK,800.0,2.0,85.0,2,10625
Panathur,2 BHK,1199.0,2.0,85.0,2,7089
Gunjur,2 BHK,1175.0,2.0,43.48,2,3700
Sahakara Nagar,3 Bedroom,2400.0,3.0,270.0,3,11250
Bellandur,3 BHK,1767.0,3.0,109.0,3,6168
Sarjapur  Road,4 BHK,1800.0,3.0,90.0,4,5000
Talaghattapura,3 BHK,1856.0,3.0,135.0,3,7273
R.T. Nagar,3 Bedroom,1200.0,3.0,140.0,3,11666
Rajaji Nagar,3 BHK,1613.0,3.0,150.0,3,9299
Jakkur,4 BHK,6830.0,5.0,795.0,4,11639
Whitefield,2 BHK,1290.0,2.0,90.0,2,6976
Kasavanhalli,2 BHK,1375.0,2.0,78.0,2,5672
other,2 BHK,985.0,2.0,57.0,2,5786
other,4 Bedroom,1200.0,4.0,200.0,4,16666
Singasandra,2 BHK,1099.0,2.0,37.5,2,3412
Electronic City,1 BHK,605.0,1.0,13.31,1,2200
other,11 BHK,6000.0,12.0,150.0,11,2500
other,7 Bedroom,3600.0,7.0,160.0,7,4444
Marathahalli,2 BHK,1200.0,2.0,53.0,2,4416
Hulimavu,2 BHK,1000.0,2.0,45.0,2,4500
Electronic City,3 BHK,1159.0,2.0,38.0,3,3278
Sarjapur  Road,5 BHK,3692.0,5.0,340.0,5,9209
other,2 Bedroom,600.0,2.0,80.0,2,13333
other,4 Bedroom,2400.0,4.0,600.0,4,25000
BTM Layout,3 BHK,1540.0,2.0,78.0,3,5064
other,3 BHK,1503.0,2.0,93.0,3,6187
Ambedkar Nagar,3 BHK,1850.0,4.0,120.0,3,6486
Uttarahalli,3 BHK,1580.0,3.0,60.0,3,3797
Mico Layout,2 BHK,1190.0,2.0,39.25,2,3298
other,2 Bedroom,1200.0,2.0,61.0,2,5083
HAL 2nd Stage,7 Bedroom,1000.0,7.0,250.0,7,25000
Jakkur,2 BHK,850.0,1.0,19.9,2,2341
Domlur,3 BHK,2180.0,3.0,285.0,3,13073
other,3 BHK,2500.0,2.0,60.0,3,2400
ITPL,3 Bedroom,1200.0,3.0,56.12,3,4676
other,3 BHK,1500.0,2.0,52.0,3,3466
other,5 Bedroom,1200.0,3.0,180.0,5,15000
Thanisandra,3 BHK,1880.0,3.0,120.0,3,6382
Padmanabhanagar,2 BHK,1150.0,2.0,45.0,2,3913
Kasavanhalli,4 Bedroom,4408.0,4.0,344.0,4,7803
Bellandur,2 BHK,924.0,2.0,35.11,2,3799
Sarjapur  Road,5 BHK,7800.0,6.0,385.0,5,4935
Kumaraswami Layout,6 Bedroom,600.0,6.0,100.0,6,16666
other,4 Bedroom,750.0,4.0,95.0,4,12666
Karuna Nagar,3 Bedroom,1962.0,3.0,175.0,3,8919
2nd Stage Nagarbhavi,5 Bedroom,1200.0,4.0,240.0,5,20000
other,3 BHK,1765.0,3.0,77.27,3,4377
Attibele,2 BHK,1174.0,2.0,29.35,2,2500
Bannerghatta Road,3 BHK,1894.0,3.0,115.0,3,6071
Ramamurthy Nagar,4 Bedroom,960.0,2.0,138.0,4,14375
Sector 7 HSR Layout,3 BHK,1760.0,3.0,142.0,3,8068
other,3 BHK,1768.0,3.0,95.0,3,5373
Kothanur,3 BHK,1790.0,3.0,97.0,3,5418
other,4 Bedroom,3500.0,5.0,800.0,4,22857
Whitefield,2 Bedroom,1200.0,2.0,45.0,2,3750
Kengeri,4 Bedroom,2540.0,4.0,160.0,4,6299
Mahadevpura,2 BHK,1137.0,2.0,53.4,2,4696
other,7 Bedroom,1200.0,6.0,115.0,7,9583
Mysore Road,3 BHK,1568.0,2.0,100.0,3,6377
Devanahalli,4 Bedroom,4920.0,4.0,393.5,4,7997
AECS Layout,2 BHK,1308.0,2.0,60.0,2,4587
other,4 BHK,3060.0,4.0,350.0,4,11437
Bisuvanahalli,3 BHK,1200.0,2.0,41.0,3,3416
other,1 BHK,800.0,1.0,32.0,1,4000
Electronic City,1 BHK,530.0,1.0,13.5,1,2547
Bisuvanahalli,3 BHK,873.0,2.0,40.0,3,4581
other,7 Bedroom,3000.0,6.0,145.0,7,4833
other,4 Bedroom,1200.0,4.0,235.0,4,19583
Subramanyapura,2 BHK,958.0,2.0,33.53,2,3500
Thigalarapalya,3 BHK,2072.0,4.0,160.0,3,7722
Talaghattapura,3 BHK,2273.0,3.0,159.0,3,6995
Banashankari Stage VI,3 Bedroom,600.0,3.0,90.0,3,15000
Thanisandra,1 BHK,760.0,1.0,50.0,1,6578
Dommasandra,2 BHK,1090.0,2.0,43.5,2,3990
Thigalarapalya,3 BHK,2072.0,4.0,155.0,3,7480
Bellandur,3 BHK,2666.0,3.0,230.0,3,8627
Sarjapur  Road,3 BHK,1550.0,2.0,89.0,3,5741
Ramagondanahalli,2 BHK,1251.0,2.0,50.0,2,3996
Electronic City,2 BHK,1070.0,2.0,53.0,2,4953
Kalena Agrahara,2 BHK,1200.0,2.0,45.0,2,3750
CV Raman Nagar,3 BHK,1435.0,2.0,60.0,3,4181
Bommasandra,3 BHK,1400.0,2.0,45.0,3,3214
Judicial Layout,4 BHK,2330.0,3.0,94.71,4,4064
Thanisandra,2 BHK,1200.0,2.0,60.0,2,5000
Panathur,2 BHK,1000.0,2.0,40.0,2,4000
Bannerghatta Road,3 BHK,1655.0,2.0,62.89,3,3800
Raja Rajeshwari Nagar,9 Bedroom,3600.0,9.0,240.0,9,6666
other,1 Bedroom,600.0,1.0,35.0,1,5833
Electronics City Phase 1,2 BHK,1175.0,2.0,50.0,2,4255
Uttarahalli,2 BHK,1155.0,2.0,40.43,2,3500
Magadi Road,3 Bedroom,1200.0,3.0,98.0,3,8166
Budigere,2 BHK,1149.0,2.0,65.0,2,5657
Choodasandra,4 Bedroom,3200.0,4.0,375.0,4,11718
other,2 BHK,1010.0,2.0,55.0,2,5445
Hegde Nagar,3 BHK,1718.0,3.0,119.0,3,6926
other,2 BHK,650.0,1.0,20.0,2,3076
Banashankari,2 BHK,1330.0,2.0,77.0,2,5789
Hormavu,2 BHK,1200.0,2.0,68.0,2,5666
Bannerghatta Road,3 BHK,1693.0,3.0,89.0,3,5256
Koramangala,2 BHK,1249.0,2.0,165.0,2,13210
Thanisandra,3 BHK,1800.0,4.0,125.0,3,6944
Margondanahalli,2 Bedroom,900.0,2.0,55.0,2,6111
NGR Layout,2 BHK,1020.0,2.0,46.0,2,4509
Amruthahalli,3 BHK,1700.0,3.0,77.0,3,4529
Panathur,1 BHK,661.0,1.0,40.0,1,6051
Bannerghatta Road,2 BHK,1280.0,2.0,60.16,2,4700
other,5 Bedroom,1000.0,4.0,90.0,5,9000
Hosur Road,2 BHK,1170.0,2.0,48.0,2,4102
Thanisandra,1 BHK,933.0,2.0,58.0,1,6216
Sarjapur  Road,2 BHK,1150.0,2.0,44.0,2,3826
Benson Town,4 BHK,3633.0,6.0,550.0,4,15139
Rajaji Nagar,3 BHK,1380.0,3.0,130.0,3,9420
Banjara Layout,2 Bedroom,1050.0,2.0,64.8,2,6171
other,2 BHK,1080.0,2.0,37.8,2,3499
Harlur,2 BHK,1303.0,2.0,90.0,2,6907
Raja Rajeshwari Nagar,2 BHK,1196.0,2.0,40.54,2,3389
Nagarbhavi,5 Bedroom,2500.0,3.0,180.0,5,7200
Sarjapur  Road,5 Bedroom,1200.0,4.0,95.0,5,7916
Bommanahalli,3 BHK,1250.0,3.0,41.0,3,3280
Akshaya Nagar,3 BHK,1419.0,2.0,87.0,3,6131
Banashankari,2 BHK,1200.0,2.0,49.0,2,4083
7th Phase JP Nagar,2 BHK,1120.0,2.0,65.0,2,5803
Electronics City Phase 1,1 BHK,762.5,1.0,45.4,1,5954
Electronics City Phase 1,2 BHK,1245.0,2.0,75.0,2,6024
other,4 Bedroom,3978.0,4.0,428.0,4,10759
Channasandra,2 BHK,1093.0,2.0,34.0,2,3110
Electronic City,2 BHK,1025.0,1.0,26.63,2,2598
Hebbal,2 BHK,1345.0,2.0,97.0,2,7211
other,2 Bedroom,1500.0,2.0,45.0,2,3000
Vijayanagar,2 BHK,1180.0,2.0,86.0,2,7288
NGR Layout,2 BHK,1020.0,2.0,48.45,2,4750
other,3 BHK,1800.0,3.0,85.0,3,4722
Konanakunte,4 BHK,3621.0,4.0,511.0,4,14112
other,3 BHK,2008.0,3.0,170.0,3,8466
Kothannur,3 BHK,1240.0,2.0,39.85,3,3213
CV Raman Nagar,2 BHK,1020.0,2.0,55.0,2,5392
other,3 BHK,2095.0,3.0,142.0,3,6778
Gunjur,3 BHK,1362.0,3.0,62.63,3,4598
Yelahanka,1 BHK,694.0,1.0,30.84,1,4443
Vittasandra,2 BHK,1404.0,2.0,70.0,2,4985
Electronic City Phase II,3 BHK,1192.0,2.0,47.0,3,3942
other,8 Bedroom,1850.0,12.0,300.0,8,16216
Mahalakshmi Layout,4 Bedroom,1050.0,1.0,35.0,4,3333
Vittasandra,2 BHK,1259.0,2.0,67.48,2,5359
Kengeri,2 BHK,1000.0,2.0,34.0,2,3400
Rachenahalli,2 BHK,985.0,2.0,49.97,2,5073
Indira Nagar,1 BHK,850.0,1.0,60.0,1,7058
Uttarahalli,3 BHK,1650.0,2.0,50.0,3,3030
Rajaji Nagar,5 Bedroom,2400.0,3.0,320.0,5,13333
Sarjapur  Road,3 BHK,1900.0,4.0,180.0,3,9473
Ambedkar Nagar,2 BHK,1367.0,2.0,86.0,2,6291
Rajaji Nagar,3 BHK,1210.0,2.0,81.0,3,6694
other,2 BHK,1190.0,2.0,48.6,2,4084
Sarjapur,2 BHK,1175.0,2.0,42.0,2,3574
R.T. Nagar,4 BHK,2800.0,4.0,400.0,4,14285
Whitefield,2 BHK,1140.0,2.0,40.0,2,3508
Hoskote,2 BHK,1095.0,2.0,33.6,2,3068
Attibele,1 BHK,782.0,1.0,19.55,1,2500
Gubbalala,3 BHK,1435.0,2.0,80.0,3,5574
Choodasandra,2 BHK,725.0,2.0,36.0,2,4965
other,3 BHK,1600.0,2.0,88.0,3,5500
9th Phase JP Nagar,5 Bedroom,1200.0,5.0,160.0,5,13333
Hosakerehalli,4 BHK,3205.0,4.0,300.0,4,9360
Vishwapriya Layout,4 Bedroom,720.0,3.0,95.0,4,13194
Nehru Nagar,3 BHK,1674.0,3.0,90.0,3,5376
other,4 Bedroom,860.0,4.0,125.0,4,14534
Haralur Road,2 BHK,1243.0,2.0,46.0,2,3700
Kanakpura Road,3 BHK,1703.0,3.0,130.0,3,7633
Padmanabhanagar,3 BHK,1350.0,3.0,71.55,3,5300
Horamavu Agara,2 BHK,1145.0,2.0,41.21,2,3599
Whitefield,2 BHK,1216.0,2.0,75.0,2,6167
7th Phase JP Nagar,2 BHK,1100.0,3.0,60.0,2,5454
Sarjapur  Road,4 BHK,2425.0,5.0,201.0,4,8288
Kothanur,4 Bedroom,2700.0,4.0,175.0,4,6481
Raja Rajeshwari Nagar,2 BHK,1196.0,2.0,40.54,2,3389
Kanakpura Road,3 BHK,1600.0,3.0,90.0,3,5625
Banashankari Stage V,3 BHK,1510.0,3.0,47.57,3,3150
Hosakerehalli,4 BHK,3033.0,4.0,326.0,4,10748
Pai Layout,2 Bedroom,780.0,2.0,72.0,2,9230
Haralur Road,2 BHK,1194.0,2.0,47.0,2,3936
Banashankari Stage II,2 BHK,1250.0,2.0,110.0,2,8800
Sarjapur  Road,5 Bedroom,1480.0,4.0,169.0,5,11418
Horamavu Agara,4 Bedroom,1250.0,4.0,100.0,4,8000
other,4 Bedroom,1350.0,4.0,160.0,4,11851
Laggere,1 Bedroom,620.0,2.0,48.0,1,7741
Raja Rajeshwari Nagar,3 BHK,1630.0,3.0,60.65,3,3720
Hennur Road,2 BHK,1020.0,2.0,45.0,2,4411
Hosakerehalli,3 BHK,1410.0,2.0,53.58,3,3800
Bhoganhalli,2 BHK,1181.7,2.0,100.0,2,8462
other,9 Bedroom,1200.0,8.0,270.0,9,22500
Sarjapur  Road,3 BHK,1691.0,3.0,113.0,3,6682
KR Puram,4 Bedroom,1200.0,4.0,83.0,4,6916
Banaswadi,2 BHK,1200.0,2.0,60.0,2,5000
Old Airport Road,2 BHK,1050.0,2.0,50.0,2,4761
Kathriguppe,3 BHK,1300.0,3.0,77.99,3,5999
Whitefield,2 BHK,1356.0,2.0,94.84,2,6994
Whitefield,2 BHK,1338.0,2.0,103.0,2,7698
other,1 BHK,500.0,1.0,20.0,1,4000
other,3 BHK,1280.0,2.0,100.0,3,7812
Raja Rajeshwari Nagar,3 BHK,1531.0,3.0,51.77,3,3381
other,4 BHK,11000.0,4.0,1600.0,4,14545
Anekal,2 BHK,620.0,1.0,23.5,2,3790
8th Phase JP Nagar,3 BHK,1605.0,2.0,65.0,3,4049
Thanisandra,3 BHK,1917.0,3.0,110.0,3,5738
Nagarbhavi,4 Bedroom,1200.0,3.0,340.0,4,28333
other,3 BHK,2750.0,3.0,943.0,3,34290
Sarjapur  Road,3 BHK,1400.0,3.0,55.0,3,3928
Thubarahalli,3 BHK,2625.0,3.0,175.0,3,6666
Sahakara Nagar,2 BHK,1100.0,2.0,55.0,2,5000
Hennur Road,3 BHK,2041.0,3.0,128.0,3,6271
Sarjapur  Road,3 BHK,2070.0,4.0,160.0,3,7729
Hebbal,3 BHK,1110.0,3.0,59.0,3,5315
other,3 BHK,1650.0,3.0,110.0,3,6666
Ambedkar Nagar,4 Bedroom,3565.0,4.0,255.0,4,7152
Varthur Road,3 BHK,1033.0,2.0,30.47,3,2949
HSR Layout,2 BHK,1142.0,2.0,65.0,2,5691
other,3 BHK,1500.0,2.0,75.0,3,5000
Electronic City Phase II,3 BHK,1502.0,3.0,50.0,3,3328
other,3 BHK,1176.0,2.0,45.0,3,3826
Hosa Road,2 BHK,1063.0,2.0,32.8,2,3085
Kammanahalli,2 BHK,1100.0,2.0,60.0,2,5454
other,2 Bedroom,1044.0,2.0,85.0,2,8141
Sarjapur  Road,3 Bedroom,5400.0,3.0,700.0,3,12962
6th Phase JP Nagar,2 BHK,1216.0,2.0,60.0,2,4934
other,3 BHK,1425.0,3.0,55.0,3,3859
other,8 Bedroom,1200.0,8.0,150.0,8,12500
Hosur Road,2 BHK,1345.0,2.0,108.0,2,8029
Marathahalli,2 BHK,1248.0,2.0,130.0,2,10416
other,4 Bedroom,1830.0,4.0,325.0,4,17759
other,2 BHK,900.0,1.0,130.0,2,14444
Bellandur,4 Bedroom,3600.0,4.0,245.0,4,6805
Uttarahalli,2 BHK,1025.0,2.0,35.88,2,3500
other,2 Bedroom,840.0,1.0,99.0,2,11785
Lakshminarayana Pura,3 BHK,3050.0,2.0,90.0,3,2950
Kadubeesanahalli,3 BHK,1424.0,2.0,75.0,3,5266
EPIP Zone,3 BHK,2500.0,3.0,268.0,3,10720
ISRO Layout,6 Bedroom,1200.0,6.0,230.0,6,19166
other,2 BHK,1285.0,2.0,45.0,2,3501
Babusapalaya,2 BHK,1450.0,2.0,55.99,2,3861
Raja Rajeshwari Nagar,2 BHK,1160.0,2.0,70.0,2,6034
Yelahanka New Town,3 Bedroom,2700.0,3.0,350.0,3,12962
other,3 Bedroom,2000.0,2.0,95.0,3,4750
Uttarahalli,2 BHK,1125.0,2.0,47.65,2,4235
Old Madras Road,4 Bedroom,3900.0,4.0,175.0,4,4487
Raja Rajeshwari Nagar,2 BHK,1133.0,2.0,38.37,2,3386
Arekere,2 BHK,1240.0,2.0,60.0,2,4838
Devanahalli,3 BHK,1466.0,3.0,77.59,3,5292
other,4 BHK,3870.0,4.0,411.0,4,10620
Ramagondanahalli,3 BHK,1610.0,2.0,115.0,3,7142
Sarjapur  Road,3 BHK,1220.0,3.0,56.0,3,4590
Kaggadasapura,2 BHK,1225.0,2.0,48.0,2,3918
Akshaya Nagar,2 BHK,1300.0,2.0,54.0,2,4153
Thanisandra,3 BHK,1996.0,4.0,123.0,3,6162
other,3 BHK,1320.0,2.0,55.0,3,4166
Rajaji Nagar,3 BHK,1800.0,3.0,240.0,3,13333
other,3 BHK,1835.0,3.0,175.0,3,9536
Sarjapur  Road,3 BHK,1435.0,3.0,90.0,3,6271
Whitefield,3 BHK,4097.0,3.0,400.0,3,9763
Rayasandra,3 BHK,1601.0,3.0,70.0,3,4372
Kadubeesanahalli,2 BHK,1257.0,2.0,93.0,2,7398
Iblur Village,4 BHK,3596.0,5.0,260.0,4,7230
Kasavanhalli,2 BHK,1309.0,2.0,95.0,2,7257
Yeshwanthpur,2 Bedroom,1362.0,2.0,100.0,2,7342
Whitefield,2 BHK,1115.0,2.0,55.0,2,4932
Whitefield,3 BHK,1870.0,3.0,90.0,3,4812
other,2 BHK,1322.0,2.0,61.99,2,4689
other,3 BHK,1450.0,2.0,65.0,3,4482
Banashankari,2 BHK,1185.0,2.0,67.0,2,5654
7th Phase JP Nagar,3 BHK,1790.0,3.0,120.0,3,6703
Whitefield,3 Bedroom,3450.0,4.0,208.0,3,6028
Rachenahalli,2 BHK,1050.0,2.0,52.5,2,5000
Budigere,1 BHK,693.0,1.0,27.375,1,3950
Sarjapur  Road,4 Bedroom,5400.0,4.0,750.0,4,13888
Devanahalli,1 BHK,775.0,1.0,39.0,1,5032
Hoodi,2 BHK,1000.0,2.0,48.0,2,4800
Thanisandra,2 BHK,1183.0,2.0,77.25,2,6530
Binny Pete,3 BHK,1282.0,3.0,178.0,3,13884
other,1 BHK,425.0,1.0,15.0,1,3529
other,2 BHK,820.0,2.0,30.0,2,3658
other,4 BHK,2230.0,4.0,792.0,4,35515
Hebbal,2 BHK,1200.0,2.0,52.0,2,4333
Ramagondanahalli,2 BHK,1151.0,2.0,43.6,2,3788
Hulimavu,3 BHK,1320.0,2.0,59.0,3,4469
other,3 BHK,1450.0,3.0,68.0,3,4689
Poorna Pragna Layout,3 BHK,1270.0,2.0,50.79,3,3999
Hegde Nagar,3 BHK,1847.0,3.0,123.0,3,6659
Devanahalli,1 BHK,1020.0,2.0,49.87,1,4889
Somasundara Palya,3 BHK,1650.0,3.0,95.0,3,5757
other,4 BHK,2380.0,4.0,242.0,4,10168
Electronic City,3 BHK,2400.0,2.0,78.0,3,3250
Begur Road,2 BHK,1160.0,2.0,44.0,2,3793
Haralur Road,4 BHK,2805.0,4.0,154.0,4,5490
Mysore Road,3 BHK,1360.0,3.0,47.6,3,3500
Brookefield,2 BHK,1089.0,2.0,49.0,2,4499
Yelahanka,2 BHK,1115.0,2.0,60.0,2,5381
Kudlu Gate,2 BHK,1215.0,2.0,45.0,2,3703
Sarjapur  Road,2 BHK,1194.0,2.0,51.0,2,4271
Kannamangala,3 BHK,1536.0,3.0,104.0,3,6770
Hennur,2 BHK,1255.0,2.0,55.5,2,4422
Electronic City,3 BHK,1571.0,3.0,83.0,3,5283
other,2 Bedroom,900.0,1.0,70.0,2,7777
other,2 BHK,1517.0,2.0,125.0,2,8239
Budigere,1 BHK,664.0,1.0,35.4,1,5331
Devarachikkanahalli,2 BHK,1230.0,2.0,58.0,2,4715
Hegde Nagar,2 BHK,1340.0,2.0,72.0,2,5373
Attibele,2 BHK,1175.0,2.0,29.38,2,2500
Benson Town,4 Bedroom,3569.0,4.0,600.0,4,16811
Sarjapur,2 BHK,1240.0,2.0,43.1,2,3475
Electronic City,3 BHK,1470.0,2.0,44.1,3,3000
7th Phase JP Nagar,3 BHK,1680.0,3.0,117.0,3,6964
other,2 BHK,1000.0,2.0,35.0,2,3500
Gunjur,2 BHK,1071.0,2.0,51.0,2,4761
Sector 2 HSR Layout,3 BHK,1512.0,3.0,80.0,3,5291
Frazer Town,4 BHK,3435.0,2.0,341.0,4,9927
other,3 BHK,3144.0,3.0,455.0,3,14472
Kalyan nagar,2 BHK,1190.0,2.0,60.0,2,5042
Pattandur Agrahara,3 BHK,1550.0,2.0,80.0,3,5161
Kanakpura Road,2 BHK,1283.0,2.0,63.22,2,4927
Mallasandra,2 BHK,1325.0,2.0,73.0,2,5509
Whitefield,2 BHK,1105.0,2.0,35.36,2,3200
Banaswadi,3 BHK,1600.0,3.0,74.5,3,4656
other,4 Bedroom,400.0,2.0,35.0,4,8750
Hennur,2 BHK,1100.0,2.0,44.55,2,4050
Talaghattapura,3 BHK,2099.0,3.0,147.0,3,7003
Yelachenahalli,3 BHK,1365.0,2.0,72.0,3,5274
Hebbal Kempapura,2 BHK,1485.0,3.0,132.0,2,8888
Hosur Road,2 BHK,1250.0,2.0,55.0,2,4400
Chikkalasandra,3 BHK,1290.0,2.0,56.12,3,4350
Bellandur,2 BHK,1170.0,2.0,46.79,2,3999
Kanakpura Road,2 BHK,1328.0,2.0,91.06,2,6856
Hennur Road,2 BHK,1232.0,2.0,87.0,2,7061
1st Phase JP Nagar,2 BHK,1180.0,2.0,88.5,2,7500
Electronic City,3 BHK,1350.0,3.0,66.0,3,4888
other,4 BHK,3602.0,4.0,280.0,4,7773
Nagasandra,4 BHK,2400.0,4.0,145.0,4,6041
Electronic City Phase II,2 BHK,1020.0,2.0,32.0,2,3137
Begur Road,3 BHK,1410.0,2.0,49.34,3,3499
other,3 BHK,2159.0,3.0,120.0,3,5558
TC Palaya,3 BHK,1200.0,2.0,66.0,3,5500
Hennur,2 BHK,1255.0,2.0,57.5,2,4581
Whitefield,2 BHK,1074.0,2.0,42.79,2,3984
other,3 Bedroom,1200.0,3.0,72.0,3,6000
Sarjapur,3 BHK,1170.0,2.0,35.41,3,3026
Tumkur Road,1 BHK,728.5,1.0,31.315,1,4298
Bommanahalli,3 BHK,1660.0,2.0,100.0,3,6024
other,5 Bedroom,900.0,6.0,95.0,5,10555
other,6 Bedroom,1850.0,3.0,175.0,6,9459
other,1 Bedroom,510.0,1.0,34.0,1,6666
Mahadevpura,3 BHK,1601.0,3.0,95.0,3,5933
Begur Road,2 BHK,952.0,2.0,42.0,2,4411
other,2 BHK,1166.0,2.0,65.0,2,5574
other,3 BHK,2700.0,2.0,260.0,3,9629
Electronic City Phase II,4 Bedroom,1900.0,3.0,97.0,4,5105
other,3 Bedroom,1350.0,2.0,175.0,3,12962
Haralur Road,3 BHK,1520.0,2.0,96.0,3,6315
other,4 Bedroom,2100.0,6.0,100.0,4,4761
Yelenahalli,2 BHK,1175.0,2.0,50.0,2,4255
Bharathi Nagar,2 BHK,1432.0,2.0,68.0,2,4748
Bommasandra Industrial Area,2 BHK,1005.0,2.0,39.77,2,3957
Rajiv Nagar,2 BHK,1309.0,2.0,80.0,2,6111
other,3 BHK,2251.0,4.0,370.0,3,16437
7th Phase JP Nagar,2 BHK,1050.0,2.0,77.0,2,7333
Kanakpura Road,3 BHK,1100.0,3.0,53.0,3,4818
Sanjay nagar,3 Bedroom,2400.0,3.0,310.0,3,12916
Begur Road,3 BHK,1565.0,2.0,56.34,3,3600
other,2 BHK,940.0,2.0,56.0,2,5957
other,2 BHK,1015.0,2.0,40.1,2,3950
BTM 2nd Stage,3 Bedroom,2400.0,2.0,360.0,3,15000
Electronic City,3 BHK,1652.0,3.0,110.0,3,6658
Whitefield,2 BHK,1100.0,2.0,52.0,2,4727
Rajiv Nagar,4 BHK,2340.0,4.0,145.0,4,6196
Koramangala,2 BHK,1084.0,2.0,179.0,2,16512
Kammanahalli,2 Bedroom,1200.0,1.0,81.0,2,6750
other,3 BHK,1400.0,3.0,69.0,3,4928
HBR Layout,4 Bedroom,2400.0,5.0,320.0,4,13333
Electronic City Phase II,2 BHK,1020.0,2.0,27.43,2,2689
Whitefield,2 BHK,1267.0,2.0,90.33,2,7129
Bhoganhalli,4 BHK,2439.0,4.0,185.0,4,7585
other,3 BHK,1645.0,3.0,95.0,3,5775
Sahakara Nagar,3 BHK,1100.0,2.0,75.0,3,6818
Electronic City,3 BHK,1320.0,2.0,38.12,3,2887
Ananth Nagar,2 BHK,1000.0,2.0,29.7,2,2970
Whitefield,4 Bedroom,5000.0,5.0,550.0,4,11000
KR Puram,6 Bedroom,2500.0,6.0,58.0,6,2320
Kengeri Satellite Town,2 Bedroom,2480.0,2.0,160.0,2,6451
Kalyan nagar,2 BHK,1086.0,2.0,44.5,2,4097
Kengeri,1 BHK,416.0,1.0,18.0,1,4326
Hennur Road,2 BHK,1066.0,2.0,52.23,2,4899
OMBR Layout,3 BHK,1730.0,3.0,93.0,3,5375
Balagere,2 BHK,1205.0,2.0,63.86,2,5299
Marathahalli,4 BHK,2500.0,5.0,175.0,4,7000
Subramanyapura,2 BHK,960.0,2.0,55.0,2,5729
Haralur Road,3 BHK,1510.0,3.0,83.05,3,5500
other,4 Bedroom,1600.0,4.0,90.0,4,5625
Old Madras Road,3 Bedroom,2300.0,3.0,142.0,3,6173
other,2 BHK,1000.0,2.0,63.0,2,6300
Poorna Pragna Layout,2 BHK,1160.0,2.0,46.39,2,3999
Hormavu,3 BHK,1553.0,2.0,58.23,3,3749
Frazer Town,4 BHK,3436.0,5.0,345.0,4,10040
Seegehalli,2 BHK,1134.0,2.0,45.0,2,3968
Nagasandra,4 Bedroom,600.0,4.0,80.0,4,13333
Horamavu Agara,5 Bedroom,2400.0,3.0,80.0,5,3333
Whitefield,4 Bedroom,2400.0,4.0,143.0,4,5958
other,2 BHK,1350.0,2.0,65.0,2,4814
Sarjapur,2 BHK,1032.0,2.0,42.0,2,4069
other,3 BHK,1998.0,3.0,117.0,3,5855
Murugeshpalya,2 BHK,1225.0,2.0,48.0,2,3918
Sarjapur  Road,2 BHK,1262.0,2.0,67.0,2,5309
Badavala Nagar,3 BHK,1842.0,3.0,115.0,3,6243
other,2 BHK,1300.0,2.0,60.0,2,4615
Kothannur,2 BHK,1070.0,2.0,38.6,2,3607
other,2 BHK,950.0,2.0,33.0,2,3473
Balagere,2 BHK,1012.0,2.0,53.58,2,5294
TC Palaya,2 Bedroom,1000.0,2.0,70.0,2,7000
Kadugodi,2 BHK,1394.0,2.0,56.0,2,4017
Amruthahalli,3 BHK,1450.0,2.0,85.0,3,5862
Banaswadi,1 RK,527.0,1.0,35.0,1,6641
Sarjapur  Road,3 BHK,1693.0,3.0,125.0,3,7383
Thigalarapalya,4 BHK,3122.0,6.0,235.0,4,7527
Horamavu Agara,2 BHK,1000.0,2.0,46.0,2,4600
Kogilu,2 BHK,952.0,2.0,49.0,2,5147
Malleshwaram,3 BHK,2050.0,4.0,270.0,3,13170
Kengeri,1 BHK,1200.0,1.0,14.0,1,1166
Mysore Road,3 BHK,1410.0,2.0,64.0,3,4539
Hebbal Kempapura,2 BHK,1400.0,2.0,108.0,2,7714
Chikka Tirupathi,4 Bedroom,3056.0,5.0,100.0,4,3272
Sarjapur  Road,3 BHK,1826.0,3.0,125.0,3,6845
Hosur Road,3 BHK,1689.0,3.0,103.0,3,6098
Bisuvanahalli,3 BHK,1075.0,2.0,31.0,3,2883
Bannerghatta Road,5 BHK,2500.0,4.0,1400.0,5,56000
Hennur,2 BHK,960.0,2.0,48.0,2,5000
Channasandra,2 BHK,808.0,2.0,40.0,2,4950
Kodichikkanahalli,2 BHK,907.0,2.0,38.5,2,4244
Uttarahalli,2 BHK,1300.0,2.0,55.0,2,4230
other,3 BHK,1804.0,3.0,150.0,3,8314
other,3 BHK,1350.0,3.0,81.0,3,6000
Whitefield,3 BHK,1836.0,3.0,125.0,3,6808
other,2 BHK,826.0,2.0,36.0,2,4358
Hormavu,2 BHK,1200.0,2.0,68.0,2,5666
other,4 BHK,3000.0,3.0,300.0,4,10000
Indira Nagar,4 Bedroom,2400.0,4.0,405.0,4,16875
6th Phase JP Nagar,2 BHK,1460.0,2.0,80.0,2,5479
Hoskote,4 Bedroom,1200.0,4.0,140.0,4,11666
BTM 2nd Stage,2 BHK,965.0,2.0,56.0,2,5803
Kothannur,4 Bedroom,2000.0,4.0,95.0,4,4750
Bhoganhalli,3 BHK,1395.0,2.0,70.0,3,5017
Sarjapur  Road,3 BHK,1660.0,3.0,116.0,3,6987
other,2 BHK,935.0,2.0,35.0,2,3743
other,2 BHK,1180.0,2.0,39.0,2,3305
Electronic City,2 BHK,1090.0,2.0,31.48,2,2888
Kanakpura Road,2 BHK,1204.0,2.0,82.0,2,6810
Thigalarapalya,3 BHK,2072.0,4.0,153.0,3,7384
Haralur Road,2 BHK,1056.0,2.0,65.0,2,6155
Hebbal,2 BHK,985.0,2.0,62.0,2,6294
Banashankari Stage V,3 BHK,1315.0,2.0,49.0,3,3726
KR Puram,2 BHK,734.0,2.0,22.0,2,2997
Kammanahalli,4 Bedroom,2900.0,4.0,200.0,4,6896
Yelahanka,3 BHK,2317.0,4.0,140.0,3,6042
Singasandra,1 Bedroom,600.0,1.0,45.0,1,7500
other,4 Bedroom,2200.0,3.0,115.0,4,5227
Attibele,2 BHK,656.0,2.0,25.0,2,3810
Attibele,1 BHK,400.0,1.0,11.5,1,2875
Yeshwanthpur,1 BHK,666.0,1.0,36.8,1,5525
Kadubeesanahalli,4 Bedroom,3293.0,5.0,425.0,4,12906
7th Phase JP Nagar,2 BHK,1174.0,2.0,47.0,2,4003
Budigere,2 BHK,1153.0,2.0,56.7,2,4917
Jalahalli,4 Bedroom,1000.0,4.0,90.0,4,9000
Narayanapura,5 BHK,2400.0,5.0,85.0,5,3541
Marathahalli,2 BHK,957.0,2.0,47.0,2,4911
Banashankari Stage III,2 BHK,910.0,2.0,160.0,2,17582
Sarjapur  Road,2 BHK,1060.0,2.0,50.0,2,4716
Banashankari,2 BHK,1000.0,2.0,50.0,2,5000
Kanakpura Road,3 BHK,1843.0,3.0,88.0,3,4774
Kanakpura Road,3 BHK,1150.0,2.0,40.25,3,3500
Chandapura,3 BHK,1230.0,2.0,31.37,3,2550
Kodihalli,4 BHK,3197.0,5.0,335.0,4,10478
Kanakpura Road,3 BHK,1703.0,3.0,130.0,3,7633
Kaval Byrasandra,2 BHK,1125.0,2.0,41.5,2,3688
Battarahalli,3 BHK,1516.0,3.0,65.0,3,4287
Brookefield,2 BHK,1130.0,2.0,56.0,2,4955
Iblur Village,4 BHK,3596.0,5.0,279.0,4,7758
Hosa Road,3 BHK,1512.0,3.0,103.0,3,6812
Thanisandra,3 BHK,1491.0,3.0,88.0,3,5902
Cox Town,2 BHK,1250.0,2.0,95.0,2,7600
Banashankari,3 BHK,1650.0,3.0,90.75,3,5500
Raja Rajeshwari Nagar,2 BHK,1140.0,2.0,48.7,2,4271
other,4 Bedroom,1200.0,4.0,110.0,4,9166
Thanisandra,3 BHK,1881.0,3.0,95.0,3,5050
Kanakpura Road,3 BHK,1100.0,3.0,56.57,3,5142
Iblur Village,3 BHK,2666.0,4.0,200.0,3,7501
other,2 BHK,1500.0,2.0,50.0,2,3333
other,2 BHK,1217.0,2.0,164.0,2,13475
Whitefield,2 BHK,1197.0,2.0,50.0,2,4177
other,6 Bedroom,2400.0,4.0,170.0,6,7083
Gollarapalya Hosahalli,3 BHK,1345.0,3.0,60.0,3,4460
KR Puram,4 Bedroom,1200.0,4.0,100.0,4,8333
Bannerghatta Road,4 BHK,3230.0,5.0,165.0,4,5108
Banashankari Stage II,3 BHK,1260.0,2.0,95.0,3,7539
Sarjapur  Road,4 Bedroom,2750.0,5.0,240.0,4,8727
9th Phase JP Nagar,2 Bedroom,1200.0,2.0,80.0,2,6666
other,3 Bedroom,1650.0,3.0,52.0,3,3151
other,1 BHK,600.0,1.0,45.0,1,7500
Uttarahalli,2 BHK,1025.0,2.0,35.88,2,3500
Hennur Road,2 BHK,973.0,2.0,56.0,2,5755
other,4 BHK,4110.0,4.0,590.0,4,14355
Rajaji Nagar,3 BHK,2533.0,3.0,425.0,3,16778
Balagere,2 BHK,1007.0,2.0,75.0,2,7447
Hosa Road,1 BHK,800.0,1.0,39.99,1,4998
Sarjapur  Road,3 Bedroom,1830.0,3.0,119.0,3,6502
Nagavara,2 BHK,1077.0,2.0,50.0,2,4642
other,1 Bedroom,672.0,1.0,27.0,1,4017
other,3 Bedroom,1280.0,3.0,87.0,3,6796
Electronics City Phase 1,2 BHK,865.0,2.0,40.0,2,4624
Kalena Agrahara,2 BHK,1000.0,2.0,33.0,2,3300
CV Raman Nagar,2 BHK,1285.0,2.0,65.0,2,5058
Thanisandra,1 BHK,662.0,1.0,29.0,1,4380
Whitefield,5 Bedroom,2297.0,5.0,269.0,5,11710
Sarjapura - Attibele Road,5 Bedroom,3750.0,6.0,295.0,5,7866
other,3 BHK,1850.0,3.0,95.0,3,5135
Harlur,3 BHK,1754.0,3.0,132.0,3,7525
other,5 Bedroom,1200.0,4.0,150.0,5,12500
1st Phase JP Nagar,5 Bedroom,2200.0,7.0,350.0,5,15909
other,3 BHK,1567.0,3.0,100.0,3,6381
Hoskote,3 BHK,1695.0,3.0,50.0,3,2949
other,2 Bedroom,360.0,2.0,40.0,2,11111
Bellandur,2 BHK,1150.0,2.0,69.44,2,6038
Akshaya Nagar,3 BHK,1690.0,3.0,85.0,3,5029
other,3 BHK,2350.0,3.0,210.0,3,8936
Karuna Nagar,2 BHK,1057.0,2.0,65.0,2,6149
Hebbal,2 BHK,1320.0,2.0,91.0,2,6893
Bhoganhalli,4 BHK,2119.0,4.0,111.0,4,5238
other,2 BHK,1125.0,2.0,48.0,2,4266
Kanakpura Road,3 BHK,2254.0,3.0,170.0,3,7542
Balagere,1 BHK,645.0,1.0,34.18,1,5299
Rachenahalli,2 BHK,1050.0,2.0,52.08,2,4960
Panathur,2 BHK,1040.0,2.0,41.6,2,4000
Sarjapur  Road,3 BHK,1700.0,3.0,194.0,3,11411
1st Block Jayanagar,3 BHK,1760.0,3.0,115.0,3,6534
Bellandur,3 BHK,1138.0,3.0,128.0,3,11247
Sarjapur  Road,2 BHK,984.0,2.0,45.91,2,4665
Raja Rajeshwari Nagar,2 BHK,1140.0,2.0,38.65,2,3390
Whitefield,3 BHK,3596.0,3.0,200.0,3,5561
other,3 BHK,1365.0,3.0,54.0,3,3956
other,1 BHK,560.0,1.0,19.0,1,3392
Vittasandra,2 BHK,1404.0,2.0,75.0,2,5341
Bellandur,4 BHK,1540.0,4.0,45.0,4,2922
Electronic City,3 BHK,1050.0,2.0,39.0,3,3714
other,4 Bedroom,360.0,4.0,37.0,4,10277
Rachenahalli,3 BHK,1550.0,3.0,72.5,3,4677
Yelahanka,3 BHK,1653.0,3.0,93.0,3,5626
Parappana Agrahara,2 BHK,1194.0,2.0,47.0,2,3936
other,3 BHK,1471.0,2.0,75.0,3,5098
Kanakpura Road,2 BHK,700.0,2.0,35.0,2,5000
Kudlu Gate,2 BHK,812.0,2.0,38.25,2,4710
8th Phase JP Nagar,3 BHK,1504.0,2.0,83.0,3,5518
Mahadevpura,3 BHK,1450.0,3.0,75.0,3,5172
Kannamangala,2 BHK,957.0,2.0,54.5,2,5694
Attibele,1 BHK,460.0,1.0,13.0,1,2826
other,2 Bedroom,900.0,2.0,128.0,2,14222
7th Phase JP Nagar,2 BHK,1050.0,2.0,75.0,2,7142
Budigere,3 BHK,1820.0,3.0,85.0,3,4670
Varthur Road,2 BHK,1033.0,2.0,32.0,2,3097
Kumaraswami Layout,2 Bedroom,800.0,1.0,90.0,2,11250
other,1 Bedroom,520.0,2.0,29.0,1,5576
OMBR Layout,2 BHK,1165.0,2.0,88.5,2,7596
Gottigere,2 BHK,1280.0,2.0,61.0,2,4765
other,3 BHK,1950.0,3.0,230.0,3,11794
other,1 Bedroom,600.0,1.0,58.0,1,9666
Kanakpura Road,3 BHK,1550.0,3.0,65.1,3,4199
Nagavarapalya,2 BHK,1392.0,2.0,130.0,2,9339
Jigani,3 BHK,1221.0,3.0,75.0,3,6142
other,1 BHK,1000.0,1.0,19.0,1,1900
other,2 BHK,1300.0,2.0,60.0,2,4615
Hoskote,2 BHK,1065.0,2.0,31.5,2,2957
Yelahanka New Town,3 BHK,1209.0,2.0,65.0,3,5376
other,1 Bedroom,375.0,1.0,40.0,1,10666
other,3 Bedroom,7800.0,3.0,2000.0,3,25641
Sarjapur  Road,3 BHK,1929.0,4.0,101.0,3,5235
Hennur Road,2 BHK,1255.0,2.0,84.0,2,6693
Jalahalli,2 BHK,1035.0,2.0,31.05,2,3000
other,4 Bedroom,1200.0,5.0,270.0,4,22500
Banashankari,3 BHK,1650.0,3.0,101.0,3,6121
Whitefield,3 BHK,1275.0,2.0,65.0,3,5098
Electronic City,2 BHK,1128.0,2.0,65.58,2,5813
Raja Rajeshwari Nagar,2 BHK,1140.0,2.0,38.55,2,3381
OMBR Layout,4 Bedroom,1200.0,4.0,200.0,4,16666
Thanisandra,2 BHK,1183.0,2.0,58.56,2,4950
Whitefield,2 BHK,1155.0,2.0,39.0,2,3376
Kengeri,3 BHK,1245.0,2.0,43.57,3,3499
Akshaya Nagar,2 BHK,1000.0,2.0,38.0,2,3800
7th Phase JP Nagar,2 BHK,1040.0,2.0,75.0,2,7211
Ardendale,4 BHK,2062.0,3.0,140.0,4,6789
other,3 BHK,1477.0,2.0,75.0,3,5077
other,4 Bedroom,1050.0,3.0,85.0,4,8095
Sarjapur  Road,3 BHK,1525.0,2.0,65.0,3,4262
other,2 BHK,1220.0,2.0,65.0,2,5327
Hebbal,3 BHK,2388.0,3.0,195.0,3,8165
Electronic City,2 BHK,1355.0,2.0,97.6,2,7202
Whitefield,2 BHK,1141.0,2.0,62.0,2,5433
Vijayanagar,2 BHK,1046.0,2.0,49.06,2,4690
Bannerghatta Road,1 BHK,630.0,1.0,35.0,1,5555
Pattandur Agrahara,2 BHK,1247.0,2.0,62.35,2,5000
Marathahalli,4 BHK,3855.0,4.0,220.0,4,5706
other,2 BHK,850.0,2.0,32.5,2,3823
KR Puram,2 BHK,1120.0,2.0,36.0,2,3214
HSR Layout,8 Bedroom,800.0,8.0,285.0,8,35625
Jakkur,3 BHK,1798.0,3.0,110.0,3,6117
Bisuvanahalli,3 BHK,1180.0,2.0,45.0,3,3813
Kereguddadahalli,2 BHK,1000.0,2.0,34.0,2,3400
Whitefield,2 BHK,1216.0,2.0,82.0,2,6743
Margondanahalli,2 Bedroom,900.0,2.0,65.0,2,7222
TC Palaya,2 Bedroom,1100.0,2.0,65.0,2,5909
Badavala Nagar,3 BHK,1842.0,3.0,115.0,3,6243
Bisuvanahalli,3 BHK,1075.0,2.0,43.0,3,4000
Whitefield,2 BHK,1495.0,3.0,88.0,2,5886
Kanakpura Road,3 BHK,1700.0,3.0,120.0,3,7058
Whitefield,3 BHK,1564.0,3.0,103.0,3,6585
other,2 BHK,1256.0,2.0,65.0,2,5175
Kundalahalli,2 BHK,1047.0,2.0,89.0,2,8500
Bhoganhalli,3 BHK,1458.0,3.0,78.45,3,5380
Hoodi,2 BHK,1063.0,2.0,61.17,2,5754
other,4 BHK,7400.0,5.0,1850.0,4,25000
Garudachar Palya,2 BHK,1150.0,2.0,52.5,2,4565
Chandapura,1 BHK,450.0,1.0,9.0,1,2000
Ramamurthy Nagar,2 BHK,1360.0,2.0,45.0,2,3308
Marathahalli,3 BHK,1605.0,3.0,76.0,3,4735
Nagarbhavi,3 BHK,1200.0,4.0,205.0,3,17083
Haralur Road,3 BHK,1810.0,3.0,100.0,3,5524
LB Shastri Nagar,2 BHK,1250.0,2.0,48.5,2,3880
Poorna Pragna Layout,3 BHK,1475.0,2.0,58.99,3,3999
R.T. Nagar,2 Bedroom,1100.0,3.0,125.0,2,11363
Dommasandra,2 BHK,850.0,2.0,25.4,2,2988
Electronic City Phase II,2 BHK,1025.0,2.0,26.65,2,2600
Uttarahalli,2 BHK,980.0,2.0,45.0,2,4591
other,3 BHK,1428.0,2.0,70.0,3,4901
other,2 BHK,1130.0,2.0,74.0,2,6548
Kanakpura Road,3 BHK,1592.0,3.0,125.0,3,7851
JP Nagar,3 BHK,2050.0,3.0,82.0,3,4000
other,2 BHK,1407.0,2.0,83.87,2,5960
Cunningham Road,3 BHK,4170.0,3.0,800.0,3,19184
Whitefield,3 BHK,1585.0,3.0,84.0,3,5299
Jakkur,3 BHK,1785.0,3.0,130.0,3,7282
other,2 BHK,900.0,2.0,45.0,2,5000
other,4 Bedroom,1350.0,4.0,160.0,4,11851
Kanakpura Road,3 BHK,1843.0,3.0,96.2,3,5219
other,4 BHK,2375.0,3.0,260.0,4,10947
other,2 BHK,1200.0,2.0,67.0,2,5583
Akshaya Nagar,2 BHK,1113.0,2.0,52.0,2,4672
Uttarahalli,2 BHK,1125.0,2.0,47.0,2,4177
Whitefield,3 Bedroom,1200.0,3.0,56.0,3,4666
Anekal,3 BHK,1150.0,3.0,45.0,3,3913
Yeshwanthpur,2 BHK,770.0,1.0,70.0,2,9090
Kodichikkanahalli,2 BHK,1181.0,2.0,57.0,2,4826
Kanakpura Road,3 BHK,1550.0,3.0,64.7,3,4174
1st Phase JP Nagar,2 BHK,1200.0,2.0,86.0,2,7166
Raja Rajeshwari Nagar,2 BHK,1295.0,2.0,55.47,2,4283
Kenchenahalli,3 Bedroom,480.0,3.0,65.0,3,13541
Lingadheeranahalli,3 BHK,1506.0,3.0,95.0,3,6308
other,3 BHK,1872.0,3.0,152.0,3,8119
Sarjapur  Road,4 BHK,2900.0,4.0,240.0,4,8275
Abbigere,3 BHK,1326.0,2.0,35.0,3,2639
Talaghattapura,2 BHK,1040.0,2.0,33.28,2,3200
other,5 Bedroom,1200.0,5.0,90.0,5,7500
Hennur,2 BHK,1020.0,2.0,48.0,2,4705
other,3 BHK,1400.0,3.0,90.0,3,6428
Neeladri Nagar,10 BHK,4000.0,12.0,160.0,10,4000
other,1 Bedroom,812.0,1.0,26.0,1,3201
Vidyaranyapura,2 BHK,1100.0,2.0,54.0,2,4909
other,2 BHK,829.0,2.0,22.8,2,2750
Vijayanagar,5 Bedroom,750.0,4.0,95.0,5,12666
Sanjay nagar,4 Bedroom,2700.0,4.0,350.0,4,12962
Kanakpura Road,3 BHK,1500.0,3.0,72.99,3,4865
Ananth Nagar,2 BHK,850.0,2.0,25.99,2,3057
Kaval Byrasandra,2 BHK,1180.0,2.0,60.0,2,5084
Cunningham Road,4 BHK,5108.0,5.0,995.0,4,19479
Thigalarapalya,3 BHK,2215.0,4.0,150.0,3,6772
Channasandra,2 BHK,1010.0,2.0,36.0,2,3564
Marathahalli,2 BHK,1350.0,2.0,60.0,2,4444
Talaghattapura,2 BHK,1100.0,2.0,50.0,2,4545
Kanakpura Road,3 BHK,1450.0,2.0,62.93,3,4340
Raja Rajeshwari Nagar,2 BHK,1133.0,2.0,40.0,2,3530
Thanisandra,3 BHK,1564.0,3.0,100.0,3,6393
Doddaballapur,2 Bedroom,640.0,1.0,10.5,2,1640
Thigalarapalya,3 BHK,2072.0,4.0,157.0,3,7577
Rajaji Nagar,2 Bedroom,432.0,2.0,65.0,2,15046
Kanakapura,3 BHK,1460.0,2.0,51.1,3,3500
other,1 BHK,600.0,2.0,35.0,1,5833
Rachenahalli,2 BHK,1050.0,2.0,52.07,2,4959
other,2 BHK,700.0,2.0,35.0,2,5000
TC Palaya,3 Bedroom,1200.0,3.0,66.0,3,5500
Yelahanka,1 BHK,654.0,1.0,38.0,1,5810
other,2 BHK,1105.0,2.0,50.0,2,4524
Hebbal Kempapura,3 BHK,1800.0,3.0,150.0,3,8333
Yelachenahalli,2 Bedroom,1200.0,2.0,160.0,2,13333
Kogilu,1 BHK,700.0,1.0,30.84,1,4405
other,2 BHK,1000.0,2.0,51.0,2,5100
HSR Layout,5 Bedroom,3500.0,5.0,350.0,5,10000
other,3 BHK,1240.0,2.0,63.0,3,5080
Bellandur,2 BHK,1200.0,2.0,45.0,2,3750
Yelahanka New Town,3 BHK,1400.0,2.0,80.0,3,5714
other,3 BHK,1393.0,3.0,54.0,3,3876
Whitefield,2 BHK,1140.0,2.0,50.0,2,4385
Sahakara Nagar,5 Bedroom,1000.0,5.0,130.0,5,13000
other,3 BHK,1640.0,3.0,70.0,3,4268
other,2 Bedroom,1200.0,2.0,62.0,2,5166
Rajaji Nagar,2 BHK,720.0,2.0,65.0,2,9027
Yelenahalli,3 BHK,1650.0,3.0,68.0,3,4121
Nagavara,4 BHK,2496.0,4.0,125.0,4,5008
9th Phase JP Nagar,2 BHK,905.0,2.0,32.0,2,3535
HAL 2nd Stage,3 BHK,1600.0,3.0,125.0,3,7812
Chikkalasandra,2 BHK,1070.0,2.0,45.48,2,4250
other,4 Bedroom,8400.0,5.0,1675.0,4,19940
Koramangala,2 BHK,1200.0,2.0,100.0,2,8333
Indira Nagar,5 Bedroom,1800.0,5.0,350.0,5,19444
8th Phase JP Nagar,4 Bedroom,2600.0,4.0,115.0,4,4423
Old Airport Road,2 BHK,1145.0,2.0,75.0,2,6550
other,3 BHK,1495.0,3.0,55.0,3,3678
other,3 Bedroom,1200.0,2.0,300.0,3,25000
other,3 BHK,1400.0,3.0,53.0,3,3785
Whitefield,3 BHK,1558.0,3.0,90.0,3,5776
Balagere,2 BHK,1007.0,2.0,53.0,2,5263
Ananth Nagar,2 BHK,902.0,2.0,25.26,2,2800
other,2 BHK,1054.0,2.0,53.0,2,5028
Uttarahalli,3 BHK,1135.0,2.0,39.73,3,3500
Gollarapalya Hosahalli,2 BHK,996.0,2.0,49.0,2,4919
other,2 BHK,1275.0,2.0,47.0,2,3686
NRI Layout,2 Bedroom,1250.0,2.0,75.0,2,6000
Hennur,4 BHK,2502.0,4.0,170.0,4,6794
other,2 BHK,1394.0,2.0,52.0,2,3730
Haralur Road,2 BHK,1243.0,2.0,46.0,2,3700
Whitefield,2 BHK,1216.0,2.0,75.0,2,6167
other,2 BHK,1075.0,2.0,57.0,2,5302
Indira Nagar,3 BHK,1500.0,3.0,120.0,3,8000
Rayasandra,4 BHK,1708.0,3.0,76.0,4,4449
Whitefield,3 Bedroom,1500.0,3.0,250.0,3,16666
other,3 BHK,1800.0,4.0,115.0,3,6388
Harlur,2 BHK,1532.0,2.0,65.0,2,4242
Hosa Road,3 BHK,1726.0,3.0,81.99,3,4750
other,2 BHK,1220.0,2.0,54.89,2,4499
Chikkalasandra,2 BHK,1100.0,2.0,50.0,2,4545
Electronics City Phase 1,3 BHK,1530.0,3.0,40.0,3,2614
Hosa Road,2 BHK,840.0,2.0,28.0,2,3333
Bommanahalli,3 BHK,1250.0,4.0,39.0,3,3120
CV Raman Nagar,2 BHK,1030.0,2.0,35.0,2,3398
other,3 BHK,1499.0,3.0,89.0,3,5937
other,3 BHK,1280.0,2.0,55.0,3,4296
Munnekollal,2 BHK,1170.0,2.0,59.0,2,5042
Chikka Tirupathi,3 Bedroom,2646.0,3.0,99.0,3,3741
other,2 BHK,1161.0,2.0,36.69,2,3160
Yelahanka,2 Bedroom,1633.0,2.0,72.0,2,4409
other,2 BHK,1050.0,2.0,65.0,2,6190
Hennur Road,4 Bedroom,1650.0,5.0,309.0,4,18727
Varthur Road,1 BHK,552.5,1.0,13.26,1,2400
Attibele,1 BHK,418.0,2.0,11.5,1,2751
other,2 BHK,905.0,2.0,50.0,2,5524
Raja Rajeshwari Nagar,3 BHK,1510.0,2.0,55.0,3,3642
Chikkalasandra,3 BHK,1390.0,2.0,60.47,3,4350
Kasturi Nagar,2 BHK,1101.0,2.0,66.0,2,5994
other,4 BHK,2736.0,4.0,130.0,4,4751
EPIP Zone,3 BHK,2350.0,4.0,163.0,3,6936
other,2 BHK,1056.0,2.0,55.0,2,5208
Kanakpura Road,3 BHK,1425.0,2.0,49.88,3,3500
Electronics City Phase 1,3 BHK,1450.0,3.0,70.0,3,4827
Pai Layout,2 BHK,1000.0,2.0,34.0,2,3400
Haralur Road,1 BHK,575.0,1.0,38.0,1,6608
Uttarahalli,3 BHK,1350.0,2.0,47.24,3,3499
Kanakpura Road,3 Bedroom,2000.0,2.0,180.0,3,9000
other,3 BHK,1770.0,3.0,90.0,3,5084
other,3 BHK,3155.0,3.0,315.0,3,9984
Anandapura,2 Bedroom,616.0,3.0,45.0,2,7305
JP Nagar,1 BHK,745.0,1.0,34.27,1,4600
other,7 Bedroom,6500.0,7.0,104.0,7,1600
HAL 2nd Stage,4 Bedroom,2400.0,4.0,650.0,4,27083
Kudlu Gate,2 BHK,1547.0,2.0,103.0,2,6658
Panathur,3 BHK,1713.0,3.0,120.0,3,7005
Sarjapur,4 Bedroom,3300.0,4.0,430.0,4,13030
Mahalakshmi Layout,8 Bedroom,1135.0,4.0,190.0,8,16740
Yeshwanthpur,3 BHK,1856.0,4.0,180.0,3,9698
other,2 BHK,1152.0,2.0,57.0,2,4947
Dodda Nekkundi,2 BHK,1100.0,2.0,46.75,2,4250
Narayanapura,3 BHK,2357.0,3.0,135.0,3,5727
Akshaya Nagar,3 BHK,1820.0,4.0,95.0,3,5219
Kannamangala,3 BHK,1865.0,3.0,139.0,3,7453
Kanakpura Road,2 BHK,1240.0,2.0,65.0,2,5241
Padmanabhanagar,3 BHK,1360.0,3.0,75.0,3,5514
ISRO Layout,4 Bedroom,4200.0,4.0,255.0,4,6071
Electronics City Phase 1,3 BHK,1450.0,2.0,63.51,3,4380
Budigere,2 BHK,1139.0,2.0,72.0,2,6321
Begur Road,3 BHK,1565.0,2.0,56.0,3,3578
Anjanapura,3 Bedroom,1200.0,3.0,165.0,3,13750
other,4 Bedroom,600.0,4.0,77.0,4,12833
other,2 BHK,1200.0,2.0,39.0,2,3250
Yeshwanthpur,3 BHK,1855.0,3.0,140.0,3,7547
Margondanahalli,2 Bedroom,1100.0,2.0,55.0,2,5000
other,3 BHK,1350.0,2.0,65.0,3,4814
Old Madras Road,3 BHK,2266.0,3.0,207.0,3,9135
Hosur Road,1 BHK,660.0,1.0,23.0,1,3484
Kodihalli,2 BHK,1490.0,2.0,150.0,2,10067
Kambipura,3 BHK,1083.0,2.0,48.0,3,4432
Hennur Road,2 BHK,1232.0,2.0,83.0,2,6737
Raja Rajeshwari Nagar,3 BHK,1550.0,3.0,86.8,3,5600
Whitefield,2 BHK,1227.0,2.0,70.0,2,5704
Sarjapur  Road,2 BHK,1059.0,2.0,68.0,2,6421
Electronic City Phase II,2 BHK,545.0,1.0,29.0,2,5321
Mahadevpura,2 BHK,1152.0,2.0,54.0,2,4687
Kenchenahalli,2 BHK,870.0,1.0,45.0,2,5172
Panathur,2 BHK,1210.0,2.0,78.0,2,6446
Thanisandra,1 BHK,663.0,1.0,46.1,1,6953
other,3 BHK,1464.0,2.0,135.0,3,9221
Kadubeesanahalli,3 BHK,1365.0,3.0,80.0,3,5860
Whitefield,3 BHK,1704.0,3.0,120.0,3,7042
Whitefield,2 BHK,1256.0,2.0,73.0,2,5812
Kengeri,2 BHK,883.0,2.0,49.0,2,5549
Bisuvanahalli,3 BHK,1080.0,2.0,45.0,3,4166
Marsur,2 BHK,497.0,1.0,25.0,2,5030
9th Phase JP Nagar,2 BHK,1073.0,2.0,52.36,2,4879
Rajaji Nagar,6 BHK,3000.0,6.0,250.0,6,8333
other,2 BHK,1300.0,2.0,32.0,2,2461
Thanisandra,1 BHK,662.0,1.0,45.0,1,6797
other,2 BHK,1200.0,2.0,75.0,2,6250
5th Phase JP Nagar,3 BHK,1550.0,3.0,75.0,3,4838
other,3 BHK,1707.0,3.0,171.0,3,10017
Banashankari,2 BHK,1310.0,2.0,80.43,2,6139
Sarjapur  Road,3 BHK,1577.0,3.0,65.0,3,4121
Kodigehaali,3 BHK,1324.0,3.0,65.0,3,4909
Kothanur,3 BHK,1786.0,2.0,106.0,3,5935
Harlur,2 BHK,1174.0,2.0,75.0,2,6388
other,1 BHK,565.0,1.0,33.0,1,5840
7th Phase JP Nagar,3 BHK,1400.0,2.0,95.0,3,6785
Electronic City,1 BHK,635.0,1.0,28.0,1,4409
Kalyan nagar,6 Bedroom,3000.0,8.0,400.0,6,13333
8th Phase JP Nagar,3 BHK,1368.0,2.0,54.71,3,3999
Kanakpura Road,3 BHK,1450.0,3.0,55.0,3,3793
Devanahalli,2 BHK,1045.0,2.0,48.0,2,4593
Electronic City,2 BHK,1128.0,2.0,65.4,2,5797
Bhoganhalli,2 BHK,1125.0,2.0,54.0,2,4800
Whitefield,2 BHK,1195.0,2.0,62.38,2,5220
other,1 Bedroom,630.0,2.0,70.0,1,11111
Kaggadasapura,3 Bedroom,1350.0,2.0,105.0,3,7777
Domlur,3 BHK,1740.0,3.0,120.0,3,6896
Bommasandra,2 BHK,877.0,2.0,29.0,2,3306
Whitefield,3 BHK,1770.0,4.0,68.0,3,3841
Yelahanka New Town,1 BHK,450.0,1.0,15.0,1,3333
other,2 Bedroom,1000.0,2.0,55.0,2,5500
Whitefield,2 BHK,1300.0,2.0,78.0,2,6000
Doddathoguru,3 BHK,1300.0,2.0,42.0,3,3230
Abbigere,2 BHK,985.0,2.0,40.39,2,4100
other,2 BHK,1045.0,2.0,42.0,2,4019
other,2 BHK,1250.0,2.0,62.4,2,4992
other,3 Bedroom,2400.0,4.0,130.0,3,5416
Haralur Road,2 BHK,1194.0,2.0,46.0,2,3852
Kanakpura Road,3 BHK,1420.0,3.0,68.0,3,4788
other,3 BHK,1720.0,3.0,83.0,3,4825
Rachenahalli,3 BHK,1530.0,2.0,74.4,3,4862
Anandapura,3 Bedroom,1200.0,2.0,66.0,3,5500
Whitefield,3 BHK,1570.0,3.0,88.0,3,5605
Sarjapur  Road,3 BHK,1157.0,2.0,74.0,3,6395
Sarjapur,2 BHK,535.0,1.0,20.0,2,3738
Kanakapura,2 BHK,1120.0,2.0,39.2,2,3500
Kenchenahalli,2 BHK,1150.0,2.0,60.0,2,5217
Hebbal Kempapura,5 Bedroom,2400.0,5.0,200.0,5,8333
Bannerghatta Road,2 BHK,1215.0,2.0,68.0,2,5596
Bhoganhalli,2 BHK,1444.0,2.0,75.97,2,5261
6th Phase JP Nagar,3 BHK,2003.0,3.0,200.0,3,9985
other,2 BHK,1100.0,2.0,70.0,2,6363
Kodichikkanahalli,2 BHK,900.0,2.0,42.0,2,4666
Talaghattapura,3 BHK,1372.0,2.0,43.9,3,3199
Chikka Tirupathi,4 Bedroom,4500.0,5.0,150.0,4,3333
Bisuvanahalli,3 BHK,1180.0,2.0,55.5,3,4703
Electronic City,2 BHK,1096.0,2.0,39.0,2,3558
Yeshwanthpur,3 BHK,2504.0,3.0,138.0,3,5511
2nd Phase Judicial Layout,3 BHK,1350.0,2.0,47.25,3,3500
other,1 BHK,420.0,1.0,23.5,1,5595
Hennur Road,3 BHK,1832.0,3.0,116.0,3,6331
Anandapura,6 Bedroom,1380.0,7.0,130.0,6,9420
Vasanthapura,2 BHK,940.0,2.0,40.0,2,4255
Whitefield,4 Bedroom,4290.0,5.0,625.0,4,14568
Electronic City Phase II,2 BHK,1020.0,2.0,29.45,2,2887
Abbigere,6 Bedroom,2500.0,6.0,81.0,6,3240
Raja Rajeshwari Nagar,2 BHK,1400.0,2.0,45.0,2,3214
other,3 BHK,1675.0,3.0,65.0,3,3880
Tumkur Road,2 BHK,708.0,2.0,32.0,2,4519
Channasandra,2 BHK,1123.0,2.0,36.0,2,3205
Talaghattapura,2 BHK,1100.0,2.0,49.49,2,4499
other,2 BHK,1225.0,2.0,71.05,2,5800
Lingadheeranahalli,3 BHK,1683.0,3.0,140.0,3,8318
Kengeri Satellite Town,1 BHK,410.0,1.0,15.0,1,3658
Banashankari Stage II,3 Bedroom,2400.0,2.0,300.0,3,12500
other,4 Bedroom,1350.0,4.0,90.0,4,6666
Gubbalala,2 BHK,1475.0,2.0,80.0,2,5423
Kengeri Satellite Town,2 BHK,1120.0,2.0,49.0,2,4375
KR Puram,2 BHK,1150.0,2.0,42.0,2,3652
Yeshwanthpur,2 BHK,1165.0,2.0,85.0,2,7296
Harlur,3 BHK,1756.0,3.0,134.0,3,7630
Ambedkar Nagar,2 BHK,1409.0,2.0,95.0,2,6742
Mahadevpura,2 BHK,1212.0,2.0,56.9,2,4694
Doddathoguru,2 BHK,1140.0,2.0,26.49,2,2323
Hegde Nagar,3 BHK,2087.01,3.0,115.0,3,5510
Jalahalli East,1 BHK,775.0,1.0,34.1,1,4400
Sarjapur  Road,3 BHK,1700.0,3.0,118.0,3,6941
other,3 BHK,1732.5,3.0,91.855,3,5301
Raja Rajeshwari Nagar,2 BHK,1110.0,2.0,52.0,2,4684
Electronic City Phase II,2 BHK,1000.0,2.0,28.88,2,2888
Marathahalli,7 Bedroom,1550.0,7.0,160.0,7,10322
Chikka Tirupathi,4 Bedroom,2704.0,5.0,105.0,4,3883
other,2 Bedroom,3760.0,2.0,280.0,2,7446
Hormavu,5 Bedroom,1600.0,5.0,140.0,5,8750
Bhoganhalli,2 BHK,1205.0,2.0,64.47,2,5350
Thigalarapalya,2 BHK,1418.0,2.0,105.0,2,7404
Ardendale,3 BHK,1650.0,3.0,82.0,3,4969
other,7 Bedroom,1200.0,7.0,139.0,7,11583
Electronic City,2 BHK,1150.0,2.0,38.0,2,3304
other,3 BHK,3071.0,4.0,300.0,3,9768
Dasanapura,4 Bedroom,2400.0,5.0,100.0,4,4166
other,2 BHK,850.0,2.0,34.0,2,4000
Raja Rajeshwari Nagar,3 BHK,1580.0,3.0,83.74,3,5299
Whitefield,2 BHK,1216.0,2.0,84.03,2,6910
Kanakpura Road,1 BHK,381.0,1.0,28.0,1,7349
KR Puram,9 BHK,4600.0,9.0,200.0,9,4347
Hebbal,2 BHK,1040.0,2.0,55.0,2,5288
Choodasandra,4 Bedroom,2429.0,3.0,210.0,4,8645
other,3 BHK,1260.0,2.0,49.0,3,3888
Kanakpura Road,3 BHK,1498.0,3.0,74.9,3,5000
Vidyaranyapura,3 BHK,1200.0,2.0,65.0,3,5416
Electronic City,2 BHK,1355.0,2.0,72.9,2,5380
other,3 BHK,1450.0,3.0,65.0,3,4482
Yeshwanthpur,3 BHK,1860.0,3.0,168.0,3,9032
Whitefield,2 BHK,1240.0,2.0,70.0,2,5645
Kanakpura Road,3 BHK,1300.0,2.0,55.0,3,4230
Kundalahalli,4 Bedroom,2500.0,5.0,245.0,4,9800
Ambedkar Nagar,3 BHK,1950.0,4.0,120.0,3,6153
Ambalipura,3 BHK,1625.0,2.0,145.0,3,8923
Electronic City,2 BHK,1200.0,2.0,30.0,2,2500
OMBR Layout,2 BHK,1101.0,2.0,66.0,2,5994
Whitefield,2 BHK,1216.0,2.0,72.5,2,5962
Uttarahalli,3 BHK,1250.0,2.0,50.0,3,4000
other,3 Bedroom,2100.0,3.0,70.0,3,3333
Ananth Nagar,3 BHK,1470.0,3.0,54.0,3,3673
Bannerghatta Road,3 BHK,1365.0,3.0,76.18,3,5580
Hosa Road,2 BHK,1364.0,2.0,64.79,2,4750
Banashankari Stage VI,2 BHK,1180.0,2.0,59.47,2,5039
7th Phase JP Nagar,3 BHK,1800.0,3.0,130.0,3,7222
Kothannur,2 BHK,1350.0,2.0,42.0,2,3111
Thanisandra,3 BHK,1573.0,3.0,98.0,3,6230
other,3 BHK,1620.0,3.0,95.0,3,5864
Doddakallasandra,2 BHK,1072.0,2.0,42.87,2,3999
other,1 BHK,560.0,1.0,21.0,1,3750
other,2 BHK,1156.0,2.0,46.0,2,3979
Kudlu Gate,2 BHK,1336.0,2.0,86.0,2,6437
Electronic City,2 BHK,1200.0,2.0,20.0,2,1666
Marathahalli,4 BHK,3400.0,4.0,235.0,4,6911
Raja Rajeshwari Nagar,3 BHK,1608.0,3.0,54.51,3,3389
JP Nagar,4 BHK,4000.0,4.0,441.0,4,11025
Whitefield,2 BHK,1215.0,2.0,40.0,2,3292
Cox Town,2 BHK,1100.0,2.0,166.0,2,15090
Green Glen Layout,3 BHK,1517.0,3.0,100.0,3,6591
Mallasandra,2 BHK,1565.0,2.0,69.0,2,4408
other,6 Bedroom,1680.0,4.0,250.0,6,14880
Whitefield,3 BHK,2025.0,4.0,115.0,3,5679
Jalahalli,2 BHK,1057.0,2.0,34.71,2,3283
other,3 BHK,3850.0,3.0,650.0,3,16883
Sarjapur,2 BHK,1044.0,2.0,34.0,2,3256
Domlur,3 BHK,1695.0,3.0,125.0,3,7374
Ramagondanahalli,3 BHK,1635.0,3.0,61.0,3,3730
other,2 BHK,1140.0,2.0,56.0,2,4912
Electronic City,2 BHK,1150.0,2.0,35.0,2,3043
Yelachenahalli,2 BHK,1110.0,2.0,110.0,2,9909
Yeshwanthpur,2 BHK,1163.0,2.0,64.08,2,5509
Thubarahalli,3 BHK,1418.0,2.0,65.0,3,4583
Varthur,2 BHK,1155.0,2.0,60.0,2,5194
Bannerghatta Road,2 BHK,1115.0,2.0,78.0,2,6995
other,3 BHK,2250.0,3.0,180.0,3,8000
NGR Layout,2 BHK,1020.0,2.0,48.45,2,4750
other,3 BHK,1485.0,3.0,78.0,3,5252
Whitefield,4 Bedroom,1920.0,5.0,250.0,4,13020
other,1 Bedroom,1350.0,2.0,45.0,1,3333
other,2 BHK,1209.0,2.0,100.0,2,8271
8th Phase JP Nagar,2 BHK,1089.0,2.0,43.55,2,3999
Yelahanka,2 BHK,1445.0,2.0,82.0,2,5674
Uttarahalli,3 BHK,1250.0,2.0,50.0,3,4000
other,3 BHK,1500.0,3.0,95.0,3,6333
Devarachikkanahalli,3 BHK,1705.0,3.0,75.0,3,4398
Indira Nagar,3 BHK,2200.0,3.0,160.0,3,7272
Cunningham Road,5 Bedroom,2925.0,5.0,936.0,5,32000
Laggere,8 Bedroom,1800.0,8.0,110.0,8,6111
Whitefield,5 Bedroom,7200.0,5.0,900.0,5,12500
Bellandur,3 BHK,1830.0,3.0,90.0,3,4918
Electronics City Phase 1,3 BHK,1450.0,3.0,58.0,3,4000
Choodasandra,3 BHK,1220.0,3.0,56.0,3,4590
Hoodi,2 BHK,1259.0,2.0,69.45,2,5516
Subramanyapura,3 BHK,1260.0,2.0,73.0,3,5793
Kaggadasapura,3 Bedroom,2500.0,5.0,120.0,3,4800
Kudlu Gate,3 BHK,1919.0,3.0,121.0,3,6305
Vishveshwarya Layout,6 Bedroom,1400.0,6.0,160.0,6,11428
Hennur Road,3 BHK,2050.0,3.0,120.0,3,5853
Basavangudi,2 BHK,1050.0,2.0,103.0,2,9809
other,2 BHK,1000.0,2.0,60.0,2,6000
Raja Rajeshwari Nagar,3 BHK,1640.0,3.0,86.8,3,5292
other,3 BHK,1488.0,2.0,65.0,3,4368
Chikkalasandra,3 BHK,1265.0,2.0,53.12,3,4199
Hennur,3 BHK,1640.0,3.0,120.0,3,7317
9th Phase JP Nagar,3 BHK,1719.3,2.0,88.0,3,5118
other,4 Bedroom,1200.0,4.0,130.0,4,10833
Kaggadasapura,2 BHK,1180.0,2.0,70.0,2,5932
HBR Layout,3 BHK,1700.0,3.0,110.0,3,6470
Kudlu Gate,2 BHK,1464.0,3.0,56.0,2,3825
JP Nagar,6 Bedroom,1800.0,6.0,68.0,6,3777
Cox Town,4 Bedroom,1495.0,2.0,300.0,4,20066
Yelahanka,2 BHK,1225.0,2.0,62.0,2,5061
7th Phase JP Nagar,2 BHK,1050.0,2.0,68.0,2,6476
Sarjapur  Road,3 BHK,1691.0,3.0,120.0,3,7096
other,3 BHK,2250.0,3.0,199.0,3,8844
other,4 BHK,3700.0,4.0,314.0,4,8486
Jalahalli,8 Bedroom,1244.0,6.0,130.0,8,10450
other,4 Bedroom,750.0,4.0,120.0,4,16000
other,2 BHK,1190.0,2.0,42.25,2,3550
Haralur Road,2 BHK,1243.0,2.0,46.0,2,3700
Hormavu,2 BHK,765.0,2.0,37.49,2,4900
Chamrajpet,2 BHK,730.0,1.0,52.0,2,7123
other,3 BHK,1405.0,3.0,169.0,3,12028
Old Madras Road,3 BHK,1350.0,3.0,45.9,3,3400
Yelahanka New Town,1 BHK,500.0,1.0,18.0,1,3600
Amruthahalli,2 BHK,1200.0,2.0,55.0,2,4583
Electronic City,2 BHK,1165.0,2.0,33.64,2,2887
7th Phase JP Nagar,3 BHK,1400.0,2.0,57.0,3,4071
other,4 Bedroom,4800.0,6.0,420.0,4,8750
Hormavu,3 BHK,1668.0,2.0,120.0,3,7194
Kenchenahalli,1 BHK,700.0,1.0,35.0,1,5000
5th Phase JP Nagar,3 BHK,1485.0,2.0,66.0,3,4444
Kanakpura Road,2 BHK,700.0,2.0,41.0,2,5857
Whitefield,3 BHK,1346.0,2.0,100.0,3,7429
Bisuvanahalli,3 BHK,1075.0,2.0,47.0,3,4372
other,2 BHK,1050.0,2.0,43.5,2,4142
other,3 BHK,1800.0,3.0,53.0,3,2944
Electronic City,3 BHK,1609.0,3.0,115.0,3,7147
Electronic City,2 BHK,770.0,1.0,35.0,2,4545
Kudlu,2 BHK,1076.0,2.0,50.0,2,4646
Kengeri,2 BHK,1052.0,2.0,51.0,2,4847
Channasandra,2 BHK,1000.0,2.0,35.0,2,3500
Bommasandra Industrial Area,2 BHK,1000.0,2.0,28.88,2,2888
Thanisandra,3 BHK,1262.0,2.0,83.0,3,6576
other,3 Bedroom,720.0,3.0,48.0,3,6666
other,1 BHK,500.0,1.0,16.0,1,3200
other,4 BHK,3500.0,4.0,700.0,4,20000
other,3 BHK,1600.0,3.0,180.0,3,11250
Ramagondanahalli,2 BHK,1251.0,2.0,53.0,2,4236
Nagarbhavi,3 Bedroom,1050.0,3.0,175.0,3,16666
other,2 BHK,1140.0,2.0,49.0,2,4298
Old Madras Road,2 BHK,1157.0,2.0,47.32,2,4089
other,1 BHK,648.0,1.0,34.0,1,5246
Harlur,2 BHK,1197.0,2.0,78.0,2,6516
Raja Rajeshwari Nagar,3 BHK,1500.0,3.0,51.04,3,3402
Raja Rajeshwari Nagar,2 BHK,1185.0,2.0,51.15,2,4316
Kumaraswami Layout,3 Bedroom,2400.0,2.0,125.0,3,5208
Kundalahalli,3 BHK,1397.0,3.0,105.0,3,7516
Uttarahalli,2 BHK,1175.0,2.0,47.0,2,4000
Basaveshwara Nagar,2 BHK,900.0,2.0,60.0,2,6666
Rachenahalli,3 BHK,2710.0,3.0,170.0,3,6273
other,3 BHK,1720.0,2.0,126.0,3,7325
other,3 BHK,1756.0,3.0,307.0,3,17482
Yelahanka,1 BHK,602.0,1.0,38.0,1,6312
Sarjapur  Road,3 BHK,1770.0,3.0,180.0,3,10169
Kothannur,6 Bedroom,930.0,6.0,150.0,6,16129
Raja Rajeshwari Nagar,2 BHK,1295.0,2.0,43.9,2,3389
Rachenahalli,2 BHK,1200.0,2.0,46.5,2,3875
other,2 BHK,1100.0,2.0,45.0,2,4090
Haralur Road,2 BHK,1194.0,2.0,46.0,2,3852
Marsur,3 BHK,720.0,2.0,32.0,3,4444
Kasavanhalli,3 BHK,1476.0,3.0,105.0,3,7113
Thanisandra,2 BHK,1056.0,2.0,65.0,2,6155
other,2 BHK,825.0,2.0,35.0,2,4242
EPIP Zone,3 BHK,1860.0,3.0,130.0,3,6989
Gunjur,2 BHK,1063.0,2.0,40.0,2,3762
Yelahanka,1 BHK,697.0,1.0,35.0,1,5021
Vijayanagar,1 Bedroom,2400.0,2.0,230.0,1,9583
Anekal,2 Bedroom,1200.0,1.0,36.0,2,3000
CV Raman Nagar,3 BHK,1835.0,3.0,160.0,3,8719
Bannerghatta Road,3 BHK,1532.5,3.0,79.465,3,5185
BTM 2nd Stage,2 BHK,1147.0,2.0,75.0,2,6538
Uttarahalli,3 BHK,1360.0,2.0,47.59,3,3499
Kothanur,4 Bedroom,3400.0,5.0,265.0,4,7794
other,2 BHK,1080.0,2.0,64.0,2,5925
Doddakallasandra,2 BHK,1010.0,2.0,40.4,2,4000
other,4 Bedroom,2600.0,4.0,523.0,4,20115
Brookefield,2 BHK,1270.0,2.0,95.0,2,7480
other,2 BHK,1100.0,2.0,61.0,2,5545
Bellandur,2 BHK,1410.0,3.0,92.0,2,6524
Yelahanka,2 BHK,1065.0,2.0,41.38,2,3885
Anekal,2 Bedroom,1200.0,1.0,36.0,2,3000
Hebbal,3 BHK,1315.0,3.0,65.0,3,4942
Indira Nagar,3 BHK,1650.0,3.0,200.0,3,12121
Haralur Road,2 BHK,1140.0,2.0,43.0,2,3771
Kaggadasapura,2 BHK,960.0,2.0,45.0,2,4687
other,2 BHK,800.0,1.0,29.0,2,3625
other,2 BHK,1225.0,2.0,47.0,2,3836
Whitefield,2 BHK,1020.0,2.0,45.0,2,4411
Magadi Road,2 BHK,750.0,2.0,27.0,2,3600
other,3 Bedroom,3000.0,3.0,550.0,3,18333
other,1 BHK,535.0,1.0,35.0,1,6542
Varthur,2 BHK,1035.0,2.0,60.0,2,5797
Bommasandra Industrial Area,2 BHK,1020.0,2.0,29.45,2,2887
GM Palaya,3 BHK,1315.0,3.0,64.0,3,4866
Basavangudi,3 BHK,1350.0,2.0,130.0,3,9629
Bisuvanahalli,3 BHK,1075.0,2.0,48.0,3,4465
Kengeri Satellite Town,3 BHK,1450.0,2.0,50.0,3,3448
Gottigere,1 Bedroom,1806.0,1.0,150.0,1,8305
Marathahalli,2 BHK,1350.0,2.0,99.0,2,7333
Binny Pete,3 Bedroom,1282.0,3.0,178.0,3,13884
Sarjapur  Road,3 BHK,1929.0,3.0,103.0,3,5339
Varthur,2 BHK,1100.0,2.0,72.0,2,6545
Kundalahalli,3 BHK,1800.0,3.0,110.0,3,6111
Gubbalala,2 BHK,1223.0,2.0,42.81,2,3500
Bharathi Nagar,3 BHK,1664.0,3.0,69.89,3,4200
Uttarahalli,3 BHK,1360.0,2.0,47.59,3,3499
Kasavanhalli,2 BHK,1158.0,2.0,67.0,2,5785
Hormavu,3 BHK,1420.0,3.0,58.0,3,4084
Rajaji Nagar,2 BHK,1268.0,2.0,127.0,2,10015
Sarjapur  Road,2 BHK,1308.0,2.0,75.0,2,5733
Basavangudi,3 BHK,1485.0,3.0,140.0,3,9427
other,2 BHK,1067.0,2.0,33.99,2,3185
Hulimavu,4 Bedroom,2000.0,4.0,100.0,4,5000
Rajaji Nagar,3 BHK,2450.0,3.0,330.0,3,13469
other,2 BHK,1152.0,2.0,42.0,2,3645
Uttarahalli,3 BHK,1525.0,2.0,62.82,3,4119
Whitefield,2 BHK,935.0,2.0,32.72,2,3499
other,3 BHK,1465.0,2.0,88.0,3,6006
HBR Layout,2 BHK,1198.0,2.0,65.0,2,5425
other,2 BHK,1145.0,2.0,63.0,2,5502
Attibele,2 Bedroom,1000.0,2.0,42.0,2,4200
Anekal,1 BHK,400.0,1.0,11.5,1,2875
other,4 BHK,3420.0,3.0,410.0,4,11988
Marathahalli,3 BHK,1310.0,3.0,63.22,3,4825
Panathur,3 BHK,1861.0,3.0,92.0,3,4943
Uttarahalli,3 BHK,1265.0,2.0,65.0,3,5138
other,2 BHK,1280.0,2.0,75.0,2,5859
Raja Rajeshwari Nagar,3 BHK,1279.0,2.0,53.76,3,4203
TC Palaya,3 Bedroom,1200.0,2.0,66.0,3,5500
other,2 Bedroom,680.0,3.0,60.0,2,8823
Uttarahalli,3 BHK,1590.0,3.0,57.0,3,3584
Hoodi,3 BHK,1837.0,3.0,142.0,3,7729
Hegde Nagar,3 BHK,1703.0,3.0,130.0,3,7633
Kudlu,3 BHK,1293.0,2.0,80.0,3,6187
Kanakpura Road,2 BHK,700.0,2.0,35.0,2,5000
other,3 BHK,1450.0,3.0,75.0,3,5172
5th Phase JP Nagar,2 BHK,1030.0,2.0,57.0,2,5533
Kanakpura Road,3 BHK,1570.0,3.0,65.93,3,4199
7th Phase JP Nagar,2 BHK,1000.0,2.0,60.0,2,6000
Bisuvanahalli,3 BHK,1075.0,2.0,37.0,3,3441
Varthur,2 BHK,1091.0,2.0,33.82,2,3099
EPIP Zone,3 BHK,2210.0,3.0,165.0,3,7466
other,3 BHK,1738.0,3.0,110.0,3,6329
Rajaji Nagar,3 BHK,1640.0,3.0,262.0,3,15975
Gottigere,2 BHK,1205.0,2.0,56.58,2,4695
Kengeri,2 BHK,1009.0,2.0,30.27,2,3000
2nd Stage Nagarbhavi,5 Bedroom,1200.0,4.0,240.0,5,20000
Thigalarapalya,4 BHK,3362.0,5.0,249.0,4,7406
Rajaji Nagar,4 Bedroom,1600.0,3.0,140.0,4,8750
Sanjay nagar,2 BHK,960.0,2.0,55.0,2,5729
Uttarahalli,3 BHK,1330.0,2.0,46.55,3,3500
Whitefield,2 BHK,1024.0,2.0,32.0,2,3125
Raja Rajeshwari Nagar,2 BHK,1196.0,2.0,40.54,2,3389
Mico Layout,1 BHK,800.0,1.0,29.5,1,3687
Sarjapur  Road,3 BHK,2650.0,3.0,260.0,3,9811
Jakkur,2 BHK,1080.0,2.0,60.0,2,5555
Hebbal,2 BHK,1150.0,2.0,55.0,2,4782
Dasanapura,3 BHK,1286.0,2.0,68.0,3,5287
Whitefield,4 Bedroom,2990.0,3.0,198.0,4,6622
JP Nagar,2 BHK,1150.0,2.0,90.0,2,7826
Cooke Town,2 BHK,1310.0,2.0,111.0,2,8473
Prithvi Layout,2 BHK,1352.0,2.0,72.0,2,5325
Bommanahalli,2 BHK,1307.0,2.0,50.0,2,3825
Channasandra,4 Bedroom,1500.0,4.0,89.45,4,5963
other,3 Bedroom,1500.0,3.0,90.0,3,6000
Thanisandra,2 BHK,1131.0,2.0,55.975,2,4949
Whitefield,3 Bedroom,1600.0,3.0,69.1,3,4318
Hosur Road,4 BHK,2835.0,4.0,241.0,4,8500
Yelahanka,1 BHK,682.0,1.0,36.23,1,5312
Whitefield,4 Bedroom,2500.0,4.0,420.0,4,16800
Vittasandra,3 BHK,1650.0,3.0,85.5,3,5181
Haralur Road,3 BHK,1850.0,3.0,120.0,3,6486
Gubbalala,4 Bedroom,2000.0,4.0,85.0,4,4250
other,2 BHK,982.0,2.0,25.0,2,2545
other,2 BHK,2015.0,4.0,325.0,2,16129
Thanisandra,3 BHK,1702.0,3.0,115.0,3,6756
Amruthahalli,2 BHK,1250.0,2.0,79.0,2,6320
Raja Rajeshwari Nagar,2 BHK,1240.0,2.0,40.54,2,3269
other,2 Bedroom,1050.0,2.0,110.0,2,10476
Yelahanka,2 BHK,1023.0,2.0,52.59,2,5140
other,2 BHK,1541.0,2.0,181.0,2,11745
other,2 BHK,1275.0,2.0,105.0,2,8235
Jakkur,2 BHK,1260.0,2.0,71.195,2,5650
Sarjapur  Road,3 BHK,1580.0,3.0,125.0,3,7911
Thanisandra,2 BHK,1343.0,2.0,75.0,2,5584
Bommasandra,3 BHK,1365.0,3.0,52.81,3,3868
Kundalahalli,2 BHK,1047.0,2.0,72.0,2,6876
Shampura,2 BHK,1187.0,2.0,42.0,2,3538
Banashankari Stage III,4 Bedroom,1050.0,3.0,147.0,4,14000
Frazer Town,1 Bedroom,560.0,1.0,90.0,1,16071
Banashankari Stage III,3 Bedroom,1000.0,2.0,120.0,3,12000
other,5 Bedroom,1200.0,5.0,260.0,5,21666
Choodasandra,2 BHK,1075.0,2.0,45.0,2,4186
other,3 BHK,2048.0,3.0,206.0,3,10058
Raja Rajeshwari Nagar,2 BHK,1090.0,2.0,57.0,2,5229
other,3 BHK,1119.0,2.0,48.5,3,4334
other,3 BHK,1355.0,2.0,58.25,3,4298
Hennur Road,3 BHK,1445.0,2.0,86.69,3,5999
Electronics City Phase 1,6 Bedroom,1314.0,5.0,120.0,6,9132
Malleshwaram,4 BHK,6500.0,4.0,1400.0,4,21538
Yeshwanthpur,2 BHK,1165.0,2.0,64.08,2,5500
Hormavu,2 BHK,1081.5,2.0,38.665,2,3575
Kothanur,3 BHK,1787.0,3.0,107.0,3,5987
Mico Layout,4 Bedroom,3600.0,4.0,330.0,4,9166
Battarahalli,6 Bedroom,1200.0,6.0,100.0,6,8333
Electronic City,2 BHK,1020.0,2.0,29.45,2,2887
Sarjapur  Road,3 BHK,1797.0,4.0,95.0,3,5286
Whitefield,2 BHK,1205.0,2.0,58.0,2,4813
other,2 BHK,1095.0,2.0,55.85,2,5100
Yelahanka,3 BHK,1560.0,2.0,98.0,3,6282
5th Phase JP Nagar,2 BHK,1200.0,2.0,68.0,2,5666
Rajaji Nagar,2 Bedroom,1200.0,1.0,200.0,2,16666
Kanakpura Road,3 BHK,1450.0,3.0,65.0,3,4482
Whitefield,3 BHK,1496.0,2.0,718.0,3,47994
Thanisandra,3 BHK,1935.0,4.0,122.0,3,6304
Hosa Road,2 BHK,1016.0,2.0,45.0,2,4429
Jakkur,3 BHK,1816.0,3.0,119.0,3,6552
7th Phase JP Nagar,3 BHK,1680.0,3.0,125.0,3,7440
Hoodi,4 BHK,2065.5,4.0,91.915,4,4450
Kengeri,2 BHK,1220.0,2.0,63.0,2,5163
Hosur Road,4 Bedroom,3000.0,4.0,160.0,4,5333
7th Phase JP Nagar,2 BHK,980.0,2.0,86.0,2,8775
Nagarbhavi,4 Bedroom,1200.0,4.0,210.0,4,17500
Hormavu,2 BHK,1250.0,2.0,60.0,2,4800
Jakkur,2 BHK,1424.0,2.0,80.0,2,5617
Hennur Road,2 BHK,1297.0,2.0,49.9,2,3847
Karuna Nagar,3 Bedroom,1300.0,3.0,170.0,3,13076
Harlur,2 BHK,1290.0,2.0,89.9,2,6968
Koramangala,3 BHK,1720.0,2.0,130.0,3,7558
Judicial Layout,7 BHK,1200.0,7.0,199.0,7,16583
other,3 Bedroom,2320.0,2.0,160.0,3,6896
other,3 BHK,1237.0,3.0,78.0,3,6305
Whitefield,3 BHK,1720.0,3.0,98.97,3,5754
Chikkalasandra,2 BHK,1224.0,2.0,45.0,2,3676
Yelahanka,3 BHK,1780.0,3.0,107.0,3,6011
Seegehalli,4 Bedroom,4700.0,5.0,800.0,4,17021
Electronics City Phase 1,3 BHK,1800.0,3.0,93.0,3,5166
Budigere,2 BHK,1139.0,2.0,56.5,2,4960
other,1 Bedroom,588.0,1.0,88.2,1,15000
2nd Phase Judicial Layout,2 BHK,700.0,2.0,35.0,2,5000
Kengeri,1 Bedroom,1150.0,1.0,52.0,1,4521
Channasandra,3 BHK,1330.0,2.0,51.83,3,3896
Old Airport Road,4 BHK,2732.0,4.0,194.0,4,7101
Kodigehaali,2 BHK,965.0,2.0,38.0,2,3937
Whitefield,3 BHK,1600.0,3.0,95.0,3,5937
other,2 BHK,1464.0,2.0,56.0,2,3825
Frazer Town,3 BHK,3100.0,3.0,400.0,3,12903
Bisuvanahalli,3 BHK,1075.0,2.0,45.0,3,4186
other,2 Bedroom,1840.0,1.0,95.0,2,5163
Gubbalala,3 BHK,1223.0,3.0,39.14,3,3200
Ramamurthy Nagar,2 BHK,1170.0,2.0,44.0,2,3760
Old Madras Road,4 BHK,3630.0,6.0,207.0,4,5702
Bellandur,2 BHK,950.0,2.0,35.0,2,3684
Frazer Town,3 BHK,5400.0,3.0,400.0,3,7407
Electronic City,2 BHK,865.0,2.0,40.0,2,4624
other,4 Bedroom,2400.0,4.0,89.0,4,3708
Lakshminarayana Pura,2 BHK,1169.0,2.0,75.0,2,6415
Hormavu,2 BHK,1206.0,2.0,40.0,2,3316
other,4 Bedroom,10961.0,4.0,80.0,4,729
Lingadheeranahalli,3 BHK,1683.0,3.0,110.0,3,6535
Mico Layout,3 BHK,1250.0,2.0,44.0,3,3520
Green Glen Layout,4 BHK,3150.0,4.0,218.0,4,6920
other,4 Bedroom,1200.0,2.0,140.0,4,11666
Bhoganhalli,3 BHK,1234.6,3.0,104.0,3,8423
Vijayanagar,1 BHK,492.0,1.0,40.0,1,8130
Banashankari,2 BHK,600.0,1.0,27.0,2,4500
Kalena Agrahara,3 BHK,1902.0,3.0,89.0,3,4679
other,6 Bedroom,1200.0,6.0,125.0,6,10416
other,3 BHK,1639.0,3.0,77.89,3,4752
Kasavanhalli,2 BHK,1214.0,2.0,67.0,2,5518
other,2 BHK,565.0,2.0,15.0,2,2654
Basaveshwara Nagar,5 Bedroom,2520.0,4.0,378.0,5,15000
Harlur,2 BHK,1386.0,2.0,85.0,2,6132
Bannerghatta Road,3 BHK,1535.0,3.0,60.0,3,3908
Marathahalli,3 BHK,1720.0,3.0,175.0,3,10174
7th Phase JP Nagar,3 BHK,1370.0,2.0,54.79,3,3999
Bannerghatta Road,3 BHK,1655.0,3.0,90.0,3,5438
Vidyaranyapura,2 BHK,1200.0,2.0,52.0,2,4333
Horamavu Banaswadi,3 BHK,1610.0,3.0,68.0,3,4223
Ramamurthy Nagar,4 Bedroom,2200.0,4.0,130.0,4,5909
TC Palaya,3 Bedroom,1200.0,2.0,66.0,3,5500
Mahalakshmi Layout,2 BHK,1105.0,2.0,80.0,2,7239
Banjara Layout,2 Bedroom,753.0,3.0,59.5,2,7901
Whitefield,3 BHK,1655.0,3.0,108.0,3,6525
other,3 Bedroom,1700.0,3.0,250.0,3,14705
Vishwapriya Layout,2 BHK,770.0,2.0,30.0,2,3896
Balagere,2 BHK,1210.0,2.0,69.0,2,5702
Hormavu,2 BHK,965.0,2.0,37.15,2,3849
Kanakpura Road,2 BHK,700.0,1.0,41.0,2,5857
Whitefield,3 BHK,1740.0,3.0,95.0,3,5459
Electronic City Phase II,3 BHK,1310.0,2.0,37.84,3,2888
Sarjapur  Road,3 BHK,1763.25,3.0,98.0,3,5557
Basaveshwara Nagar,3 BHK,1900.0,3.0,150.0,3,7894
Kammanahalli,6 Bedroom,1000.0,6.0,150.0,6,15000
JP Nagar,3 BHK,1450.0,2.0,55.0,3,3793
CV Raman Nagar,3 BHK,1836.0,3.0,148.0,3,8061
other,2 BHK,1160.0,2.0,75.0,2,6465
Hennur Road,2 BHK,1153.0,2.0,43.0,2,3729
Marathahalli,2 BHK,1285.0,2.0,55.0,2,4280
other,3 BHK,2500.0,3.0,400.0,3,16000
Haralur Road,2 BHK,1194.0,2.0,47.0,2,3936
Kadubeesanahalli,2 BHK,1257.0,2.0,105.0,2,8353
Whitefield,3 Bedroom,1200.0,3.0,56.72,3,4726
Hormavu,1 BHK,583.0,1.0,28.275,1,4849
Tumkur Road,3 BHK,1240.0,2.0,79.0,3,6370
other,3 BHK,1294.0,2.0,52.0,3,4018
other,2 BHK,1095.0,2.0,45.0,2,4109
HSR Layout,2 BHK,1145.0,2.0,46.0,2,4017
Sarjapur  Road,3 BHK,1826.0,3.0,130.0,3,7119
Electronics City Phase 1,2 BHK,1314.0,2.0,60.0,2,4566
Koramangala,3 BHK,1900.0,2.0,120.0,3,6315
other,2 BHK,1040.0,2.0,36.84,2,3542
Jakkur,2 BHK,1290.0,2.0,85.0,2,6589
Kodigehaali,4 Bedroom,1200.0,4.0,130.0,4,10833
other,3 BHK,1325.0,2.0,47.0,3,3547
Hebbal,3 BHK,2600.0,3.0,195.0,3,7500
Hormavu,2 BHK,1129.0,2.0,44.0,2,3897
Balagere,1 BHK,790.5,1.0,42.295,1,5350
Hebbal,2 BHK,1075.0,2.0,52.0,2,4837
other,2 BHK,1108.0,2.0,48.0,2,4332
Ananth Nagar,2 BHK,810.0,2.0,25.5,2,3148
other,2 BHK,1155.0,2.0,65.0,2,5627
Rajaji Nagar,6 BHK,3900.0,6.0,260.0,6,6666
Rajaji Nagar,3 BHK,2367.0,3.0,375.0,3,15842
Sarjapur  Road,3 BHK,1550.0,2.0,98.0,3,6322
Babusapalaya,2 BHK,1100.0,2.0,38.0,2,3454
Kanakpura Road,3 BHK,1452.0,3.0,54.2,3,3732
Kothanur,2 BHK,1141.0,2.0,54.0,2,4732
other,3 BHK,1464.0,3.0,56.0,3,3825
other,3 BHK,1500.0,3.0,67.49,3,4499
Anekal,1 BHK,456.0,1.0,15.0,1,3289
Ramamurthy Nagar,4 Bedroom,1200.0,4.0,125.0,4,10416
Thigalarapalya,4 BHK,3122.0,6.0,245.0,4,7847
Horamavu Agara,2 BHK,760.0,2.0,37.49,2,4932
Koramangala,6 Bedroom,2400.0,6.0,480.0,6,20000
Electronic City,2 BHK,800.0,2.0,32.0,2,4000
Choodasandra,2 BHK,1180.0,2.0,71.22,2,6035
Kaggadasapura,2 BHK,775.0,2.0,29.5,2,3806
Sarjapur  Road,2 BHK,1145.0,2.0,49.5,2,4323
other,2 BHK,1060.0,2.0,42.5,2,4009
Cooke Town,4 BHK,3900.0,5.0,415.0,4,10641
Yelahanka New Town,3 BHK,1420.0,2.0,75.0,3,5281
Bhoganhalli,4 BHK,2439.0,4.0,190.0,4,7790
1st Phase JP Nagar,4 Bedroom,1200.0,4.0,300.0,4,25000
Hoodi,2 BHK,1111.0,2.0,49.43,2,4449
other,2 Bedroom,984.0,2.0,125.0,2,12703
Begur Road,2 BHK,1200.0,2.0,42.0,2,3500
Bannerghatta Road,2 BHK,1247.0,2.0,47.39,2,3800
7th Phase JP Nagar,2 BHK,1050.0,2.0,77.47,2,7378
Devanahalli,3 BHK,1550.0,3.0,85.0,3,5483
7th Phase JP Nagar,2 BHK,1270.0,2.0,80.0,2,6299
Balagere,2 BHK,1012.0,2.0,65.0,2,6422
Electronic City,2 BHK,970.0,2.0,34.0,2,3505
Thanisandra,2 BHK,933.0,2.0,55.0,2,5894
Sarjapur,3 Bedroom,1200.0,3.0,76.13,3,6344
other,2 BHK,1148.0,2.0,69.0,2,6010
7th Phase JP Nagar,3 BHK,1463.0,2.0,61.45,3,4200
other,3 BHK,1719.0,3.0,110.0,3,6399
Whitefield,2 BHK,1350.0,2.0,88.0,2,6518
Kaggadasapura,3 BHK,1250.0,2.0,47.0,3,3760
1st Block Jayanagar,8 Bedroom,1700.0,3.0,50.0,8,2941
Mahalakshmi Layout,9 Bedroom,4320.0,7.0,821.0,9,19004
Lakshminarayana Pura,2 BHK,1200.0,2.0,75.0,2,6250
Whitefield,2 BHK,1030.0,2.0,44.0,2,4271
other,3 BHK,1420.0,3.0,70.0,3,4929
Ramagondanahalli,5 Bedroom,9600.0,6.0,1800.0,5,18750
Kanakpura Road,3 BHK,1703.0,3.0,126.0,3,7398
Old Madras Road,2 BHK,1210.0,2.0,80.0,2,6611
Horamavu Banaswadi,2 BHK,1210.0,2.0,50.0,2,4132
other,4 Bedroom,3000.0,4.0,264.0,4,8800
other,3 BHK,1398.0,3.0,51.0,3,3648
Nagavarapalya,2 BHK,1335.0,2.0,110.0,2,8239
Hennur,2 BHK,1255.0,2.0,52.32,2,4168
Mysore Road,4 Bedroom,540.0,4.0,85.0,4,15740
Babusapalaya,3 BHK,1213.0,2.0,36.37,3,2998
Raja Rajeshwari Nagar,2 BHK,1240.0,2.0,42.04,2,3390
Yelahanka,2 BHK,1076.0,2.0,78.0,2,7249
other,2 BHK,1320.0,2.0,90.0,2,6818
Whitefield,2 BHK,1216.0,2.0,80.0,2,6578
Thanisandra,2 BHK,1100.0,2.0,42.0,2,3818
Thanisandra,2 BHK,1183.0,2.0,77.3,2,6534
Balagere,1 BHK,656.0,1.0,38.77,1,5910
Sarjapur  Road,1 BHK,534.0,1.0,19.65,1,3679
other,3 BHK,1285.0,2.0,45.0,3,3501
Hegde Nagar,3 BHK,2087.01,4.0,135.0,3,6468
other,2 BHK,1161.0,2.0,59.0,2,5081
Begur,2 BHK,1306.0,2.0,65.0,2,4977
Jakkur,4 BHK,2249.81,4.0,245.0,4,10889
other,3 BHK,1504.0,3.0,67.0,3,4454
Balagere,1 BHK,645.0,1.0,39.0,1,6046
other,2 Bedroom,1230.0,2.0,68.0,2,5528
Hebbal,2 BHK,1100.0,2.0,42.0,2,3818
Yelahanka,1 BHK,1000.0,1.0,14.0,1,1400
other,2 BHK,1260.0,2.0,62.0,2,4920
Ambalipura,3 BHK,1817.0,3.0,102.0,3,5613
Kodihalli,4 BHK,3626.0,5.0,785.0,4,21649
Basavangudi,2 BHK,1180.0,2.0,124.0,2,10508
KR Puram,2 BHK,1050.0,2.0,55.0,2,5238
Whitefield,4 Bedroom,2900.0,4.0,225.0,4,7758
Kasavanhalli,3 Bedroom,2110.0,4.0,120.0,3,5687
Mysore Road,1 BHK,500.0,1.0,17.5,1,3500
Murugeshpalya,3 BHK,1930.0,3.0,75.0,3,3886
other,4 Bedroom,4560.0,5.0,430.0,4,9429
Mahadevpura,3 BHK,1505.0,3.0,64.0,3,4252
other,4 Bedroom,2400.0,3.0,750.0,4,31250
Ananth Nagar,2 BHK,890.0,2.0,27.0,2,3033
Anekal,2 Bedroom,1200.0,1.0,36.0,2,3000
other,2 BHK,849.0,2.0,25.4,2,2991
Chikka Tirupathi,3 Bedroom,1555.0,3.0,100.0,3,6430
other,3 Bedroom,600.0,2.0,85.0,3,14166
KR Puram,2 BHK,1057.0,2.0,30.0,2,2838
7th Phase JP Nagar,3 BHK,1680.0,3.0,122.0,3,7261
other,2 BHK,799.0,2.0,35.0,2,4380
Sarjapur  Road,3 BHK,1450.0,2.0,75.0,3,5172
Sarjapur  Road,3 BHK,1819.0,3.0,100.0,3,5497
Kanakpura Road,2 BHK,1135.0,2.0,56.74,2,4999
HSR Layout,3 BHK,2020.0,3.0,160.0,3,7920
other,5 BHK,2700.0,4.0,170.0,5,6296
Hosa Road,3 BHK,1730.0,3.0,72.6,3,4196
Dasarahalli,3 BHK,1950.0,3.0,125.0,3,6410
other,2 BHK,1077.0,2.0,34.46,2,3199
Brookefield,3 BHK,1706.0,3.0,115.0,3,6740
other,5 BHK,3000.0,4.0,300.0,5,10000
Channasandra,2 BHK,1093.0,2.0,37.0,2,3385
Ambalipura,4 BHK,2750.0,4.0,142.0,4,5163
Whitefield,2 BHK,1340.0,2.0,101.0,2,7537
Sarjapur  Road,2 BHK,984.0,2.0,59.0,2,5995
AECS Layout,2 BHK,1135.0,2.0,68.0,2,5991
Bannerghatta Road,4 Bedroom,2400.0,4.0,344.0,4,14333
Varthur,2 BHK,1155.0,2.0,39.0,2,3376
Kaggalipura,1 BHK,700.0,1.0,35.0,1,5000
Horamavu Agara,2 BHK,1150.0,2.0,42.0,2,3652
Hormavu,3 BHK,1385.0,2.0,69.25,3,5000
Akshaya Nagar,3 BHK,2061.0,4.0,175.0,3,8491
Kathriguppe,3 BHK,1335.0,2.0,73.43,3,5500
Bellandur,2 BHK,1015.0,2.0,60.0,2,5911
Banashankari Stage III,6 Bedroom,3000.0,6.0,150.0,6,5000
Electronic City,2 BHK,1108.0,2.0,63.0,2,5685
Frazer Town,2 BHK,1180.0,2.0,105.0,2,8898
Hennur Road,3 BHK,1445.0,2.0,83.0,3,5743
Vasanthapura,2 BHK,1500.0,2.0,65.0,2,4333
KR Puram,2 BHK,1085.0,2.0,46.5,2,4285
Rachenahalli,2 BHK,1050.0,2.0,52.08,2,4960
Kadugodi,4 Bedroom,2760.0,4.0,290.0,4,10507
Sarjapura - Attibele Road,3 BHK,1330.0,2.0,49.0,3,3684
Ramagondanahalli,2 Bedroom,845.0,2.0,45.82,2,5422
Kadugodi,3 BHK,1762.0,3.0,112.0,3,6356
Ambalipura,2 BHK,1303.0,2.0,80.0,2,6139
other,3 BHK,1560.0,2.0,125.0,3,8012
Chandapura,3 BHK,1110.0,2.0,29.97,3,2700
other,3 BHK,1290.0,3.0,45.33,3,3513
Bannerghatta Road,2 BHK,1237.5,2.0,49.49,2,3999
9th Phase JP Nagar,3 BHK,1780.0,3.0,88.0,3,4943
Budigere,3 BHK,1636.0,3.0,83.0,3,5073
Sector 2 HSR Layout,2 BHK,1332.0,2.0,101.0,2,7582
other,5 BHK,7500.0,5.0,1500.0,5,20000
other,2 BHK,1012.0,2.0,59.0,2,5830
Kengeri,2 BHK,1192.0,2.0,53.5,2,4488
Abbigere,2 BHK,795.0,2.0,32.54,2,4093
Uttarahalli,2 BHK,1213.0,2.0,48.0,2,3957
Sarjapur  Road,3 BHK,1489.0,3.0,88.0,3,5910
Sarjapur,2 BHK,1200.0,2.0,62.0,2,5166
Electronic City Phase II,2 BHK,1140.0,2.0,28.5,2,2500
Devanahalli,3 BHK,1200.0,2.0,55.08,3,4590
Ananth Nagar,2 BHK,1200.0,2.0,36.0,2,3000
Vidyaranyapura,5 Bedroom,2640.0,6.0,225.0,5,8522
Bannerghatta Road,3 BHK,1650.0,3.0,94.0,3,5696
other,2 Bedroom,1245.0,2.0,120.0,2,9638
Ardendale,2 BHK,1224.0,2.0,67.0,2,5473
Gunjur,2 BHK,1235.0,2.0,52.76,2,4272
Yelenahalli,2 BHK,1200.0,2.0,45.6,2,3800
Yelahanka,3 BHK,1342.0,2.0,75.0,3,5588
9th Phase JP Nagar,3 Bedroom,600.0,3.0,79.0,3,13166
TC Palaya,2 BHK,1800.0,2.0,80.0,2,4444
HAL 2nd Stage,4 Bedroom,2280.0,4.0,615.0,4,26973
Rajaji Nagar,4 Bedroom,2440.0,2.0,415.0,4,17008
Abbigere,2 BHK,1130.0,2.0,46.33,2,4100
Somasundara Palya,2 BHK,1186.0,2.0,72.0,2,6070
Hennur,2 BHK,1295.0,2.0,64.0,2,4942
Kudlu Gate,3 BHK,1535.0,3.0,85.0,3,5537
Bannerghatta Road,2 BHK,905.0,2.0,65.0,2,7182
Bhoganhalli,2 BHK,1447.0,2.0,75.97,2,5250
Hormavu,2 BHK,1165.0,2.0,40.0,2,3433
Kanakpura Road,2 BHK,1328.0,2.0,107.0,2,8057
other,3 BHK,1485.0,2.0,110.0,3,7407
Uttarahalli,3 BHK,1475.0,2.0,59.0,3,4000
Vidyaranyapura,2 BHK,1175.0,2.0,35.0,2,2978
Thigalarapalya,2 BHK,1245.0,2.0,100.0,2,8032
other,4 Bedroom,720.0,4.0,95.0,4,13194
Rachenahalli,2 BHK,985.0,2.0,49.97,2,5073
Koramangala,5 Bedroom,4000.0,5.0,800.0,5,20000
Hoskote,3 BHK,1395.0,2.0,63.0,3,4516
Hoodi,1 BHK,711.0,1.0,42.65,1,5998
Electronic City,1 BHK,630.0,1.0,40.5,1,6428
other,4 BHK,2400.0,4.0,88.0,4,3666
Gottigere,2 BHK,1170.0,2.0,40.0,2,3418
Marathahalli,4 BHK,2500.0,5.0,181.0,4,7240
Gubbalala,2 BHK,1100.0,2.0,49.5,2,4500
Hulimavu,4 BHK,3560.0,5.0,318.0,4,8932
Margondanahalli,2 Bedroom,900.0,2.0,55.0,2,6111
Bommasandra Industrial Area,2 BHK,1160.0,2.0,33.51,2,2888
Varthur,2 BHK,1180.0,2.0,55.0,2,4661
Tumkur Road,3 BHK,1441.0,2.0,95.0,3,6592
other,2 BHK,1115.0,2.0,50.0,2,4484
Yeshwanthpur,3 BHK,1825.0,3.0,145.0,3,7945
other,2 BHK,1489.0,2.0,101.0,2,6783
other,4 BHK,2800.0,4.0,365.0,4,13035
Subramanyapura,3 BHK,1278.0,3.0,62.71,3,4906
Subramanyapura,3 BHK,1223.0,2.0,42.81,3,3500
Hormavu,2 BHK,1065.0,2.0,50.92,2,4781
Whitefield,2 BHK,1190.0,2.0,64.0,2,5378
Koramangala,2 BHK,1130.0,2.0,63.0,2,5575
Banashankari,2 BHK,1430.0,2.0,87.8,2,6139
HRBR Layout,4 BHK,1900.0,3.0,90.0,4,4736
Sarjapur  Road,2 BHK,1104.0,2.0,34.22,2,3099
Begur Road,2 BHK,1240.0,2.0,45.88,2,3700
Choodasandra,2 BHK,1140.0,2.0,62.0,2,5438
Haralur Road,2 BHK,1194.0,2.0,47.0,2,3936
other,3 BHK,1464.0,3.0,56.0,3,3825
Sarjapur,4 BHK,3508.0,6.0,425.0,4,12115
KR Puram,3 BHK,1700.0,3.0,105.0,3,6176
Old Airport Road,4 BHK,2658.0,5.0,189.0,4,7110
other,3 BHK,1600.0,2.0,90.0,3,5625
OMBR Layout,4 BHK,3400.0,4.0,275.0,4,8088
other,3 BHK,1508.0,3.0,75.0,3,4973
Balagere,1 BHK,645.0,1.0,41.0,1,6356
Hormavu,2 BHK,1046.0,2.0,60.0,2,5736
Munnekollal,10 Bedroom,1200.0,8.0,230.0,10,19166
other,1 Bedroom,1500.0,1.0,300.0,1,20000
other,6 Bedroom,3000.0,6.0,300.0,6,10000
Hosa Road,2 BHK,1079.0,2.0,32.37,2,2999
Thubarahalli,2 BHK,975.0,2.0,50.0,2,5128
ISRO Layout,2 BHK,1050.0,2.0,51.0,2,4857
HSR Layout,9 Bedroom,1200.0,9.0,350.0,9,29166
other,3 BHK,1903.0,3.0,165.0,3,8670
other,2 BHK,1200.0,2.0,38.0,2,3166
other,4 BHK,2996.0,4.0,200.0,4,6675
other,4 BHK,3400.0,4.0,200.0,4,5882
Whitefield,4 Bedroom,2000.0,4.0,312.0,4,15600
Jigani,4 Bedroom,1453.0,4.0,160.0,4,11011
Parappana Agrahara,2 BHK,1092.0,2.0,43.0,2,3937
Hormavu,2 BHK,1400.0,2.0,75.0,2,5357
Bommanahalli,3 BHK,1416.0,2.0,85.0,3,6002
other,3 BHK,1602.0,2.0,75.0,3,4681
Electronic City,2 BHK,1100.0,2.0,41.0,2,3727
Kothanur,3 BHK,1760.0,3.0,110.0,3,6250
Marathahalli,4 BHK,3800.0,4.0,235.0,4,6184
Vasanthapura,2 BHK,1037.0,2.0,36.28,2,3498
Yeshwanthpur,3 BHK,1384.0,2.0,76.18,3,5504
other,1 BHK,551.0,1.0,30.0,1,5444
Indira Nagar,4 Bedroom,2400.0,5.0,1250.0,4,52083
7th Phase JP Nagar,1 Bedroom,500.0,1.0,70.0,1,14000
Sarjapur,2 BHK,900.0,2.0,27.98,2,3108
Anandapura,3 Bedroom,1415.0,2.0,74.0,3,5229
other,2 BHK,1400.0,2.0,44.8,2,3200
other,7 Bedroom,1300.0,4.0,52.0,7,4000
other,3 BHK,2850.0,4.0,460.0,3,16140
Kudlu,2 BHK,1030.0,2.0,75.0,2,7281
Kambipura,2 BHK,883.0,2.0,39.0,2,4416
other,1 BHK,550.0,1.0,16.0,1,2909
Anjanapura,4 BHK,1800.0,4.0,55.0,4,3055
other,5 Bedroom,10000.0,5.0,1950.0,5,19500
Old Madras Road,2 BHK,1210.0,2.0,77.0,2,6363
Vishveshwarya Layout,4 Bedroom,2000.0,3.0,90.0,4,4500
Kanakpura Road,3 BHK,1100.0,3.0,58.0,3,5272
other,3 BHK,1780.0,3.0,100.0,3,5617
other,5 Bedroom,950.0,5.0,55.0,5,5789
Panathur,2 BHK,1050.0,2.0,60.0,2,5714
Hosur Road,3 BHK,1370.0,3.0,38.36,3,2800
Marathahalli,4 BHK,3951.0,4.0,230.0,4,5821
Bellandur,3 BHK,2500.0,4.0,112.0,3,4480
other,2 BHK,1050.0,2.0,49.5,2,4714
Anjanapura,3 BHK,1843.0,3.0,87.54,3,4749
other,4 Bedroom,1200.0,4.0,175.0,4,14583
Somasundara Palya,2 BHK,1186.0,2.0,76.0,2,6408
Haralur Road,1 BHK,560.0,1.0,45.0,1,8035
Narayanapura,3 BHK,2357.0,3.0,135.0,3,5727
Mahalakshmi Layout,4 Bedroom,1200.0,3.0,160.0,4,13333
Yelahanka,3 BHK,1823.0,3.0,103.0,3,5650
other,3 BHK,1310.0,3.0,90.0,3,6870
Varthur,3 BHK,1520.0,2.0,60.0,3,3947
Bhoganhalli,3 BHK,1610.0,3.0,74.03,3,4598
Whitefield,3 Bedroom,1500.0,3.0,78.57,3,5237
LB Shastri Nagar,3 BHK,1400.0,2.0,80.0,3,5714
HRBR Layout,3 BHK,1625.0,3.0,120.0,3,7384
Chikkalasandra,5 Bedroom,1500.0,5.0,265.0,5,17666
other,2 BHK,900.0,2.0,24.0,2,2666
Talaghattapura,2 BHK,1100.0,2.0,32.0,2,2909
Bannerghatta Road,2 BHK,1000.0,2.0,50.0,2,5000
Old Airport Road,4 BHK,2774.0,4.0,197.0,4,7101
Yeshwanthpur,1 BHK,994.0,2.0,54.67,1,5500
Kaggadasapura,3 BHK,1455.0,3.0,75.0,3,5154
Chandapura,5 Bedroom,1200.0,3.0,79.0,5,6583
Lakshminarayana Pura,2 BHK,1172.0,2.0,75.0,2,6399
Hennur,3 BHK,1935.0,3.0,102.0,3,5271
Rachenahalli,3 BHK,1856.0,3.0,106.0,3,5711
Old Airport Road,3 BHK,2392.0,3.0,249.0,3,10409
Uttarahalli,2 BHK,900.0,2.0,35.0,2,3888
Raja Rajeshwari Nagar,3 BHK,1757.0,3.0,98.0,3,5577
Kudlu,2 BHK,1027.0,2.0,44.0,2,4284
Whitefield,3 BHK,1626.0,3.0,85.0,3,5227
other,2 BHK,1200.0,2.0,54.0,2,4500
Hoodi,3 BHK,1559.0,3.0,81.55,3,5230
Amruthahalli,2 BHK,1190.0,2.0,50.0,2,4201
KR Puram,2 BHK,1200.0,2.0,72.0,2,6000
Harlur,2 BHK,1174.0,2.0,74.0,2,6303
Kanakpura Road,2 BHK,1140.0,2.0,65.0,2,5701
other,3 BHK,1603.0,2.0,125.0,3,7797
Subramanyapura,3 BHK,1278.0,3.0,70.0,3,5477
other,3 BHK,2180.0,3.0,285.0,3,13073
Banashankari Stage III,2 BHK,1085.0,2.0,50.0,2,4608
Whitefield,2 BHK,1195.0,2.0,61.36,2,5134
other,1 Bedroom,450.0,1.0,30.0,1,6666
Rajaji Nagar,4 BHK,6500.0,5.0,1400.0,4,21538
Attibele,1 BHK,550.0,1.0,11.5,1,2090
Kenchenahalli,2 BHK,1165.0,2.0,45.0,2,3862
other,4 BHK,3000.0,4.0,301.0,4,10033
Kanakpura Road,3 BHK,1452.0,3.0,60.0,3,4132
Sarjapur  Road,1 BHK,702.0,1.0,52.0,1,7407
Banashankari,2 BHK,1020.0,2.0,42.83,2,4199
Devanahalli,2 BHK,1340.0,2.0,65.0,2,4850
Marathahalli,2 BHK,1125.0,2.0,54.8,2,4871
other,2 BHK,1160.0,2.0,40.53,2,3493
Vidyaranyapura,2 BHK,1050.0,2.0,50.0,2,4761
JP Nagar,3 BHK,1960.0,3.0,138.0,3,7040
other,2 BHK,1200.0,3.0,52.0,2,4333
Judicial Layout,3 Bedroom,1200.0,4.0,220.0,3,18333
Old Airport Road,4 BHK,3496.0,4.0,262.0,4,7494
Electronic City,2 BHK,975.0,2.0,32.0,2,3282
Kudlu,3 BHK,1700.0,3.0,125.0,3,7352
8th Phase JP Nagar,4 Bedroom,1800.0,4.0,230.0,4,12777
other,4 Bedroom,612.0,3.0,47.0,4,7679
other,2 BHK,1275.0,2.0,69.0,2,5411
Yeshwanthpur,1 BHK,650.0,1.0,40.0,1,6153
Kasavanhalli,2 BHK,1495.0,2.0,110.0,2,7357
Hebbal,2 BHK,1200.0,2.0,48.0,2,4000
Sarjapura - Attibele Road,5 Bedroom,3750.0,6.0,295.0,5,7866
Hennur Road,3 BHK,1679.0,3.0,94.86,3,5649
Kaval Byrasandra,3 BHK,1450.0,3.0,75.0,3,5172
Balagere,2 BHK,1210.0,2.0,82.0,2,6776
other,3 BHK,1300.0,2.0,51.0,3,3923
Judicial Layout,2 BHK,1165.0,2.0,49.61,2,4258
other,3 BHK,1623.0,3.0,155.0,3,9550
Channasandra,2 BHK,1115.0,2.0,35.8,2,3210
BTM Layout,1 BHK,450.0,1.0,20.0,1,4444
Electronics City Phase 1,2 BHK,1170.0,2.0,62.0,2,5299
KR Puram,4 Bedroom,2065.0,4.0,248.0,4,12009
Marathahalli,3 BHK,1700.0,3.0,126.0,3,7411
Uttarahalli,6 Bedroom,1080.0,5.0,160.0,6,14814
Malleshpalya,2 BHK,1245.0,2.0,55.0,2,4417
Kanakapura,2 BHK,1017.0,2.0,66.0,2,6489
7th Phase JP Nagar,6 Bedroom,650.0,6.0,80.0,6,12307
Sarjapur  Road,2 BHK,1242.0,2.0,75.0,2,6038
Yelahanka,3 BHK,1792.0,3.0,104.0,3,5803
other,3 BHK,1275.0,3.0,55.0,3,4313
8th Phase JP Nagar,6 Bedroom,1200.0,6.0,260.0,6,21666
other,3 Bedroom,2400.0,3.0,408.0,3,17000
Electronic City,1 BHK,605.0,1.0,22.0,1,3636
HBR Layout,4 Bedroom,1200.0,4.0,115.0,4,9583
Electronics City Phase 1,3 BHK,1450.0,3.0,79.0,3,5448
Bommenahalli,3 Bedroom,1232.0,3.0,96.0,3,7792
Laggere,5 Bedroom,1800.0,5.0,70.0,5,3888
other,1 Bedroom,900.0,1.0,52.3,1,5811
Kaggadasapura,2 BHK,1200.0,2.0,40.0,2,3333
Kadugodi,4 BHK,2100.0,3.0,204.0,4,9714
Uttarahalli,2 BHK,1065.0,2.0,42.6,2,4000
other,2 BHK,1075.0,2.0,48.38,2,4500
other,6 Bedroom,1508.0,5.0,240.0,6,15915
other,2 BHK,780.0,2.0,31.0,2,3974
7th Phase JP Nagar,2 BHK,1070.0,2.0,42.79,2,3999
Haralur Road,3 BHK,1464.0,3.0,56.0,3,3825
Ramagondanahalli,3 BHK,1910.0,3.0,150.0,3,7853
Cox Town,2 BHK,1280.0,2.0,120.0,2,9375
Raja Rajeshwari Nagar,3 BHK,1530.0,3.0,64.0,3,4183
Bhoganhalli,4 BHK,2439.0,5.0,170.0,4,6970
other,2 BHK,1088.0,2.0,43.5,2,3998
Harlur,2 BHK,1174.0,2.0,78.0,2,6643
Raja Rajeshwari Nagar,3 BHK,1587.0,3.0,73.8,3,4650
Kaggadasapura,2 BHK,1045.0,2.0,55.0,2,5263
Yelahanka,2 BHK,1304.0,2.0,69.97,2,5365
Whitefield,2 BHK,1100.0,2.0,65.0,2,5909
Hennur Road,3 BHK,1482.0,2.0,68.9,3,4649
Devanahalli,3 Bedroom,2125.0,3.0,107.0,3,5035
Varthur,5 BHK,1000.0,4.0,65.0,5,6500
Parappana Agrahara,2 BHK,1194.0,2.0,45.0,2,3768
Kengeri,2 BHK,1200.0,2.0,42.0,2,3500
other,2 BHK,1070.0,2.0,32.99,2,3083
Whitefield,2 BHK,1100.0,2.0,46.5,2,4227
Whitefield,3 BHK,1270.0,2.0,65.0,3,5118
other,2 BHK,1000.0,2.0,31.0,2,3100
Vittasandra,2 BHK,1246.0,2.0,69.0,2,5537
Raja Rajeshwari Nagar,3 BHK,1529.0,3.0,51.77,3,3385
Thanisandra,2 BHK,1185.0,2.0,42.0,2,3544
Gubbalala,2 BHK,1060.0,2.0,46.0,2,4339
Ananth Nagar,3 BHK,1319.0,3.0,37.75,3,2862
Kaikondrahalli,2 BHK,991.0,2.0,64.0,2,6458
Thanisandra,1 BHK,662.0,1.0,38.0,1,5740
other,2 Bedroom,1200.0,2.0,110.0,2,9166
other,3 Bedroom,1400.0,2.0,78.0,3,5571
Talaghattapura,3 BHK,1575.0,3.0,80.0,3,5079
Hosakerehalli,2 Bedroom,600.0,2.0,65.0,2,10833
Electronic City,5 Bedroom,717.0,5.0,78.0,5,10878
Ananth Nagar,3 BHK,1319.0,3.0,42.0,3,3184
Brookefield,3 Bedroom,2000.0,3.0,300.0,3,15000
9th Phase JP Nagar,3 Bedroom,600.0,3.0,85.0,3,14166
Electronics City Phase 1,2 BHK,940.0,2.0,48.0,2,5106
Singasandra,1 Bedroom,600.0,1.0,45.0,1,7500
Chandapura,2 BHK,985.0,2.0,25.12,2,2550
Kodigehalli,2 Bedroom,500.0,1.0,55.0,2,11000
Bharathi Nagar,2 BHK,1349.0,2.0,56.7,2,4203
Whitefield,3 Bedroom,3117.0,3.0,261.0,3,8373
HSR Layout,3 BHK,1844.0,3.0,95.0,3,5151
Thigalarapalya,3 BHK,1830.0,4.0,150.0,3,8196
Sarjapur  Road,3 BHK,2275.0,4.0,185.0,3,8131
Varthur Road,2 BHK,1140.0,2.0,49.11,2,4307
Banjara Layout,3 Bedroom,900.0,3.0,79.0,3,8777
other,3 BHK,2648.0,3.0,238.0,3,8987
Judicial Layout,5 Bedroom,5400.0,4.0,700.0,5,12962
Banashankari,3 BHK,1372.0,2.0,48.02,3,3500
other,3 BHK,1455.0,2.0,45.0,3,3092
Hebbal,2 BHK,1100.0,2.0,38.0,2,3454
Raja Rajeshwari Nagar,2 BHK,1295.0,2.0,55.47,2,4283
other,3 BHK,1475.0,3.0,80.0,3,5423
Nagavarapalya,3 BHK,1788.0,3.0,164.0,3,9172
Haralur Road,2 BHK,1243.0,2.0,46.0,2,3700
Yeshwanthpur,3 BHK,1855.0,3.0,140.0,3,7547
Kudlu Gate,3 BHK,1850.0,3.0,120.0,3,6486
Marathahalli,3 BHK,1315.0,2.0,60.0,3,4562
Whitefield,2 BHK,1216.0,2.0,73.0,2,6003
other,3 BHK,1721.0,3.0,93.0,3,5403
Sonnenahalli,2 BHK,1059.0,2.0,62.0,2,5854
Kadubeesanahalli,2 BHK,1184.0,2.0,55.0,2,4645
Yelahanka,7 Bedroom,1400.0,5.0,175.0,7,12500
Koramangala,3 BHK,2223.0,3.0,300.0,3,13495
other,3 BHK,1480.0,3.0,75.0,3,5067
Kodigehalli,2 BHK,635.0,1.0,28.0,2,4409
BEML Layout,2 Bedroom,2400.0,2.0,228.0,2,9500
JP Nagar,4 Bedroom,1500.0,2.0,87.0,4,5800
Hosa Road,2 BHK,1143.0,2.0,34.2,2,2992
Sarjapur  Road,4 Bedroom,3200.0,4.0,350.0,4,10937
Kanakpura Road,4 Bedroom,3170.0,4.0,250.0,4,7886
other,4 Bedroom,4800.0,4.0,420.0,4,8750
Kothannur,3 BHK,1485.0,2.0,59.4,3,4000
Rajaji Nagar,3 BHK,1640.0,3.0,251.0,3,15304
Hennur,3 BHK,1655.0,2.0,70.5,3,4259
2nd Phase Judicial Layout,2 BHK,900.0,2.0,41.0,2,4555
Somasundara Palya,3 BHK,1600.0,3.0,69.0,3,4312
other,2 BHK,1163.0,2.0,68.0,2,5846
Hennur Road,3 BHK,1654.0,3.0,97.8,3,5912
other,1 BHK,801.0,1.0,33.645,1,4200
other,4 Bedroom,2100.0,3.0,1000.0,4,47619
Bellandur,3 BHK,1685.0,3.0,101.0,3,5994
Banashankari,3 BHK,1540.0,3.0,80.0,3,5194
Thanisandra,2 BHK,1265.0,2.0,72.0,2,5691
Kasavanhalli,3 BHK,1200.0,2.0,38.4,3,3200
other,3 BHK,1300.0,2.0,80.0,3,6153
Sultan Palaya,2 BHK,1009.0,2.0,65.0,2,6442
other,3 BHK,1540.0,3.0,65.0,3,4220
Bellandur,3 BHK,1735.0,3.0,105.0,3,6051
HSR Layout,4 Bedroom,2500.0,4.0,300.0,4,12000
Marsur,2 BHK,497.0,1.0,20.0,2,4024
Hosakerehalli,3 BHK,2480.0,4.0,265.0,3,10685
Electronic City,2 BHK,1142.0,2.0,35.0,2,3064
other,3 BHK,1215.0,2.0,46.8,3,3851
Electronic City,2 BHK,1070.0,2.0,48.0,2,4485
Electronic City Phase II,2 BHK,1125.0,2.0,32.49,2,2888
Bommasandra Industrial Area,2 BHK,1125.0,2.0,32.49,2,2888
Kasavanhalli,2 BHK,1575.0,2.0,85.0,2,5396
Hosa Road,2 BHK,1360.0,2.0,77.61,2,5706
Haralur Road,3 BHK,1710.0,3.0,85.0,3,4970
Electronic City Phase II,2 BHK,1200.0,2.0,34.65,2,2887
Bannerghatta Road,4 BHK,2185.0,3.0,120.0,4,5491
other,4 BHK,2400.0,4.0,135.0,4,5625
Banashankari,2 BHK,1195.0,2.0,35.84,2,2999
Vittasandra,2 BHK,1246.0,2.0,67.0,2,5377
Bannerghatta Road,2 BHK,1012.0,2.0,45.0,2,4446
other,2 BHK,998.0,2.0,65.0,2,6513
Doddaballapur,3 Bedroom,2400.0,3.0,250.0,3,10416
7th Phase JP Nagar,2 BHK,1130.0,2.0,85.0,2,7522
Koramangala,3 BHK,1744.0,4.0,210.0,3,12041
Rajaji Nagar,4 BHK,2733.0,5.0,259.0,4,9476
Karuna Nagar,3 BHK,1354.0,2.0,98.0,3,7237
other,10 Bedroom,3300.0,9.0,450.0,10,13636
Brookefield,2 BHK,1389.0,2.0,94.0,2,6767
Sarjapur,4 Bedroom,2970.0,3.0,130.0,4,4377
Cooke Town,3 BHK,2560.0,4.0,310.0,3,12109
Lakshminarayana Pura,2 BHK,1172.0,2.0,75.0,2,6399
other,3 Bedroom,500.0,2.0,75.0,3,15000
Kaggadasapura,3 BHK,1495.0,3.0,60.0,3,4013
Varthur,2 BHK,1738.0,2.0,57.35,2,3299
Dodda Nekkundi,2 BHK,1315.0,2.0,50.0,2,3802
Marathahalli,2 BHK,1270.0,2.0,73.0,2,5748
Electronic City Phase II,2 BHK,1160.0,2.0,52.2,2,4500
Raja Rajeshwari Nagar,3 BHK,1450.0,2.0,67.0,3,4620
Rayasandra,1 BHK,470.0,1.0,21.0,1,4468
other,2 BHK,1040.0,2.0,78.0,2,7500
Sarjapur,2 BHK,1205.0,2.0,29.0,2,2406
other,3 BHK,1215.0,2.0,50.0,3,4115
Whitefield,2 BHK,1135.0,2.0,58.0,2,5110
8th Phase JP Nagar,3 BHK,1224.0,2.0,48.95,3,3999
Budigere,1 BHK,664.0,1.0,34.0,1,5120
Electronic City,3 BHK,1160.0,2.0,42.0,3,3620
Babusapalaya,2 Bedroom,1200.0,2.0,105.0,2,8750
Bommasandra Industrial Area,3 BHK,1220.0,2.0,35.2,3,2885
Bharathi Nagar,2 BHK,1328.0,2.0,69.0,2,5195
other,2 BHK,975.0,2.0,45.0,2,4615
other,3 BHK,1907.0,2.0,200.0,3,10487
Malleshwaram,2 BHK,900.0,2.0,72.0,2,8000
other,2 BHK,1330.0,2.0,80.0,2,6015
Rajiv Nagar,4 BHK,2330.0,5.0,175.0,4,7510
Munnekollal,3 BHK,1222.0,3.0,95.0,3,7774
other,3 Bedroom,4446.0,3.0,410.0,3,9221
Sarjapur  Road,3 BHK,2100.0,3.0,135.0,3,6428
Sarjapur  Road,4 BHK,3335.0,4.0,290.0,4,8695
Whitefield,3 Bedroom,3150.0,3.0,290.0,3,9206
other,3 Bedroom,1320.0,2.0,95.0,3,7196
9th Phase JP Nagar,7 Bedroom,820.0,7.0,140.0,7,17073
other,5 Bedroom,8000.0,5.0,550.0,5,6875
Hennur Road,4 Bedroom,2400.0,5.0,500.0,4,20833
Uttarahalli,3 BHK,1330.0,2.0,46.55,3,3500
Hosur Road,4 Bedroom,2000.0,4.0,130.0,4,6500
Electronic City Phase II,2 BHK,1244.0,2.0,55.0,2,4421
other,2 BHK,1000.0,2.0,50.0,2,5000
other,3 BHK,1400.0,3.0,45.0,3,3214
other,2 BHK,1000.0,2.0,35.0,2,3500
Rajaji Nagar,2 Bedroom,1056.0,1.0,250.0,2,23674
Uttarahalli,3 BHK,1250.0,2.0,50.0,3,4000
Hebbal,4 BHK,2630.0,5.0,188.0,4,7148
other,2 BHK,1225.0,2.0,48.0,2,3918
Hebbal,4 BHK,2790.0,5.0,198.0,4,7096
Indira Nagar,4 BHK,2800.0,4.0,365.0,4,13035
Seegehalli,3 BHK,1683.0,3.0,80.0,3,4753
Yelahanka,2 BHK,1250.0,2.0,54.0,2,4320
Hennur Road,3 BHK,1305.0,2.0,110.0,3,8429
Sarjapur  Road,3 Bedroom,3009.0,3.0,331.0,3,11000
Kaggalipura,3 BHK,1150.0,2.0,57.5,3,5000
Kaggadasapura,3 BHK,1654.0,3.0,70.0,3,4232
other,4 BHK,4500.0,3.0,225.0,4,5000
Kadugodi,3 BHK,1890.0,4.0,125.0,3,6613
Kasavanhalli,2 BHK,1069.0,2.0,55.0,2,5144
Chandapura,2 BHK,750.0,1.0,18.5,2,2466
Banashankari,4 Bedroom,675.0,4.0,140.0,4,20740
Hormavu,3 Bedroom,2400.0,3.0,177.0,3,7375
2nd Stage Nagarbhavi,5 Bedroom,1200.0,4.0,240.0,5,20000
Koramangala,4 Bedroom,1410.0,2.0,280.0,4,19858
7th Phase JP Nagar,3 BHK,1680.0,3.0,130.0,3,7738
other,3 BHK,3500.0,4.0,500.0,3,14285
Sarjapur  Road,3 BHK,1700.0,3.0,93.5,3,5500
other,4 Bedroom,2200.0,2.0,150.0,4,6818
Thanisandra,3 BHK,1595.0,3.0,105.0,3,6583
Hennur Road,3 BHK,1480.0,3.0,100.0,3,6756
other,3 Bedroom,1092.0,3.0,95.0,3,8699
Koramangala,3 BHK,1866.0,4.0,245.0,3,13129
Hebbal,3 BHK,3895.0,3.0,390.0,3,10012
Banashankari,3 BHK,1683.0,3.0,115.0,3,6833
Dasanapura,2 BHK,545.0,2.0,22.34,2,4099
Banaswadi,2 BHK,1340.0,2.0,73.0,2,5447
other,3 BHK,1100.0,3.0,53.0,3,4818
Akshaya Nagar,1 Bedroom,2000.0,1.0,200.0,1,10000
Whitefield,2 BHK,1185.0,2.0,56.0,2,4725
Chandapura,2 BHK,1025.0,2.0,27.68,2,2700
Haralur Road,3 BHK,1730.0,3.0,95.0,3,5491
Hennur Road,3 BHK,1450.0,3.0,88.6,3,6110
Sarjapur  Road,3 BHK,1862.0,4.0,110.0,3,5907
Sarjapur  Road,4 Bedroom,4000.0,5.0,578.0,4,14450
Old Madras Road,3 BHK,2430.0,5.0,180.0,3,7407
Hoodi,1 BHK,863.0,1.0,40.55,1,4698
other,2 BHK,1170.0,2.0,50.0,2,4273
other,2 BHK,1075.0,2.0,50.0,2,4651
Kodichikkanahalli,2 BHK,1060.0,2.0,34.0,2,3207
Sarjapur,2 BHK,850.0,2.0,32.0,2,3764
other,2 BHK,935.0,2.0,45.9,2,4909
Gollarapalya Hosahalli,4 BHK,1905.0,3.0,86.0,4,4514
other,6 Bedroom,2000.0,6.0,300.0,6,15000
Uttarahalli,2 BHK,1125.0,2.0,47.0,2,4177
9th Phase JP Nagar,6 Bedroom,600.0,6.0,75.0,6,12500
Chikkabanavar,4 Bedroom,1500.0,4.0,105.0,4,7000
Bannerghatta Road,3 BHK,1725.0,3.0,85.0,3,4927
other,2 BHK,1120.0,2.0,36.0,2,3214
Brookefield,3 BHK,1750.0,2.0,75.0,3,4285
7th Phase JP Nagar,3 BHK,2000.0,3.0,77.0,3,3850
Koramangala,3 BHK,1745.0,3.0,98.0,3,5616
other,5 Bedroom,750.0,3.0,95.0,5,12666
Uttarahalli,3 BHK,1270.0,2.0,55.25,3,4350
Iblur Village,5 BHK,5384.0,5.0,420.0,5,7800
Sahakara Nagar,2 BHK,1180.0,2.0,72.0,2,6101
other,2 BHK,1130.0,2.0,75.0,2,6637
Old Airport Road,4 BHK,2774.0,4.0,197.0,4,7101
Ardendale,3 BHK,1723.0,3.0,95.0,3,5513
Mysore Road,2 BHK,1175.0,2.0,70.5,2,6000
Ambalipura,3 BHK,1607.0,2.0,112.0,3,6969
other,2 BHK,1334.0,2.0,62.0,2,4647
Vishwapriya Layout,5 Bedroom,2800.0,5.0,130.0,5,4642
Bellandur,3 BHK,1138.0,3.0,128.0,3,11247
Bannerghatta,2 BHK,1113.0,2.0,70.0,2,6289
9th Phase JP Nagar,3 BHK,1300.0,2.0,58.0,3,4461
Judicial Layout,4 Bedroom,2400.0,4.0,350.0,4,14583
Sahakara Nagar,2 BHK,1200.0,2.0,60.0,2,5000
other,3 BHK,1900.0,3.0,190.0,3,10000
Jakkur,4 BHK,5150.0,4.0,559.0,4,10854
2nd Stage Nagarbhavi,5 Bedroom,1200.0,5.0,290.0,5,24166
Whitefield,4 BHK,2956.0,5.0,234.0,4,7916
Begur Road,5 Bedroom,1100.0,5.0,165.0,5,15000
Panathur,2 BHK,1398.0,2.0,97.0,2,6938
Margondanahalli,2 Bedroom,1152.0,1.0,66.0,2,5729
other,4 Bedroom,1200.0,5.0,325.0,4,27083
Nagavara,2 BHK,1110.0,2.0,44.95,2,4049
Devarachikkanahalli,2 BHK,1116.0,2.0,47.0,2,4211
Babusapalaya,2 BHK,1061.0,2.0,33.84,2,3189
Malleshwaram,1 BHK,620.0,1.0,86.8,1,14000
7th Phase JP Nagar,3 BHK,1300.0,2.0,52.0,3,4000
Yelahanka,2 BHK,1270.0,2.0,57.0,2,4488
other,2 BHK,966.0,2.0,50.0,2,5175
other,4 Bedroom,1200.0,4.0,110.0,4,9166
Kodigehalli,4 Bedroom,1320.0,4.0,249.0,4,18863
Iblur Village,4 BHK,3596.0,5.0,297.0,4,8259
other,3 BHK,1630.0,3.0,131.0,3,8036
other,5 BHK,600.0,6.0,72.0,5,12000
other,4 Bedroom,600.0,4.0,60.0,4,10000
Amruthahalli,2 BHK,1202.0,2.0,65.0,2,5407
other,4 Bedroom,1200.0,4.0,250.0,4,20833
Jakkur,2 BHK,1300.0,2.0,80.0,2,6153
Uttarahalli,6 Bedroom,1200.0,6.0,330.0,6,27500
Electronic City,3 BHK,1360.0,2.0,70.0,3,5147
Hennur Road,2 BHK,1155.0,2.0,69.18,2,5989
Hosakerehalli,3 BHK,2378.0,3.0,310.0,3,13036
Hulimavu,2 BHK,1011.0,2.0,38.0,2,3758
HSR Layout,2 BHK,1203.0,2.0,72.0,2,5985
Hebbal,3 BHK,1645.0,3.0,113.0,3,6869
Horamavu Agara,3 BHK,1364.0,2.0,50.0,3,3665
Anekal,2 Bedroom,1200.0,1.0,36.0,2,3000
other,1 BHK,1050.0,2.0,41.0,1,3904
Kothanur,3 BHK,1365.0,3.0,66.0,3,4835
Jigani,3 BHK,1252.0,3.0,70.0,3,5591
Jigani,3 BHK,1221.0,3.0,65.0,3,5323
Panathur,2 BHK,1065.0,2.0,58.0,2,5446
Uttarahalli,3 BHK,1270.0,2.0,55.25,3,4350
Sahakara Nagar,2 BHK,1032.0,2.0,45.0,2,4360
Frazer Town,4 BHK,2900.0,4.0,325.0,4,11206
Sector 2 HSR Layout,3 BHK,1515.0,2.0,69.0,3,4554
other,2 BHK,1427.0,2.0,52.0,2,3644
Magadi Road,2 BHK,940.0,2.0,50.0,2,5319
other,5 Bedroom,800.0,5.0,95.0,5,11875
Bannerghatta Road,3 BHK,1460.0,2.0,80.0,3,5479
Sarjapur  Road,3 BHK,2275.0,4.0,175.0,3,7692
9th Phase JP Nagar,3 BHK,1522.0,3.0,90.0,3,5913
other,3 Bedroom,3210.0,4.0,380.0,3,11838
Thanisandra,2 BHK,1185.0,2.0,42.67,2,3600
other,13 BHK,5425.0,13.0,275.0,13,5069
Banashankari,3 BHK,1800.0,3.0,125.0,3,6944
Electronics City Phase 1,2 BHK,940.0,2.0,44.0,2,4680
other,4 Bedroom,1200.0,4.0,135.0,4,11250
Marathahalli,3 BHK,1530.0,3.0,71.0,3,4640
other,2 BHK,1050.0,2.0,48.0,2,4571
Subramanyapura,2 BHK,929.0,2.0,49.0,2,5274
Banjara Layout,4 Bedroom,3750.0,5.0,145.0,4,3866
Whitefield,3 BHK,1530.0,3.0,75.0,3,4901
other,2 Bedroom,900.0,2.0,48.0,2,5333
Vasanthapura,2 BHK,978.0,2.0,34.22,2,3498
other,4 BHK,4041.0,4.0,345.0,4,8537
Ambalipura,1 BHK,770.0,1.0,43.82,1,5690
Kaval Byrasandra,2 BHK,1066.0,2.0,55.0,2,5159
Electronic City,1 BHK,635.0,1.0,28.0,1,4409
Whitefield,2 BHK,1015.0,2.0,45.0,2,4433
Hebbal,2 BHK,1175.0,2.0,54.05,2,4600
Chikkabanavar,4 Bedroom,1200.0,4.0,130.0,4,10833
Vidyaranyapura,2 BHK,1188.0,2.0,50.0,2,4208
Indira Nagar,5 Bedroom,2400.0,5.0,700.0,5,29166
other,1 BHK,620.0,1.0,25.0,1,4032
other,2 BHK,1210.0,2.0,75.0,2,6198
Attibele,3 Bedroom,1000.0,3.0,90.0,3,9000
Rajaji Nagar,3 BHK,1500.0,2.0,99.0,3,6600
Electronic City,4 Bedroom,1800.0,3.0,190.0,4,10555
Thanisandra,2 BHK,1185.0,2.0,43.66,2,3684
Basavangudi,3 BHK,1600.0,3.0,130.0,3,8125
Gottigere,2 BHK,1100.0,2.0,28.0,2,2545
Yelahanka,2 BHK,1300.0,2.0,69.23,2,5325
Kasavanhalli,2 BHK,1100.0,2.0,68.0,2,6181
Ambalipura,2 BHK,860.0,2.0,28.93,2,3363
Panathur,2 BHK,1007.0,2.0,64.0,2,6355
Akshaya Nagar,2 BHK,1100.0,2.0,48.0,2,4363
Whitefield,2 BHK,1187.0,2.0,38.0,2,3201
R.T. Nagar,2 BHK,1150.0,2.0,72.0,2,6260
Nagavara,3 BHK,2400.0,3.0,252.0,3,10500
other,2 BHK,1180.0,2.0,41.94,2,3554
Budigere,3 BHK,1820.0,3.0,85.2,3,4681
other,6 Bedroom,1200.0,6.0,180.0,6,15000
other,3 Bedroom,2400.0,6.0,775.0,3,32291
KR Puram,4 BHK,3628.0,4.0,246.0,4,6780
other,3 BHK,1430.0,3.0,70.0,3,4895
Kanakpura Road,2 BHK,700.0,2.0,40.0,2,5714
other,2 Bedroom,1200.0,1.0,70.0,2,5833
Old Madras Road,2 BHK,1165.0,2.0,40.77,2,3499
Jalahalli,3 BHK,1569.0,3.0,118.0,3,7520
other,3 BHK,1950.0,4.0,95.0,3,4871
other,2 Bedroom,1200.0,3.0,70.0,2,5833
Yelahanka New Town,3 BHK,2437.0,3.0,125.0,3,5129
Marathahalli,2 BHK,950.0,2.0,46.7,2,4915
Bellandur,3 BHK,1685.0,3.0,115.0,3,6824
Green Glen Layout,4 BHK,3270.0,4.0,205.0,4,6269
Sarjapur  Road,2 BHK,1314.0,2.0,110.0,2,8371
other,3 BHK,1250.0,3.0,40.0,3,3200
other,2 BHK,1130.0,2.0,68.0,2,6017
other,6 Bedroom,1200.0,9.0,122.0,6,10166
Sultan Palaya,6 Bedroom,890.0,5.0,160.0,6,17977
other,3 Bedroom,857.0,3.0,95.0,3,11085
Whitefield,3 BHK,2247.0,3.0,145.0,3,6453
Marathahalli,3 BHK,1550.0,3.0,85.0,3,5483
Hoodi,3 BHK,1715.0,2.0,100.0,3,5830
Hennur Road,2 BHK,1053.0,2.0,53.0,2,5033
Begur Road,3 Bedroom,1200.0,3.0,50.0,3,4166
HBR Layout,3 Bedroom,1800.0,2.0,230.0,3,12777
Chikkabanavar,5 Bedroom,2400.0,4.0,97.0,5,4041
other,2 BHK,1245.0,1.0,60.0,2,4819
Koramangala,2 BHK,1320.0,2.0,160.0,2,12121
Whitefield,4 Bedroom,3940.0,5.0,265.0,4,6725
Thanisandra,3 BHK,1241.0,2.0,65.0,3,5237
7th Phase JP Nagar,3 BHK,2980.0,4.0,260.0,3,8724
other,6 Bedroom,600.0,5.0,130.0,6,21666
Raja Rajeshwari Nagar,2 BHK,1090.0,2.0,57.0,2,5229
other,3 BHK,1390.0,2.0,50.0,3,3597
other,8 Bedroom,600.0,4.0,175.0,8,29166
other,3 BHK,2750.0,5.0,275.0,3,10000
Kanakpura Road,3 BHK,1100.0,2.0,52.97,3,4815
Koramangala,2 BHK,1350.0,2.0,103.0,2,7629
other,2 BHK,800.0,2.0,55.0,2,6875
Sarjapur  Road,3 BHK,1787.0,3.0,116.0,3,6491
BTM Layout,3 BHK,2400.0,3.0,220.0,3,9166
Haralur Road,4 BHK,3700.0,4.0,325.0,4,8783
Sarjapur  Road,3 BHK,1691.0,3.0,93.01,3,5500
Bhoganhalli,4 BHK,2119.0,4.0,111.0,4,5238
Chikka Tirupathi,3 Bedroom,2325.0,3.0,95.0,3,4086
other,5 Bedroom,570.0,5.0,80.0,5,14035
Akshaya Nagar,3 BHK,1896.0,3.0,102.0,3,5379
Banashankari Stage V,3 BHK,1540.0,3.0,48.51,3,3150
other,2 BHK,1244.0,2.0,95.0,2,7636
Hormavu,3 BHK,1365.0,2.0,65.0,3,4761
Thanisandra,1 BHK,580.0,1.0,27.5,1,4741
Hebbal,3 BHK,1355.0,3.0,83.87,3,6189
other,3 Bedroom,1500.0,4.0,180.0,3,12000
8th Phase JP Nagar,2 BHK,1513.0,2.0,62.0,2,4097
Banashankari,2 BHK,1175.0,2.0,41.13,2,3500
Thanisandra,3 BHK,2087.0,4.0,139.0,3,6660
Talaghattapura,3 BHK,2273.0,3.0,145.0,3,6379
Thanisandra,3 BHK,2050.0,3.0,150.0,3,7317
other,2 Bedroom,1200.0,2.0,75.0,2,6250
Whitefield,2 BHK,1190.0,2.0,59.0,2,4957
Kodigehalli,4 BHK,600.0,3.0,85.0,4,14166
Marathahalli,2 BHK,1019.0,2.0,49.86,2,4893
Thigalarapalya,3 BHK,1830.0,3.0,133.0,3,7267
Channasandra,7 Bedroom,4278.0,7.0,299.0,7,6989
Sarjapur  Road,2 BHK,1028.0,2.0,56.0,2,5447
Old Madras Road,2 BHK,1300.0,2.0,100.0,2,7692
Whitefield,2 BHK,1160.0,2.0,67.0,2,5775
Mahadevpura,3 BHK,1505.0,3.0,78.0,3,5182
Kogilu,2 BHK,1250.0,2.0,55.55,2,4444
other,1 Bedroom,400.0,2.0,50.0,1,12500
Sarjapur,2 BHK,1175.0,2.0,41.68,2,3547
other,2 BHK,1100.0,2.0,57.6,2,5236
other,3 BHK,2159.0,3.0,120.0,3,5558
Devanahalli,2 BHK,1230.0,2.0,56.45,2,4589
other,3 Bedroom,1200.0,4.0,180.0,3,15000
other,5 Bedroom,3100.0,6.0,165.0,5,5322
Old Madras Road,5 BHK,5020.0,7.0,287.0,5,5717
other,1 Bedroom,1200.0,1.0,48.0,1,4000
other,2 BHK,1340.0,2.0,71.0,2,5298
Banashankari,3 BHK,1750.0,2.0,89.0,3,5085
Banashankari Stage II,2 BHK,1400.0,2.0,100.0,2,7142
Nagarbhavi,3 BHK,1350.0,3.0,60.0,3,4444
other,3 BHK,1675.0,3.0,62.0,3,3701
Thanisandra,2 BHK,1340.0,2.0,66.33,2,4950
Malleshwaram,2 Bedroom,600.0,1.0,68.0,2,11333
Hulimavu,2 BHK,1276.0,2.0,76.0,2,5956
Ramagondanahalli,3 BHK,1738.0,2.0,76.0,3,4372
other,2 BHK,1190.0,2.0,59.9,2,5033
Electronic City,2 BHK,1128.0,2.0,65.45,2,5802
other,2 BHK,1250.0,2.0,55.0,2,4400
other,2 BHK,1450.0,2.0,90.0,2,6206
Kudlu,2 BHK,1084.0,2.0,53.0,2,4889
Kaggadasapura,4 BHK,2150.0,4.0,90.0,4,4186
Koramangala,4 Bedroom,6000.0,4.0,625.0,4,10416
Nagarbhavi,4 Bedroom,600.0,3.0,77.0,4,12833
Hosur Road,2 BHK,1250.0,2.0,65.0,2,5200
Seegehalli,3 Bedroom,2400.0,4.0,240.0,3,10000
Begur,3 BHK,1304.0,3.0,65.0,3,4984
Rajaji Nagar,3 BHK,1640.0,3.0,268.0,3,16341
7th Phase JP Nagar,2 BHK,980.0,2.0,77.44,2,7902
other,2 BHK,1275.0,2.0,49.5,2,3882
other,1 Bedroom,10030.0,1.0,150.0,1,1495
Banjara Layout,3 Bedroom,2500.0,4.0,140.0,3,5600
other,4 Bedroom,3600.0,4.0,225.0,4,6250
Anekal,3 Bedroom,2400.0,4.0,95.0,3,3958
other,3 BHK,2300.0,3.0,280.0,3,12173
Kanakpura Road,3 BHK,1450.0,2.0,70.18,3,4840
Electronic City,2 BHK,890.0,2.0,40.0,2,4494
Bannerghatta Road,2 BHK,1340.0,2.0,85.0,2,6343
Vidyaranyapura,3 BHK,1100.0,2.0,75.0,3,6818
Kengeri,2 BHK,750.0,2.0,40.0,2,5333
Basaveshwara Nagar,4 Bedroom,700.0,4.0,125.0,4,17857
Akshaya Nagar,2 BHK,1300.0,2.0,66.0,2,5076
HRBR Layout,2 BHK,1301.0,2.0,120.0,2,9223
other,9 Bedroom,3200.0,8.0,130.0,9,4062
other,3 Bedroom,1900.0,2.0,92.0,3,4842
Kanakpura Road,2 BHK,1299.0,2.0,105.0,2,8083
other,3 BHK,1250.0,3.0,40.0,3,3200
Whitefield,4 Bedroom,1600.0,4.0,300.0,4,18750
other,2 BHK,925.0,2.0,55.0,2,5945
other,3 Bedroom,3900.0,3.0,250.0,3,6410
Uttarahalli,2 BHK,1107.0,2.0,45.0,2,4065
other,2 Bedroom,1400.0,2.0,160.0,2,11428
Electronics City Phase 1,2 BHK,1300.0,2.0,67.0,2,5153
Yeshwanthpur,2 BHK,570.0,2.0,62.0,2,10877
other,3 BHK,1715.0,3.0,95.0,3,5539
Uttarahalli,2 BHK,1200.0,2.0,55.0,2,4583
other,6 Bedroom,1200.0,5.0,280.0,6,23333
other,2 BHK,703.0,2.0,60.0,2,8534
Talaghattapura,2 BHK,921.0,2.0,29.47,2,3199
Marathahalli,2 BHK,1144.0,2.0,65.0,2,5681
Kanakpura Road,1 BHK,525.0,1.0,26.01,1,4954
other,3 BHK,1400.0,2.0,56.0,3,4000
Balagere,1 BHK,661.0,1.0,33.0,1,4992
other,6 Bedroom,1200.0,6.0,160.0,6,13333
Kengeri Satellite Town,5 BHK,1200.0,5.0,70.0,5,5833
Yelahanka,3 BHK,1705.0,3.0,91.0,3,5337
other,5 Bedroom,1035.0,5.0,170.0,5,16425
Kadugodi,3 BHK,1430.0,2.0,55.0,3,3846
Marathahalli,2 BHK,1240.0,2.0,55.0,2,4435
Garudachar Palya,3 BHK,1325.0,2.0,60.8,3,4588
Thanisandra,2 BHK,1230.0,2.0,44.0,2,3577
Electronics City Phase 1,2 BHK,1000.0,2.0,28.0,2,2800
other,4 Bedroom,740.0,2.0,24.5,4,3310
Uttarahalli,3 BHK,1330.0,2.0,46.45,3,3492
Sarjapur  Road,4 BHK,3005.0,5.0,275.0,4,9151
Shivaji Nagar,3 BHK,1300.0,3.0,170.0,3,13076
Haralur Road,2 BHK,1243.0,2.0,46.0,2,3700
Kengeri Satellite Town,4 Bedroom,936.0,2.0,60.84,4,6500
Banashankari Stage III,3 BHK,1500.0,2.0,100.0,3,6666
Kodigehalli,3 Bedroom,1200.0,4.0,156.0,3,13000
Cox Town,2 BHK,1073.0,2.0,72.0,2,6710
Kudlu,2 BHK,1300.0,2.0,56.0,2,4307
Sarjapur,4 Bedroom,4201.0,4.0,435.0,4,10354
Electronic City,2 BHK,1065.0,2.0,30.75,2,2887
Whitefield,2 BHK,1216.0,2.0,66.88,2,5500
Jigani,3 BHK,1252.0,3.0,59.0,3,4712
Hormavu,2 BHK,1196.0,2.0,44.8,2,3745
Kudlu,2 BHK,1027.0,2.0,44.0,2,4284
Padmanabhanagar,3 Bedroom,610.0,2.0,75.0,3,12295
other,3 BHK,1270.0,2.0,57.15,3,4500
Mico Layout,2 BHK,1171.0,2.0,39.0,2,3330
Kothannur,3 BHK,1215.0,2.0,48.6,3,4000
other,1 Bedroom,600.0,1.0,20.0,1,3333
Whitefield,2 BHK,1180.0,2.0,77.0,2,6525
Electronic City Phase II,2 BHK,1160.0,2.0,33.51,2,2888
other,4 Bedroom,4382.0,4.0,400.0,4,9128
Whitefield,3 BHK,2140.0,3.0,139.0,3,6495
Yeshwanthpur,3 BHK,1500.0,3.0,100.0,3,6666
Banaswadi,2 BHK,1008.0,2.0,52.0,2,5158
Ambedkar Nagar,3 BHK,1921.0,4.0,129.0,3,6715
Electronic City,2 BHK,1127.0,2.0,32.0,2,2839
Rajaji Nagar,3 BHK,1621.0,3.0,124.0,3,7649
other,3 Bedroom,2400.0,3.0,360.0,3,15000
other,3 BHK,1030.0,2.0,77.25,3,7500
Whitefield,2 BHK,1314.0,2.0,84.0,2,6392
Choodasandra,3 BHK,1220.0,3.0,56.0,3,4590
Banashankari Stage VI,4 Bedroom,600.0,3.0,97.0,4,16166
JP Nagar,3 BHK,1590.0,2.0,85.0,3,5345
Shivaji Nagar,2 BHK,703.0,2.0,49.5,2,7041
Hosakerehalli,3 BHK,1795.0,3.0,64.62,3,3600
Gottigere,4 Bedroom,3000.0,3.0,132.0,4,4400
Sarjapur  Road,4 BHK,4395.0,4.0,242.0,4,5506
Electronic City,2 BHK,825.0,2.0,35.0,2,4242
Yelahanka,4 BHK,4025.0,6.0,350.0,4,8695
other,3 BHK,1254.0,2.0,70.0,3,5582
Ramagondanahalli,3 BHK,2040.0,3.0,114.0,3,5588
Thigalarapalya,3 BHK,2215.0,4.0,154.0,3,6952
Jalahalli,2 BHK,1478.0,2.0,125.0,2,8457
Malleshwaram,3 BHK,2475.0,4.0,337.0,3,13616
other,3 BHK,1669.0,3.0,298.0,3,17855
other,3 BHK,1820.0,3.0,79.5,3,4368
Horamavu Agara,3 BHK,1623.0,3.0,89.0,3,5483
Haralur Road,2 BHK,1243.0,2.0,46.0,2,3700
Electronic City,2 BHK,1200.0,2.0,40.5,2,3375
Gottigere,3 BHK,1613.0,3.0,72.0,3,4463
other,2 BHK,1293.0,2.0,80.0,2,6187
Hoskote,2 BHK,1065.0,2.0,28.5,2,2676
Kudlu Gate,2 BHK,940.0,2.0,69.57,2,7401
Sarjapur  Road,3 BHK,1181.0,2.0,65.0,3,5503
Malleshwaram,3 BHK,2215.0,3.0,275.0,3,12415
Marathahalli,1 BHK,1100.0,2.0,75.0,1,6818
other,2 BHK,1050.0,2.0,36.0,2,3428
Subramanyapura,3 BHK,1260.0,2.0,75.0,3,5952
Banashankari Stage II,5 BHK,2800.0,3.0,425.0,5,15178
other,3 BHK,1500.0,3.0,56.0,3,3733
KR Puram,6 Bedroom,1200.0,6.0,132.0,6,11000
other,2 Bedroom,1000.0,2.0,66.0,2,6600
Whitefield,3 BHK,1530.0,3.0,73.0,3,4771
HSR Layout,2 BHK,1140.0,2.0,46.0,2,4035
other,3 BHK,1864.0,3.0,115.0,3,6169
Sarjapur,3 BHK,1575.0,3.0,55.0,3,3492
Attibele,3 Bedroom,1200.0,3.0,105.0,3,8750
Raja Rajeshwari Nagar,2 BHK,945.0,2.0,45.0,2,4761
Sarjapur  Road,3 BHK,1691.0,3.0,100.0,3,5913
other,2 Bedroom,1200.0,2.0,135.0,2,11250
Electronic City Phase II,3 BHK,1220.0,3.0,35.25,3,2889
Kengeri,1 BHK,600.0,1.0,35.0,1,5833
Whitefield,2 BHK,1160.0,2.0,39.0,2,3362
other,3 BHK,1950.0,2.0,89.5,3,4589
Vittasandra,3 BHK,1650.0,3.0,85.0,3,5151
other,4 Bedroom,2400.0,4.0,150.0,4,6250
Abbigere,2 BHK,795.0,1.0,32.6,2,4100
other,5 Bedroom,800.0,5.0,95.0,5,11875
TC Palaya,5 Bedroom,1400.0,5.0,97.0,5,6928
Kalyan nagar,3 BHK,2285.0,4.0,165.0,3,7221
Haralur Road,2 BHK,1243.0,2.0,46.0,2,3700
other,2 BHK,1263.0,2.0,51.5,2,4077
other,5 BHK,1200.0,4.0,115.0,5,9583
Hennur,3 BHK,1345.0,2.0,69.0,3,5130
other,5 Bedroom,4400.0,5.0,240.0,5,5454
Electronic City,3 BHK,1360.0,2.0,58.0,3,4264
Sarjapur  Road,2 BHK,1026.0,2.0,60.55,2,5901
other,2 BHK,1050.0,2.0,38.0,2,3619
Uttarahalli,3 BHK,1460.0,3.0,76.0,3,5205
other,3 BHK,1525.0,2.0,91.48,3,5998
Kanakpura Road,3 BHK,1843.0,3.0,85.0,3,4612
Ananth Nagar,2 BHK,1000.0,2.0,25.0,2,2500
Koramangala,3 BHK,1642.0,2.0,115.0,3,7003
TC Palaya,3 Bedroom,1200.0,2.0,66.0,3,5500
other,2 BHK,975.0,2.0,60.0,2,6153
Bisuvanahalli,3 BHK,1075.0,2.0,44.0,3,4093
Panathur,2 BHK,1210.0,2.0,77.93,2,6440
other,3 BHK,2000.0,3.0,100.0,3,5000
Akshaya Nagar,2 BHK,1225.0,2.0,68.0,2,5551
Jakkur,2 BHK,1290.0,2.0,80.04,2,6204
other,2 BHK,1220.0,2.0,56.0,2,4590
other,1 Bedroom,1500.0,2.0,85.0,1,5666
Kaggadasapura,2 BHK,1361.0,2.0,79.0,2,5804
Frazer Town,5 Bedroom,2801.25,4.0,462.0,5,16492
1st Phase JP Nagar,2 BHK,1394.0,2.0,85.0,2,6097
Yelahanka,2 BHK,870.0,2.0,27.0,2,3103
other,4 Bedroom,5000.0,5.0,300.0,4,6000
Rajaji Nagar,2 BHK,1222.0,2.0,98.0,2,8019
Hebbal,2 BHK,1420.0,2.0,99.39,2,6999
Bellandur,3 BHK,1846.0,3.0,135.0,3,7313
Benson Town,2 Bedroom,1688.12,2.0,270.0,2,15994
Yeshwanthpur,2 BHK,1167.0,2.0,64.08,2,5491
Ananth Nagar,2 BHK,902.0,2.0,25.0,2,2771
Kannamangala,4 BHK,2422.0,4.0,135.0,4,5573
Rachenahalli,2 BHK,1050.0,2.0,52.5,2,5000
other,2 BHK,1000.0,2.0,50.0,2,5000
Kundalahalli,2 BHK,1260.0,2.0,58.0,2,4603
Banashankari Stage II,3 Bedroom,3000.0,3.0,450.0,3,15000
Jigani,2 BHK,939.0,2.0,39.0,2,4153
Konanakunte,9 Bedroom,1590.0,6.0,150.0,9,9433
Jalahalli,1 Bedroom,600.0,1.0,37.0,1,6166
8th Phase JP Nagar,2 BHK,1298.0,2.0,59.0,2,4545
Marathahalli,3 BHK,1710.0,3.0,135.0,3,7894
Subramanyapura,2 BHK,1000.0,2.0,45.0,2,4500
Ramagondanahalli,3 BHK,3350.0,3.0,150.0,3,4477
other,2 Bedroom,1200.0,4.0,138.0,2,11500
other,2 BHK,1100.0,2.0,68.0,2,6181
Jalahalli,1 BHK,615.0,1.0,46.0,1,7479
Sarjapur  Road,2 BHK,1350.0,2.0,85.0,2,6296
Sultan Palaya,4 Bedroom,5000.0,5.0,325.0,4,6500
Cunningham Road,3 BHK,2700.0,3.0,501.0,3,18555
other,6 Bedroom,4000.0,6.0,460.0,6,11500
Hegde Nagar,3 BHK,1718.0,3.0,130.0,3,7566
7th Phase JP Nagar,3 BHK,1680.0,3.0,112.0,3,6666
Hulimavu,2 Bedroom,1500.0,2.0,90.0,2,6000
Sarjapur  Road,3 BHK,1157.0,2.0,72.0,3,6222
Akshaya Nagar,2 BHK,1179.0,2.0,51.0,2,4325
Hulimavu,2 BHK,1315.0,2.0,60.48,2,4599
other,2 BHK,1105.0,2.0,28.18,2,2550
Banashankari,3 BHK,1800.0,3.0,175.0,3,9722
Jalahalli East,2 BHK,1010.0,2.0,52.0,2,5148
Electronic City,1 BHK,630.0,1.0,46.0,1,7301
Anekal,2 Bedroom,1200.0,2.0,36.1,2,3008
Kaval Byrasandra,2 BHK,1025.0,2.0,60.0,2,5853
Hennur Road,2 BHK,1195.0,2.0,59.0,2,4937
Whitefield,2 BHK,1012.0,2.0,58.0,2,5731
Hegde Nagar,3 BHK,1884.0,4.0,118.0,3,6263
Sarjapura - Attibele Road,2 BHK,1090.0,2.0,37.0,2,3394
Kathriguppe,3 BHK,1300.0,3.0,77.99,3,5999
other,4 BHK,2950.0,4.0,250.0,4,8474
Harlur,3 BHK,1710.0,3.0,85.0,3,4970
Uttarahalli,3 BHK,1500.0,3.0,68.0,3,4533
Kodichikkanahalli,3 BHK,1310.0,2.0,53.0,3,4045
Dommasandra,2 BHK,674.0,1.0,19.9,2,2952
Hoodi,3 BHK,1350.0,3.0,70.0,3,5185
other,3 BHK,1464.0,2.0,115.0,3,7855
Marsur,3 BHK,715.0,2.0,29.0,3,4055
Electronic City,3 BHK,1400.0,2.0,40.45,3,2889
Electronic City Phase II,3 BHK,1625.0,2.0,48.75,3,3000
Electronic City,2 BHK,1065.0,2.0,30.76,2,2888
Kumaraswami Layout,6 Bedroom,510.0,4.0,70.0,6,13725
other,3 BHK,1385.0,3.0,90.0,3,6498
CV Raman Nagar,2 BHK,1040.0,2.0,50.0,2,4807
Haralur Road,3 BHK,1817.0,3.0,110.0,3,6053
other,2 BHK,1205.0,2.0,47.0,2,3900
other,2 BHK,1070.0,2.0,33.43,2,3124
other,2 BHK,1070.0,2.0,52.0,2,4859
Amruthahalli,2 BHK,1340.0,2.0,75.0,2,5597
Kalena Agrahara,2 BHK,980.0,2.0,35.0,2,3571
8th Phase JP Nagar,2 BHK,1006.0,2.0,44.0,2,4373
Thanisandra,2 BHK,1098.0,2.0,71.0,2,6466
Mico Layout,2 BHK,1200.0,2.0,53.5,2,4458
Thigalarapalya,4 BHK,4190.0,4.0,325.0,4,7756
Sarjapur  Road,2 BHK,1019.0,2.0,62.0,2,6084
Whitefield,2 BHK,1150.0,2.0,40.21,2,3496
Anekal,3 Bedroom,1733.5,3.0,120.0,3,6922
other,2 BHK,1415.0,2.0,120.0,2,8480
other,3 BHK,1750.0,3.0,58.0,3,3314
other,3 BHK,1537.0,3.0,81.0,3,5270
other,2 BHK,1050.0,2.0,45.0,2,4285
Kanakpura Road,3 BHK,1452.0,3.0,55.6,3,3829
Domlur,2 BHK,1246.0,2.0,95.0,2,7624
Yelahanka,3 BHK,1890.0,4.0,85.0,3,4497
Electronic City Phase II,2 BHK,972.0,2.0,40.0,2,4115
other,4 Bedroom,10624.0,4.0,2340.0,4,22025
Brookefield,4 Bedroom,1500.0,5.0,160.0,4,10666
Electronic City,1 BHK,630.0,1.0,33.7,1,5349
Kothanur,8 Bedroom,1050.0,5.0,80.0,8,7619
other,3 BHK,1255.0,2.0,60.0,3,4780
Ramamurthy Nagar,4 Bedroom,1900.0,4.0,185.0,4,9736
Doddaballapur,3 Bedroom,2440.0,3.0,142.0,3,5819
Kanakpura Road,3 BHK,1938.0,3.0,92.69,3,4782
Bannerghatta Road,3 BHK,1711.0,3.0,125.0,3,7305
Nagarbhavi,5 Bedroom,1200.0,4.0,240.0,5,20000
CV Raman Nagar,2 BHK,1392.0,2.0,95.0,2,6824
Haralur Road,2 BHK,1225.0,2.0,77.0,2,6285
Sarjapur  Road,4 BHK,3418.0,3.0,235.0,4,6875
Green Glen Layout,3 BHK,1750.0,3.0,120.0,3,6857
Subramanyapura,3 BHK,1260.0,2.0,80.0,3,6349
Sarjapur  Road,1 BHK,691.0,1.0,31.1,1,4500
BTM 2nd Stage,4 Bedroom,2400.0,3.0,400.0,4,16666
Haralur Road,2 BHK,1285.0,2.0,78.0,2,6070
Hebbal,3 BHK,1590.0,3.0,96.0,3,6037
9th Phase JP Nagar,3 BHK,1571.0,3.0,65.0,3,4137
Hebbal Kempapura,5 Bedroom,2800.0,5.0,220.0,5,7857
Garudachar Palya,2 BHK,1060.0,2.0,48.66,2,4590
Hennur,2 BHK,1255.0,2.0,58.0,2,4621
other,4 Bedroom,702.0,4.0,65.0,4,9259
other,3 Bedroom,2400.0,2.0,185.0,3,7708
Bannerghatta Road,3 BHK,1520.0,2.0,81.0,3,5328
other,2 BHK,1170.0,2.0,45.51,2,3889
OMBR Layout,3 BHK,1580.0,3.0,75.0,3,4746
Ardendale,3 BHK,1777.26,3.0,89.0,3,5007
Raja Rajeshwari Nagar,2 BHK,1303.0,2.0,55.78,2,4280
Kodichikkanahalli,2 BHK,1680.0,2.0,59.0,2,3511
Whitefield,3 BHK,1495.0,2.0,70.0,3,4682
Yelahanka,3 BHK,1835.0,3.0,108.0,3,5885
Banjara Layout,3 Bedroom,610.0,3.0,60.0,3,9836
other,2 BHK,1080.0,2.0,38.0,2,3518
Electronic City,2 BHK,1025.0,2.0,29.61,2,2888
Haralur Road,2 BHK,1225.0,2.0,67.48,2,5508
Balagere,1 BHK,656.0,1.0,34.77,1,5300
Sarjapur  Road,2 BHK,994.0,2.0,45.0,2,4527
other,2 BHK,1025.0,2.0,65.0,2,6341
Marathahalli,2 BHK,1156.0,2.0,52.0,2,4498
JP Nagar,2 BHK,1250.0,2.0,60.0,2,4800
Kundalahalli,4 Bedroom,3092.0,4.0,230.0,4,7438
Hegde Nagar,3 BHK,1930.0,4.0,122.0,3,6321
other,2 BHK,1020.0,2.0,70.0,2,6862
Kumaraswami Layout,3 BHK,1310.0,2.0,70.0,3,5343
Jalahalli,1 BHK,612.5,1.0,27.565,1,4500
Kanakpura Road,2 BHK,1322.0,2.0,70.0,2,5295
Kothanur,2 BHK,1160.0,2.0,57.0,2,4913
5th Block Hbr Layout,2 BHK,1100.0,2.0,48.0,2,4363
Lakshminarayana Pura,3 BHK,1720.0,3.0,150.0,3,8720
Hosur Road,3 BHK,1689.0,3.0,103.0,3,6098
Chikkalasandra,2 BHK,1090.0,2.0,47.42,2,4350
KR Puram,2 BHK,2000.0,2.0,75.0,2,3750
other,2 BHK,1155.0,2.0,69.0,2,5974
Vishveshwarya Layout,6 Bedroom,4000.0,6.0,230.0,6,5750
Sarjapur  Road,4 Bedroom,3913.0,5.0,176.0,4,4497
Hebbal,3 BHK,3520.0,5.0,320.0,3,9090
JP Nagar,2 BHK,1125.0,2.0,39.98,2,3553
Yelahanka,3 BHK,1847.0,3.0,92.0,3,4981
7th Phase JP Nagar,3 BHK,1515.0,2.0,76.0,3,5016
Raja Rajeshwari Nagar,8 Bedroom,3450.0,7.0,225.0,8,6521
Yelahanka,5 Bedroom,2500.0,3.0,120.0,5,4800
Sarjapur  Road,2 BHK,1130.0,2.0,55.0,2,4867
Bellandur,2 BHK,1049.0,2.0,51.0,2,4861
Amruthahalli,3 BHK,2650.0,4.0,175.0,3,6603
Varthur Road,2 BHK,850.0,2.0,25.4,2,2988
Badavala Nagar,3 BHK,1842.0,3.0,115.0,3,6243
other,5 Bedroom,2600.0,5.0,370.0,5,14230
Jakkur,2 BHK,1282.0,2.0,72.0,2,5616
Kalena Agrahara,2 BHK,800.0,2.0,40.0,2,5000
other,4 Bedroom,1200.0,4.0,240.0,4,20000
Chandapura,3 BHK,1190.0,2.0,30.35,3,2550
Magadi Road,3 BHK,1318.0,2.0,68.0,3,5159
Kanakpura Road,2 BHK,1698.0,2.0,105.0,2,6183
other,3 BHK,2030.0,2.0,182.0,3,8965
Kammasandra,3 BHK,1276.0,2.0,31.9,3,2500
other,3 BHK,1555.0,2.0,67.0,3,4308
Sarjapur  Road,3 BHK,1881.0,3.0,145.0,3,7708
other,3 BHK,2900.0,3.0,325.0,3,11206
Yelahanka,3 BHK,1890.0,3.0,109.0,3,5767
other,5 BHK,900.0,5.0,135.0,5,15000
Old Madras Road,2 BHK,935.0,2.0,32.0,2,3422
Cox Town,2 BHK,1000.0,2.0,58.0,2,5800
Singasandra,3 BHK,1510.0,3.0,75.0,3,4966
Uttarahalli,2 BHK,1120.0,2.0,50.0,2,4464
Ananth Nagar,2 BHK,930.0,2.0,25.99,2,2794
Electronic City,3 BHK,1160.0,2.0,42.0,3,3620
Raja Rajeshwari Nagar,2 BHK,1419.0,2.0,48.1,2,3389
Talaghattapura,2 BHK,1175.0,2.0,62.0,2,5276
Bellandur,3 BHK,1350.0,2.0,63.0,3,4666
Malleshwaram,3 BHK,2215.0,3.0,330.0,3,14898
Attibele,2 BHK,996.0,2.0,24.9,2,2500
Anjanapura,2 BHK,1076.0,2.0,30.13,2,2800
other,4 BHK,2180.0,4.0,115.0,4,5275
Kasavanhalli,3 BHK,1715.0,3.0,112.0,3,6530
Hebbal,2 BHK,1072.0,2.0,45.0,2,4197
Panathur,2 BHK,1230.0,2.0,80.0,2,6504
Murugeshpalya,3 BHK,1600.0,2.0,65.0,3,4062
Thanisandra,3 BHK,1411.0,3.0,93.25,3,6608
Rayasandra,5 BHK,3600.0,5.0,145.0,5,4027
Jalahalli,3 BHK,2404.0,3.0,138.0,3,5740
Sahakara Nagar,4 Bedroom,1200.0,4.0,190.0,4,15833
Nagarbhavi,3 Bedroom,1200.0,3.0,226.0,3,18833
Doddathoguru,3 BHK,1208.0,3.0,45.0,3,3725
Yeshwanthpur,5 Bedroom,1700.0,5.0,300.0,5,17647
Haralur Road,2 BHK,1225.0,2.0,69.84,2,5701
Hebbal,2 BHK,1162.0,2.0,69.0,2,5938
Thanisandra,3 BHK,1262.0,2.0,65.0,3,5150
other,2 BHK,1050.0,2.0,35.0,2,3333
Electronic City Phase II,2 BHK,1160.0,2.0,33.51,2,2888
Channasandra,3 Bedroom,1500.0,3.0,61.95,3,4130
other,3 BHK,1610.0,3.0,85.0,3,5279
Whitefield,2 BHK,1340.0,2.0,41.0,2,3059
Sarjapur  Road,3 BHK,2275.0,4.0,182.0,3,8000
Budigere,2 BHK,1162.0,2.0,58.0,2,4991
other,2 Bedroom,1500.0,2.0,145.0,2,9666
other,4 Bedroom,3150.0,3.0,135.0,4,4285
Hosa Road,2 BHK,1170.0,2.0,65.0,2,5555
Sarjapur  Road,2 BHK,980.0,2.0,25.0,2,2551
Channasandra,3 BHK,1800.0,3.0,60.0,3,3333
Banashankari Stage VI,3 BHK,1423.0,2.0,71.73,3,5040
Kaggadasapura,2 BHK,1106.0,2.0,40.0,2,3616
Kereguddadahalli,2 BHK,1105.0,2.0,27.6,2,2497
Cunningham Road,4 Bedroom,7500.0,6.0,1800.0,4,24000
other,2 BHK,1255.0,2.0,56.7,2,4517
Sarjapur  Road,3 BHK,1525.0,2.0,65.0,3,4262
Babusapalaya,2 BHK,1061.0,2.0,38.95,2,3671
other,2 Bedroom,600.0,2.0,29.5,2,4916
Electronic City Phase II,3 BHK,1400.0,2.0,40.45,3,2889
Domlur,3 BHK,2100.0,3.0,145.0,3,6904
Electronic City,2 BHK,660.0,1.0,16.5,2,2500
Kengeri,2 BHK,633.0,1.0,18.0,2,2843
Indira Nagar,2 BHK,1260.0,2.0,120.0,2,9523
ITPL,2 BHK,907.0,2.0,55.43,2,6111
Hennur Road,3 BHK,2047.0,3.0,67.55,3,3299
other,4 BHK,2597.0,4.0,189.0,4,7277
Bellandur,2 BHK,1211.0,2.0,63.0,2,5202
Hennur Road,3 BHK,1904.0,3.0,129.0,3,6775
Balagere,2 BHK,1205.0,2.0,73.87,2,6130
Yeshwanthpur,2 BHK,1164.0,2.0,64.08,2,5505
other,2 BHK,1410.0,2.0,73.0,2,5177
other,2 BHK,1150.0,2.0,42.0,2,3652
other,2 BHK,1170.0,2.0,53.11,2,4539
Hennur,2 BHK,1420.0,2.0,62.0,2,4366
Panathur,2 BHK,1125.0,2.0,45.0,2,4000
NRI Layout,3 BHK,1789.0,3.0,75.0,3,4192
Hulimavu,2 BHK,1021.0,2.0,76.0,2,7443
Ananth Nagar,2 BHK,1074.0,2.0,26.85,2,2500
Kalena Agrahara,2 BHK,1222.0,2.0,48.0,2,3927
Hennur Road,3 BHK,1981.0,3.0,134.0,3,6764
Yelachenahalli,2 BHK,1103.0,2.0,55.0,2,4986
other,2 BHK,1201.0,2.0,65.0,2,5412
Sarjapur  Road,2 BHK,970.0,2.0,36.0,2,3711
Sarjapur,2 BHK,920.0,2.0,33.0,2,3586
Yeshwanthpur,3 BHK,1520.0,3.0,85.0,3,5592
other,3 Bedroom,675.0,3.0,75.0,3,11111
Sonnenahalli,3 BHK,1268.0,2.0,75.0,3,5914
other,1 BHK,660.0,1.0,30.0,1,4545
Harlur,3 BHK,1755.0,3.0,115.0,3,6552
Hoskote,2 BHK,945.0,2.0,35.0,2,3703
8th Phase JP Nagar,1 BHK,500.0,1.0,33.0,1,6600
Harlur,4 BHK,2820.0,4.0,153.5,4,5443
Bommenahalli,4 Bedroom,1632.0,3.0,140.0,4,8578
Rachenahalli,2 BHK,1167.0,2.0,37.5,2,3213
Domlur,2 BHK,1276.0,2.0,80.0,2,6269
Thanisandra,2 BHK,1234.0,2.0,51.0,2,4132
Lakshminarayana Pura,2 BHK,1180.0,2.0,75.0,2,6355
ISRO Layout,3 BHK,1370.0,2.0,87.0,3,6350
Chandapura,2 BHK,1015.0,2.0,25.88,2,2549
other,9 Bedroom,1200.0,9.0,230.0,9,19166
Electronic City,3 Bedroom,1400.0,3.0,90.0,3,6428
Whitefield,5 Bedroom,1500.0,6.0,155.0,5,10333
other,2 Bedroom,1050.0,2.0,40.0,2,3809
Ramagondanahalli,2 BHK,1251.0,2.0,46.0,2,3677
Yeshwanthpur,3 BHK,1677.0,3.0,92.13,3,5493
Sarjapur  Road,2 BHK,1346.0,2.0,67.38,2,5005
Electronic City Phase II,2 BHK,1090.0,2.0,31.48,2,2888
2nd Stage Nagarbhavi,5 Bedroom,1200.0,4.0,240.0,5,20000
Kanakapura,3 BHK,1290.0,2.0,51.1,3,3961
Rajaji Nagar,5 BHK,7514.0,7.0,1500.0,5,19962
Sarjapur  Road,3 BHK,1881.0,3.0,145.0,3,7708
Garudachar Palya,2 BHK,1150.0,2.0,52.0,2,4521
Sarjapur,3 Bedroom,2690.0,3.0,295.0,3,10966
Kanakpura Road,3 BHK,1200.0,2.0,42.0,3,3500
other,3 BHK,2735.0,4.0,340.0,3,12431
other,2 BHK,950.0,2.0,65.0,2,6842
Bommasandra Industrial Area,3 BHK,1491.0,3.0,40.5,3,2716
Ramamurthy Nagar,2 BHK,1050.0,2.0,51.66,2,4920
Hoodi,3 BHK,1660.0,3.0,131.0,3,7891
Nagavarapalya,2 BHK,1134.0,2.0,55.0,2,4850
Hebbal,2 BHK,1100.0,2.0,60.0,2,5454
Magadi Road,4 Bedroom,800.0,4.0,60.0,4,7500
Anekal,2 Bedroom,1200.0,2.0,36.1,2,3008
Nagarbhavi,4 Bedroom,1200.0,4.0,240.0,4,20000
other,3 Bedroom,980.0,3.0,80.0,3,8163
other,2 BHK,1080.0,2.0,37.8,2,3499
CV Raman Nagar,1 BHK,705.0,2.0,50.0,1,7092
other,3 BHK,1200.0,2.0,35.0,3,2916
Anandapura,2 BHK,1200.0,2.0,50.0,2,4166
Talaghattapura,3 BHK,1804.0,3.0,127.0,3,7039
other,2 BHK,750.0,2.0,20.0,2,2666
other,2 BHK,420.0,1.0,25.0,2,5952
Malleshwaram,4 BHK,2800.0,4.0,252.0,4,9000
Battarahalli,2 BHK,1090.0,2.0,38.37,2,3520
R.T. Nagar,2 Bedroom,1500.0,2.0,230.0,2,15333
Electronic City,3 BHK,1287.0,3.0,55.0,3,4273
Thigalarapalya,3 BHK,2215.0,4.0,157.0,3,7088
Kadubeesanahalli,4 Bedroom,2800.0,4.0,375.0,4,13392
Sarjapur  Road,2 BHK,1045.0,2.0,42.15,2,4033
Hormavu,2 BHK,1263.0,2.0,47.36,2,3749
Whitefield,3 BHK,1431.0,3.0,85.0,3,5939
Dommasandra,2 BHK,999.0,2.0,32.9,2,3293
Vittasandra,3 BHK,1650.0,3.0,85.5,3,5181
other,3 BHK,1535.0,3.0,72.0,3,4690
Marathahalli,4 BHK,2524.0,5.0,180.0,4,7131
Kanakpura Road,3 BHK,1300.0,3.0,69.0,3,5307
other,3 Bedroom,600.0,3.0,125.0,3,20833
7th Phase JP Nagar,3 BHK,2095.0,3.0,150.0,3,7159
Yelahanka New Town,1 BHK,284.0,1.0,8.0,1,2816
Ananth Nagar,1 BHK,500.0,1.0,14.0,1,2800
other,2 BHK,1265.0,2.0,68.0,2,5375
Kudlu Gate,2 BHK,1215.0,2.0,44.96,2,3700
Yelahanka New Town,2 Bedroom,2200.0,2.0,180.0,2,8181
Electronics City Phase 1,2 BHK,1028.0,2.0,51.0,2,4961
Hegde Nagar,3 BHK,1835.0,2.0,92.0,3,5013
Munnekollal,3 BHK,1800.0,3.0,65.0,3,3611
1st Phase JP Nagar,3 BHK,2077.0,3.0,175.0,3,8425
Sarjapur  Road,4 Bedroom,2880.0,3.0,211.0,4,7326
other,6 BHK,499.0,5.0,89.0,6,17835
Dasanapura,2 BHK,708.0,2.0,31.15,2,4399
Basavangudi,2 BHK,1200.0,2.0,120.0,2,10000
Hosakerehalli,2 BHK,925.0,2.0,46.25,2,5000
Bannerghatta Road,3 Bedroom,2826.0,3.0,148.0,3,5237
Kanakpura Road,1 BHK,458.0,1.0,33.66,1,7349
other,4 Bedroom,3500.0,4.0,150.0,4,4285
other,4 Bedroom,5400.0,4.0,250.0,4,4629
other,4 Bedroom,1000.0,5.0,200.0,4,20000
Amruthahalli,3 BHK,1450.0,2.0,90.0,3,6206
Kasavanhalli,2 BHK,1100.0,2.0,65.0,2,5909
Kadubeesanahalli,2 BHK,1185.0,2.0,55.0,2,4641
9th Phase JP Nagar,4 Bedroom,1900.0,4.0,80.0,4,4210
TC Palaya,2 Bedroom,1200.0,2.0,70.0,2,5833
other,1 BHK,660.0,1.0,35.0,1,5303
Ambedkar Nagar,3 BHK,1920.0,4.0,121.0,3,6302
Jalahalli,2 Bedroom,1200.0,2.0,130.0,2,10833
1st Phase JP Nagar,4 BHK,4550.0,2.0,240.0,4,5274
Bhoganhalli,2 BHK,1205.0,2.0,68.32,2,5669
Haralur Road,2 BHK,1200.0,2.0,46.0,2,3833
Akshaya Nagar,3 BHK,1421.0,2.0,73.0,3,5137
6th Phase JP Nagar,3 BHK,1250.0,3.0,48.9,3,3912
Whitefield,1 BHK,524.0,1.0,29.0,1,5534
HSR Layout,5 Bedroom,4200.0,5.0,245.0,5,5833
Thubarahalli,3 BHK,1540.0,3.0,90.0,3,5844
Sarjapur  Road,2 BHK,984.0,2.0,45.91,2,4665
Rachenahalli,2 BHK,985.0,2.0,50.17,2,5093
Whitefield,4 BHK,2830.0,4.0,161.0,4,5689
other,2 BHK,1140.0,2.0,60.0,2,5263
5th Block Hbr Layout,5 Bedroom,3600.0,5.0,130.0,5,3611
Begur Road,3 BHK,1410.0,2.0,44.42,3,3150
Raja Rajeshwari Nagar,3 BHK,1700.0,2.0,86.0,3,5058
Devanahalli,2 BHK,1360.0,2.0,62.425,2,4590
Electronic City,4 Bedroom,1800.0,3.0,700.0,4,38888
Dasarahalli,2 BHK,1333.0,2.0,86.65,2,6500
Electronic City,2 BHK,1258.0,2.0,85.5,2,6796
other,3 BHK,1458.0,2.0,87.0,3,5967
Electronic City,2 BHK,1386.0,2.0,85.0,2,6132
Electronic City Phase II,2 BHK,1140.0,2.0,32.92,2,2887
Electronic City,1 RK,550.0,1.0,27.0,1,4909
Lakshminarayana Pura,2 BHK,1200.0,2.0,83.0,2,6916
other,2 BHK,1081.0,2.0,39.0,2,3607
Whitefield,4 BHK,2135.0,3.0,149.0,4,6978
Whitefield,4 Bedroom,2500.0,4.0,100.0,4,4000
other,2 BHK,1178.0,2.0,80.0,2,6791
Devarachikkanahalli,2 BHK,991.0,2.0,40.0,2,4036
Gottigere,3 BHK,1493.0,2.0,90.0,3,6028
Subramanyapura,3 BHK,2495.0,3.0,130.0,3,5210
Kammasandra,2 BHK,1080.0,2.0,31.99,2,2962
Dasanapura,2 BHK,814.0,2.0,53.0,2,6511
Electronic City,2 BHK,1025.0,2.0,50.0,2,4878
Bannerghatta Road,2 BHK,1115.0,2.0,61.0,2,5470
Whitefield,4 BHK,3075.0,2.0,188.0,4,6113
other,7 Bedroom,1900.0,3.0,162.0,7,8526
Whitefield,3 BHK,1760.0,3.0,160.0,3,9090
other,7 BHK,4000.0,8.0,150.0,7,3750
Marathahalli,4 Bedroom,3090.0,4.0,350.0,4,11326
other,4 Bedroom,2700.0,4.0,295.0,4,10925
other,2 BHK,1175.0,2.0,52.86,2,4498
Hebbal,4 BHK,4235.0,5.0,364.0,4,8595
KR Puram,2 BHK,1100.0,2.0,45.0,2,4090
Banashankari Stage VI,8 Bedroom,1200.0,7.0,175.0,8,14583
Banashankari,2 BHK,1450.0,2.0,50.75,2,3500
6th Phase JP Nagar,3 BHK,1810.0,3.0,131.0,3,7237
other,3 BHK,1426.0,2.0,71.0,3,4978
Sector 7 HSR Layout,3 BHK,2400.0,2.0,195.0,3,8125
Kalyan nagar,2 BHK,1100.0,2.0,70.0,2,6363
Electronic City Phase II,1 BHK,650.0,1.0,35.0,1,5384
Old Madras Road,3 BHK,1350.0,3.0,47.25,3,3500
Kaikondrahalli,2 BHK,900.0,2.0,27.0,2,3000
Billekahalli,2 BHK,1360.0,2.0,110.0,2,8088
Yeshwanthpur,1 BHK,674.0,1.0,36.85,1,5467
Whitefield,3 BHK,1655.0,3.0,110.0,3,6646
other,2 Bedroom,1300.0,2.0,95.0,2,7307
other,1 BHK,936.0,1.0,53.0,1,5662
other,2 BHK,1277.0,2.0,92.0,2,7204
other,2 BHK,1200.0,2.0,54.0,2,4500
Hormavu,2 BHK,1015.0,2.0,39.08,2,3850
Budigere,2 BHK,1153.0,2.0,60.0,2,5203
Kengeri,2 BHK,1035.0,2.0,51.5,2,4975
KR Puram,5 BHK,3300.0,3.0,255.0,5,7727
Uttarahalli,2 BHK,1150.0,2.0,46.0,2,4000
Malleshwaram,2 BHK,1020.0,2.0,80.0,2,7843
Old Madras Road,4 BHK,3715.0,6.0,212.5,4,5720
8th Phase JP Nagar,2 BHK,1160.0,2.0,95.0,2,8189
Bisuvanahalli,3 BHK,1075.0,2.0,31.0,3,2883
Brookefield,2 BHK,1225.0,2.0,72.0,2,5877
Benson Town,4 Bedroom,5000.0,5.0,950.0,4,19000
other,2 Bedroom,1600.0,2.0,160.0,2,10000
Malleshwaram,2 Bedroom,600.0,1.0,90.0,2,15000
other,3 BHK,1350.0,3.0,60.0,3,4444
Kanakpura Road,3 BHK,1300.0,3.0,69.0,3,5307
Channasandra,2 BHK,830.0,2.0,36.28,2,4371
Hennur Road,2 BHK,1155.0,2.0,69.18,2,5989
Sarakki Nagar,4 BHK,3126.0,5.0,345.0,4,11036
Jalahalli,2 BHK,905.0,2.0,55.0,2,6077
Vijayanagar,3 BHK,1375.0,2.0,75.0,3,5454
Electronic City,2 BHK,1140.0,2.0,33.84,2,2968
Hormavu,3 BHK,1550.0,2.0,60.0,3,3870
Kannamangala,2 BHK,957.0,2.0,55.0,2,5747
9th Phase JP Nagar,2 BHK,890.0,2.0,45.0,2,5056
KR Puram,4 BHK,3000.0,4.0,235.0,4,7833
Electronic City,2 BHK,1070.0,2.0,55.0,2,5140
other,5 Bedroom,2400.0,5.0,150.0,5,6250
other,4 Bedroom,2400.0,4.0,595.0,4,24791
Rachenahalli,2 BHK,925.0,2.0,37.0,2,4000
Kammasandra,2 BHK,1415.0,2.0,60.0,2,4240
other,1 Bedroom,500.0,2.0,80.0,1,16000
Thanisandra,2 BHK,1225.0,2.0,37.0,2,3020
Hennur Road,4 Bedroom,2000.0,4.0,90.0,4,4500
Whitefield,1 BHK,840.0,1.0,57.0,1,6785
Banashankari Stage III,9 Bedroom,1560.0,9.0,200.0,9,12820
Thanisandra,3 BHK,1732.0,3.0,112.0,3,6466
Parappana Agrahara,1 Bedroom,1200.0,1.0,45.0,1,3750
Bhoganhalli,2 BHK,1260.0,2.0,69.9,2,5547
other,2 BHK,1092.0,2.0,49.0,2,4487
other,2 BHK,827.0,2.0,34.0,2,4111
Horamavu Banaswadi,2 BHK,1357.0,2.0,54.0,2,3979
Chamrajpet,2 BHK,650.0,1.0,40.0,2,6153
7th Phase JP Nagar,3 BHK,1370.0,2.0,54.79,3,3999
Whitefield,3 BHK,1725.0,3.0,106.0,3,6144
Jalahalli,2 BHK,1000.0,2.0,70.0,2,7000
TC Palaya,1 Bedroom,1350.0,1.0,55.0,1,4074
7th Phase JP Nagar,4 Bedroom,3200.0,4.0,350.0,4,10937
Bellandur,3 BHK,1717.0,3.0,110.0,3,6406
Kanakpura Road,2 BHK,700.0,2.0,34.0,2,4857
other,2 BHK,800.0,2.0,32.0,2,4000
Rajaji Nagar,3 BHK,2367.0,4.0,340.0,3,14364
Sonnenahalli,3 BHK,1415.0,2.0,65.0,3,4593
7th Phase JP Nagar,3 BHK,1976.0,3.0,145.0,3,7338
Kambipura,2 BHK,883.0,2.0,37.01,2,4191
Yelahanka,1 BHK,700.0,1.0,38.0,1,5428
Sarjapur,3 BHK,1445.0,3.0,50.0,3,3460
Battarahalli,3 BHK,2082.0,3.0,104.0,3,4995
HSR Layout,2 BHK,1140.0,2.0,46.0,2,4035
Thanisandra,2 BHK,933.0,2.0,66.0,2,7073
NRI Layout,4 Bedroom,800.0,5.0,95.5,4,11937
Electronic City Phase II,2 BHK,1155.0,2.0,57.0,2,4935
Whitefield,2 BHK,1130.0,2.0,36.0,2,3185
Bannerghatta,2 BHK,1200.0,2.0,73.2,2,6100
other,2 BHK,1055.0,2.0,43.0,2,4075
Rajaji Nagar,3 BHK,1555.0,3.0,125.0,3,8038
Bellandur,3 BHK,1420.0,3.0,62.0,3,4366
other,3 BHK,2214.0,3.0,350.0,3,15808
Devanahalli,1 BHK,658.0,1.0,26.91,1,4089
Bhoganhalli,2 BHK,896.9,2.0,78.74,2,8779
Giri Nagar,4 Bedroom,4000.0,3.0,750.0,4,18750
KR Puram,2 BHK,1035.0,2.0,40.0,2,3864
Whitefield,4 Bedroom,2064.0,5.0,225.0,4,10901
other,3 BHK,1839.0,3.0,160.0,3,8700
Begur Road,2 BHK,1200.0,2.0,44.0,2,3666
Yelahanka New Town,1 BHK,488.0,1.0,20.0,1,4098
Banashankari Stage II,3 BHK,1200.0,2.0,198.0,3,16500
other,5 Bedroom,2760.0,5.0,140.0,5,5072
other,2 BHK,993.0,2.0,42.0,2,4229
Hoodi,2 BHK,1225.0,2.0,45.0,2,3673
TC Palaya,2 Bedroom,900.0,2.0,55.0,2,6111
Electronic City,9 Bedroom,1200.0,13.0,150.0,9,12500
Sompura,2 BHK,825.0,2.0,32.0,2,3878
Bellandur,4 BHK,3596.0,4.0,359.0,4,9983
other,6 Bedroom,1350.0,4.0,203.0,6,15037
Electronic City Phase II,3 BHK,1800.0,3.0,92.0,3,5111
other,2 BHK,925.0,2.0,100.0,2,10810
Kanakpura Road,2 BHK,1167.0,2.0,63.0,2,5398
Sarjapur,2 BHK,1150.0,2.0,41.0,2,3565
other,1 BHK,650.0,1.0,50.0,1,7692
Sarjapur  Road,1 BHK,675.0,1.0,27.0,1,4000
other,2 Bedroom,2260.0,3.0,90.0,2,3982
other,1 BHK,720.0,1.0,32.39,1,4498
Whitefield,3 Bedroom,4800.0,4.0,600.0,3,12500
Yelahanka New Town,2 BHK,1100.0,2.0,58.0,2,5272
other,2 BHK,550.0,1.0,20.0,2,3636
Uttarahalli,3 BHK,1330.0,2.0,58.0,3,4360
Koramangala,2 BHK,1320.0,2.0,160.0,2,12121
Ulsoor,3 BHK,1685.0,4.0,185.0,3,10979
other,6 BHK,2400.0,5.0,75.0,6,3125
Banashankari,3 BHK,1770.0,3.0,108.0,3,6101
Prithvi Layout,3 BHK,2100.0,3.0,125.0,3,5952
Pai Layout,2 BHK,1000.0,2.0,42.0,2,4200
Banashankari Stage VI,4 Bedroom,600.0,5.0,105.0,4,17500
Ramamurthy Nagar,4 Bedroom,2916.0,5.0,230.0,4,7887
Kanakpura Road,3 BHK,1419.59,2.0,65.0,3,4578
Kodihalli,3 BHK,1400.0,3.0,81.0,3,5785
Horamavu Agara,2 BHK,750.0,2.0,37.49,2,4998
Bannerghatta,2 BHK,1070.0,2.0,87.0,2,8130
Kannamangala,3 BHK,1536.0,3.0,107.0,3,6966
Koramangala,3 BHK,1710.0,3.0,210.0,3,12280
JP Nagar,3 BHK,2400.0,3.0,196.0,3,8166
Electronic City,2 BHK,1342.0,2.0,72.0,2,5365
other,3 BHK,1600.0,3.0,130.0,3,8125
Kanakapura,1 BHK,938.0,1.0,50.0,1,5330
Budigere,3 BHK,1820.0,3.0,99.0,3,5439
Bhoganhalli,4 BHK,2439.0,4.0,170.0,4,6970
Kanakpura Road,3 BHK,1665.0,3.0,78.0,3,4684
Basavangudi,3 BHK,1850.0,3.0,150.0,3,8108
Bommanahalli,2 BHK,1091.0,2.0,40.0,2,3666
Sarjapur  Road,2 BHK,1346.0,2.0,74.03,2,5500
Rajaji Nagar,3 BHK,1720.0,3.0,224.0,3,13023
Brookefield,4 Bedroom,3675.0,4.0,176.0,4,4789
JP Nagar,2 BHK,1060.0,2.0,33.95,2,3202
Whitefield,2 BHK,1059.0,2.0,57.0,2,5382
Hebbal,2 BHK,1299.0,2.0,97.0,2,7467
Yelahanka,3 BHK,1491.0,3.0,85.0,3,5700
Basaveshwara Nagar,2 BHK,1200.0,2.0,85.0,2,7083
Yelahanka,4 Bedroom,2088.0,5.0,110.0,4,5268
Hulimavu,1 BHK,581.0,1.0,29.0,1,4991
other,2 BHK,1296.0,2.0,70.0,2,5401
other,2 BHK,1125.0,2.0,55.0,2,4888
Electronic City,3 BHK,1275.0,3.0,70.0,3,5490
Kothanur,2 BHK,1095.0,2.0,55.0,2,5022
Sarjapur  Road,5 Bedroom,10000.0,4.0,975.0,5,9750
Old Madras Road,2 BHK,1330.0,2.0,74.0,2,5563
Marathahalli,2 BHK,1135.0,2.0,60.0,2,5286
Whitefield,1 BHK,516.0,1.0,39.0,1,7558
other,2 BHK,1194.0,2.0,51.0,2,4271
Banaswadi,5 Bedroom,2250.0,5.0,240.0,5,10666
Electronic City,2 BHK,770.0,1.0,38.0,2,4935
Sarjapur  Road,4 BHK,3930.0,5.0,330.0,4,8396
Chikkalasandra,3 BHK,1355.0,3.0,54.2,3,4000
Abbigere,8 Bedroom,3000.0,8.0,150.0,8,5000
Harlur,4 BHK,2569.0,4.0,175.0,4,6811
Hebbal,4 Bedroom,3758.0,4.0,450.0,4,11974
Whitefield,2 BHK,1116.0,2.0,51.91,2,4651
Bhoganhalli,3 BHK,1707.0,3.0,115.0,3,6736
other,5 Bedroom,1200.0,5.0,225.0,5,18750
Sultan Palaya,2 BHK,1070.0,2.0,49.0,2,4579
Whitefield,4 Bedroom,6000.0,4.0,410.0,4,6833
Tumkur Road,3 BHK,1240.0,2.0,83.0,3,6693
JP Nagar,2 BHK,1300.0,2.0,91.65,2,7050
Ambalipura,2 BHK,1700.0,2.0,76.0,2,4470
Yeshwanthpur,3 BHK,1383.0,2.0,76.18,3,5508
other,5 Bedroom,3000.0,4.0,170.0,5,5666
Electronic City,2 BHK,1200.0,2.0,34.65,2,2887
Hulimavu,2 BHK,935.0,2.0,45.0,2,4812
Thigalarapalya,3 BHK,2072.0,4.0,151.0,3,7287
other,4 Bedroom,2200.0,3.0,225.0,4,10227
Jalahalli,1 BHK,615.0,1.0,46.0,1,7479
KR Puram,2 BHK,1116.0,2.0,45.0,2,4032
Whitefield,3 BHK,1639.0,3.0,107.0,3,6528
Marathahalli,3 BHK,1839.0,3.0,130.0,3,7069
other,2 BHK,1300.0,2.0,63.0,2,4846
Horamavu Agara,1 BHK,625.0,1.0,27.0,1,4320
Thanisandra,1 BHK,933.0,1.0,61.0,1,6538
Hosur Road,3 BHK,2289.0,3.0,180.0,3,7863
Ramamurthy Nagar,3 Bedroom,2316.0,3.0,125.0,3,5397
Electronic City,2 BHK,1005.0,2.0,39.77,2,3957
Horamavu Agara,2 BHK,1080.0,2.0,38.0,2,3518
Munnekollal,7 Bedroom,1200.0,5.0,185.0,7,15416
Vishwapriya Layout,2 BHK,910.0,2.0,36.0,2,3956
Hormavu,2 Bedroom,1200.0,2.0,73.0,2,6083
Electronic City,2 BHK,1110.0,2.0,39.9,2,3594
Kothanur,2 BHK,1142.0,2.0,56.0,2,4903
2nd Stage Nagarbhavi,5 Bedroom,4000.0,4.0,240.0,5,6000
Hormavu,3 Bedroom,1800.0,3.0,204.0,3,11333
AECS Layout,2 BHK,1005.0,2.0,55.0,2,5472
other,3 BHK,1626.6,3.0,133.0,3,8176
Kothanur,2 BHK,1074.0,2.0,54.0,2,5027
Basavangudi,2 BHK,1230.0,2.0,80.0,2,6504
Electronic City,2 BHK,660.0,1.0,23.0,2,3484
other,2 BHK,1007.0,2.0,48.0,2,4766
other,4 Bedroom,1050.0,6.0,74.0,4,7047
Whitefield,3 BHK,1820.0,3.0,113.0,3,6208
other,2 BHK,1100.0,2.0,60.0,2,5454
other,2 Bedroom,1348.0,2.0,100.0,2,7418
Kasavanhalli,2 BHK,1230.0,2.0,66.25,2,5386
Horamavu Agara,3 BHK,1176.0,2.0,37.62,3,3198
Uttarahalli,3 BHK,1590.0,3.0,57.0,3,3584
Frazer Town,2 BHK,1200.0,2.0,78.0,2,6500
other,2 BHK,1298.0,2.0,65.0,2,5007
Hosa Road,2 BHK,1161.0,2.0,55.15,2,4750
other,4 Bedroom,1350.0,4.0,175.0,4,12962
other,5 Bedroom,1200.0,5.0,138.0,5,11500
Kothanur,2 BHK,1285.0,2.0,60.0,2,4669
Electronic City,3 BHK,1644.0,3.0,92.59,3,5631
Kasavanhalli,3 BHK,1819.0,3.0,150.0,3,8246
1st Phase JP Nagar,2 BHK,1205.0,2.0,85.0,2,7053
Basavangudi,3 BHK,2350.0,3.0,300.0,3,12765
Uttarahalli,2 BHK,1050.0,2.0,42.0,2,4000
Raja Rajeshwari Nagar,2 BHK,1270.0,2.0,62.0,2,4881
9th Phase JP Nagar,3 BHK,1240.0,2.0,39.89,3,3216
Jigani,3 BHK,1221.0,3.0,70.0,3,5733
other,3 BHK,840.0,2.0,35.0,3,4166
Kanakpura Road,2 BHK,1339.0,2.0,63.0,2,4705
Thigalarapalya,4 BHK,4303.0,5.0,300.0,4,6971
Haralur Road,2 BHK,1315.0,2.0,85.0,2,6463
Raja Rajeshwari Nagar,2 BHK,1250.0,2.0,68.0,2,5440
Bisuvanahalli,2 BHK,850.0,2.0,27.0,2,3176
7th Phase JP Nagar,3 BHK,1600.0,2.0,110.0,3,6875
Electronic City,2 BHK,1250.0,2.0,62.0,2,4960
other,3 BHK,1312.0,2.0,56.5,3,4306
Kadugodi,4 BHK,3000.0,4.0,134.0,4,4466
Electronic City Phase II,2 BHK,1300.0,2.0,28.0,2,2153
Whitefield,3 BHK,1990.0,3.0,98.0,3,4924
other,2 BHK,760.0,2.0,38.5,2,5065
Ambedkar Nagar,4 Bedroom,5000.0,4.0,536.0,4,10720
Kadugodi,3 BHK,1614.0,2.0,59.0,3,3655
Judicial Layout,4 Bedroom,1500.0,3.0,162.0,4,10800
other,3 BHK,1182.0,2.0,60.1,3,5084
other,2 BHK,1053.0,2.0,46.29,2,4396
6th Phase JP Nagar,1 BHK,750.0,1.0,105.0,1,14000
Gubbalala,3 BHK,2000.0,3.0,125.0,3,6250
Electronic City Phase II,3 BHK,1940.0,4.0,116.0,3,5979
Hennur Road,3 BHK,1590.0,3.0,105.0,3,6603
other,2 BHK,1035.0,2.0,52.79,2,5100
Kothanur,2 BHK,1075.0,2.0,53.0,2,4930
TC Palaya,2 Bedroom,900.0,2.0,55.0,2,6111
Raja Rajeshwari Nagar,3 BHK,1260.0,2.0,62.0,3,4920
other,6 Bedroom,1200.0,5.0,260.0,6,21666
Thanisandra,3 BHK,1702.0,3.0,107.0,3,6286
Bisuvanahalli,3 BHK,1080.0,2.0,65.0,3,6018
Kasavanhalli,2 BHK,1189.0,2.0,45.0,2,3784
Tumkur Road,2 BHK,1035.0,2.0,39.0,2,3768
Hennur Road,2 BHK,1200.0,2.0,36.0,2,3000
Mahalakshmi Layout,8 Bedroom,4482.0,6.0,852.0,8,19009
other,3 Bedroom,1200.0,3.0,200.0,3,16666
Kanakpura Road,3 BHK,1300.0,3.0,69.0,3,5307
Yelahanka,2 BHK,1330.0,2.0,93.65,2,7041
Kaggalipura,3 BHK,1210.0,2.0,65.0,3,5371
Chikkalasandra,3 BHK,1350.0,2.0,54.2,3,4014
Kaval Byrasandra,3 BHK,1900.0,2.0,65.0,3,3421
Ambedkar Nagar,3 BHK,1935.0,4.0,125.0,3,6459
other,3 BHK,1410.0,3.0,90.0,3,6382
Narayanapura,2 BHK,1469.0,2.0,99.14,2,6748
Varthur,2 BHK,986.0,2.0,30.0,2,3042
other,3 BHK,1460.0,2.0,87.0,3,5958
other,3 BHK,1260.0,2.0,85.05,3,6750
Chandapura,3 BHK,1345.0,2.0,39.5,3,2936
Panathur,2 BHK,980.0,2.0,57.0,2,5816
Kudlu Gate,2 BHK,1164.0,2.0,75.0,2,6443
Subramanyapura,2 BHK,929.0,1.0,51.0,2,5489
Kundalahalli,2 BHK,1065.0,2.0,70.0,2,6572
Budigere,2 BHK,1172.0,2.0,56.725,2,4840
other,3 BHK,3000.0,3.0,265.0,3,8833
Electronic City,2 BHK,1020.0,2.0,30.27,2,2967
Gunjur,3 BHK,1362.0,3.0,62.63,3,4598
Yeshwanthpur,1 BHK,672.0,1.0,36.85,1,5483
Brookefield,3 BHK,1746.0,3.0,115.0,3,6586
Electronic City,2 BHK,1025.0,2.0,29.6,2,2887
Harlur,2 BHK,1532.0,2.0,65.0,2,4242
other,3 BHK,1200.0,2.0,120.0,3,10000
other,2 BHK,1250.0,2.0,65.0,2,5200
Thanisandra,2 BHK,1188.0,2.0,39.0,2,3282
Whitefield,4 Bedroom,3565.0,4.0,218.0,4,6115
Marathahalli,2 BHK,1170.0,2.0,85.0,2,7264
other,3 BHK,1800.0,3.0,150.0,3,8333
other,2 Bedroom,1000.0,2.0,35.0,2,3500
Cooke Town,3 Bedroom,2600.0,3.0,375.0,3,14423
Nagarbhavi,5 Bedroom,1200.0,4.0,240.0,5,20000
Kasturi Nagar,4 BHK,2507.0,4.0,155.0,4,6182
Babusapalaya,2 BHK,1061.0,2.0,34.0,2,3204
KR Puram,2 BHK,1290.0,2.0,69.0,2,5348
Sarjapur,2 BHK,1200.0,2.0,42.6,2,3550
Bharathi Nagar,2 BHK,1120.0,2.0,39.29,2,3508
Electronic City,1 BHK,418.0,1.0,12.0,1,2870
Sarjapur,3 BHK,1333.0,3.0,50.0,3,3750
Sarjapur  Road,3 BHK,1691.0,3.0,119.0,3,7037
Thanisandra,1 BHK,747.0,1.0,34.355,1,4599
Koramangala,3 BHK,1700.0,3.0,250.0,3,14705
Thanisandra,3 BHK,2030.0,4.0,125.0,3,6157
Balagere,2 BHK,1012.0,2.0,70.0,2,6916
Kudlu,3 BHK,1245.0,2.0,60.0,3,4819
7th Phase JP Nagar,3 BHK,1765.0,3.0,118.0,3,6685
Gottigere,2 BHK,1110.0,2.0,36.25,2,3265
other,2 BHK,967.0,2.0,47.8,2,4943
Sarjapur,2 BHK,1044.0,2.0,36.0,2,3448
Brookefield,3 BHK,1518.0,3.0,80.0,3,5270
KR Puram,3 BHK,1300.0,2.0,57.0,3,4384
other,3 BHK,1798.0,3.0,73.0,3,4060
KR Puram,2 BHK,866.28,2.0,36.38,2,4199
Whitefield,4 BHK,3410.0,5.0,253.0,4,7419
Singasandra,2 BHK,1179.0,2.0,56.0,2,4749
Rajaji Nagar,5 BHK,3600.0,5.0,510.0,5,14166
Uttarahalli,2 BHK,1125.0,2.0,48.0,2,4266
other,3 BHK,2098.0,3.0,210.0,3,10009
other,3 BHK,2400.0,3.0,270.0,3,11250
Haralur Road,3 BHK,1976.0,3.0,92.0,3,4655
5th Phase JP Nagar,2 BHK,1190.0,2.0,61.8,2,5193
other,3 BHK,1800.0,3.0,192.0,3,10666
Jakkur,3 BHK,1761.0,3.0,84.52,3,4799
Cooke Town,3 BHK,1700.0,3.0,120.0,3,7058
Ramamurthy Nagar,4 Bedroom,900.0,4.0,70.0,4,7777
Electronic City,2 BHK,1128.0,2.0,64.8,2,5744
other,3 BHK,2000.0,3.0,85.0,3,4250
Electronic City,3 BHK,1220.0,2.0,35.25,3,2889
other,1 BHK,509.0,1.0,20.0,1,3929
Ambedkar Nagar,3 BHK,1852.0,4.0,122.0,3,6587
Varthur,2 BHK,1045.0,2.0,40.65,2,3889
other,4 BHK,3535.0,4.0,215.0,4,6082
Marathahalli,2 BHK,1102.0,2.0,53.67,2,4870
other,4 BHK,1800.0,3.0,82.0,4,4555
Hebbal Kempapura,3 BHK,1436.0,2.0,75.0,3,5222
other,3 BHK,1397.0,2.0,52.0,3,3722
Kothanur,2 BHK,1140.0,2.0,56.0,2,4912
Balagere,1 BHK,645.0,1.0,55.0,1,8527
other,5 Bedroom,2350.0,5.0,325.0,5,13829
Nagarbhavi,3 Bedroom,1200.0,3.0,225.0,3,18750
other,3 BHK,2050.0,3.0,135.0,3,6585
Sarjapur  Road,2 BHK,1335.0,2.0,95.0,2,7116
other,2 Bedroom,2100.0,1.0,200.0,2,9523
Hebbal,3 BHK,1255.0,3.0,77.68,3,6189
Thanisandra,2 BHK,1140.0,2.0,59.0,2,5175
Hosakerehalli,3 BHK,2376.0,3.0,240.0,3,10101
Thanisandra,1 BHK,663.0,1.0,46.0,1,6938
other,2 BHK,1276.0,2.0,146.0,2,11442
Kanakpura Road,3 BHK,1450.0,3.0,60.9,3,4200
Kodichikkanahalli,3 BHK,1400.0,3.0,66.0,3,4714
other,2 BHK,1339.0,2.0,53.56,2,4000
Electronic City,3 BHK,1220.0,2.0,35.23,3,2887
Malleshwaram,3 BHK,2200.0,3.0,275.0,3,12500
other,2 BHK,1305.0,2.0,52.0,2,3984
Green Glen Layout,3 BHK,1740.0,3.0,80.0,3,4597
Anekal,2 BHK,680.0,1.0,21.0,2,3088
Bannerghatta Road,2 BHK,1154.0,2.0,47.0,2,4072
Munnekollal,2 BHK,1200.0,2.0,40.0,2,3333
Hennur,2 BHK,1155.0,2.0,54.5,2,4718
Sarjapur  Road,3 BHK,1525.0,2.0,63.0,3,4131
Giri Nagar,1 Bedroom,600.0,1.0,125.0,1,20833
Hebbal,4 BHK,2470.0,5.0,192.0,4,7773
Padmanabhanagar,2 BHK,1100.0,2.0,60.0,2,5454
Hormavu,2 BHK,1250.0,2.0,65.0,2,5200
Kothanur,2 BHK,1400.0,2.0,70.0,2,5000
other,3 BHK,1600.0,3.0,100.0,3,6250
Electronic City,2 BHK,1060.0,2.0,55.0,2,5188
Horamavu Banaswadi,2 BHK,1272.0,2.0,50.0,2,3930
Kanakpura Road,2 BHK,1296.0,2.0,105.0,2,8101
other,4 BHK,2100.0,3.0,280.0,4,13333
Karuna Nagar,3 BHK,1945.0,3.0,115.0,3,5912
Chandapura,3 BHK,1033.0,2.0,30.47,3,2949
other,3 BHK,2300.0,2.0,69.0,3,3000
other,2 BHK,1235.0,2.0,52.76,2,4272
other,3 BHK,2100.0,3.0,270.0,3,12857
other,2 BHK,850.0,2.0,42.5,2,5000
Subramanyapura,3 BHK,1880.0,3.0,110.0,3,5851
1st Block Jayanagar,4 BHK,2450.0,4.0,368.0,4,15020
Kenchenahalli,2 BHK,1060.0,2.0,52.0,2,4905
Hennur Road,3 BHK,2041.0,3.0,138.0,3,6761
Hulimavu,2 Bedroom,1050.0,2.0,125.0,2,11904
Vasanthapura,2 BHK,1037.0,2.0,36.28,2,3498
other,2 BHK,1150.0,2.0,39.0,2,3391
LB Shastri Nagar,2 BHK,1184.0,2.0,62.0,2,5236
Sarjapur  Road,3 BHK,1680.0,3.0,126.0,3,7500
Thanisandra,3 BHK,1411.0,3.0,100.0,3,7087
Hosa Road,5 Bedroom,900.0,5.0,55.0,5,6111
Kanakpura Road,2 BHK,900.0,2.0,41.0,2,4555
other,4 BHK,3500.0,4.0,320.0,4,9142
Thanisandra,3 BHK,1702.0,3.0,120.0,3,7050
Yelahanka,2 BHK,1185.0,2.0,45.0,2,3797
Kanakpura Road,3 BHK,1300.0,3.0,64.0,3,4923
other,3 BHK,1545.0,3.0,65.0,3,4207
6th Phase JP Nagar,3 BHK,1515.0,2.0,135.0,3,8910
Jigani,3 BHK,1300.0,3.0,55.0,3,4230
JP Nagar,2 BHK,1107.0,2.0,57.0,2,5149
Old Madras Road,3 BHK,1720.0,3.0,106.0,3,6162
other,3 BHK,1540.0,3.0,150.0,3,9740
Bannerghatta Road,2 BHK,1000.0,2.0,44.5,2,4450
Electronic City,1 BHK,710.0,1.0,40.42,1,5692
Thanisandra,2 BHK,1100.0,2.0,38.0,2,3454
other,4 Bedroom,3200.0,4.0,270.0,4,8437
other,2 Bedroom,900.0,2.0,160.0,2,17777
Bannerghatta Road,2 BHK,1250.0,2.0,48.0,2,3840
Kaggadasapura,5 Bedroom,2400.0,4.0,130.0,5,5416
Hegde Nagar,3 BHK,1584.01,3.0,104.0,3,6565
other,4 Bedroom,875.0,4.0,150.0,4,17142
Kammasandra,4 Bedroom,1200.0,6.0,60.0,4,5000
Kaggadasapura,3 BHK,1485.0,2.0,70.0,3,4713
Hennur Road,3 BHK,1100.0,2.0,78.0,3,7090
8th Phase JP Nagar,2 BHK,1245.0,2.0,59.0,2,4738
Hennur Road,4 Bedroom,1880.0,4.0,154.0,4,8191
Frazer Town,3 BHK,2214.0,3.0,350.0,3,15808
other,2 BHK,1050.0,2.0,45.0,2,4285
Vishveshwarya Layout,4 Bedroom,600.0,4.0,95.0,4,15833
Bannerghatta Road,3 BHK,1640.0,3.0,105.0,3,6402
other,2 BHK,1165.0,2.0,65.0,2,5579
Kengeri Satellite Town,2 BHK,1110.0,2.0,43.0,2,3873
other,7 BHK,1200.0,5.0,200.0,7,16666
other,4 BHK,4000.0,4.0,630.0,4,15750
Uttarahalli,4 Bedroom,4500.0,4.0,150.0,4,3333
other,4 Bedroom,600.0,4.0,65.0,4,10833
Haralur Road,2 BHK,1300.0,2.0,75.0,2,5769
other,2 Bedroom,1200.0,1.0,42.0,2,3500
Hennur Road,2 BHK,1232.0,2.0,69.61,2,5650
Jalahalli,3 BHK,1704.0,3.0,94.36,3,5537
7th Phase JP Nagar,3 BHK,1275.0,2.0,88.0,3,6901
Sarjapur  Road,3 BHK,1222.0,2.0,68.0,3,5564
Varthur,2 BHK,1155.0,2.0,39.0,2,3376
Electronic City,2 BHK,1000.0,2.0,29.95,2,2995
Horamavu Banaswadi,2 BHK,1200.0,2.0,45.0,2,3750
Yelahanka,3 BHK,1517.0,3.0,71.28,3,4698
Horamavu Agara,2 BHK,1166.0,2.0,37.29,2,3198
Electronic City,2 BHK,870.0,2.0,24.0,2,2758
5th Phase JP Nagar,5 BHK,4000.0,5.0,170.0,5,4250
Rajaji Nagar,4 Bedroom,1350.0,4.0,230.0,4,17037
Yelahanka,4 BHK,3019.0,6.0,190.0,4,6293
Whitefield,3 BHK,1345.0,3.0,58.0,3,4312
Dommasandra,3 Bedroom,900.0,3.0,75.0,3,8333
EPIP Zone,4 BHK,3800.0,5.0,170.0,4,4473
Chandapura,3 BHK,971.0,2.0,38.0,3,3913
Hennur,2 BHK,1255.0,2.0,52.32,2,4168
EPIP Zone,2 BHK,1810.0,2.0,65.0,2,3591
other,2 BHK,1300.0,2.0,58.0,2,4461
Thanisandra,2 BHK,1185.0,2.0,45.0,2,3797
Pattandur Agrahara,2 BHK,1350.0,2.0,68.0,2,5037
Budigere,3 BHK,1590.0,3.0,76.16,3,4789
other,7 BHK,3000.0,6.0,180.0,7,6000
Brookefield,4 Bedroom,1560.0,3.0,149.0,4,9551
Somasundara Palya,3 BHK,1650.0,3.0,126.0,3,7636
other,3 BHK,1645.0,3.0,87.0,3,5288
Hennur Road,3 BHK,1748.0,3.0,103.0,3,5892
Channasandra,3 Bedroom,1200.0,3.0,67.77,3,5647
Kengeri Satellite Town,2 BHK,1170.0,2.0,50.0,2,4273
Sarjapur  Road,4 BHK,3420.0,4.0,225.0,4,6578
Hulimavu,3 BHK,1320.0,2.0,65.0,3,4924
Bhoganhalli,3 BHK,1610.0,3.0,84.53,3,5250
Frazer Town,3 BHK,1870.0,3.0,180.0,3,9625
Sarjapur  Road,3 BHK,1510.0,2.0,110.0,3,7284
other,6 Bedroom,2700.0,7.0,110.0,6,4074
other,3 BHK,1305.0,2.0,45.68,3,3500
Mahadevpura,3 BHK,1500.0,2.0,70.0,3,4666
Indira Nagar,2 BHK,1149.0,2.0,130.0,2,11314
Thanisandra,4 BHK,1948.0,4.0,129.0,4,6622
Margondanahalli,3 Bedroom,1448.0,3.0,70.0,3,4834
Sarjapur  Road,2 BHK,1242.0,2.0,78.0,2,6280
Sonnenahalli,2 BHK,1268.0,2.0,58.0,2,4574
Ramamurthy Nagar,2 BHK,1101.0,2.0,50.0,2,4541
other,2 BHK,1255.0,2.0,65.0,2,5179
Gunjur,3 BHK,1362.0,3.0,62.63,3,4598
Sarjapur  Road,3 BHK,984.0,3.0,140.0,3,14227
Hosa Road,2 BHK,1089.0,2.0,32.67,2,3000
other,2 BHK,800.0,1.0,95.0,2,11875
Uttarahalli,3 BHK,1345.0,2.0,57.0,3,4237
Thanisandra,8 Bedroom,3600.0,9.0,125.0,8,3472
other,4 Bedroom,1650.0,2.0,140.0,4,8484
Kaggalipura,2 BHK,950.0,2.0,45.0,2,4736
HBR Layout,2 BHK,1068.0,2.0,43.0,2,4026
Yelahanka New Town,2 BHK,860.0,1.0,31.0,2,3604
Kengeri,8 Bedroom,3000.0,5.0,130.0,8,4333
Yelahanka,3 BHK,1756.0,3.0,66.73,3,3800
Whitefield,2 BHK,1205.0,2.0,41.0,2,3402
Hoodi,8 Bedroom,1120.0,8.0,155.0,8,13839
Banashankari,3 BHK,1800.0,3.0,125.0,3,6944
other,4 BHK,8321.0,5.0,2912.0,4,34995
Hormavu,3 BHK,1604.0,3.0,68.0,3,4239
Old Airport Road,4 BHK,2732.0,4.0,204.0,4,7467
Chandapura,2 BHK,1015.0,2.0,25.88,2,2549
Banashankari,3 BHK,1200.0,2.0,42.0,3,3500
Ananth Nagar,2 BHK,1010.0,2.0,38.0,2,3762
Nagarbhavi,2 Bedroom,1250.0,3.0,170.0,2,13600
Doddakallasandra,2 BHK,1010.0,2.0,41.0,2,4059
Yelahanka,2 BHK,1050.0,2.0,33.1,2,3152
Harlur,3 BHK,1710.0,3.0,85.0,3,4970
Mysore Road,3 BHK,1248.52,3.0,115.0,3,9210
Attibele,1 BHK,410.0,1.0,10.0,1,2439
Bannerghatta Road,3 BHK,1656.0,2.0,62.93,3,3800
other,3 BHK,1451.0,3.0,68.0,3,4686
Giri Nagar,6 BHK,1200.0,3.0,175.0,6,14583
JP Nagar,2 BHK,1250.0,2.0,93.0,2,7440
other,3 Bedroom,4395.0,3.0,240.0,3,5460
Hebbal,4 BHK,4600.0,6.0,650.0,4,14130
Sonnenahalli,3 BHK,1340.0,2.0,42.0,3,3134
other,2 BHK,1100.0,2.0,38.5,2,3500
Whitefield,4 BHK,6000.0,6.0,700.0,4,11666
Devarachikkanahalli,3 BHK,1700.0,3.0,71.0,3,4176
Yelahanka,6 Bedroom,1200.0,5.0,150.0,6,12500
KR Puram,2 BHK,1096.0,2.0,43.0,2,3923
Electronic City Phase II,2 BHK,1160.0,2.0,33.5,2,2887
other,3 BHK,1936.0,3.0,139.0,3,7179
Nehru Nagar,3 BHK,1269.0,2.0,63.0,3,4964
other,3 BHK,3500.0,3.0,630.0,3,18000
Panathur,3 BHK,1586.0,3.0,125.0,3,7881
Rajaji Nagar,2 BHK,1376.0,2.0,124.0,2,9011
Kanakpura Road,3 BHK,1452.0,3.0,56.5,3,3891
Hosa Road,3 BHK,1538.0,3.0,107.0,3,6957
Hosur Road,3 BHK,1427.0,2.0,130.0,3,9110
other,2 BHK,935.0,2.0,40.0,2,4278
other,2 BHK,1220.0,2.0,65.0,2,5327
Harlur,2 BHK,1197.0,2.0,77.0,2,6432
Varthur,2 BHK,674.0,1.0,19.9,2,2952
Kengeri,1 BHK,410.0,1.0,15.0,1,3658
Thanisandra,3 BHK,1573.0,3.0,90.0,3,5721
Dodda Nekkundi,2 BHK,1135.0,2.0,58.0,2,5110
other,3 BHK,1400.0,2.0,40.43,3,2887
other,1 BHK,1200.0,1.0,65.0,1,5416
other,3 Bedroom,1800.0,3.0,95.0,3,5277
other,3 BHK,2045.0,3.0,108.0,3,5281
Sonnenahalli,2 BHK,1268.0,2.0,56.0,2,4416
Kambipura,2 BHK,883.0,2.0,39.0,2,4416
Yelahanka,3 BHK,1600.0,3.0,100.0,3,6250
Thanisandra,2 BHK,1056.0,2.0,70.0,2,6628
Jigani,10 Bedroom,1200.0,11.0,105.0,10,8750
Whitefield,4 Bedroom,4500.0,4.0,330.0,4,7333
KR Puram,2 BHK,1204.0,2.0,58.0,2,4817
other,3 BHK,1410.0,2.0,54.0,3,3829
Hennur Road,2 BHK,1460.0,2.0,80.3,2,5500
Dodda Nekkundi,2 BHK,1100.0,2.0,41.18,2,3743
other,2 BHK,1310.0,2.0,88.0,2,6717
Sector 2 HSR Layout,3 BHK,1600.0,3.0,90.0,3,5625
other,2 BHK,1250.0,2.0,55.0,2,4400
Sarjapur  Road,3 BHK,1665.0,3.0,74.93,3,4500
Uttarahalli,3 BHK,1345.0,2.0,57.0,3,4237
other,5 Bedroom,2000.0,5.0,240.0,5,12000
Singasandra,2 BHK,1416.0,2.0,67.0,2,4731
HSR Layout,2 BHK,1145.0,2.0,46.0,2,4017
Munnekollal,10 Bedroom,7200.0,10.0,200.0,10,2777
Jigani,3 BHK,1231.0,3.0,58.0,3,4711
Yelahanka,3 BHK,1664.0,3.0,73.95,3,4444
TC Palaya,3 Bedroom,2000.0,3.0,76.0,3,3800
Rajiv Nagar,2 BHK,972.0,2.0,43.0,2,4423
other,4 BHK,4500.0,4.0,345.0,4,7666
Bellandur,2 BHK,1185.0,2.0,47.39,2,3999
Green Glen Layout,3 BHK,1750.0,3.0,81.0,3,4628
other,3 BHK,3155.0,4.0,316.0,3,10015
Balagere,2 BHK,1007.0,2.0,68.0,2,6752
Rajaji Nagar,3 BHK,1700.0,3.0,200.0,3,11764
Whitefield,2 BHK,1250.0,2.0,85.0,2,6800
other,2 BHK,1200.0,2.0,54.0,2,4500
Horamavu Agara,2 BHK,994.0,2.0,42.28,2,4253
Ramamurthy Nagar,2 BHK,1150.0,2.0,40.0,2,3478
Thanisandra,2 BHK,1220.0,2.0,46.0,2,3770
other,2 BHK,1441.0,2.0,115.0,2,7980
Old Madras Road,2 BHK,1157.0,2.0,47.32,2,4089
other,6 BHK,1300.0,6.0,150.0,6,11538
Thigalarapalya,2 BHK,1400.0,2.0,105.0,2,7500
Tindlu,3 BHK,1370.07,3.0,65.0,3,4744
Magadi Road,5 Bedroom,1200.0,4.0,130.0,5,10833
Judicial Layout,2 BHK,900.0,2.0,40.0,2,4444
other,1 BHK,418.0,1.0,17.0,1,4066
other,3 Bedroom,1200.0,3.0,80.0,3,6666
ISRO Layout,2 BHK,1200.0,2.0,43.0,2,3583
Talaghattapura,2 BHK,1062.0,2.0,42.48,2,4000
Kanakpura Road,2 BHK,1120.0,2.0,51.0,2,4553
Bannerghatta Road,2 BHK,1012.0,2.0,44.0,2,4347
other,3 Bedroom,1500.0,2.0,200.0,3,13333
ISRO Layout,4 Bedroom,950.0,4.0,180.0,4,18947
other,2 BHK,675.0,2.0,13.5,2,2000
Kanakpura Road,2 BHK,1430.0,2.0,78.0,2,5454
Channasandra,3 BHK,1195.0,2.0,56.0,3,4686
Old Madras Road,3 BHK,2640.0,5.0,150.0,3,5681
Bisuvanahalli,3 BHK,1075.0,2.0,42.0,3,3906
Hennur Road,3 BHK,1570.0,3.0,75.99,3,4840
Kammanahalli,6 Bedroom,782.0,4.0,85.0,6,10869
Sonnenahalli,2 BHK,1011.0,2.0,59.0,2,5835
7th Phase JP Nagar,3 BHK,1450.0,2.0,102.0,3,7034
Sarjapur  Road,2 BHK,1129.0,2.0,69.0,2,6111
Sarjapura - Attibele Road,2 BHK,1126.0,2.0,36.0,2,3197
Attibele,3 Bedroom,1200.0,3.0,110.0,3,9166
Sarjapur  Road,3 BHK,1180.0,2.0,45.0,3,3813
other,3 Bedroom,600.0,4.0,125.0,3,20833
Bannerghatta,4 Bedroom,940.0,4.0,45.0,4,4787
Hennur Road,3 BHK,1933.0,3.0,140.0,3,7242
TC Palaya,2 BHK,1082.0,2.0,64.0,2,5914
LB Shastri Nagar,8 Bedroom,2400.0,8.0,250.0,8,10416
other,1 BHK,610.0,1.0,28.0,1,4590
Raja Rajeshwari Nagar,4 Bedroom,1560.0,4.0,160.0,4,10256
Kathriguppe,3 BHK,1300.0,3.0,77.99,3,5999
Kudlu Gate,4 Bedroom,3850.0,6.0,205.0,4,5324
CV Raman Nagar,2 BHK,1125.0,2.0,46.0,2,4088
other,6 Bedroom,2400.0,7.0,107.0,6,4458
Kengeri Satellite Town,2 BHK,777.4,2.0,25.0,2,3215
Battarahalli,4 BHK,2500.0,3.0,65.0,4,2600
Old Airport Road,3 BHK,1798.0,3.0,150.0,3,8342
other,3 BHK,1864.0,3.0,120.0,3,6437
Whitefield,2 BHK,1115.0,2.0,51.91,2,4655
Kengeri,1 BHK,400.0,1.0,25.0,1,6250
Rajaji Nagar,2 BHK,1200.0,2.0,70.0,2,5833
Ananth Nagar,2 BHK,992.0,2.0,24.8,2,2500
other,2 BHK,950.0,2.0,57.0,2,6000
R.T. Nagar,4 Bedroom,1500.0,4.0,185.0,4,12333
Bellandur,2 BHK,1200.0,2.0,62.0,2,5166
GM Palaya,3 BHK,1735.0,3.0,70.0,3,4034
BTM 2nd Stage,3 BHK,1750.0,3.0,172.0,3,9828
HSR Layout,2 BHK,1009.0,2.0,56.0,2,5550
Jalahalli,3 BHK,1470.0,2.0,102.0,3,6938
Bannerghatta Road,3 BHK,1527.0,3.0,115.0,3,7531
other,2 BHK,985.0,2.0,45.0,2,4568
Chandapura,4 Bedroom,3300.0,4.0,135.0,4,4090
Whitefield,2 BHK,1277.0,2.0,77.0,2,6029
other,2 BHK,1100.0,2.0,62.0,2,5636
Whitefield,3 BHK,1820.0,3.0,140.0,3,7692
other,4 BHK,1530.0,3.0,87.0,4,5686
Banashankari,3 BHK,1886.0,3.0,205.0,3,10869
other,3 Bedroom,1200.0,2.0,190.0,3,15833
Jakkur,3 BHK,1880.0,4.0,110.0,3,5851
Hennur,2 BHK,1255.0,2.0,52.35,2,4171
Kanakpura Road,2 BHK,700.0,2.0,36.0,2,5142
Somasundara Palya,2 BHK,1140.0,2.0,46.0,2,4035
Electronic City Phase II,2 BHK,1025.0,2.0,29.6,2,2887
Whitefield,4 Bedroom,3453.0,4.0,247.0,4,7153
Uttarahalli,3 BHK,1490.0,2.0,59.6,3,4000
other,4 Bedroom,1200.0,4.0,165.0,4,13750
Sarjapur  Road,3 BHK,1850.0,3.0,140.0,3,7567
Kaikondrahalli,2 BHK,1250.0,2.0,78.0,2,6240
other,4 BHK,2400.0,3.0,108.0,4,4500
Kodihalli,4 BHK,3522.0,5.0,716.0,4,20329
Jalahalli,2 BHK,1244.0,2.0,88.0,2,7073
Kumaraswami Layout,4 Bedroom,623.0,4.0,75.0,4,12038
Kengeri Satellite Town,2 BHK,1025.0,2.0,28.57,2,2787
Sarjapur  Road,2 BHK,1034.0,2.0,38.0,2,3675
Kengeri Satellite Town,3 BHK,1191.0,2.0,33.34,3,2799
Kasavanhalli,3 BHK,1719.0,3.0,71.0,3,4130
7th Phase JP Nagar,2 BHK,1035.0,2.0,39.33,2,3800
2nd Stage Nagarbhavi,4 Bedroom,1500.0,3.0,230.0,4,15333
Hosa Road,3 BHK,1470.0,2.0,84.22,3,5729
Yelahanka,2 BHK,1140.0,2.0,50.66,2,4443
Harlur,3 BHK,1752.12,3.0,125.0,3,7134
other,6 Bedroom,600.0,6.0,80.0,6,13333
2nd Stage Nagarbhavi,4 Bedroom,600.0,3.0,84.0,4,14000
Old Madras Road,3 BHK,2990.0,3.0,170.0,3,5685
Bannerghatta Road,3 BHK,1660.0,3.0,85.0,3,5120
Electronic City,3 BHK,1275.0,3.0,63.43,3,4974
Bommasandra Industrial Area,3 BHK,1365.0,3.0,52.81,3,3868
other,3 Bedroom,1200.0,2.0,90.0,3,7500
Kanakpura Road,3 BHK,2546.0,3.0,170.0,3,6677
Banashankari Stage VI,3 BHK,1392.0,3.0,69.46,3,4989
Whitefield,2 BHK,1245.0,2.0,59.76,2,4800
Rayasandra,5 Bedroom,1200.0,3.0,65.0,5,5416
Yeshwanthpur,2 BHK,1162.0,2.0,64.08,2,5514
Singasandra,3 BHK,1476.0,2.0,73.0,3,4945
KR Puram,1 BHK,700.0,1.0,21.5,1,3071
other,6 Bedroom,3600.0,6.0,170.0,6,4722
Jigani,2 BHK,943.0,2.0,48.0,2,5090
Pai Layout,2 Bedroom,1150.0,2.0,170.0,2,14782
Sarjapur  Road,2 BHK,1113.0,2.0,44.5,2,3998
Varthur,3 BHK,1615.0,3.0,69.43,3,4299
other,2 BHK,1279.0,2.0,100.0,2,7818
other,2 BHK,1256.0,2.0,60.0,2,4777
Benson Town,3 BHK,1805.0,3.0,280.0,3,15512
Sompura,3 BHK,1350.0,3.0,65.0,3,4814
other,3 BHK,1525.0,3.0,68.63,3,4500
Rajaji Nagar,2 BHK,1357.0,2.0,123.0,2,9064
Hebbal,3 BHK,1645.0,3.0,117.0,3,7112
Ardendale,3 BHK,1777.26,3.0,92.0,3,5176
Bisuvanahalli,2 BHK,845.0,2.0,32.0,2,3786
other,2 BHK,900.0,2.0,25.0,2,2777
Varthur,2 BHK,1210.0,2.0,64.15,2,5301
Electronic City Phase II,2 BHK,829.0,2.0,22.8,2,2750
Sarjapur  Road,3 BHK,1879.0,3.0,155.0,3,8249
Hosur Road,3 BHK,1250.0,2.0,50.0,3,4000
other,4 Bedroom,1200.0,2.0,35.0,4,2916
Marathahalli,3 BHK,1108.0,2.0,60.0,3,5415
Kanakpura Road,3 BHK,1401.0,3.0,69.0,3,4925
Sarjapur  Road,3 BHK,1157.0,2.0,75.0,3,6482
other,2 BHK,1185.0,2.0,42.66,2,3600
Green Glen Layout,3 BHK,1751.0,3.0,115.0,3,6567
other,5 Bedroom,1550.0,3.0,250.0,5,16129
Bannerghatta Road,3 BHK,1880.0,3.0,96.5,3,5132
Whitefield,3 BHK,1322.5,3.0,40.985,3,3099
Channasandra,2 BHK,1175.0,2.0,38.0,2,3234
Jalahalli,2 BHK,1407.0,2.0,98.49,2,7000
Yelachenahalli,2 BHK,1000.0,2.0,50.0,2,5000
Electronic City,3 BHK,1518.0,3.0,75.0,3,4940
other,4 Bedroom,3600.0,4.0,300.0,4,8333
Tumkur Road,2 BHK,1027.0,2.0,67.0,2,6523
other,4 BHK,3730.0,6.0,430.0,4,11528
HSR Layout,2 BHK,1145.0,2.0,46.0,2,4017
Rajaji Nagar,2 BHK,1260.0,2.0,113.0,2,8968
Begur Road,3 BHK,1500.0,2.0,47.25,3,3150
Varthur,3 BHK,1737.0,2.0,57.32,3,3299
other,3 Bedroom,1600.0,4.0,120.0,3,7500
Hegde Nagar,3 BHK,1500.0,3.0,135.0,3,9000
other,2 Bedroom,1120.0,2.0,60.0,2,5357
Banashankari Stage V,3 Bedroom,1200.0,4.0,240.0,3,20000
Hennur Road,2 BHK,1030.0,2.0,46.5,2,4514
Gubbalala,6 Bedroom,2500.0,6.0,88.0,6,3520
Kammasandra,2 BHK,1057.0,2.0,35.0,2,3311
Old Airport Road,4 BHK,2774.0,4.0,208.0,4,7498
Thanisandra,2 BHK,1460.0,2.0,38.5,2,2636
Hormavu,2 BHK,1075.0,2.0,53.7,2,4995
Mysore Road,3 BHK,1080.0,2.0,60.0,3,5555
Kodihalli,4 BHK,3626.0,5.0,788.0,4,21731
other,6 BHK,5100.0,7.0,225.0,6,4411
Singasandra,2 BHK,1139.7,2.0,45.0,2,3948
Whitefield,2 BHK,1074.0,2.0,42.8,2,3985
Whitefield,2 Bedroom,1200.0,2.0,46.13,2,3844
Sarjapur  Road,4 BHK,2425.0,5.0,175.0,4,7216
Sarjapur  Road,2 BHK,1323.0,2.0,74.0,2,5593
Hennur Road,2 BHK,1232.0,2.0,69.61,2,5650
Kalena Agrahara,2 BHK,900.0,2.0,40.0,2,4444
Sarjapura - Attibele Road,1 BHK,740.0,1.0,23.65,1,3195
Hebbal,3 BHK,1920.0,3.0,134.0,3,6979
Bannerghatta Road,3 BHK,1550.0,3.0,96.0,3,6193
Gunjur,3 BHK,1588.0,2.0,90.0,3,5667
Rajaji Nagar,3 Bedroom,2300.0,4.0,240.0,3,10434
R.T. Nagar,9 Bedroom,3600.0,8.0,165.0,9,4583
other,2 BHK,1308.0,2.0,86.0,2,6574
Hennur,3 BHK,1340.0,2.0,54.27,3,4050
Kogilu,10 Bedroom,3280.0,9.0,450.0,10,13719
other,4 BHK,2170.0,3.0,265.0,4,12211
Thanisandra,3 BHK,1719.0,3.0,135.0,3,7853
Horamavu Agara,2 BHK,1107.83,2.0,41.51,2,3746
Whitefield,2 Bedroom,1200.0,2.0,45.84,2,3820
Electronic City,1 BHK,1090.0,2.0,31.48,1,2888
other,5 Bedroom,1500.0,5.0,210.0,5,14000
Jakkur,3 BHK,1660.0,3.0,104.0,3,6265
Nagavara,2 BHK,1247.0,2.0,58.56,2,4696
Thanisandra,3 BHK,1595.0,3.0,96.0,3,6018
Kaggadasapura,2 BHK,1175.0,2.0,50.0,2,4255
9th Phase JP Nagar,8 Bedroom,1200.0,8.0,135.0,8,11250
Old Airport Road,2 BHK,1184.0,2.0,78.0,2,6587
other,2 BHK,1100.0,2.0,55.0,2,5000
Kaval Byrasandra,2 Bedroom,935.0,2.0,78.0,2,8342
Yelahanka,2 BHK,1200.0,2.0,60.0,2,5000
Rajaji Nagar,5 Bedroom,2400.0,5.0,408.0,5,17000
other,2 BHK,1000.0,2.0,45.0,2,4500
Uttarahalli,2 BHK,1000.0,2.0,52.0,2,5200
Yelahanka,3 BHK,1705.0,3.0,85.0,3,4985
Tumkur Road,3 BHK,1060.0,2.0,58.3,3,5500
Chikka Tirupathi,4 Bedroom,2325.0,4.0,120.0,4,5161
other,3 BHK,1347.0,2.0,55.0,3,4083
other,8 Bedroom,1200.0,8.0,250.0,8,20833
Varthur,2 BHK,1402.0,2.0,95.0,2,6776
Whitefield,3 BHK,2280.0,4.0,125.0,3,5482
other,2 BHK,1225.0,2.0,71.05,2,5800
5th Phase JP Nagar,2 BHK,1080.0,2.0,53.0,2,4907
other,7 Bedroom,4000.0,7.0,90.0,7,2250
other,3 BHK,1919.0,3.0,300.0,3,15633
Begur Road,2 BHK,1100.0,2.0,42.6,2,3872
Sarjapur  Road,4 BHK,2500.0,4.0,225.0,4,9000
Bannerghatta Road,2 BHK,1200.0,2.0,78.0,2,6500
Begur Road,3 BHK,1565.0,2.0,57.91,3,3700
other,2 Bedroom,1200.0,3.0,78.0,2,6500
Kothanur,3 BHK,1581.0,3.0,76.0,3,4807
Whitefield,4 BHK,2882.0,5.0,200.0,4,6939
TC Palaya,5 Bedroom,1440.0,5.0,97.0,5,6736
other,3 BHK,2180.0,3.0,273.0,3,12522
Nagasandra,4 Bedroom,7000.0,8.0,450.0,4,6428
Abbigere,2 BHK,985.0,2.0,38.92,2,3951
Sarjapur,2 BHK,950.0,2.0,35.0,2,3684
other,2 BHK,1339.0,2.0,55.0,2,4107
other,3 BHK,1449.0,3.0,72.0,3,4968
other,4 Bedroom,1200.0,4.0,180.0,4,15000
other,2 BHK,1100.0,2.0,38.68,2,3516
Hoskote,2 BHK,1003.5,2.0,28.095,2,2799
Kaval Byrasandra,3 Bedroom,2700.0,3.0,200.0,3,7407
other,2 BHK,1100.0,2.0,55.0,2,5000
Electronic City,2 BHK,1140.0,2.0,32.92,2,2887
Sarjapur,3 Bedroom,3009.0,3.0,330.0,3,10967
other,8 BHK,3300.0,8.0,310.0,8,9393
Chandapura,2 BHK,800.0,1.0,30.0,2,3750
Chikkabanavar,5 Bedroom,2000.0,4.0,65.0,5,3250
other,4 Bedroom,1200.0,4.0,190.0,4,15833
other,2 BHK,1300.0,2.0,95.0,2,7307
Kanakpura Road,2 BHK,1220.0,2.0,60.0,2,4918
Sarjapur  Road,3 BHK,2289.0,3.0,178.0,3,7776
Whitefield,2 BHK,1170.0,2.0,56.0,2,4786
other,2 BHK,1350.0,2.0,89.5,2,6629
Kambipura,2 BHK,883.0,2.0,45.0,2,5096
Thanisandra,3 BHK,1698.0,3.0,125.0,3,7361
Electronic City Phase II,4 BHK,2187.5,4.0,105.0,4,4800
Ambalipura,2 BHK,1060.0,2.0,58.5,2,5518
Somasundara Palya,2 BHK,1185.0,2.0,75.0,2,6329
Padmanabhanagar,3 BHK,2051.0,3.0,180.0,3,8776
Vittasandra,2 BHK,1404.0,2.0,72.0,2,5128
Electronic City Phase II,2 BHK,1252.0,2.0,67.0,2,5351
Bannerghatta Road,3 BHK,1885.0,3.0,125.0,3,6631
Kaikondrahalli,6 BHK,3381.0,6.0,225.0,6,6654
Hormavu,2 BHK,980.0,2.0,42.0,2,4285
Rajaji Nagar,3 BHK,2500.0,3.0,340.0,3,13600
Banashankari,3 BHK,1420.0,3.0,66.0,3,4647
other,6 Bedroom,1166.0,6.0,135.0,6,11578
other,3 BHK,2312.0,3.0,156.0,3,6747
Kaggadasapura,4 BHK,3000.0,4.0,130.0,4,4333
other,2 BHK,1175.0,2.0,70.0,2,5957
other,2 BHK,1145.0,2.0,51.0,2,4454
Bhoganhalli,2 BHK,970.0,2.0,52.5,2,5412
Electronic City,2 BHK,1125.0,2.0,32.49,2,2888
Whitefield,3 BHK,1639.5,3.0,92.63,3,5649
9th Phase JP Nagar,4 BHK,5000.0,4.0,300.0,4,6000
other,3 BHK,2150.0,3.0,240.0,3,11162
Ramagondanahalli,3 BHK,1635.0,3.0,57.0,3,3486
Harlur,3 Bedroom,3425.0,3.0,320.0,3,9343
Hennur Road,4 Bedroom,2950.0,4.0,253.0,4,8576
Jalahalli,1 BHK,1200.0,2.0,66.0,1,5500
Kathriguppe,3 BHK,1335.0,2.0,73.43,3,5500
Hebbal,4 BHK,4450.0,6.0,449.0,4,10089
Yeshwanthpur,2 BHK,1166.0,2.0,64.08,2,5495
Rajaji Nagar,3 BHK,2390.0,3.0,372.0,3,15564
Shivaji Nagar,3 BHK,1460.0,2.0,90.0,3,6164
other,3 BHK,1690.0,3.0,50.0,3,2958
other,4 Bedroom,1386.0,3.0,93.0,4,6709
other,2 BHK,1090.0,2.0,32.0,2,2935
Thanisandra,3 BHK,1732.0,3.0,85.73,3,4949
Whitefield,2 BHK,1390.0,2.0,72.0,2,5179
Sector 2 HSR Layout,2 BHK,1095.0,2.0,85.41,2,7800
Kengeri,1 BHK,416.0,1.0,17.19,1,4132
other,3 BHK,2400.0,2.0,50.0,3,2083
Yelahanka,3 BHK,1350.0,2.0,85.0,3,6296
Bannerghatta Road,3 BHK,1453.0,2.0,73.0,3,5024
Thanisandra,2 BHK,1226.0,2.0,65.0,2,5301
Kasturi Nagar,4 Bedroom,784.0,2.0,165.0,4,21045
Hegde Nagar,3 BHK,1168.0,2.0,68.0,3,5821
Bannerghatta,3 BHK,1776.0,3.0,124.0,3,6981
Whitefield,3 BHK,1634.0,3.0,68.0,3,4161
Hormavu,2 Bedroom,860.0,2.0,56.0,2,6511
other,3 Bedroom,2800.0,3.0,400.0,3,14285
Nagasandra,3 BHK,1650.0,3.0,95.0,3,5757
Whitefield,2 BHK,1190.0,2.0,70.0,2,5882
JP Nagar,2 BHK,1133.0,2.0,33.99,2,3000
other,3 BHK,1559.0,3.0,62.0,3,3976
Horamavu Banaswadi,3 Bedroom,1448.0,4.0,118.0,3,8149
other,4 BHK,6652.0,6.0,510.0,4,7666
Kodichikkanahalli,2 BHK,976.0,2.0,50.0,2,5122
other,4 Bedroom,1350.0,4.0,240.0,4,17777
Yelahanka,3 BHK,1590.0,2.0,54.0,3,3396
Binny Pete,4 BHK,2940.0,6.0,201.0,4,6836
Yelahanka,3 BHK,1633.0,3.0,110.0,3,6736
Ramagondanahalli,2 BHK,1151.0,2.0,46.0,2,3996
Kereguddadahalli,3 BHK,1400.0,2.0,42.0,3,3000
Uttarahalli,2 BHK,1037.0,2.0,36.0,2,3471
Thanisandra,2 BHK,1188.0,2.0,36.0,2,3030
other,2 BHK,1060.0,2.0,52.0,2,4905
Thigalarapalya,3 BHK,2215.0,4.0,162.0,3,7313
Hennur,2 BHK,1255.0,2.0,54.5,2,4342
other,3 Bedroom,2500.0,2.0,300.0,3,12000
Hormavu,2 BHK,1075.0,2.0,53.7,2,4995
Harlur,4 BHK,2569.0,5.0,180.0,4,7006
ITPL,2 BHK,900.0,2.0,27.0,2,3000
Kudlu Gate,2 BHK,1183.0,2.0,79.54,2,6723
HSR Layout,2 BHK,1100.0,2.0,44.0,2,4000
GM Palaya,3 BHK,1643.0,2.0,70.0,3,4260
other,3 BHK,1500.0,2.0,120.0,3,8000
Jakkur,2 BHK,1291.0,2.0,75.0,2,5809
Hennur,3 Bedroom,1150.0,3.0,80.0,3,6956
Electronic City,2 BHK,1060.0,2.0,55.0,2,5188
7th Phase JP Nagar,3 BHK,1575.0,2.0,110.0,3,6984
other,5 Bedroom,1200.0,4.0,160.0,5,13333
other,4 Bedroom,1700.0,3.0,55.0,4,3235
Mahadevpura,2 BHK,1236.0,2.0,58.0,2,4692
Ramamurthy Nagar,3 BHK,1350.0,2.0,39.0,3,2888
Malleshwaram,3 BHK,2475.0,4.0,326.0,3,13171
Pai Layout,6 Bedroom,3800.0,6.0,175.0,6,4605
Haralur Road,4 BHK,4694.0,5.0,375.0,4,7988
Electronic City,2 BHK,1200.0,2.0,34.66,2,2888
Hulimavu,4 Bedroom,1200.0,4.0,75.0,4,6250
Harlur,2 BHK,1225.0,2.0,56.0,2,4571
Thanisandra,2 BHK,1056.0,2.0,75.0,2,7102
9th Phase JP Nagar,3 BHK,1240.0,2.0,42.85,3,3455
Banashankari,3 BHK,1200.0,3.0,74.52,3,6210
Kudlu Gate,3 BHK,1850.0,3.0,110.0,3,5945
other,4 BHK,3150.0,4.0,180.0,4,5714
other,1 Bedroom,1200.0,1.0,39.82,1,3318
other,4 Bedroom,1018.0,4.0,150.0,4,14734
other,5 Bedroom,3520.0,6.0,460.0,5,13068
Chandapura,3 BHK,1185.0,2.0,30.22,3,2550
Jigani,2 BHK,918.0,2.0,56.0,2,6100
other,3 BHK,1330.0,2.0,68.0,3,5112
Munnekollal,3 BHK,1540.0,2.0,70.0,3,4545
Lakshminarayana Pura,2 BHK,1200.0,2.0,80.0,2,6666
other,2 BHK,927.0,2.0,37.0,2,3991
Sarjapur,4 Bedroom,2585.5,4.0,115.0,4,4447
CV Raman Nagar,3 BHK,1400.0,2.0,78.0,3,5571
Hosur Road,2 BHK,1345.0,2.0,106.0,2,7881
1st Phase JP Nagar,4 Bedroom,1900.0,3.0,400.0,4,21052
8th Phase JP Nagar,2 BHK,1510.0,2.0,80.0,2,5298
Malleshwaram,2 BHK,302.0,2.0,25.0,2,8278
ITPL,3 Bedroom,1500.0,3.0,61.52,3,4101
Yelahanka,1 BHK,827.5,1.0,42.535,1,5140
Vidyaranyapura,3 BHK,1100.0,2.0,72.0,3,6545
other,3 Bedroom,1200.0,3.0,68.0,3,5666
Kanakpura Road,3 BHK,1100.0,3.0,53.0,3,4818
Hormavu,4 BHK,2282.0,4.0,115.0,4,5039
other,2 BHK,760.0,2.0,37.49,2,4932
Sarjapur,3 Bedroom,2172.0,3.0,100.0,3,4604
other,6 BHK,3800.0,6.0,390.0,6,10263
Kumaraswami Layout,4 Bedroom,600.0,2.0,72.0,4,12000
8th Phase JP Nagar,3 BHK,3300.0,3.0,165.0,3,5000
Thubarahalli,3 BHK,1242.0,3.0,51.0,3,4106
other,1 Bedroom,410.0,1.0,48.0,1,11707
Uttarahalli,3 BHK,1245.0,2.0,43.58,3,3500
Kadugodi,2 BHK,1088.0,2.0,44.0,2,4044
Sahakara Nagar,2 BHK,1200.0,2.0,46.0,2,3833
TC Palaya,3 Bedroom,1995.0,3.0,100.0,3,5012
Haralur Road,2 BHK,1309.0,2.0,82.0,2,6264
other,3 BHK,1717.0,3.0,200.0,3,11648
7th Phase JP Nagar,3 BHK,2100.0,3.0,200.0,3,9523
other,2 Bedroom,880.0,1.0,90.0,2,10227
Abbigere,6 Bedroom,1200.0,6.0,95.0,6,7916
other,3 BHK,1250.0,3.0,42.0,3,3360
Hebbal,3 BHK,3520.0,5.0,240.0,3,6818
5th Phase JP Nagar,2 BHK,1041.0,2.0,54.0,2,5187
Sarjapur  Road,3 BHK,1505.0,2.0,60.0,3,3986
Bannerghatta,3 BHK,1776.0,3.0,150.0,3,8445
Bellandur,3 BHK,1767.0,3.0,89.0,3,5036
other,2 BHK,1293.0,2.0,80.0,2,6187
Kengeri,2 BHK,1320.0,2.0,50.0,2,3787
other,2 BHK,1280.0,2.0,68.48,2,5350
Thanisandra,2 BHK,1245.0,2.0,83.05,2,6670
Haralur Road,2 BHK,1027.0,1.0,44.0,2,4284
Yelenahalli,2 BHK,1200.0,2.0,46.17,2,3847
Kanakpura Road,3 BHK,1100.0,3.0,58.0,3,5272
other,3 Bedroom,1065.0,3.0,72.0,3,6760
Kannamangala,3 BHK,1814.0,3.0,129.0,3,7111
Ramagondanahalli,3 BHK,1610.0,2.0,112.0,3,6956
Kumaraswami Layout,1 Bedroom,850.0,1.0,78.0,1,9176
Kasavanhalli,3 BHK,1600.0,3.0,79.0,3,4937
other,3 BHK,3200.0,4.0,140.0,3,4375
other,3 BHK,1250.0,2.0,77.13,3,6170
Haralur Road,2 BHK,1243.0,2.0,46.0,2,3700
Kanakpura Road,3 BHK,1452.0,3.0,60.98,3,4199
other,3 BHK,1475.0,3.0,75.0,3,5084
Electronic City,2 BHK,1353.0,2.0,75.0,2,5543
other,4 Bedroom,2100.0,3.0,210.0,4,10000
Electronics City Phase 1,3 BHK,1750.0,3.0,58.0,3,3314
Whitefield,4 Bedroom,4800.0,4.0,550.0,4,11458
Sahakara Nagar,2 BHK,1270.0,2.0,96.0,2,7559
Battarahalli,4 Bedroom,2700.0,3.0,75.0,4,2777
8th Phase JP Nagar,4 Bedroom,1200.0,4.0,270.0,4,22500
Thanisandra,4 BHK,2695.0,4.0,188.0,4,6975
other,3 BHK,1330.0,3.0,65.0,3,4887
Electronic City,2 BHK,660.0,1.0,18.0,2,2727
BTM 2nd Stage,2 BHK,1400.0,2.0,90.0,2,6428
Gubbalala,4 BHK,4000.0,5.0,195.0,4,4875
Kanakpura Road,3 BHK,1700.0,3.0,75.0,3,4411
Kasavanhalli,2 BHK,1585.0,2.0,90.0,2,5678
Kothanur,3 BHK,1385.0,3.0,67.0,3,4837
Electronic City,2 BHK,1128.0,2.0,67.0,2,5939
Whitefield,4 Bedroom,60.0,4.0,218.0,4,363333
other,18 Bedroom,1200.0,18.0,200.0,18,16666
HSR Layout,2 BHK,1203.0,2.0,60.0,2,4987
Mallasandra,2 BHK,905.0,2.0,35.0,2,3867
Hennur Road,3 BHK,1450.0,3.0,79.0,3,5448
Varthur,2 BHK,1105.0,2.0,38.0,2,3438
Channasandra,2 Bedroom,1200.0,2.0,46.13,2,3844
Jalahalli,3 BHK,1530.0,2.0,89.0,3,5816
Whitefield,2 BHK,1105.0,2.0,35.4,2,3203
Dasarahalli,2 BHK,1333.0,2.0,78.56,2,5893
other,2 BHK,1065.0,2.0,45.0,2,4225
Haralur Road,2 BHK,1243.0,2.0,46.0,2,3700
Old Madras Road,3 BHK,1530.0,3.0,76.34,3,4989
Margondanahalli,2 Bedroom,1200.0,2.0,65.0,2,5416
Whitefield,2 BHK,1270.0,2.0,94.0,2,7401
Bellandur,3 BHK,1830.0,3.0,89.0,3,4863
Chandapura,2 BHK,900.0,2.0,35.0,2,3888
Attibele,4 Bedroom,2400.0,2.0,250.0,4,10416
other,6 Bedroom,7000.0,6.0,560.0,6,8000
Bannerghatta Road,3 BHK,1450.0,2.0,78.0,3,5379
other,3 BHK,1525.0,3.0,65.0,3,4262
Mahalakshmi Layout,2 BHK,1080.0,2.0,67.0,2,6203
other,2 BHK,1220.0,2.0,72.0,2,5901
Begur Road,2 BHK,1200.0,2.0,44.4,2,3700
Bommanahalli,2 BHK,1355.0,2.0,73.17,2,5400
R.T. Nagar,3 Bedroom,2400.0,3.0,348.0,3,14500
other,2 BHK,1500.0,2.0,60.0,2,4000
Koramangala,2 BHK,1355.0,2.0,95.0,2,7011
Electronic City,2 BHK,1070.0,2.0,57.0,2,5327
CV Raman Nagar,3 BHK,1980.0,4.0,180.0,3,9090
Bommasandra,2 BHK,842.0,2.0,30.0,2,3562
Bannerghatta Road,1 BHK,700.0,1.0,31.2,1,4457
other,3 BHK,2500.0,4.0,65.0,3,2600
other,3 Bedroom,1884.0,3.0,95.0,3,5042
other,2 BHK,1100.0,2.0,75.0,2,6818
Raja Rajeshwari Nagar,2 BHK,1196.0,2.0,40.54,2,3389
Budigere,3 BHK,1820.0,3.0,85.0,3,4670
other,8 Bedroom,1200.0,8.0,145.0,8,12083
Thanisandra,2 BHK,1185.0,2.0,43.5,2,3670
Varthur,3 BHK,2145.0,3.0,170.0,3,7925
Whitefield,2 BHK,1220.0,2.0,89.0,2,7295
Whitefield,2 BHK,1116.0,2.0,51.91,2,4651
6th Phase JP Nagar,2 BHK,1192.0,2.0,75.0,2,6291
Marathahalli,3 BHK,1449.0,3.0,110.0,3,7591
Ardendale,3 Bedroom,2500.0,3.0,250.0,3,10000
Electronic City,3 BHK,1700.0,3.0,90.0,3,5294
Hebbal,3 BHK,1920.0,3.0,150.0,3,7812
other,2 Bedroom,1350.0,2.0,90.0,2,6666
other,4 Bedroom,4800.0,5.0,350.0,4,7291
other,3 BHK,1700.0,3.0,74.0,3,4352
Hoodi,2 BHK,1447.0,3.0,76.0,2,5252
Kanakpura Road,2 BHK,1329.0,2.0,107.0,2,8051
Sarjapur  Road,3 BHK,1314.0,2.0,70.44,3,5360
other,3 Bedroom,1200.0,3.0,110.0,3,9166
Basavangudi,6 Bedroom,1214.0,3.0,220.0,6,18121
Hosa Road,3 BHK,1513.0,3.0,103.0,3,6807
Vittasandra,2 BHK,1246.0,2.0,67.5,2,5417
other,1 BHK,705.0,1.0,65.0,1,9219
Dasarahalli,2 BHK,1220.0,2.0,62.0,2,5081
Yelahanka New Town,3 BHK,1800.0,2.0,45.0,3,2500
Kaval Byrasandra,3 BHK,1400.0,3.0,75.0,3,5357
Kasavanhalli,2 BHK,1245.0,2.0,80.0,2,6425
other,3 BHK,1495.0,2.0,63.0,3,4214
Nagavara,3 Bedroom,2000.0,2.0,275.0,3,13750
Gollarapalya Hosahalli,2 BHK,996.0,2.0,36.5,2,3664
Yelahanka New Town,2 BHK,1000.0,2.0,44.0,2,4400
other,2 BHK,1250.0,2.0,75.0,2,6000
Chandapura,2 BHK,975.0,2.0,24.86,2,2549
Thanisandra,3 BHK,1261.0,2.0,80.0,3,6344
Electronics City Phase 1,2 BHK,1080.0,2.0,46.0,2,4259
other,2 BHK,1230.0,2.0,50.0,2,4065
Kammasandra,2 BHK,982.0,2.0,25.53,2,2599
Bommasandra,3 BHK,1260.0,3.0,49.36,3,3917
HSR Layout,3 BHK,1590.0,2.0,135.0,3,8490
Horamavu Agara,2 BHK,1220.0,2.0,39.0,2,3196
Electronic City Phase II,3 BHK,1320.0,2.0,38.12,3,2887
Green Glen Layout,3 BHK,1750.0,3.0,105.0,3,6000
Whitefield,3 BHK,1436.0,2.0,62.15,3,4327
TC Palaya,3 Bedroom,1200.0,2.0,66.0,3,5500
Malleshwaram,2 BHK,1500.0,2.0,140.0,2,9333
Varthur,3 BHK,1560.0,3.0,90.0,3,5769
Whitefield,3 BHK,3758.0,3.0,300.0,3,7982
other,2 BHK,1275.0,2.0,59.0,2,4627
other,2 BHK,1040.0,2.0,36.4,2,3500
Bellandur,3 BHK,2400.0,3.0,115.0,3,4791
Chamrajpet,6 Bedroom,1500.0,9.0,230.0,6,15333
Kaggalipura,4 Bedroom,3150.0,5.0,212.0,4,6730
Karuna Nagar,3 BHK,1960.0,3.0,165.0,3,8418
JP Nagar,4 BHK,3200.0,5.0,309.0,4,9656
Yeshwanthpur,2 BHK,1195.0,2.0,100.0,2,8368
Yelahanka,2 BHK,1195.0,2.0,53.105,2,4443
Hosakerehalli,5 BHK,4500.0,5.0,145.0,5,3222
Ulsoor,3 Bedroom,450.0,3.0,70.0,3,15555
other,4 Bedroom,4304.0,4.0,699.0,4,16240
other,1 BHK,648.0,1.0,40.0,1,6172
Sarjapur  Road,3 BHK,2100.0,3.0,130.0,3,6190
other,4 BHK,4209.0,4.0,602.0,4,14302
Kudlu,2 BHK,1027.0,2.0,44.0,2,4284
Whitefield,4 Bedroom,1575.0,4.0,250.0,4,15873
Bommasandra,3 BHK,1365.0,3.0,52.82,3,3869
Yelahanka New Town,1 BHK,650.0,1.0,18.0,1,2769
Chandapura,3 Bedroom,1200.0,2.0,52.0,3,4333
Cooke Town,2 BHK,1100.0,2.0,90.0,2,8181
7th Phase JP Nagar,3 BHK,1075.0,2.0,57.0,3,5302
Banashankari,2 BHK,1100.0,2.0,63.0,2,5727
other,3 BHK,1320.0,2.0,58.0,3,4393
Kammasandra,3 BHK,1500.0,3.0,65.0,3,4333
other,4 Bedroom,1200.0,5.0,150.0,4,12500
other,4 Bedroom,1125.0,3.0,126.0,4,11200
Vittasandra,2 BHK,1246.0,2.0,67.5,2,5417
Bhoganhalli,3 BHK,1718.0,3.0,90.2,3,5250
Bisuvanahalli,3 BHK,1075.0,2.0,32.0,3,2976
other,3 BHK,1480.0,3.0,75.48,3,5100
other,1 BHK,500.0,1.0,13.0,1,2600
other,5 Bedroom,2400.0,4.0,325.0,5,13541
Kaikondrahalli,4 Bedroom,1200.0,4.0,125.0,4,10416
Kanakpura Road,2 BHK,1041.0,2.0,36.44,2,3500
other,2 BHK,1128.0,2.0,48.4,2,4290
Hosur Road,3 BHK,1685.0,3.0,90.0,3,5341
other,2 BHK,1180.0,2.0,65.0,2,5508
other,2 BHK,1200.0,2.0,60.0,2,5000
other,3 BHK,1250.0,3.0,39.5,3,3160
Kanakpura Road,2 BHK,900.0,2.0,41.0,2,4555
other,2 BHK,1000.0,2.0,45.0,2,4500
Yelahanka New Town,2 Bedroom,600.0,2.0,80.0,2,13333
2nd Stage Nagarbhavi,4 Bedroom,600.0,3.0,82.0,4,13666
Vittasandra,2 BHK,1404.0,2.0,70.0,2,4985
Ramagondanahalli,3 BHK,1610.0,2.0,115.0,3,7142
Varthur,2 BHK,1760.0,2.0,85.0,2,4829
other,2 BHK,975.0,2.0,55.0,2,5641
other,2 BHK,1190.0,2.0,42.0,2,3529
Kudlu,2 BHK,1110.0,2.0,46.0,2,4144
Rajaji Nagar,3 BHK,2415.0,3.0,399.0,3,16521
Gunjur,2 BHK,1235.0,2.0,44.5,2,3603
other,2 BHK,1275.0,2.0,55.0,2,4313
other,2 BHK,1115.0,2.0,115.0,2,10313
Harlur,2 BHK,1197.0,2.0,76.0,2,6349
Begur Road,1 BHK,644.0,1.0,32.0,1,4968
Bellandur,2 BHK,1251.0,2.0,47.0,2,3756
Neeladri Nagar,2 BHK,1105.0,2.0,30.0,2,2714
other,2 BHK,1080.0,2.0,38.0,2,3518
Kudlu,4 BHK,2100.0,3.0,114.0,4,5428
Kammanahalli,4 Bedroom,1600.0,4.0,200.0,4,12500
NRI Layout,3 BHK,1731.0,3.0,64.11,3,3703
Hormavu,2 BHK,1206.0,2.0,55.0,2,4560
Choodasandra,3 BHK,1560.0,3.0,78.0,3,5000
7th Phase JP Nagar,1 BHK,750.0,1.0,47.0,1,6266
Billekahalli,2 BHK,1035.0,2.0,90.0,2,8695
Ramamurthy Nagar,7 Bedroom,1200.0,6.0,150.0,7,12500
Sarjapur  Road,1 BHK,702.0,1.0,35.8,1,5099
Attibele,3 Bedroom,1700.0,3.0,65.0,3,3823
HRBR Layout,2 BHK,1145.0,2.0,68.5,2,5982
Ramamurthy Nagar,3 BHK,1208.0,2.0,43.5,3,3600
Brookefield,3 BHK,1589.0,3.0,74.0,3,4657
other,2 BHK,560.0,2.0,22.0,2,3928
Yelahanka New Town,3 BHK,1426.0,2.0,61.5,3,4312
Green Glen Layout,3 BHK,1885.0,3.0,135.0,3,7161
Old Madras Road,2 BHK,935.0,2.0,37.77,2,4039
Old Madras Road,2 BHK,1171.0,2.0,73.0,2,6233
Singasandra,2 BHK,1465.0,2.0,60.0,2,4095
Budigere,3 BHK,1820.0,3.0,95.0,3,5219
Raja Rajeshwari Nagar,3 BHK,1580.0,2.0,66.85,3,4231
Whitefield,3 BHK,1452.0,2.0,46.29,3,3188
Hennur Road,3 BHK,1445.0,2.0,89.56,3,6197
other,2 BHK,1170.0,2.0,39.0,2,3333
other,2 BHK,1070.0,2.0,50.0,2,4672
Kanakpura Road,3 BHK,1100.0,3.0,53.0,3,4818
Kothanur,2 BHK,1140.0,2.0,56.0,2,4912
Rachenahalli,3 BHK,2600.0,3.0,160.0,3,6153
other,3 BHK,1532.0,3.0,59.75,3,3900
other,2 BHK,1150.0,2.0,46.0,2,4000
Jakkur,2 BHK,1473.0,2.0,85.0,2,5770
Kathriguppe,3 BHK,1350.0,3.0,80.99,3,5999
Prithvi Layout,4 BHK,4040.0,4.0,500.0,4,12376
CV Raman Nagar,3 BHK,1980.0,4.0,166.0,3,8383
Hoodi,3 BHK,1660.0,3.0,128.0,3,7710
Kogilu,7 Bedroom,2456.0,7.0,85.0,7,3460
Chamrajpet,2 BHK,650.0,2.0,45.0,2,6923
Neeladri Nagar,5 Bedroom,4000.0,5.0,425.0,5,10625
Malleshwaram,3 BHK,2475.0,4.0,340.0,3,13737
other,3 BHK,1495.0,3.0,90.0,3,6020
R.T. Nagar,2 BHK,1080.0,2.0,48.0,2,4444
Electronic City,3 BHK,1521.0,2.0,57.5,3,3780
other,2 BHK,1053.0,2.0,51.0,2,4843
Binny Pete,3 BHK,2406.0,5.0,289.0,3,12011
Kanakpura Road,4 Bedroom,800.0,5.0,115.0,4,14375
7th Phase JP Nagar,2 BHK,1050.0,2.0,42.0,2,4000
Thanisandra,3 BHK,1430.0,2.0,51.48,3,3600
Begur,3 BHK,2400.0,3.0,12.0,3,500
other,3 BHK,1250.0,3.0,39.0,3,3120
other,4 Bedroom,1200.0,2.0,75.0,4,6250
Ulsoor,1 Bedroom,840.0,1.0,150.0,1,17857
Uttarahalli,2 BHK,1000.0,2.0,39.5,2,3950
Bannerghatta Road,3 BHK,1630.0,3.0,68.0,3,4171
Kammasandra,2 BHK,985.0,2.0,40.5,2,4111
BTM 2nd Stage,2 BHK,1200.0,2.0,70.0,2,5833
Ambalipura,3 BHK,1650.0,3.0,100.0,3,6060
Mico Layout,3 BHK,1670.0,3.0,59.0,3,3532
5th Block Hbr Layout,5 BHK,1200.0,5.0,205.0,5,17083
Sarjapur  Road,4 Bedroom,1240.0,4.0,110.0,4,8870
other,2 BHK,1000.0,2.0,60.0,2,6000
other,2 BHK,830.0,2.0,26.0,2,3132
Ramagondanahalli,3 Bedroom,1200.0,3.0,56.1,3,4675
other,5 Bedroom,9600.0,7.0,2736.0,5,28500
Begur Road,4 BHK,2500.0,6.0,122.5,4,4900
Haralur Road,2 BHK,1140.0,2.0,43.0,2,3771
Nagarbhavi,3 BHK,1635.0,3.0,85.0,3,5198
HSR Layout,2 BHK,1120.0,2.0,70.0,2,6250
other,6 Bedroom,2750.0,6.0,200.0,6,7272
Sector 7 HSR Layout,3 BHK,1760.0,3.0,185.0,3,10511
Harlur,3 BHK,2137.0,3.0,110.0,3,5147
Hennur Road,2 BHK,973.0,2.0,50.74,2,5214
Kundalahalli,3 BHK,1724.0,3.0,146.0,3,8468
Banashankari,2 BHK,1020.0,2.0,40.79,2,3999
other,4 Bedroom,1200.0,4.0,375.0,4,31250
Electronic City,1 BHK,750.0,1.0,35.0,1,4666
Kambipura,2 BHK,883.0,2.0,49.0,2,5549
other,2 Bedroom,660.0,2.0,76.0,2,11515
Ramagondanahalli,4 BHK,2787.0,5.0,220.0,4,7893
other,3 BHK,1470.0,2.0,60.0,3,4081
Electronics City Phase 1,2 BHK,891.0,2.0,24.95,2,2800
other,2 BHK,1275.0,2.0,76.0,2,5960
Kothanur,3 BHK,1787.0,3.0,120.0,3,6715
8th Phase JP Nagar,5 Bedroom,1730.0,5.0,75.0,5,4335
Sarjapur  Road,2 BHK,1320.0,2.0,110.0,2,8333
Yeshwanthpur,3 BHK,1692.0,3.0,108.0,3,6382
Bannerghatta Road,3 BHK,1400.0,3.0,69.0,3,4928
Sarjapur  Road,2 BHK,1115.0,2.0,50.0,2,4484
Whitefield,2 BHK,1370.0,2.0,56.0,2,4087
other,3 BHK,1602.0,3.0,56.0,3,3495
other,1 BHK,793.0,1.0,40.0,1,5044
GM Palaya,1 BHK,675.0,1.0,35.0,1,5185
Chikka Tirupathi,4 Bedroom,3250.0,4.0,136.0,4,4184
Whitefield,2 BHK,1050.0,2.0,43.0,2,4095
Jalahalli East,2 BHK,1035.0,2.0,42.5,2,4106
Sarjapur  Road,3 BHK,1200.0,2.0,44.0,3,3666
Sarjapur  Road,3 BHK,1800.0,3.0,128.0,3,7111
Green Glen Layout,3 BHK,1625.0,3.0,70.0,3,4307
Electronics City Phase 1,2 BHK,1032.0,2.0,31.99,2,3099
Whitefield,3 BHK,1496.0,2.0,71.81,3,4800
other,2 BHK,1100.0,2.0,38.0,2,3454
Kundalahalli,3 BHK,1500.0,3.0,50.0,3,3333
Electronic City Phase II,2 BHK,1125.0,2.0,32.63,2,2900
other,3 BHK,1555.0,3.0,73.0,3,4694
Amruthahalli,2 BHK,1025.0,2.0,42.0,2,4097
Hennur Road,3 BHK,1482.0,2.0,83.73,3,5649
other,3 BHK,1630.0,2.0,130.0,3,7975
Arekere,2 BHK,900.0,2.0,44.5,2,4944
Yelahanka,3 BHK,1355.0,2.0,75.0,3,5535
other,4 Bedroom,2360.0,4.0,601.0,4,25466
Akshaya Nagar,2 BHK,1200.0,2.0,45.0,2,3750
other,3 BHK,1563.0,3.0,63.0,3,4030
other,4 Bedroom,1344.0,4.0,84.0,4,6250
other,3 BHK,3000.0,3.0,400.0,3,13333
5th Phase JP Nagar,2 BHK,812.0,2.0,42.0,2,5172
Basaveshwara Nagar,2 BHK,1200.0,2.0,70.0,2,5833
Yelahanka,4 BHK,3175.0,4.0,156.0,4,4913
Marathahalli,2 BHK,1360.0,2.0,101.0,2,7426
Hegde Nagar,3 BHK,1835.0,3.0,88.0,3,4795
Rachenahalli,3 BHK,1756.0,3.0,110.0,3,6264
8th Phase JP Nagar,3 BHK,1296.0,2.0,51.83,3,3999
Haralur Road,2 BHK,1300.0,2.0,75.0,2,5769
Anandapura,2 BHK,1141.0,2.0,50.0,2,4382
Poorna Pragna Layout,3 BHK,1270.0,2.0,50.79,3,3999
Electronic City,2 BHK,1210.0,2.0,55.0,2,4545
Hormavu,2 BHK,1310.0,2.0,67.0,2,5114
other,5 Bedroom,1356.0,5.0,139.0,5,10250
Magadi Road,3 BHK,1187.0,2.0,55.195,3,4649
Gubbalala,3 BHK,1745.0,3.0,140.0,3,8022
KR Puram,5 Bedroom,1200.0,5.0,125.0,5,10416
Raja Rajeshwari Nagar,2 BHK,1050.0,2.0,37.0,2,3523
Banashankari,4 Bedroom,2400.0,4.0,500.0,4,20833
Electronic City Phase II,2 BHK,1140.0,2.0,34.0,2,2982
Varthur,2 BHK,1210.0,2.0,54.0,2,4462
Kasavanhalli,3 BHK,2035.0,3.0,142.0,3,6977
Sarjapur,1 BHK,633.0,1.0,17.09,1,2699
Amruthahalli,1 BHK,485.0,1.0,19.5,1,4020
Varthur,2 BHK,1112.0,2.0,40.0,2,3597
other,2 Bedroom,1200.0,2.0,82.0,2,6833
Sarjapur  Road,3 BHK,2275.0,4.0,187.0,3,8219
other,3 BHK,1419.0,3.0,85.0,3,5990
other,2 BHK,1195.0,2.0,42.4,2,3548
other,3 Bedroom,2020.0,3.0,270.0,3,13366
Kundalahalli,2 BHK,1047.0,2.0,91.0,2,8691
Thanisandra,3 BHK,1884.0,4.0,117.0,3,6210
Malleshpalya,3 Bedroom,1200.0,3.0,149.0,3,12416
Bannerghatta Road,2 BHK,905.0,2.0,58.0,2,6408
JP Nagar,3 BHK,1500.0,2.0,108.0,3,7200
other,1 BHK,850.0,1.0,50.0,1,5882
Bommasandra Industrial Area,2 BHK,1125.0,2.0,32.49,2,2888
Tindlu,2 BHK,1165.0,2.0,59.8,2,5133
other,2 BHK,933.0,2.0,55.0,2,5894
Hennur Road,3 BHK,1305.0,2.0,77.0,3,5900
Billekahalli,2 BHK,1125.0,2.0,65.0,2,5777
other,2 BHK,865.0,2.0,33.0,2,3815
Jigani,3 BHK,1245.0,3.0,66.0,3,5301
Kalyan nagar,2 BHK,8840.0,2.0,300.0,2,3393
other,3 BHK,1145.0,2.0,60.0,3,5240
AECS Layout,3 BHK,2000.0,3.0,90.0,3,4500
Hosa Road,1 BHK,625.0,1.0,43.68,1,6988
Kundalahalli,3 BHK,1724.0,3.0,128.0,3,7424
other,1 Bedroom,400.0,1.0,75.0,1,18750
Sarjapur  Road,3 BHK,1270.0,2.0,54.0,3,4251
other,3 BHK,1250.0,3.0,39.5,3,3160
Jakkur,3 BHK,1485.0,3.0,72.0,3,4848
other,2 BHK,823.0,2.0,80.0,2,9720
7th Phase JP Nagar,2 BHK,990.0,2.0,50.0,2,5050
TC Palaya,3 Bedroom,1200.0,2.0,66.0,3,5500
Kothanur,2 BHK,1185.0,2.0,59.0,2,4978
Kadubeesanahalli,2 BHK,1140.0,2.0,78.0,2,6842
Ambedkar Nagar,3 BHK,1950.0,4.0,120.0,3,6153
Yelahanka,3 BHK,1756.0,3.0,125.0,3,7118
Kumaraswami Layout,4 Bedroom,2600.0,4.0,300.0,4,11538
Chandapura,1 BHK,520.0,1.0,14.04,1,2700
other,2 BHK,1025.0,2.0,48.0,2,4682
Whitefield,2 BHK,1216.0,2.0,72.0,2,5921
Sarjapur  Road,2 BHK,1026.0,2.0,60.8,2,5925
Sarjapur  Road,4 BHK,3335.0,4.0,300.0,4,8995
Kodichikkanahalli,5 Bedroom,2700.0,4.0,125.0,5,4629
Begur,3 BHK,1411.0,2.0,60.0,3,4252
Kanakpura Road,3 BHK,1320.0,2.0,39.6,3,3000
Bannerghatta,2 BHK,1320.0,2.0,50.0,2,3787
OMBR Layout,4 Bedroom,615.0,4.0,110.0,4,17886
Kalena Agrahara,3 BHK,1804.0,3.0,155.0,3,8592
HSR Layout,2 BHK,1140.0,2.0,46.0,2,4035
Banjara Layout,4 Bedroom,1200.0,4.0,105.0,4,8750
Hebbal,4 BHK,2375.0,4.0,189.0,4,7957
other,3 Bedroom,1750.0,3.0,120.0,3,6857
other,9 Bedroom,960.0,5.0,130.0,9,13541
TC Palaya,2 BHK,1152.0,2.0,55.0,2,4774
Thanisandra,2 BHK,965.0,2.0,54.0,2,5595
Lingadheeranahalli,3 BHK,1684.0,3.0,120.0,3,7125
Hennur Road,3 BHK,1672.0,3.0,86.0,3,5143
Panathur,2 BHK,1075.0,2.0,60.0,2,5581
Sarjapur,3 Bedroom,1500.0,5.0,95.0,3,6333
other,3 BHK,1464.0,3.0,56.0,3,3825
Hebbal,2 BHK,1420.0,2.0,99.26,2,6990
other,2 BHK,750.0,2.0,40.0,2,5333
Yeshwanthpur,2 BHK,1169.0,2.0,64.08,2,5481
Devanahalli,3 BHK,1290.0,2.0,62.0,3,4806
Kengeri,2 BHK,1160.0,2.0,40.6,2,3500
Shampura,7 Bedroom,2800.0,5.0,95.0,7,3392
Mysore Road,3 BHK,1500.0,3.0,70.0,3,4666
Domlur,3 BHK,1850.0,3.0,180.0,3,9729
other,3 BHK,1300.0,3.0,46.0,3,3538
Electronic City Phase II,3 BHK,1310.0,2.0,37.84,3,2888
other,3 BHK,2750.0,5.0,550.0,3,20000
other,3 BHK,2150.0,3.0,170.0,3,7906
5th Block Hbr Layout,6 BHK,5100.0,5.0,300.0,6,5882
Kanakpura Road,3 BHK,1300.0,2.0,69.0,3,5307
Kadugodi,3 Bedroom,1875.0,2.0,110.0,3,5866
Hulimavu,3 BHK,1260.0,2.0,82.0,3,6507
Kengeri,5 Bedroom,2500.0,4.0,100.0,5,4000
Begur Road,2 BHK,1241.0,2.0,64.0,2,5157
Sarjapur  Road,2 BHK,1025.0,2.0,45.5,2,4439
other,2 Bedroom,1500.0,1.0,218.0,2,14533
Electronic City,2 BHK,1070.0,2.0,60.0,2,5607
Begur,2 BHK,815.0,2.0,40.0,2,4907
Kanakpura Road,3 BHK,2000.0,3.0,105.0,3,5250
Malleshwaram,2 BHK,1400.0,2.0,190.0,2,13571
Varthur,4 Bedroom,1740.0,4.0,175.0,4,10057
ITPL,4 BHK,5667.5,5.0,300.0,4,5293
Sarjapur  Road,2 BHK,1216.0,2.0,70.0,2,5756
other,3 BHK,2200.0,2.0,190.0,3,8636
Hegde Nagar,4 Bedroom,3500.0,5.0,450.0,4,12857
Shampura,6 Bedroom,970.0,6.0,105.0,6,10824
Vidyaranyapura,7 Bedroom,3150.0,7.0,375.0,7,11904
Kanakpura Road,2 Bedroom,1200.0,2.0,62.0,2,5166
Marathahalli,2 BHK,1026.0,2.0,50.18,2,4890
Hulimavu,1 BHK,450.0,1.0,20.0,1,4444
KR Puram,2 BHK,1205.0,2.0,45.79,2,3800
other,2 BHK,1368.0,2.0,87.0,2,6359
Anekal,1 BHK,420.0,1.0,12.5,1,2976
Doddakallasandra,2 BHK,1270.0,2.0,42.0,2,3307
Anekal,2 BHK,614.0,1.0,28.0,2,4560
Margondanahalli,2 Bedroom,1050.0,2.0,58.0,2,5523
Yelahanka,2 BHK,1408.0,2.0,58.0,2,4119
Subramanyapura,3 BHK,1223.0,2.0,42.81,3,3500
other,3 BHK,1507.0,3.0,80.0,3,5308
other,2 BHK,1160.0,2.0,57.0,2,4913
other,3 Bedroom,1200.0,3.0,70.0,3,5833
Poorna Pragna Layout,2 BHK,960.0,2.0,50.0,2,5208
NGR Layout,2 BHK,1021.0,2.0,46.0,2,4505
Hegde Nagar,3 BHK,2162.03,4.0,129.0,3,5966
Channasandra,2 BHK,1065.0,2.0,36.0,2,3380
Attibele,1 BHK,400.0,1.0,10.25,1,2562
Bannerghatta Road,3 BHK,1465.0,2.0,75.0,3,5119
Kanakpura Road,2 BHK,900.0,2.0,41.0,2,4555
Marathahalli,3 BHK,1800.0,3.0,95.0,3,5277
other,2 BHK,1175.0,2.0,83.0,2,7063
other,3 BHK,6729.0,4.0,900.0,3,13374
Vijayanagar,4 Bedroom,1500.0,4.0,360.0,4,24000
Sanjay nagar,2 BHK,1050.0,2.0,59.0,2,5619
Electronic City Phase II,3 BHK,1400.0,2.0,40.43,3,2887
Indira Nagar,3 BHK,1650.0,3.0,150.0,3,9090
KR Puram,3 BHK,1827.0,3.0,97.0,3,5309
Thanisandra,3 BHK,1930.0,4.0,122.0,3,6321
Mysore Road,2 BHK,883.0,2.0,37.0,2,4190
Kengeri,1 BHK,550.0,1.0,16.75,1,3045
Electronic City,3 BHK,1400.0,2.0,40.44,3,2888
other,4 Bedroom,1350.0,4.0,310.0,4,22962
Kanakapura,3 BHK,1290.0,2.0,38.69,3,2999
Raja Rajeshwari Nagar,3 BHK,1021.0,3.0,59.0,3,5778
Whitefield,2 BHK,1158.0,2.0,55.0,2,4749
other,5 Bedroom,1200.0,4.0,99.0,5,8250
Nagarbhavi,3 BHK,1400.0,2.0,135.0,3,9642
Doddaballapur,4 Bedroom,2400.0,3.0,200.0,4,8333
Munnekollal,3 BHK,1200.0,2.0,57.9,3,4825
JP Nagar,3 BHK,1520.0,3.0,125.0,3,8223
Thanisandra,1 BHK,777.0,1.0,38.46,1,4949
EPIP Zone,3 BHK,2710.0,3.0,177.0,3,6531
Kenchenahalli,2 BHK,1150.0,2.0,57.0,2,4956
Yelahanka,2 BHK,1101.0,2.0,37.0,2,3360
Mahalakshmi Layout,3 BHK,1200.0,2.0,110.0,3,9166
Hebbal,2 BHK,1252.0,2.0,92.0,2,7348
Marathahalli,3 BHK,1650.0,3.0,85.0,3,5151
Marathahalli,2 BHK,1270.0,2.0,80.0,2,6299
Tumkur Road,2 BHK,992.0,2.0,70.0,2,7056
BEML Layout,3 BHK,2000.0,3.0,85.0,3,4250
other,2 BHK,893.0,2.0,34.5,2,3863
Bisuvanahalli,3 BHK,1075.0,2.0,41.49,3,3859
Kumaraswami Layout,7 Bedroom,3200.0,7.0,150.0,7,4687
Electronic City,2 BHK,1190.0,2.0,42.0,2,3529
other,3 Bedroom,800.0,2.0,60.0,3,7500
Sarjapur  Road,3 BHK,1814.0,3.0,63.0,3,3472
other,3 BHK,1400.0,3.0,98.0,3,7000
Bommasandra,5 Bedroom,1400.0,6.0,252.0,5,18000
Murugeshpalya,3 BHK,1344.0,2.0,56.0,3,4166
Whitefield,4 Bedroom,5400.0,5.0,475.0,4,8796
Kanakpura Road,3 BHK,1843.0,3.0,95.0,3,5154
other,2 BHK,1138.0,2.0,32.0,2,2811
other,3 BHK,1495.0,2.0,70.0,3,4682
Electronic City,2 BHK,880.0,2.0,19.0,2,2159
Whitefield,3 BHK,1396.0,3.0,74.0,3,5300
Garudachar Palya,3 BHK,1610.0,2.0,119.0,3,7391
Tindlu,2 BHK,1050.0,2.0,55.0,2,5238
other,3 BHK,1410.0,2.0,43.71,3,3100
Ramamurthy Nagar,2 BHK,1050.0,2.0,37.8,2,3599
Kumaraswami Layout,3 BHK,1310.0,2.0,85.0,3,6488
Yelahanka,2 BHK,1104.0,2.0,58.0,2,5253
Hennur Road,2 BHK,1165.0,2.0,67.0,2,5751
Kudlu,2 BHK,1027.0,2.0,44.0,2,4284
Hormavu,2 BHK,1310.0,2.0,60.0,2,4580
other,2 BHK,1052.0,2.0,52.0,2,4942
Singasandra,3 BHK,1464.0,3.0,56.0,3,3825
other,2 BHK,1000.0,2.0,45.0,2,4500
other,3 BHK,2240.0,3.0,300.0,3,13392
Electronic City,2 BHK,1150.0,2.0,38.08,2,3311
Thanisandra,3 BHK,2019.0,3.0,70.67,3,3500
Varthur,2 BHK,986.0,2.0,31.0,2,3144
Banashankari Stage III,1 Bedroom,1350.0,1.0,145.0,1,10740
9th Phase JP Nagar,8 BHK,800.0,8.0,140.0,8,17500
Akshaya Nagar,3 BHK,1893.0,4.0,95.0,3,5018
Gollarapalya Hosahalli,3 BHK,1318.0,3.0,56.0,3,4248
Yelahanka,3 BHK,1325.0,2.0,58.0,3,4377
Kereguddadahalli,2 BHK,800.0,2.0,33.0,2,4125
other,3 BHK,1300.0,3.0,42.25,3,3250
other,3 BHK,1300.0,2.0,60.0,3,4615
Jalahalli East,2 BHK,1010.0,2.0,52.0,2,5148
Banashankari,2 BHK,1260.0,2.0,75.0,2,5952
Ulsoor,4 Bedroom,2500.0,4.0,170.0,4,6800
other,3 Bedroom,1520.0,3.0,165.0,3,10855
Bellandur,3 BHK,1665.0,3.0,85.0,3,5105
Subramanyapura,3 BHK,1800.0,3.0,80.0,3,4444
Harlur,2 BHK,1197.0,2.0,75.0,2,6265
other,2 BHK,1050.0,2.0,45.0,2,4285
other,5 Bedroom,910.0,4.0,194.0,5,21318
Thanisandra,3 BHK,1241.0,2.0,68.0,3,5479
other,3 BHK,1567.0,2.0,62.0,3,3956
Kasavanhalli,3 BHK,1747.0,3.0,110.0,3,6296
Ramagondanahalli,3 BHK,1475.0,2.0,55.0,3,3728
other,4 Bedroom,600.0,4.0,70.0,4,11666
other,3 Bedroom,1524.0,4.0,400.0,3,26246
Hennur Road,2 BHK,980.0,2.0,51.0,2,5204
Dasanapura,2 BHK,966.0,2.0,58.0,2,6004
other,4 BHK,2360.0,3.0,185.0,4,7838
Sarjapur  Road,2 BHK,1112.0,2.0,62.0,2,5575
Electronic City,2 BHK,550.0,1.0,16.0,2,2909
other,3 BHK,1760.0,3.0,90.0,3,5113
Hennur,3 BHK,1830.0,2.0,113.0,3,6174
Yelahanka,2 BHK,1175.0,2.0,67.77,2,5767
Whitefield,2 BHK,1175.0,2.0,41.0,2,3489
Horamavu Banaswadi,2 BHK,1025.0,2.0,45.0,2,4390
Hennur Road,2 BHK,1232.0,2.0,80.0,2,6493
5th Phase JP Nagar,2 BHK,1010.0,2.0,57.0,2,5643
Sarjapur  Road,2 BHK,1273.0,2.0,60.0,2,4713
Battarahalli,2 BHK,1590.0,2.0,102.0,2,6415
other,2 BHK,1100.0,2.0,58.5,2,5318
Electronic City Phase II,3 BHK,1252.0,2.0,65.0,3,5191
1st Phase JP Nagar,4 BHK,2615.0,5.0,222.0,4,8489
other,5 Bedroom,1200.0,6.0,140.0,5,11666
Yelahanka,2 BHK,1322.0,2.0,77.9,2,5892
7th Phase JP Nagar,1 Bedroom,1000.0,1.0,60.0,1,6000
Harlur,3 BHK,1755.0,3.0,117.0,3,6666
Electronic City Phase II,2 BHK,1135.0,2.0,46.0,2,4052
Rajaji Nagar,8 Bedroom,1200.0,10.0,180.0,8,15000
Electronic City,4 Bedroom,1200.0,4.0,125.0,4,10416
Nagavarapalya,2 BHK,1260.0,2.0,85.5,2,6785
Hoodi,5 Bedroom,3250.0,5.0,395.0,5,12153
Ambedkar Nagar,4 Bedroom,3500.0,4.0,550.0,4,15714
other,5 Bedroom,1200.0,6.0,110.0,5,9166
Kanakpura Road,3 BHK,1100.0,2.0,53.0,3,4818
Electronic City Phase II,3 BHK,1305.0,3.0,65.0,3,4980
other,2 BHK,1236.0,2.0,50.0,2,4045
Uttarahalli,2 BHK,1125.0,1.0,45.0,2,4000
KR Puram,2 BHK,1075.0,2.0,32.25,2,3000
BEML Layout,3 BHK,2000.0,3.0,85.0,3,4250
other,8 Bedroom,772.0,5.0,60.0,8,7772
Thanisandra,6 Bedroom,2999.97,6.0,110.0,6,3666
Kadugodi,2 BHK,1150.0,2.0,42.0,2,3652
other,4 Bedroom,1440.0,4.0,240.0,4,16666
Hennur Road,2 BHK,1085.0,2.0,43.4,2,4000
Hebbal Kempapura,3 BHK,1785.0,3.0,165.0,3,9243
Malleshwaram,3 BHK,2475.0,4.0,300.0,3,12121
Thanisandra,3 BHK,1930.0,4.0,120.0,3,6217
other,2 BHK,600.0,4.0,70.0,2,11666
other,2 Bedroom,800.0,1.0,45.0,2,5625
Kumaraswami Layout,4 BHK,2450.0,3.0,110.0,4,4489
other,4 Bedroom,1200.0,4.0,120.0,4,10000
Bannerghatta Road,3 BHK,1920.0,3.0,122.0,3,6354
Kothannur,2 BHK,1085.0,2.0,34.2,2,3152
Thanisandra,1 BHK,760.0,1.0,50.4,1,6631
Sahakara Nagar,2 BHK,1201.0,2.0,80.0,2,6661
Electronic City,3 BHK,1615.0,3.0,97.0,3,6006
HSR Layout,2 BHK,1289.0,2.0,70.0,2,5430
Budigere,3 BHK,1991.0,4.0,100.0,3,5022
Kasavanhalli,2 BHK,1377.0,2.0,80.0,2,5809
Cunningham Road,3 BHK,2880.0,3.0,560.0,3,19444
other,3 Bedroom,1892.0,3.0,200.0,3,10570
Shivaji Nagar,3 BHK,2176.0,3.0,348.0,3,15992
Thanisandra,1 BHK,663.0,1.0,32.82,1,4950
Electronic City,2 BHK,1165.0,2.0,33.65,2,2888
Talaghattapura,3 BHK,3554.0,4.0,340.0,3,9566
Thanisandra,2 BHK,1260.0,2.0,45.5,2,3611
other,2 BHK,1095.0,2.0,39.0,2,3561
other,2 BHK,1050.0,2.0,65.0,2,6190
Seegehalli,2 BHK,1176.0,2.0,50.0,2,4251
Thanisandra,3 BHK,1801.0,3.0,115.0,3,6385
Whitefield,2 BHK,1315.0,2.0,55.0,2,4182
Parappana Agrahara,2 BHK,1194.0,2.0,46.0,2,3852
NRI Layout,2 Bedroom,2400.0,2.0,125.0,2,5208
Uttarahalli,3 BHK,1308.0,2.0,53.0,3,4051
Sarjapur  Road,4 Bedroom,4250.0,4.0,610.0,4,14352
Thanisandra,3 BHK,1806.0,6.0,116.0,3,6423
Thanisandra,4 Bedroom,3671.0,4.0,220.0,4,5992
Hoodi,3 BHK,2144.0,3.0,140.0,3,6529
Hoskote,2 BHK,1065.0,2.0,33.75,2,3169
Balagere,2 BHK,1205.0,2.0,67.0,2,5560
Uttarahalli,2 BHK,1160.0,2.0,40.58,2,3498
other,4 Bedroom,750.0,4.0,90.0,4,12000
Parappana Agrahara,2 BHK,1194.0,2.0,47.0,2,3936
other,3 Bedroom,1200.0,3.0,56.53,3,4710
2nd Phase Judicial Layout,3 BHK,1681.0,3.0,69.0,3,4104
Bommasandra Industrial Area,2 BHK,1130.0,2.0,32.64,2,2888
Iblur Village,4 BHK,3633.0,5.0,335.0,4,9221
Kanakpura Road,3 BHK,1843.0,3.0,90.0,3,4883
Bommasandra Industrial Area,2 BHK,1065.0,2.0,30.75,2,2887
Thubarahalli,3 BHK,1885.0,3.0,85.52,3,4536
R.T. Nagar,2 BHK,1500.0,2.0,45.0,2,3000
Panathur,2 BHK,1210.0,2.0,69.0,2,5702
TC Palaya,2 BHK,1126.0,2.0,39.4,2,3499
Rayasandra,3 BHK,1357.0,2.0,59.0,3,4347
Whitefield,3 BHK,1451.0,2.0,46.4,3,3197
Kodigehaali,4 Bedroom,1600.0,3.0,185.0,4,11562
Hennur Road,2 BHK,950.0,2.0,55.0,2,5789
Bellandur,2 BHK,960.0,2.0,48.0,2,5000
other,3 BHK,1625.0,3.0,92.0,3,5661
Kaggalipura,3 Bedroom,2200.0,4.0,150.0,3,6818
Ambalipura,3 BHK,1607.0,2.0,112.0,3,6969
Kanakpura Road,2 BHK,900.0,2.0,46.29,2,5143
other,2 BHK,1080.0,2.0,38.0,2,3518
Cooke Town,3 BHK,2300.0,3.0,250.0,3,10869
Hosa Road,4 BHK,1708.0,3.0,64.05,4,3750
Yeshwanthpur,6 Bedroom,2500.0,5.0,185.0,6,7400
Bellandur,3 BHK,1690.0,2.0,116.0,3,6863
other,5 Bedroom,2100.0,3.0,145.0,5,6904
other,2 BHK,1269.0,2.0,97.0,2,7643
Haralur Road,2 BHK,1243.0,2.0,46.0,2,3700
Haralur Road,3 BHK,1890.0,3.0,84.0,3,4444
Electronic City Phase II,2 BHK,1165.0,2.0,33.64,2,2887
BTM Layout,2 BHK,1196.0,2.0,56.81,2,4750
Parappana Agrahara,2 BHK,1194.0,2.0,47.0,2,3936
Old Madras Road,3 BHK,1630.0,3.0,110.0,3,6748
Whitefield,2 BHK,1063.0,2.0,42.52,2,4000
other,2 BHK,1025.0,2.0,39.98,2,3900
Chikkabanavar,2 BHK,1067.0,2.0,38.0,2,3561
Kanakpura Road,2 BHK,1299.0,2.0,89.63,2,6899
Panathur,2 BHK,1343.0,2.0,76.0,2,5658
Electronics City Phase 1,3 BHK,1330.0,2.0,44.0,3,3308
Mallasandra,3 BHK,1524.0,2.0,72.0,3,4724
Lingadheeranahalli,4 BHK,2236.0,4.0,153.0,4,6842
Domlur,3 BHK,2955.0,3.0,235.0,3,7952
other,3 BHK,1410.0,2.0,45.12,3,3200
Kothanur,3 BHK,1581.0,3.0,76.0,3,4807
Balagere,2 BHK,1205.0,2.0,79.0,2,6556
other,3 BHK,1270.0,2.0,95.25,3,7500
other,3 BHK,1353.0,2.0,110.0,3,8130
other,2 Bedroom,1200.0,2.0,68.0,2,5666
other,3 BHK,1250.0,2.0,70.0,3,5600
other,3 BHK,1950.2,3.0,261.0,3,13383
Sarjapur  Road,3 BHK,1157.0,2.0,69.0,3,5963
Kanakapura,1 BHK,825.0,1.0,35.475,1,4300
Sarjapur  Road,2 BHK,1000.0,2.0,40.0,2,4000
Yelahanka,1 BHK,602.0,1.0,33.0,1,5481
Malleshwaram,5 Bedroom,3000.0,4.0,900.0,5,30000
Hoodi,2 BHK,1108.0,2.0,66.46,2,5998
other,2 BHK,1040.0,2.0,68.0,2,6538
Sarjapur,2 BHK,1451.0,2.0,87.0,2,5995
Raja Rajeshwari Nagar,2 BHK,1196.0,2.0,40.54,2,3389
Kanakpura Road,1 BHK,850.0,1.0,37.32,1,4390
other,3 BHK,2171.66,3.0,298.0,3,13722
other,2 BHK,900.0,2.0,40.0,2,4444
Kudlu,1 BHK,720.0,1.0,33.0,1,4583
Thanisandra,3 BHK,1573.0,3.0,115.0,3,7310
Sarjapur  Road,2 BHK,1026.0,2.0,60.8,2,5925
CV Raman Nagar,3 BHK,1400.0,2.0,78.0,3,5571
Electronic City Phase II,3 BHK,1400.0,2.0,40.44,3,2888
Banashankari Stage III,4 Bedroom,2100.0,3.0,147.0,4,7000
Sarjapur  Road,2 BHK,1150.0,2.0,48.5,2,4217
Raja Rajeshwari Nagar,3 BHK,1500.0,3.0,88.0,3,5866
Malleshwaram,3 BHK,2400.0,3.0,380.0,3,15833
HRBR Layout,3 Bedroom,1200.0,3.0,240.0,3,20000
Kothanur,3 BHK,1790.0,3.0,105.0,3,5865
Ramagondanahalli,2 BHK,1125.0,2.0,75.0,2,6666
Hennur,3 BHK,1830.5,3.0,84.205,3,4600
Marathahalli,2 BHK,1100.0,2.0,50.0,2,4545
Banaswadi,2 BHK,1105.0,2.0,57.0,2,5158
Thanisandra,2 BHK,1110.0,2.0,46.5,2,4189
Dasanapura,1 BHK,708.0,1.0,41.0,1,5790
Kudlu,2 BHK,1027.0,2.0,44.0,2,4284
other,3 BHK,1600.0,3.0,100.0,3,6250
Thigalarapalya,4 BHK,3122.0,6.0,250.0,4,8007
Sarjapur  Road,2 BHK,1205.0,2.0,37.0,2,3070
5th Phase JP Nagar,3 BHK,1450.0,2.0,73.0,3,5034
other,4 Bedroom,750.0,4.0,125.0,4,16666
other,1 Bedroom,540.0,1.0,22.0,1,4074
Dasarahalli,2 BHK,1200.0,2.0,42.0,2,3500
Nagarbhavi,5 Bedroom,1200.0,5.0,260.0,5,21666
Kalena Agrahara,3 BHK,1837.0,2.0,95.0,3,5171
Whitefield,2 BHK,1346.0,2.0,76.01,2,5647
Balagere,2 BHK,1012.0,2.0,68.0,2,6719
other,3 BHK,1950.2,3.0,193.0,3,9896
other,2 BHK,1125.0,2.0,55.0,2,4888
Kaggadasapura,3 BHK,1500.0,2.0,60.0,3,4000
Harlur,2 BHK,1197.0,2.0,79.9,2,6675
Chandapura,3 BHK,1230.0,2.0,31.37,3,2550
BTM 2nd Stage,2 BHK,1100.0,2.0,47.0,2,4272
other,3 BHK,1563.0,2.0,52.0,3,3326
other,7 Bedroom,2240.0,4.0,700.0,7,31250
Hebbal Kempapura,4 BHK,3729.0,4.0,384.0,4,10297
Sarjapur  Road,3 Bedroom,1600.0,3.0,120.0,3,7500
Green Glen Layout,3 BHK,1715.0,3.0,120.0,3,6997
Bannerghatta Road,4 Bedroom,2400.0,4.0,180.0,4,7500
Ramagondanahalli,2 BHK,1151.0,2.0,42.5,2,3692
Banaswadi,6 Bedroom,1200.0,6.0,220.0,6,18333
Uttarahalli,2 BHK,1123.0,2.0,60.0,2,5342
Whitefield,2 BHK,1276.0,2.0,69.5,2,5446
Yelahanka,2 BHK,1315.0,2.0,77.0,2,5855
Poorna Pragna Layout,3 BHK,1270.0,2.0,50.79,3,3999
R.T. Nagar,2 BHK,1325.0,2.0,52.0,2,3924
Varthur,3 BHK,1600.0,3.0,63.2,3,3950
Jakkur,3 BHK,1660.0,3.0,104.0,3,6265
Kodichikkanahalli,2 BHK,1299.0,2.0,58.0,2,4464
Haralur Road,2 BHK,953.0,2.0,90.0,2,9443
Gunjur,2 BHK,1457.0,2.0,60.0,2,4118
Varthur,2 BHK,1000.0,1.0,65.0,2,6500
Mallasandra,3 BHK,1100.0,2.0,76.3,3,6936
Yelahanka,3 BHK,1075.0,2.0,33.53,3,3119
Kanakpura Road,3 BHK,1450.0,3.0,51.0,3,3517
Iblur Village,5 BHK,5515.0,6.0,425.0,5,7706
Bellandur,1 BHK,950.0,1.0,40.0,1,4210
Basaveshwara Nagar,5 Bedroom,1200.0,3.0,180.0,5,15000
other,2 BHK,1013.0,2.0,60.0,2,5923
Hennur Road,2 BHK,1140.0,2.0,56.0,2,4912
other,3 BHK,2100.0,3.0,220.0,3,10476
Yelahanka,3 BHK,1430.0,3.0,61.0,3,4265
other,4 Bedroom,600.0,3.0,95.0,4,15833
other,3 BHK,1600.0,3.0,77.0,3,4812
other,2 BHK,1190.0,2.0,48.0,2,4033
other,3 BHK,1625.0,3.0,95.0,3,5846
other,2 BHK,1210.0,2.0,53.0,2,4380
Lakshminarayana Pura,3 BHK,1615.0,3.0,150.0,3,9287
Kalena Agrahara,2 BHK,1180.0,2.0,85.0,2,7203
Nagasandra,2 Bedroom,1500.0,2.0,91.0,2,6066
Bellandur,2 BHK,850.0,2.0,45.0,2,5294
Kothanur,2 BHK,1130.0,2.0,55.0,2,4867
CV Raman Nagar,2 BHK,907.0,2.0,38.0,2,4189
HRBR Layout,2 BHK,1300.0,2.0,90.0,2,6923
Hennur Road,3 BHK,1891.0,3.0,110.0,3,5817
Hennur Road,3 BHK,1735.0,3.0,78.0,3,4495
Bisuvanahalli,3 BHK,1075.0,2.0,33.0,3,3069
Kumaraswami Layout,7 BHK,2040.0,4.0,123.0,7,6029
Uttarahalli,3 BHK,1590.0,3.0,57.0,3,3584
other,5 Bedroom,1600.0,5.0,140.0,5,8750
other,2 BHK,1160.0,2.0,55.0,2,4741
other,2 BHK,930.0,1.0,55.0,2,5913
Devanahalli,2 BHK,1010.0,2.0,58.0,2,5742
Electronic City,2 BHK,1221.0,2.0,68.0,2,5569
Banashankari,4 Bedroom,2400.0,4.0,470.0,4,19583
Hoodi,2 BHK,1108.0,2.0,87.48,2,7895
Vidyaranyapura,4 Bedroom,1000.0,4.0,120.0,4,12000
Sarjapur  Road,2 BHK,1370.0,2.0,70.0,2,5109
Banashankari Stage V,2 BHK,920.0,2.0,41.5,2,4510
Bhoganhalli,2 BHK,1277.0,2.0,79.89,2,6256
Horamavu Agara,2 BHK,1213.0,2.0,38.1,2,3140
Singasandra,3 BHK,1653.0,3.0,85.0,3,5142
Sarjapur,4 Bedroom,2400.0,3.0,140.0,4,5833
other,4 BHK,3000.0,5.0,500.0,4,16666
Kothanur,4 Bedroom,1600.0,4.0,250.0,4,15625
Electronics City Phase 1,3 BHK,1321.0,2.0,40.0,3,3028
Kanakpura Road,3 BHK,1570.0,3.0,68.0,3,4331
other,8 Bedroom,1150.0,8.0,70.0,8,6086
other,2 BHK,1053.0,2.0,44.0,2,4178
other,3 BHK,1325.0,2.0,56.0,3,4226
other,2 BHK,1169.0,2.0,50.11,2,4286
Badavala Nagar,2 BHK,1274.0,2.0,90.0,2,7064
Whitefield,2 BHK,1317.0,2.0,52.0,2,3948
Lakshminarayana Pura,2 BHK,1195.0,2.0,75.0,2,6276
Whitefield,4 BHK,2858.0,4.0,137.0,4,4793
Kammasandra,2 BHK,940.0,2.0,35.0,2,3723
Budigere,3 BHK,1820.0,3.0,95.5,3,5247
Kanakpura Road,3 BHK,1450.0,3.0,53.6,3,3696
Whitefield,3 BHK,1615.0,3.0,96.9,3,6000
Nagarbhavi,3 BHK,2200.0,2.0,99.0,3,4500
Subramanyapura,2 BHK,985.0,2.0,62.5,2,6345
Electronics City Phase 1,2 BHK,1116.0,2.0,33.0,2,2956
Attibele,2 BHK,656.0,2.0,25.0,2,3810
Kaggadasapura,2 BHK,1105.0,2.0,39.5,2,3574
Whitefield,2 BHK,1220.0,2.0,55.0,2,4508
Hoodi,3 BHK,1400.0,3.0,74.0,3,5285
other,4 Bedroom,2400.0,3.0,185.0,4,7708
Anandapura,3 Bedroom,850.0,3.0,63.0,3,7411
Uttarahalli,2 BHK,1000.0,2.0,35.0,2,3500
Bannerghatta Road,3 BHK,1660.0,3.0,80.0,3,4819
Uttarahalli,3 BHK,1330.0,2.0,57.0,3,4285
Old Madras Road,2 BHK,1165.0,2.0,52.0,2,4463
Budigere,2 BHK,1153.0,2.0,56.0,2,4856
Bannerghatta Road,2 BHK,1270.0,2.0,45.0,2,3543
other,2 BHK,1070.0,2.0,38.0,2,3551
other,3 BHK,1427.0,2.0,85.0,3,5956
Banashankari Stage III,3 BHK,1420.0,2.0,95.0,3,6690
Kothanur,4 Bedroom,1600.0,5.0,130.0,4,8125
other,2 BHK,2031.0,2.0,200.0,2,9847
Bellandur,2 BHK,1454.0,2.0,90.0,2,6189
Whitefield,2 BHK,1116.0,2.0,48.76,2,4369
other,4 Bedroom,2000.0,3.0,180.0,4,9000
Sarjapur  Road,2 BHK,1346.0,2.0,74.03,2,5500
other,2 BHK,1062.0,2.0,50.0,2,4708
other,1 BHK,700.0,1.0,25.5,1,3642
JP Nagar,2 BHK,1405.0,2.0,99.0,2,7046
Doddathoguru,2 BHK,1100.0,2.0,50.0,2,4545
Chikka Tirupathi,3 Bedroom,2153.0,4.0,120.0,3,5573
other,3 Bedroom,1200.0,3.0,120.0,3,10000
Ulsoor,3 BHK,2135.0,3.0,170.0,3,7962
Uttarahalli,2 BHK,1125.0,2.0,47.0,2,4177
2nd Phase Judicial Layout,3 BHK,1350.0,2.0,47.25,3,3500
Pai Layout,2 BHK,1068.0,2.0,48.0,2,4494
Doddathoguru,2 BHK,940.0,2.0,32.9,2,3500
Hosa Road,2 BHK,1063.0,2.0,32.79,2,3084
Brookefield,2 BHK,1382.0,2.0,84.5,2,6114
Harlur,2 BHK,936.0,2.0,45.0,2,4807
KR Puram,3 BHK,1400.0,2.0,60.0,3,4285
Sarjapur  Road,1 BHK,475.0,1.0,26.0,1,5473
other,2 BHK,1200.0,2.0,115.0,2,9583
other,3 BHK,1450.0,3.0,65.0,3,4482
Kudlu,2 BHK,1162.0,2.0,52.0,2,4475
Hosa Road,2 BHK,1161.0,2.0,48.75,2,4198
Green Glen Layout,3 BHK,1623.29,2.0,105.0,3,6468
Brookefield,2 BHK,1260.0,2.0,70.0,2,5555
Billekahalli,4 Bedroom,1672.0,3.0,190.0,4,11363
Seegehalli,3 Bedroom,3000.0,4.0,240.0,3,8000
Murugeshpalya,3 BHK,1500.0,2.0,70.0,3,4666
Budigere,1 BHK,705.0,1.0,34.545,1,4900
other,5 Bedroom,1650.0,6.0,200.0,5,12121
other,3 BHK,2000.0,3.0,200.0,3,10000
other,1 Bedroom,1675.0,1.0,241.0,1,14388
Electronic City,3 Bedroom,1200.0,3.0,150.0,3,12500
Nagarbhavi,2 BHK,1100.0,2.0,46.2,2,4200
Whitefield,3 BHK,1400.0,2.0,56.0,3,4000
Ramagondanahalli,3 BHK,2257.0,3.0,155.0,3,6867
Budigere,2 BHK,1153.0,2.0,56.5,2,4900
Sarjapur  Road,2 BHK,1197.0,2.0,56.86,2,4750
other,2 BHK,1150.0,2.0,90.0,2,7826
other,1 BHK,595.0,1.0,40.0,1,6722
Hoodi,3 BHK,1837.0,3.0,110.0,3,5988
other,4 Bedroom,3750.0,4.0,190.0,4,5066
Bellandur,3 BHK,1830.0,3.0,91.86,3,5019
Green Glen Layout,3 BHK,1715.0,3.0,115.0,3,6705
Electronic City,2 BHK,970.0,2.0,35.0,2,3608
Hebbal Kempapura,4 Bedroom,1200.0,3.0,148.0,4,12333
NRI Layout,4 Bedroom,2700.0,5.0,95.75,4,3546
other,3 Bedroom,5480.0,4.0,400.0,3,7299
Vidyaranyapura,4 BHK,2400.0,5.0,135.0,4,5625
Sarjapur  Road,4 BHK,3040.0,4.0,135.0,4,4440
Raja Rajeshwari Nagar,2 BHK,1168.0,2.0,66.0,2,5650
Begur Road,2 BHK,1160.0,2.0,36.54,2,3150
other,3 BHK,1500.0,3.0,75.0,3,5000
Kumaraswami Layout,2 Bedroom,600.0,2.0,72.0,2,12000
5th Phase JP Nagar,8 Bedroom,1200.0,7.0,250.0,8,20833
ITPL,2 BHK,850.0,2.0,25.4,2,2988
Ramamurthy Nagar,2 BHK,935.0,2.0,39.0,2,4171
other,5 Bedroom,1200.0,5.0,235.0,5,19583
Doddaballapur,1 BHK,654.0,1.0,49.0,1,7492
Domlur,3 BHK,1650.0,3.0,180.0,3,10909
other,3 Bedroom,5656.0,5.0,499.0,3,8822
other,2 BHK,1300.0,2.0,33.0,2,2538
Kaggalipura,2 BHK,950.0,2.0,60.0,2,6315
other,3 BHK,1250.0,3.0,54.0,3,4320
Doddakallasandra,2 BHK,1072.0,2.0,42.87,2,3999
other,3 BHK,1425.0,2.0,65.0,3,4561
Uttarahalli,2 BHK,1165.0,1.0,42.09,2,3612
Kathriguppe,3 BHK,1390.0,2.0,69.49,3,4999
Kalena Agrahara,3 BHK,1510.0,3.0,78.0,3,5165
Thanisandra,2 BHK,1305.0,2.0,90.0,2,6896
Vidyaranyapura,6 Bedroom,1200.0,6.0,170.0,6,14166
Kodigehaali,2 BHK,866.0,2.0,32.0,2,3695
Hulimavu,2 BHK,1231.0,2.0,84.5,2,6864
Sompura,3 BHK,1360.0,2.0,45.0,3,3308
Yeshwanthpur,3 BHK,2500.0,3.0,138.0,3,5520
Whitefield,2 BHK,1140.0,2.0,55.0,2,4824
Kanakpura Road,1 BHK,525.0,1.0,30.0,1,5714
Whitefield,2 BHK,1295.0,2.0,64.5,2,4980
Hormavu,2 BHK,1020.0,2.0,45.75,2,4485
Old Airport Road,4 BHK,3504.0,4.0,262.0,4,7477
other,2 Bedroom,1200.0,1.0,50.0,2,4166
Vidyaranyapura,2 BHK,1100.0,2.0,41.0,2,3727
Bannerghatta Road,4 Bedroom,4723.0,4.0,500.0,4,10586
Kothannur,2 BHK,820.0,2.0,37.0,2,4512
Sarjapur,4 Bedroom,3508.0,4.0,386.0,4,11003
other,4 Bedroom,3700.0,4.0,400.0,4,10810
Kalena Agrahara,2 BHK,1354.0,2.0,70.0,2,5169
Basavangudi,4 BHK,2453.0,3.0,250.0,4,10191
Garudachar Palya,2 BHK,1154.0,2.0,51.8,2,4488
7th Phase JP Nagar,3 BHK,1680.0,2.0,92.0,3,5476
Electronics City Phase 1,2 BHK,1205.0,2.0,60.0,2,4979
Raja Rajeshwari Nagar,3 BHK,1400.0,2.0,86.0,3,6142
other,3 BHK,1625.0,3.0,150.0,3,9230
Electronic City,3 Bedroom,1500.0,3.0,73.0,3,4866
Bannerghatta Road,3 BHK,1520.0,2.0,80.0,3,5263
other,2 BHK,1300.0,3.0,65.0,2,5000
other,2 BHK,1150.0,2.0,47.0,2,4086
Kadugodi,3 BHK,1762.0,3.0,109.0,3,6186
Budigere,2 BHK,1153.0,2.0,58.0,2,5030
Yelahanka,4 BHK,2650.0,4.0,223.0,4,8415
Sompura,3 BHK,1350.0,2.0,47.0,3,3481
Billekahalli,2 BHK,950.0,2.0,56.0,2,5894
Jakkur,3 BHK,3295.0,3.0,310.0,3,9408
other,3 BHK,1515.0,3.0,80.5,3,5313
Neeladri Nagar,2 BHK,1100.0,2.0,30.0,2,2727
other,2 BHK,1250.0,2.0,65.0,2,5200
HBR Layout,2 Bedroom,1200.0,2.0,120.0,2,10000
5th Phase JP Nagar,7 BHK,2500.0,8.0,95.0,7,3800
Hoskote,4 Bedroom,1200.0,4.0,100.0,4,8333
Yelahanka,3 BHK,1355.0,2.0,51.5,3,3800
other,3 BHK,1115.0,3.0,39.14,3,3510
8th Phase JP Nagar,3 BHK,1275.0,3.0,45.0,3,3529
Padmanabhanagar,3 BHK,2051.0,3.0,170.0,3,8288
BTM Layout,3 BHK,1470.0,2.0,99.0,3,6734
other,3 BHK,1550.0,2.0,70.0,3,4516
Bannerghatta Road,3 BHK,1270.0,2.0,73.0,3,5748
Bhoganhalli,4 BHK,2119.0,4.0,111.0,4,5238
Hebbal Kempapura,3 BHK,1600.0,3.0,170.0,3,10625
Kudlu Gate,3 BHK,1320.0,2.0,50.0,3,3787
other,3 BHK,2505.0,3.0,165.0,3,6586
other,3 BHK,1410.0,2.0,54.0,3,3829
Whitefield,2 BHK,1200.0,2.0,70.0,2,5833
other,3 BHK,1705.0,2.0,86.96,3,5100
Electronic City,2 BHK,995.0,2.0,48.0,2,4824
Horamavu Banaswadi,2 BHK,1081.0,2.0,52.0,2,4810
Parappana Agrahara,2 BHK,1194.0,2.0,46.0,2,3852
Sonnenahalli,1 BHK,605.0,1.0,30.24,1,4998
Yelahanka,1 BHK,567.0,1.0,25.0,1,4409
Rajaji Nagar,2 BHK,1718.0,3.0,275.0,2,16006
Banashankari Stage VI,3 BHK,1410.5,2.0,70.385,3,4990
2nd Stage Nagarbhavi,3 Bedroom,600.0,5.0,135.0,3,22500
Green Glen Layout,3 BHK,1715.0,3.0,105.0,3,6122
Doddaballapur,4 Bedroom,3206.0,5.0,270.0,4,8421
Kundalahalli,4 Bedroom,1500.0,4.0,235.0,4,15666
Bannerghatta Road,3 BHK,1846.0,3.0,120.0,3,6500
other,3 Bedroom,600.0,3.0,90.0,3,15000
Whitefield,2 BHK,1015.0,2.0,56.0,2,5517
other,4 Bedroom,4350.0,8.0,2600.0,4,59770
Thanisandra,2 BHK,1100.0,2.0,36.0,2,3272
Thanisandra,3 BHK,1965.0,4.0,125.0,3,6361
NGR Layout,2 BHK,1021.0,2.0,45.9,2,4495
Haralur Road,2 BHK,1230.0,2.0,67.65,2,5500
Uttarahalli,2 BHK,1260.0,2.0,55.0,2,4365
Indira Nagar,4 Bedroom,2400.0,4.0,700.0,4,29166
Lingadheeranahalli,3 BHK,1894.0,4.0,114.0,3,6019
2nd Stage Nagarbhavi,6 Bedroom,1500.0,4.0,233.0,6,15533
Jakkur,3 BHK,1950.0,3.0,131.0,3,6717
Sector 2 HSR Layout,2 BHK,1231.0,2.0,66.0,2,5361
Mahadevpura,2 BHK,1236.0,2.0,58.0,2,4692
R.T. Nagar,4 BHK,1800.0,3.0,120.0,4,6666
other,3 BHK,1650.0,3.0,135.0,3,8181
other,5 Bedroom,750.0,3.0,88.0,5,11733
Begur,3 BHK,1443.0,2.0,70.0,3,4851
Ramamurthy Nagar,4 Bedroom,886.0,4.0,120.0,4,13544
other,2 BHK,925.0,2.0,68.0,2,7351
Chandapura,3 BHK,1305.0,3.0,33.28,3,2550
other,2 BHK,1190.0,2.0,46.41,2,3900
Uttarahalli,2 BHK,1284.0,2.0,53.36,2,4155
other,4 Bedroom,350.0,3.0,45.0,4,12857
other,3 Bedroom,875.0,3.0,115.0,3,13142
Chikkalasandra,2 BHK,1325.0,2.0,59.0,2,4452
other,6 Bedroom,2400.0,5.0,750.0,6,31250
Kudlu Gate,2 BHK,1183.0,2.0,79.54,2,6723
other,3 BHK,1750.0,3.0,100.0,3,5714
other,4 BHK,16335.0,4.0,149.0,4,912
other,1 BHK,747.0,1.0,27.0,1,3614
Electronic City,2 BHK,1342.0,2.0,90.0,2,6706
other,2 BHK,1113.0,2.0,52.0,2,4672
other,3 BHK,1738.0,3.0,110.0,3,6329
Brookefield,2 BHK,1206.0,2.0,90.0,2,7462
Hebbal,4 BHK,2790.0,4.0,198.0,4,7096
Hebbal,2 BHK,1162.0,2.0,59.0,2,5077
Parappana Agrahara,2 BHK,1194.0,2.0,46.0,2,3852
Sarjapur  Road,3 BHK,1857.0,4.0,155.0,3,8346
other,6 Bedroom,3500.0,6.0,115.0,6,3285
Rajaji Nagar,2 BHK,1763.0,3.0,240.0,2,13613
Uttarahalli,2 BHK,1150.0,2.0,45.99,2,3999
Koramangala,4 Bedroom,2400.0,5.0,550.0,4,22916
Chikkalasandra,2 BHK,1290.0,2.0,72.0,2,5581
Thubarahalli,4 BHK,3408.0,5.0,145.0,4,4254
HBR Layout,2 Bedroom,900.0,2.0,145.0,2,16111
Sanjay nagar,2 BHK,1150.0,2.0,70.0,2,6086
Banashankari,3 Bedroom,1040.0,2.0,109.0,3,10480
Yeshwanthpur,2 BHK,1160.0,2.0,64.08,2,5524
other,2 BHK,900.0,2.0,27.0,2,3000
Sarjapur  Road,4 Bedroom,1750.0,3.0,185.0,4,10571
Electronic City,3 BHK,1320.0,2.0,38.12,3,2887
Lingadheeranahalli,4 BHK,2240.0,4.0,160.0,4,7142
Electronics City Phase 1,2 BHK,891.0,2.0,26.0,2,2918
other,5 Bedroom,1000.0,4.0,80.0,5,8000
Ambedkar Nagar,3 BHK,1852.0,3.0,125.0,3,6749
Chandapura,3 BHK,1323.0,2.0,42.0,3,3174
Vittasandra,2 BHK,1404.0,2.0,67.5,2,4807
Dodda Nekkundi,2 BHK,1370.0,2.0,60.0,2,4379
Uttarahalli,3 BHK,1385.0,2.0,48.48,3,3500
Gollarapalya Hosahalli,2 BHK,1129.0,2.0,50.0,2,4428
Mahalakshmi Layout,3 BHK,1876.0,3.0,150.0,3,7995
other,3 BHK,1720.0,3.0,95.0,3,5523
Jalahalli East,4 Bedroom,1200.0,4.0,80.0,4,6666
Mysore Road,2 BHK,1175.0,2.0,86.68,2,7377
other,1 BHK,500.0,1.0,51.0,1,10200
other,5 Bedroom,1200.0,5.0,85.0,5,7083
Electronic City Phase II,1 BHK,1200.0,1.0,295.0,1,24583
Hosa Road,3 BHK,1893.0,3.0,130.0,3,6867
Whitefield,3 BHK,1430.0,3.0,52.0,3,3636
other,2 BHK,850.0,2.0,45.0,2,5294
CV Raman Nagar,2 BHK,1392.0,2.0,110.0,2,7902
Kanakpura Road,3 BHK,1938.0,3.0,105.0,3,5417
other,2 BHK,1020.0,2.0,52.0,2,5098
other,3 Bedroom,2400.0,2.0,360.0,3,15000
HBR Layout,3 BHK,1832.0,3.0,160.0,3,8733
Konanakunte,3 BHK,1423.0,2.0,85.0,3,5973
Jakkur,2 BHK,1432.0,2.0,85.0,2,5935
other,4 Bedroom,1350.0,4.0,235.0,4,17407
BTM 2nd Stage,8 Bedroom,1500.0,6.0,270.0,8,18000
CV Raman Nagar,3 BHK,1480.0,3.0,65.0,3,4391
Kanakpura Road,3 BHK,1843.0,3.0,96.2,3,5219
other,3 Bedroom,760.0,4.0,95.0,3,12500
Hulimavu,3 BHK,1916.0,2.0,150.0,3,7828
Hennur,2 BHK,1255.0,2.0,55.5,2,4422
Yelenahalli,2 BHK,1160.0,2.0,44.08,2,3800
Old Madras Road,2 BHK,2640.0,2.0,170.0,2,6439
Bommanahalli,2 BHK,1160.0,2.0,53.0,2,4568
Lakshminarayana Pura,2 BHK,1200.0,2.0,80.0,2,6666
7th Phase JP Nagar,2 BHK,1100.0,2.0,44.0,2,4000
Ramagondanahalli,3 BHK,1500.0,3.0,100.0,3,6666
Electronic City Phase II,3 BHK,1336.0,2.0,50.35,3,3768
Choodasandra,2 BHK,1065.0,2.0,46.0,2,4319
Hebbal,2 BHK,1294.0,2.0,115.0,2,8887
Rachenahalli,3 BHK,1550.0,3.0,68.5,3,4419
Whitefield,2 BHK,1216.0,2.0,75.0,2,6167
other,2 BHK,1005.0,2.0,40.0,2,3980
7th Phase JP Nagar,3 BHK,1430.0,2.0,102.0,3,7132
Frazer Town,4 BHK,4100.0,4.0,660.0,4,16097
Hoodi,3 BHK,1660.0,3.0,125.0,3,7530
Thanisandra,3 BHK,1430.0,2.0,56.0,3,3916
Hennur Road,3 BHK,1470.0,2.0,75.0,3,5102
Jalahalli,2 BHK,1045.0,2.0,76.77,2,7346
Hebbal,4 BHK,4225.0,6.0,359.0,4,8497
Anjanapura,2 BHK,950.0,2.0,30.0,2,3157
Doddathoguru,3 BHK,1382.0,3.0,65.0,3,4703
BTM 2nd Stage,2 BHK,930.0,2.0,46.0,2,4946
Sarjapur  Road,3 BHK,1660.0,2.0,116.0,3,6987
other,2 BHK,1290.0,2.0,71.95,2,5577
Whitefield,3 BHK,1010.0,2.0,45.0,3,4455
Brookefield,4 BHK,3050.0,4.0,130.0,4,4262
Dodda Nekkundi,2 BHK,1200.0,2.0,71.0,2,5916
other,3 BHK,1650.0,3.0,140.0,3,8484
Budigere,3 BHK,1636.0,3.0,88.0,3,5378
Rajaji Nagar,3 BHK,2367.0,3.0,320.0,3,13519
Budigere,2 BHK,1153.0,2.0,56.55,2,4904
Hegde Nagar,3 BHK,2087.01,4.0,160.0,3,7666
LB Shastri Nagar,2 BHK,1200.0,2.0,47.0,2,3916
HSR Layout,3 BHK,1844.0,3.0,89.0,3,4826
Begur Road,2 BHK,1215.0,2.0,43.75,2,3600
Sarjapur  Road,2 BHK,1194.0,2.0,57.0,2,4773
other,2 BHK,1220.0,2.0,52.0,2,4262
other,4 BHK,3150.0,4.0,150.0,4,4761
Kanakpura Road,1 BHK,525.0,1.0,26.0,1,4952
Bommasandra,2 BHK,955.0,2.0,37.97,2,3975
other,5 Bedroom,2000.0,5.0,75.0,5,3750
Magadi Road,2 BHK,884.0,2.0,48.0,2,5429
Kathriguppe,2 BHK,1250.0,2.0,68.75,2,5500
Bommasandra Industrial Area,2 BHK,7000.0,2.0,135.0,2,1928
Dodda Nekkundi,3 Bedroom,2400.0,4.0,300.0,3,12500
Harlur,4 BHK,2990.0,4.0,225.0,4,7525
Bellandur,2 BHK,1195.0,2.0,52.0,2,4351
other,1 BHK,2559.0,1.0,55.0,1,2149
Begur,2 BHK,1200.0,2.0,33.0,2,2750
Sarjapur  Road,2 BHK,984.0,2.0,45.75,2,4649
Bommasandra,2 BHK,1089.0,2.0,40.0,2,3673
Hennur Road,2 BHK,1165.0,2.0,52.0,2,4463
Chandapura,1 BHK,410.0,1.0,10.0,1,2439
Hennur,3 BHK,1482.0,3.0,110.0,3,7422
Ramamurthy Nagar,2 Bedroom,1200.0,2.0,68.0,2,5666
Kanakapura,3 BHK,1938.0,3.0,113.0,3,5830
Hormavu,3 BHK,1166.0,2.0,34.97,3,2999
Chikkalasandra,3 BHK,1275.0,2.0,55.46,3,4349
Mico Layout,2 BHK,1140.0,2.0,43.0,2,3771
Hegde Nagar,4 Bedroom,3734.0,5.0,430.0,4,11515
other,1 BHK,720.0,1.0,28.0,1,3888
Nagarbhavi,3 Bedroom,1200.0,3.0,228.0,3,19000
other,2 BHK,1100.0,2.0,90.0,2,8181
Sector 7 HSR Layout,2 BHK,1162.0,2.0,98.77,2,8500
Ramamurthy Nagar,1 Bedroom,540.0,1.0,35.0,1,6481
Rachenahalli,3 BHK,1550.0,2.0,73.5,3,4741
R.T. Nagar,3 Bedroom,1300.0,3.0,145.0,3,11153
other,5 Bedroom,1200.0,5.0,130.0,5,10833
other,2 BHK,1220.0,2.0,60.0,2,4918
Doddathoguru,2 BHK,964.0,2.0,62.0,2,6431
other,3 BHK,1500.0,3.0,50.0,3,3333
Sarjapur  Road,3 BHK,1347.0,2.0,44.0,3,3266
other,4 BHK,5000.0,5.0,250.0,4,5000
Bannerghatta Road,4 Bedroom,3000.0,4.0,140.0,4,4666
Bannerghatta,2 BHK,1120.0,2.0,42.5,2,3794
other,1 BHK,600.0,1.0,22.8,1,3800
Whitefield,3 BHK,2321.0,4.0,157.0,3,6764
other,1 BHK,540.0,1.0,24.0,1,4444
Hebbal,2 BHK,1150.0,2.0,57.0,2,4956
Sonnenahalli,3 BHK,1484.0,3.0,75.0,3,5053
Ramagondanahalli,3 BHK,1680.0,2.0,68.0,3,4047
other,2 BHK,620.0,2.0,23.0,2,3709
other,3 BHK,1650.0,3.0,78.0,3,4727
other,3 BHK,1250.0,3.0,44.0,3,3520
other,3 BHK,2072.0,3.0,108.0,3,5212
Sarjapur  Road,2 BHK,1105.0,2.0,40.0,2,3619
Vidyaranyapura,2 BHK,1200.0,2.0,60.5,2,5041
Kumaraswami Layout,7 Bedroom,3100.0,7.0,145.0,7,4677
Whitefield,4 Bedroom,4007.0,4.0,530.0,4,13226
Electronic City,2 BHK,550.0,2.0,16.0,2,2909
Banashankari,3 BHK,1340.0,2.0,53.6,3,4000
Rajaji Nagar,3 BHK,2409.0,3.0,395.0,3,16396
Haralur Road,2 BHK,1194.0,2.0,46.0,2,3852
Vittasandra,2 BHK,1246.0,2.0,65.0,2,5216
Old Airport Road,2 BHK,1055.0,2.0,75.0,2,7109
Hosa Road,3 BHK,1541.0,3.0,69.61,3,4517
Harlur,2 BHK,1370.0,2.0,88.0,2,6423
Whitefield,2 BHK,1224.0,2.0,75.0,2,6127
Kengeri Satellite Town,4 Bedroom,650.0,2.0,58.0,4,8923
Kambipura,2 BHK,883.0,2.0,39.0,2,4416
Yeshwanthpur,3 BHK,1676.0,3.0,92.13,3,5497
other,2 BHK,1154.0,2.0,57.0,2,4939
Kanakpura Road,3 BHK,1610.0,3.0,78.5,3,4875
other,2 BHK,1057.0,2.0,42.0,2,3973
Raja Rajeshwari Nagar,2 BHK,1157.0,2.0,65.0,2,5617
other,4 Bedroom,1200.0,4.0,85.0,4,7083
other,7 Bedroom,1500.0,7.0,130.0,7,8666
other,2 BHK,980.0,2.0,35.0,2,3571
Sarjapur  Road,2 BHK,1140.0,2.0,40.0,2,3508
Raja Rajeshwari Nagar,2 BHK,1206.0,2.0,40.7,2,3374
Electronic City Phase II,1 BHK,575.0,1.0,16.75,1,2913
Koramangala,3 BHK,1600.0,3.0,170.0,3,10625
other,4 Bedroom,2400.0,4.0,500.0,4,20833
Horamavu Agara,3 BHK,1300.0,2.0,52.0,3,4000
other,3 BHK,1270.0,2.0,35.56,3,2800
Electronic City,2 BHK,1128.0,2.0,63.5,2,5629
Electronics City Phase 1,1 BHK,640.0,1.0,45.0,1,7031
other,3 BHK,2400.0,3.0,185.0,3,7708
TC Palaya,3 Bedroom,1200.0,2.0,66.0,3,5500
Kanakapura,2 BHK,1090.0,2.0,38.15,2,3500
Electronic City,1 BHK,589.0,1.0,29.0,1,4923
Electronic City,2 BHK,1125.0,2.0,28.13,2,2500
Kogilu,2 BHK,1140.0,2.0,50.66,2,4443
Sarjapur  Road,4 Bedroom,3150.0,4.0,500.0,4,15873
Kundalahalli,2 BHK,1315.0,2.0,74.0,2,5627
Bellandur,3 BHK,1435.0,2.0,115.0,3,8013
Haralur Road,2 BHK,1092.0,2.0,44.0,2,4029
Kadugodi,2 BHK,1314.0,2.0,80.0,2,6088
Koramangala,2 BHK,1260.0,2.0,100.0,2,7936
other,7 Bedroom,1875.0,5.0,192.0,7,10240
Nagarbhavi,2 BHK,1080.0,2.0,49.0,2,4537
Nagasandra,8 Bedroom,4000.0,8.0,160.0,8,4000
Mysore Road,3 BHK,1082.0,2.0,45.0,3,4158
Jakkur,3 BHK,1150.0,3.0,65.0,3,5652
Hennur Road,2 BHK,1052.0,2.0,34.72,2,3300
Yelahanka,2 BHK,1045.0,2.0,45.0,2,4306
Kodigehaali,3 BHK,1442.0,3.0,80.0,3,5547
Kaggadasapura,3 BHK,1500.0,2.0,75.0,3,5000
Marsur,2 BHK,497.0,1.0,20.0,2,4024
Hennur,2 BHK,1255.0,2.0,53.57,2,4268
other,3 BHK,1432.0,3.0,79.0,3,5516
Kundalahalli,3 Bedroom,2100.0,3.0,150.0,3,7142
Hennur Road,2 BHK,1450.0,2.0,80.0,2,5517
Sarjapur  Road,2 BHK,1346.0,2.0,69.61,2,5171
Kumaraswami Layout,2 BHK,1200.0,2.0,29.0,2,2416
Electronic City Phase II,3 BHK,925.0,2.0,49.82,3,5385
Dasarahalli,2 BHK,1160.0,2.0,49.0,2,4224
Lingadheeranahalli,3 BHK,1768.0,3.0,81.31,3,4598
Kothanur,2 BHK,1094.0,2.0,54.0,2,4936
other,3 BHK,1500.0,3.0,90.0,3,6000
Gubbalala,3 BHK,1745.0,3.0,130.0,3,7449
Bommanahalli,7 Bedroom,2875.0,7.0,85.0,7,2956
other,4 BHK,3500.0,3.0,425.0,4,12142
Sonnenahalli,3 BHK,1550.0,3.0,50.0,3,3225
Rajiv Nagar,4 BHK,2340.0,5.0,160.0,4,6837
Whitefield,3 BHK,1655.0,3.0,113.0,3,6827
Battarahalli,3 BHK,2024.0,3.0,103.0,3,5088
other,4 Bedroom,3100.0,5.0,425.0,4,13709
Jalahalli,3 BHK,1575.0,4.0,100.0,3,6349
Iblur Village,3 BHK,1920.0,3.0,130.0,3,6770
Sarjapur  Road,4 Bedroom,6200.0,4.0,744.0,4,12000
Kambipura,3 BHK,1082.0,2.0,55.0,3,5083
BTM 2nd Stage,3 BHK,1265.0,2.0,55.0,3,4347
other,3 BHK,1100.0,2.0,45.0,3,4090
Margondanahalli,2 Bedroom,1160.0,2.0,65.0,2,5603
Vittasandra,2 BHK,1246.0,2.0,67.4,2,5409
Hosa Road,1 BHK,615.0,1.0,43.08,1,7004
7th Phase JP Nagar,2 BHK,1128.0,2.0,60.0,2,5319
other,2 BHK,1260.0,2.0,55.0,2,4365
Bellandur,3 BHK,1830.0,3.0,89.89,3,4912
Sarjapur,3 BHK,1525.0,2.0,68.0,3,4459
Yelahanka,1 BHK,602.0,2.0,30.0,1,4983
Kumaraswami Layout,7 Bedroom,3000.0,4.0,400.0,7,13333
Hegde Nagar,3 BHK,1835.0,3.0,89.0,3,4850
Kanakpura Road,3 BHK,1250.0,3.0,62.6,3,5008
other,4 Bedroom,1200.0,4.0,110.0,4,9166
Murugeshpalya,3 BHK,1855.0,3.0,96.0,3,5175
Kanakpura Road,3 BHK,1450.0,3.0,60.91,3,4200
Sarjapur  Road,3 BHK,2089.0,3.0,148.0,3,7084
Haralur Road,3 BHK,1464.0,3.0,56.0,3,3825
Banashankari Stage VI,4 Bedroom,4800.0,3.0,200.0,4,4166
Begur Road,2 BHK,1200.0,2.0,44.73,2,3727
Sarjapur,2 BHK,1095.0,2.0,45.0,2,4109
Ambedkar Nagar,3 Bedroom,2900.0,3.0,297.0,3,10241
BTM 2nd Stage,3 BHK,2500.0,3.0,345.0,3,13800
Kothanur,3 BHK,1170.0,3.0,80.0,3,6837
6th Phase JP Nagar,2 BHK,1280.0,2.0,88.0,2,6875
Rajaji Nagar,2 BHK,1224.0,2.0,105.0,2,8578
Thanisandra,3 BHK,1694.0,3.0,125.0,3,7378
Electronic City,2 BHK,825.0,2.0,30.0,2,3636
Sarjapur  Road,3 BHK,1612.0,3.0,98.38,3,6102
Chandapura,2 BHK,985.0,2.0,25.12,2,2550
Murugeshpalya,2 BHK,1175.0,2.0,75.0,2,6382
other,1 Bedroom,600.0,1.0,24.0,1,4000
Whitefield,3 BHK,1740.0,3.0,85.0,3,4885
other,3 BHK,1770.0,3.0,99.0,3,5593
HAL 2nd Stage,8 Bedroom,1000.0,7.0,260.0,8,26000
Horamavu Agara,2 BHK,1090.0,2.0,46.0,2,4220
other,6 Bedroom,2295.0,3.0,650.0,6,28322
Whitefield,5 Bedroom,4144.0,5.0,331.0,5,7987
other,3 BHK,1800.0,3.0,130.0,3,7222
Jigani,3 BHK,1221.0,3.0,65.0,3,5323
8th Phase JP Nagar,2 BHK,1046.0,2.0,48.0,2,4588
Uttarahalli,3 BHK,1290.0,2.0,56.12,3,4350
Electronic City Phase II,2 BHK,1065.0,2.0,30.76,2,2888
Ambedkar Nagar,3 BHK,1850.0,4.0,121.0,3,6540
Battarahalli,2 BHK,1097.0,2.0,49.0,2,4466
Kalena Agrahara,2 BHK,1200.0,3.0,70.0,2,5833
Varthur,3 BHK,2160.0,3.0,170.0,3,7870
Anekal,2 BHK,1140.0,2.0,52.0,2,4561
Thigalarapalya,2 BHK,1297.0,2.0,98.0,2,7555
Padmanabhanagar,2 BHK,1150.0,2.0,50.0,2,4347
other,2 BHK,1375.0,2.0,45.0,2,3272
Babusapalaya,2 BHK,1245.0,2.0,58.0,2,4658
other,4 BHK,4750.0,6.0,948.0,4,19957
Nagarbhavi,4 Bedroom,1800.0,3.0,180.0,4,10000
Brookefield,3 BHK,1594.0,3.0,140.0,3,8782
TC Palaya,3 Bedroom,1200.0,2.0,66.0,3,5500
Chandapura,1 BHK,645.0,1.0,16.45,1,2550
Kanakpura Road,2 BHK,700.0,2.0,36.0,2,5142
Rajaji Nagar,5 Bedroom,2500.0,4.0,650.0,5,26000
Raja Rajeshwari Nagar,2 BHK,1000.0,2.0,50.0,2,5000
Whitefield,6 Bedroom,4000.0,5.0,540.0,6,13500
Begur Road,2 BHK,1160.0,2.0,42.0,2,3620
other,6 Bedroom,1200.0,6.0,85.0,6,7083
OMBR Layout,5 Bedroom,600.0,3.0,140.0,5,23333
other,3 BHK,1570.0,2.0,95.0,3,6050
Sarjapur  Road,2 BHK,1290.0,2.0,88.0,2,6821
Kudlu Gate,2 BHK,1200.0,2.0,44.4,2,3700
other,3 BHK,2777.29,5.0,649.0,3,23368
other,3 BHK,2250.0,3.0,180.0,3,8000
other,2 BHK,1270.0,2.0,172.0,2,13543
other,2 BHK,1145.0,2.0,44.0,2,3842
Chikkabanavar,3 BHK,1320.0,2.0,46.0,3,3484
Electronic City Phase II,2 BHK,1065.0,2.0,30.75,2,2887
Whitefield,3 BHK,1650.0,3.0,48.0,3,2909
Sarjapur  Road,2 BHK,1371.0,2.0,86.0,2,6272
other,5 Bedroom,2700.0,5.0,150.0,5,5555
Electronic City,3 BHK,1575.0,3.0,94.5,3,6000
Yelahanka,2 BHK,1362.0,2.0,66.0,2,4845
Haralur Road,3 BHK,1255.0,3.0,90.0,3,7171
Whitefield,2 BHK,925.0,2.0,35.0,2,3783
Electronics City Phase 1,2 BHK,1088.0,2.0,30.46,2,2799
Yelahanka,2 BHK,1234.0,2.0,80.0,2,6482
Sarjapur  Road,3 BHK,2140.0,3.0,139.0,3,6495
Thubarahalli,3 BHK,1584.0,3.0,101.0,3,6376
Whitefield,2 BHK,1175.0,2.0,50.0,2,4255
Sarjapura - Attibele Road,3 BHK,1555.0,3.0,52.0,3,3344
Chandapura,1 BHK,645.0,1.0,16.45,1,2550
Dodda Nekkundi,3 BHK,1585.0,2.0,79.0,3,4984
Kalena Agrahara,2 BHK,1354.0,2.0,71.0,2,5243
other,2 BHK,1165.0,2.0,45.0,2,3862
Kumaraswami Layout,3 Bedroom,1200.0,3.0,75.0,3,6250
Hebbal,3 BHK,1740.0,2.0,137.0,3,7873
Hebbal,9 Bedroom,1200.0,9.0,185.0,9,15416
Rachenahalli,2 BHK,985.0,2.0,50.17,2,5093
Chandapura,2 Bedroom,1200.0,2.0,36.0,2,3000
other,3 Bedroom,600.0,4.0,70.0,3,11666
other,3 BHK,1340.0,2.0,47.0,3,3507
Begur Road,3 BHK,1583.0,3.0,95.23,3,6015
other,4 Bedroom,1200.0,5.0,325.0,4,27083
Basaveshwara Nagar,2 Bedroom,1200.0,1.0,180.0,2,15000
Bannerghatta Road,2 BHK,1122.5,2.0,61.74,2,5500
Thanisandra,3 BHK,1573.0,3.0,100.0,3,6357
CV Raman Nagar,2 BHK,1227.0,2.0,42.88,2,3494
other,2 BHK,1210.0,2.0,60.0,2,4958
Basavangudi,3 BHK,1650.0,3.0,125.0,3,7575
Chikkalasandra,5 Bedroom,1000.0,4.0,300.0,5,30000
Raja Rajeshwari Nagar,3 BHK,1485.0,2.0,63.0,3,4242
Haralur Road,3 BHK,1735.0,3.0,97.0,3,5590
Hebbal Kempapura,3 BHK,1730.0,3.0,175.0,3,10115
other,3 BHK,1684.0,3.0,86.0,3,5106
other,3 BHK,1686.0,3.0,77.0,3,4567
other,3 BHK,1654.0,3.0,82.0,3,4957
Green Glen Layout,3 BHK,1600.0,3.0,105.0,3,6562
KR Puram,2 BHK,1100.0,2.0,39.9,2,3627
Bannerghatta Road,3 BHK,1655.0,3.0,95.0,3,5740
Hennur,3 BHK,1830.0,3.0,121.0,3,6612
Hebbal,3 BHK,1430.0,3.0,85.0,3,5944
Judicial Layout,3 BHK,1989.0,4.0,155.0,3,7792
Sarjapur  Road,2 BHK,1044.0,2.0,55.0,2,5268
Banashankari,3 BHK,1426.0,3.0,85.0,3,5960
Choodasandra,2 BHK,1115.0,2.0,50.0,2,4484
Yelahanka,2 BHK,1000.0,2.0,29.0,2,2900
Kathriguppe,2 Bedroom,754.0,2.0,69.99,2,9282
Electronic City Phase II,2 BHK,1140.0,2.0,33.06,2,2900
Kodichikkanahalli,2 BHK,900.0,2.0,35.0,2,3888
other,2 BHK,1200.0,2.0,50.0,2,4166
Malleshwaram,1 BHK,580.0,1.0,45.0,1,7758
TC Palaya,3 Bedroom,1200.0,2.0,66.0,3,5500
Munnekollal,2 BHK,1030.0,2.0,50.36,2,4889
Koramangala,3 BHK,1910.0,4.0,150.0,3,7853
other,3 BHK,1950.0,3.0,132.0,3,6769
Kundalahalli,3 BHK,1724.0,3.0,125.0,3,7250
other,3 Bedroom,1200.0,2.0,130.0,3,10833
Rajaji Nagar,2 Bedroom,1314.0,2.0,225.0,2,17123
Kothanur,2 BHK,1400.0,2.0,65.0,2,4642
Kaggadasapura,3 BHK,1400.0,2.0,68.0,3,4857
Hulimavu,2 BHK,1115.0,2.0,44.6,2,4000
Bharathi Nagar,2 BHK,1300.0,2.0,55.86,2,4296
other,4 Bedroom,2400.0,4.0,150.0,4,6250
Kanakpura Road,2 BHK,1332.0,2.0,108.0,2,8108
Abbigere,2 BHK,1000.0,2.0,41.0,2,4100
Whitefield,3 Bedroom,1200.0,3.0,67.75,3,5645
Whitefield,4 BHK,2444.0,4.0,145.0,4,5932
Electronic City,2 BHK,1070.0,2.0,48.0,2,4485
Sarjapur  Road,3 BHK,1600.0,3.0,85.0,3,5312
Marathahalli,2 BHK,1196.0,2.0,57.95,2,4845
Electronic City,2 BHK,1025.0,2.0,54.0,2,5268
Choodasandra,3 BHK,1220.0,3.0,56.0,3,4590
Sarjapur  Road,3 BHK,2180.0,4.0,180.0,3,8256
other,3 BHK,1600.0,3.0,102.0,3,6375
Whitefield,2 BHK,1270.0,2.0,110.0,2,8661
other,3 BHK,3290.0,4.0,460.0,3,13981
Jakkur,3 BHK,1932.47,3.0,183.0,3,9469
Jalahalli,3 BHK,2086.0,3.0,160.0,3,7670
Old Madras Road,2 BHK,1165.0,2.0,51.42,2,4413
Marathahalli,3 BHK,1310.0,3.0,63.25,3,4828
Rajaji Nagar,2 BHK,1440.0,2.0,165.0,2,11458
Electronics City Phase 1,3 BHK,1900.0,3.0,85.0,3,4473
Chandapura,2 BHK,740.0,1.0,22.0,2,2972
Yeshwanthpur,3 BHK,2503.0,3.0,138.0,3,5513
5th Phase JP Nagar,2 BHK,1075.0,2.0,60.0,2,5581
Bellandur,2 BHK,1260.0,2.0,83.5,2,6626
Yelahanka,3 BHK,1282.0,2.0,48.72,3,3800
Tumkur Road,4 Bedroom,1100.0,2.0,60.0,4,5454
Green Glen Layout,3 BHK,1717.0,3.0,125.0,3,7280
other,2 BHK,1088.0,2.0,46.0,2,4227
Raja Rajeshwari Nagar,3 BHK,1335.0,2.0,56.0,3,4194
Vidyaranyapura,2 BHK,1200.0,2.0,42.0,2,3500
Garudachar Palya,3 BHK,1325.0,2.0,60.0,3,4528
JP Nagar,2 BHK,1157.0,2.0,73.0,2,6309
Sahakara Nagar,2 BHK,1200.0,2.0,54.0,2,4500
Pai Layout,3 BHK,1510.0,2.0,65.0,3,4304
Bommanahalli,2 BHK,1050.0,2.0,54.0,2,5142
other,2 BHK,1200.0,2.0,45.0,2,3750
Binny Pete,3 BHK,1516.0,3.0,147.0,3,9696
Indira Nagar,6 Bedroom,2400.0,6.0,475.0,6,19791
Channasandra,2 BHK,1050.0,2.0,44.41,2,4229
Hegde Nagar,2 Bedroom,1000.0,2.0,66.0,2,6600
9th Phase JP Nagar,2 BHK,1050.0,2.0,35.0,2,3333
JP Nagar,4 Bedroom,600.0,5.0,110.0,4,18333
BEML Layout,2 BHK,1194.0,2.0,65.0,2,5443
Ananth Nagar,4 Bedroom,600.0,3.0,62.0,4,10333
Hormavu,2 BHK,1210.0,2.0,69.0,2,5702
Whitefield,3 BHK,1660.0,3.0,110.0,3,6626
Malleshwaram,2 BHK,1410.0,2.0,125.0,2,8865
Electronics City Phase 1,2 BHK,1305.0,2.0,67.0,2,5134
other,6 BHK,1800.0,6.0,101.0,6,5611
Bannerghatta Road,2 BHK,1206.0,2.0,53.0,2,4394
Kambipura,2 BHK,883.0,2.0,49.0,2,5549
Bannerghatta Road,2 BHK,1150.0,2.0,55.0,2,4782
other,2 BHK,1100.0,2.0,65.0,2,5909
Banashankari Stage II,2 BHK,1245.0,2.0,125.0,2,10040
Cooke Town,3 BHK,1850.0,3.0,110.0,3,5945
Hoodi,4 Bedroom,3385.0,4.0,220.0,4,6499
Sompura,4 BHK,2150.0,3.0,85.0,4,3953
Hebbal,3 BHK,3895.0,3.0,390.0,3,10012
other,2 BHK,1035.0,2.0,55.0,2,5314
other,2 BHK,600.0,3.0,72.0,2,12000
other,3 BHK,1200.0,3.0,75.0,3,6250
Hennur Road,2 BHK,1155.0,2.0,69.18,2,5989
other,7 Bedroom,1200.0,6.0,235.0,7,19583
Electronics City Phase 1,1 BHK,755.0,1.0,30.12,1,3989
Ramamurthy Nagar,2 BHK,960.0,2.0,35.0,2,3645
other,1 BHK,450.0,1.0,20.0,1,4444
other,5 BHK,4000.0,5.0,680.0,5,17000
Raja Rajeshwari Nagar,3 BHK,1571.0,3.0,53.12,3,3381
Sarjapur  Road,3 BHK,1691.0,3.0,119.0,3,7037
Kanakapura,3 BHK,1489.0,2.0,70.0,3,4701
Kengeri Satellite Town,2 BHK,800.0,2.0,32.0,2,4000
Jalahalli,3 BHK,980.0,2.0,60.0,3,6122
Bannerghatta Road,2 BHK,1215.0,2.0,65.0,2,5349
Electronic City Phase II,2 BHK,911.0,2.0,26.5,2,2908
Bellandur,3 BHK,1535.0,3.0,90.0,3,5863
5th Block Hbr Layout,5 Bedroom,1200.0,5.0,205.0,5,17083
R.T. Nagar,3 BHK,1500.0,2.0,70.0,3,4666
Brookefield,2 BHK,1588.0,2.0,58.0,2,3652
Horamavu Banaswadi,4 Bedroom,675.0,4.0,68.0,4,10074
Kannamangala,2 BHK,957.0,2.0,56.0,2,5851
Electronic City Phase II,2 BHK,1200.0,2.0,34.65,2,2887
other,3 BHK,1495.0,2.0,52.33,3,3500
Bisuvanahalli,2 BHK,845.0,2.0,34.0,2,4023
Padmanabhanagar,4 BHK,3000.0,4.0,350.0,4,11666
Sarjapur  Road,3 BHK,2206.0,4.0,180.0,3,8159
Kasavanhalli,2 BHK,1181.0,2.0,61.0,2,5165
Sarjapur  Road,2 BHK,1309.0,2.0,69.5,2,5309
Horamavu Agara,3 Bedroom,1400.0,3.0,63.0,3,4500
other,5 Bedroom,1650.0,5.0,450.0,5,27272
Kanakpura Road,3 BHK,1622.0,3.0,95.0,3,5856
CV Raman Nagar,2 BHK,1225.0,2.0,48.0,2,3918
other,3 BHK,1719.0,3.0,95.0,3,5526
Abbigere,2 BHK,995.0,2.0,40.8,2,4100
Mahadevpura,2 BHK,1120.0,2.0,53.0,2,4732
Bellandur,2 BHK,1047.0,2.0,75.0,2,7163
Devanahalli,3 BHK,1466.0,3.0,59.96,3,4090
Sarjapur  Road,3 BHK,1691.0,3.0,100.0,3,5913
Chandapura,2 BHK,876.0,2.0,28.47,2,3250
CV Raman Nagar,3 BHK,1659.0,3.0,135.0,3,8137
Whitefield,2 BHK,1270.0,2.0,105.0,2,8267
8th Phase JP Nagar,3 BHK,1269.0,2.0,50.75,3,3999
Kammanahalli,2 BHK,1200.0,3.0,80.0,2,6666
Budigere,2 BHK,1153.0,2.0,56.5,2,4900
Whitefield,2 BHK,1315.0,2.0,69.5,2,5285
Hebbal Kempapura,4 Bedroom,2485.0,4.0,198.0,4,7967
Tumkur Road,3 BHK,1354.0,3.0,85.86,3,6341
Whitefield,4 Bedroom,1344.0,3.0,130.0,4,9672
Yelahanka,2 BHK,1180.0,2.0,55.0,2,4661
Jalahalli,2 BHK,1313.0,2.0,68.0,2,5178
Nehru Nagar,2 BHK,1100.0,2.0,50.0,2,4545
Nagarbhavi,4 Bedroom,600.0,4.0,78.0,4,13000
Basaveshwara Nagar,3 Bedroom,1200.0,3.0,190.0,3,15833
other,6 Bedroom,625.0,3.0,78.0,6,12480
Vishwapriya Layout,2 BHK,890.0,2.0,37.0,2,4157
Whitefield,2 BHK,1173.0,2.0,78.2,2,6666
other,2 BHK,1190.0,2.0,55.0,2,4621
Green Glen Layout,2 BHK,1250.0,2.0,75.0,2,6000
Hoodi,3 BHK,1644.0,3.0,80.51,3,4897
Yelahanka New Town,3 Bedroom,1740.0,3.0,150.0,3,8620
Jakkur,3 BHK,1710.0,3.0,110.0,3,6432
5th Phase JP Nagar,4 Bedroom,1000.0,4.0,130.0,4,13000
Hebbal,3 BHK,1662.0,3.0,155.0,3,9326
other,2 BHK,1162.0,2.0,48.0,2,4130
Whitefield,2 BHK,1362.0,2.0,85.0,2,6240
other,1 BHK,540.0,1.0,22.5,1,4166
Thanisandra,3 BHK,1533.0,3.0,75.885,3,4950
other,5 Bedroom,900.0,4.0,62.0,5,6888
Sarjapur  Road,4 Bedroom,1250.0,4.0,135.0,4,10800
Begur Road,3 BHK,1410.0,2.0,54.99,3,3900
Chikka Tirupathi,4 Bedroom,2665.0,5.0,125.0,4,4690
other,2 BHK,970.0,2.0,45.0,2,4639
other,6 BHK,1799.0,6.0,101.0,6,5614
other,6 Bedroom,1500.0,6.0,300.0,6,20000
Uttarahalli,4 Bedroom,1200.0,4.0,155.0,4,12916
Haralur Road,3 BHK,1810.0,3.0,97.83,3,5404
other,2 BHK,1084.0,2.0,51.0,2,4704
other,3 BHK,1500.0,3.0,85.0,3,5666
Uttarahalli,3 BHK,1328.0,2.0,56.0,3,4216
Marathahalli,3 BHK,1485.0,2.0,90.0,3,6060
Pai Layout,3 BHK,1500.0,2.0,58.0,3,3866
R.T. Nagar,2 BHK,1100.0,2.0,85.0,2,7727
other,3 BHK,1360.0,2.0,73.0,3,5367
1st Phase JP Nagar,2 BHK,900.0,2.0,75.0,2,8333
other,3 Bedroom,600.0,3.0,86.0,3,14333
Electronic City Phase II,2 BHK,1031.0,2.0,54.48,2,5284
Whitefield,2 BHK,955.0,2.0,38.19,2,3998
Uttarahalli,2 BHK,1040.0,2.0,36.4,2,3500
KR Puram,7 Bedroom,2800.0,6.0,110.0,7,3928
Kothannur,3 BHK,1404.0,2.0,56.15,3,3999
Haralur Road,2 BHK,1200.0,2.0,46.0,2,3833
Yelahanka,2 BHK,1267.0,3.0,78.0,2,6156
Bannerghatta Road,3 BHK,1510.0,2.0,110.0,3,7284
Kanakpura Road,2 BHK,900.0,2.0,42.0,2,4666
Whitefield,3 BHK,1457.0,2.0,105.0,3,7206
Iblur Village,4 BHK,3596.0,4.0,268.0,4,7452
8th Phase JP Nagar,3 BHK,1408.0,3.0,80.0,3,5681
Dodda Nekkundi,2 BHK,1005.0,2.0,45.0,2,4477
other,6 Bedroom,30400.0,4.0,1824.0,6,6000
Electronic City Phase II,2 BHK,545.0,1.0,28.0,2,5137
other,2 BHK,1182.0,2.0,64.0,2,5414
Talaghattapura,3 BHK,2038.5,3.0,122.0,3,5984
Sarjapur  Road,2 BHK,1367.0,2.0,80.0,2,5852
other,5 Bedroom,1200.0,4.0,200.0,5,16666
Judicial Layout,5 BHK,1100.0,4.0,199.0,5,18090
Sarjapur  Road,3 BHK,1565.0,3.0,78.0,3,4984
Whitefield,3 BHK,1720.0,3.0,128.0,3,7441
Doddathoguru,2 BHK,1107.0,2.0,44.0,2,3974
Yelahanka,2 BHK,1360.0,2.0,78.19,2,5749
Anekal,2 BHK,625.0,1.0,25.0,2,4000
Jalahalli,2 BHK,1701.0,2.0,145.0,2,8524
other,3 Bedroom,1350.0,1.0,120.0,3,8888
Hosa Road,2 BHK,1016.0,2.0,39.95,2,3932
other,3 BHK,1500.0,3.0,90.0,3,6000
Akshaya Nagar,2 BHK,900.0,2.0,45.0,2,5000
Sanjay nagar,3 BHK,1500.0,3.0,150.0,3,10000
other,3 BHK,2292.0,2.0,285.0,3,12434
Kereguddadahalli,3 BHK,1280.0,3.0,42.0,3,3281
other,4 Bedroom,1750.0,4.0,263.0,4,15028
Electronic City,3 BHK,1500.0,2.0,70.0,3,4666
Electronic City,3 BHK,1500.0,2.0,64.5,3,4300
Gottigere,2 BHK,1245.0,2.0,59.0,2,4738
Yelahanka,2 BHK,1025.0,2.0,44.0,2,4292
other,3 BHK,1945.0,3.0,135.0,3,6940
Rajaji Nagar,3 BHK,1640.0,3.0,245.0,3,14939
Hegde Nagar,3 BHK,1348.0,2.0,80.5,3,5971
Tindlu,2 BHK,1100.0,2.0,55.0,2,5000
Kammasandra,3 BHK,1616.0,3.0,40.0,3,2475
Yeshwanthpur,3 BHK,1855.0,3.0,135.0,3,7277
Hennur,2 BHK,1255.0,2.0,56.5,2,4501
HAL 2nd Stage,5 Bedroom,2040.0,4.0,500.0,5,24509
5th Phase JP Nagar,4 Bedroom,2400.0,4.0,228.0,4,9500
Kothanur,2 BHK,1195.0,2.0,59.0,2,4937
Bannerghatta Road,2 BHK,1022.0,2.0,35.0,2,3424
Hegde Nagar,6 Bedroom,760.0,6.0,98.0,6,12894
Dodda Nekkundi,2 BHK,1264.0,2.0,52.0,2,4113
Chamrajpet,3 BHK,1650.0,3.0,115.0,3,6969
Abbigere,2 BHK,1020.0,2.0,40.8,2,4000
Sarjapur  Road,4 BHK,4395.0,4.0,242.0,4,5506
Bommasandra,2 BHK,950.0,2.0,25.0,2,2631
other,4 BHK,2710.0,5.0,142.0,4,5239
Hennur Road,3 BHK,1936.0,3.0,131.0,3,6766
other,2 BHK,1075.0,2.0,58.0,2,5395
Hosur Road,2 BHK,1223.0,2.0,93.0,2,7604
Whitefield,3 Bedroom,1500.0,3.0,61.9,3,4126
Hennur,2 BHK,1050.0,2.0,42.73,2,4069
7th Phase JP Nagar,3 BHK,2100.0,3.0,190.0,3,9047
2nd Stage Nagarbhavi,5 Bedroom,1200.0,4.0,240.0,5,20000
Uttarahalli,3 Bedroom,1650.0,2.0,130.0,3,7878
Whitefield,2 BHK,1495.0,2.0,78.0,2,5217
JP Nagar,3 BHK,1500.0,2.0,82.0,3,5466
Koramangala,3 BHK,1605.0,3.0,260.0,3,16199
Whitefield,2 BHK,1270.0,2.0,52.0,2,4094
Kumaraswami Layout,2 BHK,1081.0,2.0,60.0,2,5550
Koramangala,3 BHK,1600.0,2.0,140.0,3,8750
Kanakpura Road,3 BHK,1596.0,3.0,118.0,3,7393
other,6 Bedroom,600.0,6.0,65.0,6,10833
Horamavu Agara,3 BHK,1557.0,3.0,70.0,3,4495
other,3 BHK,1410.0,3.0,68.0,3,4822
Sarjapur  Road,3 BHK,1984.0,4.0,148.0,3,7459
Kadugodi,2 BHK,1196.0,2.0,60.0,2,5016
Thanisandra,4 BHK,2259.0,3.0,112.0,4,4957
Kenchenahalli,3 BHK,1720.0,3.0,100.0,3,5813
8th Phase JP Nagar,4 Bedroom,600.0,5.0,99.0,4,16500
other,2 BHK,1256.0,2.0,62.8,2,5000
Haralur Road,2 BHK,1243.0,2.0,46.0,2,3700
Electronic City Phase II,3 BHK,1310.0,2.0,37.83,3,2887
7th Phase JP Nagar,2 BHK,1140.0,2.0,57.0,2,5000
Cooke Town,3 BHK,1600.0,3.0,260.0,3,16250
Kothanur,3 BHK,1790.0,3.0,120.0,3,6703
Harlur,2 BHK,1335.0,2.0,72.76,2,5450
Nehru Nagar,3 BHK,1674.0,3.0,81.0,3,4838
CV Raman Nagar,5 Bedroom,1200.0,2.0,100.0,5,8333
other,3 BHK,1786.0,3.0,79.99,3,4478
Vidyaranyapura,4 Bedroom,770.0,3.0,65.25,4,8474
Sarjapur  Road,2 BHK,1035.0,2.0,47.0,2,4541
Whitefield,3 BHK,1564.0,3.0,103.0,3,6585
Margondanahalli,2 BHK,1334.0,2.0,67.76,2,5079
other,10 Bedroom,7150.0,13.0,3600.0,10,50349
Bommasandra,3 BHK,1365.0,3.0,53.96,3,3953
Choodasandra,2 BHK,1115.0,2.0,50.0,2,4484
Jalahalli East,2 BHK,1020.0,2.0,42.48,2,4164
other,3 BHK,1369.0,2.0,72.0,3,5259
Whitefield,2 BHK,1240.0,2.0,41.0,2,3306
Nagarbhavi,4 Bedroom,600.0,3.0,100.0,4,16666
other,2 BHK,1418.0,2.0,62.0,2,4372
Hebbal,3 BHK,2250.0,3.0,219.0,3,9733
Shivaji Nagar,2 BHK,500.0,1.0,20.0,2,4000
KR Puram,2 BHK,1020.0,2.0,39.0,2,3823
Yelahanka New Town,2 BHK,1290.0,2.0,70.0,2,5426
Attibele,3 Bedroom,1500.0,3.0,90.0,3,6000
Billekahalli,3 BHK,1290.0,3.0,62.0,3,4806
Jalahalli,3 BHK,1881.0,3.0,115.0,3,6113
Jigani,2 BHK,943.0,2.0,49.5,2,5249
Electronic City,2 BHK,1140.0,2.0,32.93,2,2888
other,2 BHK,1035.0,2.0,60.0,2,5797
Whitefield,3 Bedroom,1500.0,3.0,71.0,3,4733
other,5 BHK,5665.84,7.0,988.0,5,17437
other,2 BHK,1232.0,2.0,94.28,2,7652
Uttarahalli,3 BHK,1290.0,2.0,56.12,3,4350
Malleshwaram,4 BHK,2610.0,4.0,306.0,4,11724
Anandapura,3 BHK,1576.0,3.0,58.7,3,3724
other,2 BHK,980.0,2.0,41.0,2,4183
Hebbal,4 BHK,4000.0,6.0,440.0,4,11000
Sarjapur  Road,2 BHK,1346.0,2.0,74.03,2,5500
Hoodi,2 BHK,1181.0,2.0,55.0,2,4657
other,4 BHK,6652.0,6.0,660.0,4,9921
other,2 BHK,1307.0,2.0,59.0,2,4514
other,3 BHK,1215.0,2.0,49.86,3,4103
Sanjay nagar,2 BHK,1180.0,2.0,80.0,2,6779
Amruthahalli,1 Bedroom,600.0,1.0,55.0,1,9166
Sarjapur  Road,2 BHK,1115.0,2.0,50.0,2,4484
other,2 BHK,1025.0,2.0,43.04,2,4199
Ramagondanahalli,3 BHK,1910.0,3.0,131.0,3,6858
Gottigere,2 BHK,1200.0,2.0,50.0,2,4166
other,4 BHK,2920.0,4.0,536.0,4,18356
Hebbal,2 BHK,1349.0,2.0,98.0,2,7264
Electronics City Phase 1,2 BHK,1175.0,2.0,60.0,2,5106
2nd Phase Judicial Layout,2 BHK,1150.0,2.0,40.25,2,3500
Old Madras Road,2 BHK,935.0,2.0,32.72,2,3499
Whitefield,3 BHK,1404.0,2.0,59.0,3,4202
other,4 Bedroom,2700.0,3.0,230.0,4,8518
HSR Layout,2 BHK,1203.0,2.0,60.0,2,4987
Nagavara,3 Bedroom,440.0,3.0,35.0,3,7954
Somasundara Palya,3 BHK,1571.0,3.0,63.0,3,4010
other,2 BHK,1125.0,2.0,46.0,2,4088
Kammasandra,1 BHK,610.0,1.0,18.5,1,3032
Ambalipura,2 BHK,1198.0,2.0,80.0,2,6677
Chandapura,3 Bedroom,1200.0,3.0,65.0,3,5416
other,2 BHK,1200.0,2.0,45.0,2,3750
other,4 Bedroom,6688.0,6.0,700.0,4,10466
other,3 BHK,1830.0,3.0,95.0,3,5191
Kudlu Gate,3 BHK,1432.0,2.0,61.11,3,4267
Somasundara Palya,3 BHK,2300.0,2.0,70.0,3,3043
9th Phase JP Nagar,2 BHK,1331.95,2.0,69.0,2,5180
Yeshwanthpur,3 BHK,1876.0,3.0,160.0,3,8528
Whitefield,3 BHK,1639.0,3.0,107.0,3,6528
Akshaya Nagar,4 Bedroom,1200.0,4.0,125.0,4,10416
Kanakpura Road,3 BHK,1665.0,3.0,74.9,3,4498
Kogilu,2 BHK,1140.0,2.0,50.66,2,4443
Electronic City,4 BHK,2093.0,4.0,134.0,4,6402
Kanakpura Road,3 BHK,1450.0,3.0,60.9,3,4200
TC Palaya,3 BHK,1330.0,3.0,46.56,3,3500
other,4 Bedroom,1200.0,5.0,350.0,4,29166
Talaghattapura,3 Bedroom,1800.0,3.0,84.0,3,4666
other,4 BHK,2872.0,4.0,183.0,4,6371
other,3 BHK,1486.0,2.0,65.0,3,4374
Chikkalasandra,3 BHK,1425.0,1.0,54.0,3,3789
Raja Rajeshwari Nagar,2 BHK,1255.0,2.0,42.54,2,3389
Electronic City,2 BHK,750.0,2.0,19.5,2,2600
Nagarbhavi,2 BHK,1225.0,2.0,58.0,2,4734
Badavala Nagar,2 BHK,1274.0,2.0,90.0,2,7064
other,9 BHK,4600.0,9.0,150.0,9,3260
Kathriguppe,3 BHK,1350.0,3.0,80.99,3,5999
other,2 BHK,1100.0,2.0,64.08,2,5825
Whitefield,3 BHK,1760.0,3.0,139.0,3,7897
Bommasandra,3 Bedroom,1200.0,3.0,110.0,3,9166
other,7 BHK,1800.0,5.0,65.0,7,3611
Chandapura,3 BHK,1095.0,2.0,28.0,3,2557
HSR Layout,2 BHK,1140.0,2.0,46.0,2,4035
Raja Rajeshwari Nagar,3 BHK,1530.0,3.0,64.86,3,4239
Kundalahalli,3 BHK,1724.0,3.0,140.0,3,8120
Horamavu Agara,2 BHK,1058.0,2.0,47.0,2,4442
other,3 BHK,1464.0,3.0,56.0,3,3825
other,3 BHK,1975.0,3.0,90.0,3,4556
Sarjapur  Road,2 BHK,1112.0,2.0,58.0,2,5215
other,2 Bedroom,1600.0,2.0,90.0,2,5625
Vittasandra,3 BHK,1648.0,3.0,85.0,3,5157
other,2 BHK,1100.0,2.0,32.0,2,2909
Hoskote,1 BHK,509.0,1.0,23.0,1,4518
Talaghattapura,3 BHK,2265.0,3.0,159.0,3,7019
Malleshpalya,2 BHK,1200.0,2.0,46.0,2,3833
other,1 BHK,530.0,1.0,18.0,1,3396
other,2 BHK,1010.0,2.0,41.0,2,4059
Kaggadasapura,2 BHK,1150.0,2.0,45.0,2,3913
Sarjapur  Road,2 BHK,1340.0,2.0,75.0,2,5597
Raja Rajeshwari Nagar,3 BHK,1850.0,3.0,98.0,3,5297
Doddathoguru,2 BHK,907.0,2.0,40.0,2,4410
Benson Town,2 Bedroom,1688.0,2.0,280.0,2,16587
Ulsoor,2 Bedroom,1160.0,2.0,130.0,2,11206
Pai Layout,3 BHK,1600.0,2.0,65.0,3,4062
other,6 Bedroom,1800.0,5.0,140.0,6,7777
Iblur Village,4 BHK,3596.0,5.0,251.0,4,6979
Hosur Road,3 BHK,1464.0,3.0,56.0,3,3825
Hennur,2 BHK,1295.0,2.0,62.0,2,4787
other,2 BHK,1012.0,2.0,62.0,2,6126
Yelahanka New Town,3 BHK,1000.0,2.0,48.0,3,4800
Channasandra,2 BHK,1010.0,2.0,32.0,2,3168
Bharathi Nagar,2 BHK,1379.0,2.0,85.0,2,6163
Kasavanhalli,2 BHK,1121.0,2.0,75.0,2,6690
LB Shastri Nagar,2 BHK,1200.0,2.0,75.0,2,6250
Sarakki Nagar,4 BHK,3124.0,6.0,349.0,4,11171
Bannerghatta Road,2 BHK,1246.0,2.0,47.35,2,3800
7th Phase JP Nagar,3 BHK,1420.0,3.0,90.0,3,6338
Shivaji Nagar,2 BHK,600.0,1.0,65.0,2,10833
other,2 BHK,1090.0,2.0,55.0,2,5045
Electronic City,2 BHK,1355.0,2.0,73.0,2,5387
Ambedkar Nagar,2 BHK,1425.0,2.0,94.0,2,6596
Chikkabanavar,8 Bedroom,4000.0,7.0,110.0,8,2750
Electronics City Phase 1,3 BHK,1700.0,3.0,111.0,3,6529
Talaghattapura,3 BHK,2099.0,3.0,134.0,3,6383
Yelahanka,5 Bedroom,1330.0,5.0,210.0,5,15789
Devanahalli,3 BHK,1520.0,2.0,69.76,3,4589
other,3 BHK,1490.0,3.0,140.0,3,9395
Horamavu Banaswadi,2 BHK,1000.0,2.0,50.0,2,5000
Sahakara Nagar,2 BHK,1291.0,2.0,72.3,2,5600
Kanakpura Road,2 BHK,1339.0,2.0,67.0,2,5003
other,2 BHK,1330.0,2.0,56.0,2,4210
other,4 Bedroom,9200.0,4.0,2600.0,4,28260
other,8 Bedroom,1200.0,8.0,140.0,8,11666
other,2 BHK,1650.0,1.0,130.0,2,7878
other,6 Bedroom,8000.0,6.0,2800.0,6,35000
Jakkur,2 BHK,1125.0,2.0,44.75,2,3977
CV Raman Nagar,2 BHK,1310.0,2.0,62.0,2,4732
Bellandur,2 BHK,1096.0,2.0,40.0,2,3649
Dasanapura,2 BHK,814.0,2.0,43.7,2,5368
Sarjapur  Road,3 Bedroom,3500.0,3.0,275.0,3,7857
Raja Rajeshwari Nagar,2 BHK,1140.0,2.0,44.0,2,3859
other,3 BHK,1480.0,2.0,75.0,3,5067
Hebbal,4 BHK,4000.0,6.0,370.0,4,9250
other,3 BHK,1500.0,2.0,78.0,3,5200
Sarjapur  Road,3 BHK,1850.0,3.0,89.0,3,4810
other,2 BHK,745.0,2.0,36.0,2,4832
Whitefield,1 BHK,613.0,1.0,48.0,1,7830
other,4 Bedroom,1200.0,5.0,325.0,4,27083
Bannerghatta Road,3 BHK,1470.0,2.0,85.0,3,5782
Frazer Town,1 Bedroom,896.0,1.0,100.0,1,11160
other,1 BHK,250.0,2.0,40.0,1,16000
other,6 Bedroom,1200.0,3.0,125.0,6,10416
Bannerghatta Road,2 BHK,1160.0,2.0,45.0,2,3879
Laggere,7 Bedroom,1590.0,9.0,132.0,7,8301
Attibele,1 BHK,520.0,1.0,15.0,1,2884
other,9 Bedroom,1178.0,9.0,75.0,9,6366
Electronic City,3 BHK,1360.0,2.0,64.99,3,4778
other,2 BHK,1155.0,2.0,64.0,2,5541
Chandapura,1 BHK,520.0,1.0,14.04,1,2700
Ambalipura,3 BHK,1615.0,2.0,150.0,3,9287
Raja Rajeshwari Nagar,8 Bedroom,6000.0,8.0,215.0,8,3583
Raja Rajeshwari Nagar,2 BHK,1140.0,2.0,39.0,2,3421
other,3 BHK,1508.0,3.0,77.0,3,5106
Ambedkar Nagar,3 BHK,2395.0,4.0,150.0,3,6263
Uttarahalli,3 BHK,1590.0,3.0,57.0,3,3584
Thigalarapalya,3 BHK,2215.0,4.0,152.0,3,6862
R.T. Nagar,3 BHK,1667.0,3.0,130.0,3,7798
other,3 BHK,1903.0,2.0,293.0,3,15396
7th Phase JP Nagar,2 BHK,1530.0,2.0,108.0,2,7058
Whitefield,3 BHK,1730.0,3.0,125.0,3,7225
JP Nagar,2 BHK,1048.0,2.0,44.0,2,4198
Malleshpalya,2 BHK,1225.0,2.0,52.0,2,4244
other,2 BHK,1200.0,2.0,70.0,2,5833
Kundalahalli,2 BHK,1175.0,2.0,65.0,2,5531
Electronic City Phase II,3 BHK,1651.0,3.0,49.53,3,3000
8th Phase JP Nagar,3 BHK,1500.0,2.0,55.0,3,3666
Doddathoguru,3 BHK,1783.0,3.0,85.0,3,4767
Kereguddadahalli,2 BHK,1015.0,2.0,35.0,2,3448
other,1 BHK,1800.0,1.0,200.0,1,11111
Sarjapur  Road,4 Bedroom,2758.0,4.0,240.0,4,8701
other,4 Bedroom,1200.0,5.0,325.0,4,27083
Sarjapur  Road,2 BHK,1132.0,2.0,70.0,2,6183
KR Puram,2 Bedroom,1200.0,2.0,75.0,2,6250
Hoodi,3 BHK,1490.0,3.0,85.0,3,5704
Thanisandra,4 BHK,1917.0,4.0,130.0,4,6781
Bhoganhalli,3 BHK,1703.0,3.0,120.0,3,7046
Old Madras Road,2 BHK,1211.0,2.0,45.0,2,3715
Kudlu Gate,2 BHK,1310.0,2.0,53.0,2,4045
Marathahalli,2 BHK,1146.0,2.0,69.0,2,6020
Kudlu,2 BHK,1041.0,2.0,53.0,2,5091
Whitefield,3 BHK,1453.0,2.0,58.0,3,3991
Whitefield,1 BHK,877.0,1.0,59.0,1,6727
Kanakapura,3 BHK,1477.0,2.0,69.5,3,4705
Frazer Town,2 BHK,1420.0,2.0,120.0,2,8450
other,3 Bedroom,2000.0,2.0,360.0,3,18000
other,2 BHK,1140.0,1.0,185.0,2,16228
Sarjapur  Road,3 BHK,1380.0,2.0,55.0,3,3985
other,2 BHK,1095.0,2.0,57.0,2,5205
Hoodi,2 BHK,1258.5,2.0,59.135,2,4698
Varthur,3 BHK,1665.0,3.0,71.58,3,4299
KR Puram,2 BHK,1245.0,2.0,60.0,2,4819
EPIP Zone,4 BHK,3360.0,5.0,221.0,4,6577
Electronic City,3 Bedroom,2010.0,3.0,201.0,3,10000
Shivaji Nagar,3 BHK,1226.0,2.0,60.0,3,4893
Uttarahalli,2 BHK,1075.0,2.0,46.76,2,4349
Thanisandra,2 BHK,1226.0,2.0,65.59,2,5349
Raja Rajeshwari Nagar,1 BHK,510.0,1.0,22.0,1,4313
Budigere,2 BHK,1153.0,2.0,60.0,2,5203
Kothannur,4 Bedroom,1600.0,4.0,45.0,4,2812
LB Shastri Nagar,2 BHK,1000.0,2.0,49.5,2,4950
other,7 Bedroom,1400.0,7.0,218.0,7,15571
other,2 BHK,1256.0,2.0,65.0,2,5175
other,6 Bedroom,1200.0,5.0,130.0,6,10833
Sarjapur,3 BHK,1425.0,3.0,57.0,3,4000
Margondanahalli,5 Bedroom,1375.0,5.0,125.0,5,9090
Hosur Road,3 BHK,1919.0,3.0,117.0,3,6096
Neeladri Nagar,3 BHK,2111.0,3.0,103.0,3,4879
Bannerghatta Road,2 BHK,970.0,2.0,57.0,2,5876
other,2 BHK,1353.0,2.0,110.0,2,8130
Jalahalli,3 BHK,1405.0,2.0,85.0,3,6049
Gubbalala,2 BHK,1285.0,2.0,90.0,2,7003
Mahadevpura,2 BHK,1050.0,2.0,42.0,2,4000
Hebbal,2 BHK,1349.0,2.0,96.8,2,7175
Sarjapur  Road,4 BHK,4050.0,2.0,450.0,4,11111
other,1 Bedroom,812.0,1.0,26.0,1,3201
other,3 BHK,1440.0,2.0,63.93,3,4439
Sarjapur  Road,4 BHK,2425.0,5.0,195.0,4,8041
Sultan Palaya,4 BHK,2200.0,3.0,80.0,4,3636
Haralur Road,3 BHK,1810.0,3.0,112.0,3,6187
Cox Town,2 BHK,1200.0,2.0,140.0,2,11666
Electronic City,2 BHK,1060.0,2.0,52.0,2,4905
Kenchenahalli,2 BHK,1015.0,2.0,60.0,2,5911
Whitefield,4 BHK,2856.0,5.0,154.5,4,5409
Hosakerehalli,5 Bedroom,1500.0,6.0,145.0,5,9666
Kothanur,3 BHK,1454.0,3.0,71.5,3,4917
other,2 BHK,1075.0,2.0,48.0,2,4465
Vidyaranyapura,5 Bedroom,774.0,5.0,70.0,5,9043
Raja Rajeshwari Nagar,2 BHK,1187.0,2.0,40.14,2,3381
Hulimavu,1 BHK,500.0,1.0,220.0,1,44000
other,4 Bedroom,1200.0,5.0,325.0,4,27083
Billekahalli,3 BHK,1805.0,3.0,134.0,3,7423
Bannerghatta Road,3 BHK,1527.0,3.0,142.0,3,9299
Yeshwanthpur,3 BHK,1675.0,3.0,92.13,3,5500
Rachenahalli,2 BHK,1050.0,2.0,52.71,2,5020
Ramamurthy Nagar,7 Bedroom,1500.0,9.0,250.0,7,16666
Bellandur,2 BHK,1262.0,2.0,47.0,2,3724
Uttarahalli,3 BHK,1345.0,2.0,57.0,3,4237
Green Glen Layout,3 BHK,1715.0,3.0,112.0,3,6530
Whitefield,5 Bedroom,3453.0,4.0,231.0,5,6689
other,4 BHK,3600.0,5.0,400.0,4,11111
Raja Rajeshwari Nagar,2 BHK,1141.0,2.0,60.0,2,5258
Padmanabhanagar,4 BHK,4689.0,4.0,488.0,4,10407
Doddathoguru,1 BHK,550.0,1.0,17.0,1,3090




    
