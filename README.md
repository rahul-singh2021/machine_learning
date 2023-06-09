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
     
    
# 18:Outlier Removal using Standard Deviation
    This program demonstrates how to remove outliers from a dataset using the standard deviation method. It utilizes Python and various 
    libraries such as Pandas, NumPy, and Matplotlib to perform the outlier removal and visualize the results.This program demonstrates 
    how to remove outliers from a dataset using the standard deviation method. It utilizes Python and various 
    libraries such as Pandas, NumPy, and Matplotlib to perform the outlier removal and visualize the results.

## Installation

    1. Clone the repository:
    git clone https://github.com/rahul-singh21/machine_learning.git

    2. Install the required dependencies:
    pip install pandas numpy matplotlib scipy

## Usage

    1. Ensure you have the dataset file named `heights.csv` in the same directory as the program.

    2. Run the program:    
      python outlier_removal.py

    3. The program will read the dataset, plot a histogram of the heights, and remove outliers using the standard deviation method.
    

# 19:Outlier Removal using IQR (Interquartile Range)
    This program utilizes the concept of Interquartile Range (IQR) to identify and remove outliers from a dataset. The IQR is a 
    statistical measure that represents the range between the first quartile (Q1) and the third quartile (Q3) of a dataset.

## Requirements
    Python 3.x
    pandas library    
    Installation
    Make sure you have Python installed on your system. You can download it from the official Python website:             
    https://www.python.org/downloads/

    Install the pandas library by running the following command in your terminal or command prompt:

    pip install pandas
## Usage
    Import the required libraries:

        import pandas as pd
        Read the dataset into a pandas DataFrame. Replace "/content/heights.csv" with the path to your dataset file:

            name     ,height
            mohan    ,1.2
            maria    ,2.3
            sakib    ,4.9
            tao      ,5.1
            virat    ,5.2
            khusbu   ,5.4
            dmitry   ,5.5
            selena   ,5.5
            john     ,5.6
            imran    ,5.6
            jose     ,5.8
            deepika  ,5.9
            yoseph   ,6
            binod    ,6.1
            gulshan  ,6.2
            johnson  ,6.5
            donald   ,7.1
            aamir    ,14.5
            ken      ,23.2
            Liu      ,40.2


# 20:YOLOv8 Object Detection Program

    This program demonstrates object detection using the YOLOv8 model with the Ultralytics library.

## Requirements

    - NVIDIA GPU with CUDA support (Tesla T4 in this example)
    - Python 3.10.12
    - torch 2.0.1+cu118
    - CUDA 12.0

## Installation

    1. Clone the repository:

    git clone <git clone https://github.com/rahul-singh21/machine_learning.git>
    
    2. Install the required dependencies:

    pip install ultralytics==8.0.20

## Usage

1. Run the program:

    python program.py

    The program will download the YOLOv8 model, perform object detection on an example image, and display the results.

