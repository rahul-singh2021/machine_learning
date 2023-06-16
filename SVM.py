{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMRnBivpYT7OwsQ5lPfNd7N",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/rahul-singh2021/machine_learning/blob/main/SVM.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 14,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4tNx1F1xV6Ew",
        "outputId": "2fa5a4b3-8edf-49b9-e940-9a0279642a36"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.9666666666666667"
            ]
          },
          "metadata": {},
          "execution_count": 14
        }
      ],
      "source": [
        "import pandas as pd\n",
        "from sklearn.datasets import load_iris\n",
        "iris=load_iris()\n",
        "\n",
        "# adding target column to dataset\n",
        "df=pd.DataFrame(iris.data,columns=iris.feature_names)\n",
        "df['target'] = iris.target\n",
        "\n",
        "# target names\n",
        "iris.target_names\n",
        "\n",
        "# add flower name column to dataframe according to target column\n",
        "df['flower_name']=df.target.apply(lambda x:iris.target_names[x])\n",
        "\n",
        "# splitting the data in training set and test set\n",
        "from sklearn.model_selection import train_test_split\n",
        "x = df.drop(['target','flower_name'], axis='columns')\n",
        "y=df.target\n",
        "x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)\n",
        "\n",
        "# svm model\n",
        "from sklearn.svm import SVC\n",
        "model=SVC()\n",
        "model.fit(x_train,y_train)\n",
        "model.score(x_test,y_test)"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "\n"
      ],
      "metadata": {
        "id": "353hGqrmWRul"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "nfmlyk2TWqdp"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}
