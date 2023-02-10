import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import time
import warnings
warnings.filterwarnings("ignore")
from sklearn import metrics

data = pd.read_csv('D:\\UCM\\1st Semester\\Neural Network & Deep Learning\\ICP5\\NNDL_Code and Data\\glass.csv')
print(data.shape)
X_train, X_test = train_test_split(data, test_size=0.2, random_state=int(time.time()))

features = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]

gauss = GaussianNB()

gauss.fit(X_train[features].values,X_train["Type"])

y_pred = gauss.predict(X_test[features])

print("Naïve Bayes\nTotal number of points: {}\nMislabeled points : {}\nAccuracy {:05.2f}%\n\n"
      .format(
          X_test.shape[0],
          (X_test["Type"] != y_pred).sum(),
          100 * (1 - (X_test["Type"] != y_pred).sum() / X_test.shape[0])
      ))


print(metrics.classification_report(X_test["Type"], y_pred))