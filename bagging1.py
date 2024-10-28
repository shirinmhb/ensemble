import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from random import choice
from copy import deepcopy
from scipy import stats

class Bagging:
  def __init__(self, name, sp):
    self.T = 21
    with open(name) as f:
      content = [line.strip().split(sp) for line in f] 
    content = np.array(content)
    np.random.shuffle(content)
    # content = content.astype(np.float)
    last = len(content[0]) - 1
    X = content[:,:last]
    numTrain = round(.7 * len(content))
    X = X.astype(np.float)
    self.X = X[:numTrain]
    self.XTest = X[numTrain:]
    Y = content[:,last:]
    if name == 'Sonar.txt':
      Y = np.where(Y == 'M', 1, Y)
      Y = np.where(Y == 'R', -1, Y)

    elif name == 'Ionosphere.txt':
      Y = np.where(Y == 'b', 1, Y)
      Y = np.where(Y == 'g', -1, Y)

    Y = Y.astype(np.float)
    self.Y = Y[:numTrain]
    self.YTest = Y[numTrain:]

  def bagging_alg(self, X, Y, XTest, YTest):
    hypoList = []
    m = len(X)
    index = list(range(m))
    for t in range(self.T):
      xDataSet = []
      yDataSet = []
      for k in range(m):
        ind = choice(index)
        xDataSet.append(X[ind])
        yDataSet.append(Y[ind])
      xDataSet = np.array(xDataSet)
      yDataSet = np.array(yDataSet)
      tree = DecisionTreeClassifier(criterion="entropy", max_depth=6)
      hypo = tree.fit(xDataSet,yDataSet)
      hypoList.append(hypo)
      # predictions = hypo.predict(xDataSet)
      # print(accuracy_score(yDataSet, predictions))

    pred = np.zeros(YTest.shape)
    for k in range(len(XTest)):
      temp = 0
      voteList = []
      for t in range(self.T):
        p = hypoList[t].predict(XTest[k].reshape(1, -1))
        voteList.append(p[0])
      pred[k][0] = max(set(voteList), key=voteList.count)
    acc = accuracy_score(YTest, pred)
    return acc * 100

  def make_noisy_data(self, X, noisePercent):
    _, numFeatures= X.shape
    numNoise = round(numFeatures * noisePercent * 0.01)
    features = list(range(numFeatures))
    selected = []
    for i in range(numNoise):
      selected.append(choice(features))
    for i in range(len(X)):
      for k in selected:
        X[i][k] = X[i][k] + np.random.normal(0, 1)
 
def cal_accs(name, spt):
  accs = 0
  for _ in range(10): 
    ada = Bagging(name, spt)
    accs+= ada.bagging_alg(ada.X, ada.Y, ada.XTest, ada.YTest)
  print("accuracy", accs / 10, name, "dataset")
  accs = 0
  for _ in range(10): 
    ada = Bagging(name, spt)
    Xnoisy = deepcopy(ada.X)
    ada.make_noisy_data(Xnoisy, 10)
    accs += ada.bagging_alg(Xnoisy, ada.Y, ada.XTest, ada.YTest)
  print("accuracy noisy", accs / 10, name, "dataset 10 percent noise")
  accs = 0
  for _ in range(10): 
    ada = Bagging(name, spt)
    Xnoisy = deepcopy(ada.X)
    ada.make_noisy_data(Xnoisy, 20)
    accs += ada.bagging_alg(Xnoisy, ada.Y, ada.XTest, ada.YTest)
  print("accuracy noisy", accs / 10, name, "dataset 20 percent noise")
  accs = 0
  for _ in range(10): 
    ada = Bagging(name, spt)
    Xnoisy = deepcopy(ada.X)
    ada.make_noisy_data(Xnoisy, 30)
    accs += ada.bagging_alg(Xnoisy, ada.Y, ada.XTest, ada.YTest)
  print("accuracy noisy", accs / 10, name, "dataset 30 percent noise")

cal_accs('Diabetes.txt', '	')
cal_accs('Ionosphere.txt', ',')
cal_accs('Sonar.txt', ',')
cal_accs('Wine.txt', ',')
cal_accs('Glass.txt', '	')
cal_accs('BreastTissue.txt', '	')