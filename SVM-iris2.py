# 「データサイエンティスト養成読本　機械学習入門編」P.132より

from sklearn import datasets
from sklearn import svm
from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt


iris = datasets.load_iris()

# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
pca = PCA(n_components=2)
# n_componentsは主成分の数。つまり圧縮後の次元
data = pca.fit(iris.data).transform(iris.data)

datamax = data.max(axis=0)
datamin = data.min(axis=0)

n = 200
#等高線を書きたいのでそのためのメッシュを生成
X, Y = np.meshgrid(np.linspace(datamin[0]-1, datamax[0]+1, n),
	np.linspace(datamin[1]-1, datamax[1]+1, n) )

svc = svm.SVC(verbose=True)
# カーネルはデフォルトでRBFカーネル(ガウシアンカーネル)
svc.fit(data, iris.target)
Z = svc.predict(np.c_[X.ravel(), Y.ravel() ])

# 等高線にラベルをつけてみる
cont = plt.contour(X, Y, Z.reshape(X.shape), colors = "k" )
cont.clabel(fmt = '%1.2f')

for c, s in zip([0, 1, 2], ["o", "+", "x"]):
	d = data[iris.target == c]
	plt.scatter( d[:, 0], d[:, 1], c="k", marker=s)
plt.show()

"""

print(scores)
print("Accuracy:", scores.mean() )


"""