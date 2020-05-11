from sklearn.cluster import KMeans
from sklearn.decomposition import KernelPCA, PCA
from sklearn.manifold import TSNE

class Clustering():
	def __init__(self):
		self.pca = KernelPCA(1024, 'rbf', n_jobs=16)
		self.pca2 = PCA(64)
		self.kmeans = KMeans(n_clusters=2, n_jobs=-1)
	
	def fit(self, X):
		X = self.pca.fit_transform(X)
		X = self.pca2.fit_transform(X)
		X = TSNE(n_components=2, n_jobs=32, random_state=880531).fit_transform(X)
		self.kmeans.fit(X)
		return self
	
	def predict(self, X):
		X = self.pca.transform(X)
		X = self.pca2.transform(X)
		X = TSNE(n_components=2, n_jobs=8, random_state=880531).fit_transform(X)
		return self.kmeans.predict(X)
