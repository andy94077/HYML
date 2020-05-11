from sklearn.cluster import KMeans
from sklearn.decomposition import KernelPCA, PCA
from sklearn.manifold import TSNE

class Clustering():
	def __init__(self):
		self.seed = int(1e9+7)
		self.pca = KernelPCA(1024, 'rbf', n_jobs=-1, random_state=self.seed)
		self.pca2 = KernelPCA(64, 'rbf', n_jobs=-1, random_state=self.seed)
		self.kmeans = KMeans(n_clusters=2, n_jobs=-1, random_state=self.seed)
	
	def fit(self, X):
		print('fitting pca...')
		X = self.pca.fit_transform(X)
		print('fitting pca2...')
		X = self.pca2.fit_transform(X)
		print('fitting TSNE...')
		X = TSNE(n_components=2, n_jobs=-1, random_state=self.seed).fit_transform(X)
		self.kmeans.fit(X)
		return self
	
	def fit_predict(self, X):
		print('fitting pca...')
		X = self.pca.fit_transform(X)
		print('fitting pca2...')
		X = self.pca2.fit_transform(X)
		print('fitting TSNE...')
		X = TSNE(n_components=2, n_jobs=-1, random_state=self.seed).fit_transform(X)
		out = self.kmeans.fit_predict(X)
		return self, out

	def predict(self, X):
		X = self.pca.transform(X)
		X = self.pca2.transform(X)
		X = TSNE(n_components=2, n_jobs=-1, random_state=self.seed).fit_transform(X)
		return self.kmeans.predict(X)
