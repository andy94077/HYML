from sklearn.cluster import KMeans
from sklearn.decomposition import KernelPCA
#from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE

class Clustering():
	def __init__(self, dimension_reduction=False):
		self.dimension_reduction = dimension_reduction
		if dimension_reduction:
			self.pca = KernelPCA(200, 'rbf', n_jobs=-1, random_state=880301)
		self.kmeans = KMeans(n_clusters=2, n_jobs=-1, random_state=880301)
	
	def fit(self, X):
		if self.dimension_reduction:
			X = self.pca.fit_transform(X)
			X = TSNE(n_components=2, n_jobs=-1, random_state=880301).fit_transform(X)
		self.kmeans.fit(X)
		return self
	
	def predict(self, X):
		if self.dimension_reduction:
			X = self.pca.transform(X)
			X = TSNE(n_components=2, n_jobs=-1, verbose=1).fit_transform(X)
		return self.kmeans.predict(X)
