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

class GeneralClustering():
	def __init__(self, transforms):
		self.transforms = transforms
	
	def fit(self, X):
		for trsfm in self.transforms:
			print(f'fitting {type(trsfm)}...')
			X = trsfm.fit_transform(X)
		return self

	def fit_predict(self, X):
		for trsfm in self.transforms[:-1]:
			print(f'fitting {type(trsfm)}...')
			X = trsfm.fit_transform(X)
		
		print(f'fitting {type(self.transforms[-1])}...')
		out = self.transforms[-1].fit_predict(X)
		return self, out

	def predict(self, X):
		for trsfm in self.transforms[:-1]:
			if isinstance(trsfm, TSNE):
				X = trsfm.fit_transform(X)
			else:
				X = trsfm.transform(X)
		out = self.transforms[-1].predict(X)
		return out

	def fit_transform(self, X):
		# does not contain last transform
		for trsfm in self.transforms[:-1]:
			print(f'fitting {type(trsfm)}...')
			X = trsfm.fit_transform(X)
		return self, X

	def transform(self, X):
		# does not contain last transform
		for trsfm in self.transforms[:-1]:
			if isinstance(trsfm, TSNE):
				X = trsfm.fit_transform(X)
			else:
				X = trsfm.transform(X)
		return self, X

