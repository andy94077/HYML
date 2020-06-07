from sklearn.cluster import KMeans
from sklearn.decomposition import KernelPCA, PCA
from sklearn.manifold import TSNE

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
			print(f'predicting {type(trsfm)}...')
			if isinstance(trsfm, TSNE):
				X = trsfm.fit_transform(X)
			else:
				X = trsfm.transform(X)
		print(f'predicting {type(self.transforms[-1])}...')
		out = self.transforms[-1].predict(X)
		return out

	def fit_transform(self, X):
		# does not contain last transform
		for trsfm in self.transforms[:-1]:
			print(f'fitting {type(trsfm)}...')
			X = trsfm.fit_transform(X)
		print(f'fitting {type(self.transforms[-1])}...')
		pred = self.transforms[-1].fit_predict(X)
		return self, X, pred

	def transform(self, X):
		# does not contain last transform
		for trsfm in self.transforms[:-1]:
			if isinstance(trsfm, TSNE):
				X = trsfm.fit_transform(X)
			else:
				X = trsfm.transform(X)
		return X

