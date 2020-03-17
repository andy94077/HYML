import numpy as np

class DecisionTree():
	def __init__(self):
		self.fitted = False
		self.b = (0,0) #(i,theta,majority y(if base condition is meeted))
		self.left = -1
		self.right = 1
		self.height = 1
	
	def __str__(self):
		string = ''
		preorder = [self]
		while preorder:
			item = preorder.pop()
			if isinstance(item, DecisionTree):
				string += '(%d, %.3f)\n' % item.b
				preorder.append(item.right)
				preorder.append(item.left)
			else:
				string += str(item) + '\n'
		return string[:-1] #no newline in the end
	
	def g_func(self, X, b=None):
		if b is None:
			b = self.b
		#plus 1e-7 to prevent np.sum(Y) = 0, which will cause np.sign() = 0
		return np.sign(X[:, b[0]] - b[1] + 1e-7).astype(int).ravel()
	
	def __err_gini(self, b, X, Y):
		result = self.g_func(X,b)
		D = [Y[result == -1], Y[result == 1]]
		return 2 * np.sum([np.sum(Dc == 1) * (1 - np.mean(Dc == 1)) for Dc in D if Dc.shape[0] != 0])
	
	def __find_b(self, X, Y):
		if np.all(Y == Y[0]):
			self.majority_y = Y[0]
			return (0, -np.inf), True
		elif np.all(X == X[0]):  #all X are the same
			#plus 0.1 to prevent np.sum(Y) = 0, which will cause np.sign() = 0
			self.majority_y = np.sign(np.sum(Y) + 0.1).astype(np.int32)
			return (0, -np.inf), True
		else:
			min_err = np.inf
			min_err_b = (0, -np.inf)
			for i in range(X.shape[1]):
				Xi = np.unique(X[:, i])
				thetas = np.concatenate(([-np.inf], (Xi[:-1] + Xi[1:]) / 2))
				for theta in thetas:
					err = self.__err_gini((i, theta), X, Y)
					if err < min_err:
						min_err = err
						min_err_b = (i, theta)
			if min_err_b[1] == -np.inf:
				self.majority_y = np.sign(np.sum(Y) + 0.1).astype(np.int32)
			return min_err_b, min_err_b[1] == -np.inf # if min_err_b[1]==-np.inf, it means 'b' cannot split X into 2 partition
							
	def fit(self, X, Y, h=-1):
		'''
		@parameters:
			X, Y: data
			h: maximum height (the number of ancesters of the deepest node + 1), if h == -1, no constraint
		@return: self
		'''
		
		if h == 1:
			self.b = (0, -np.inf)
			#plus 0.1 to prevent np.sum(Y) = 0, which will cause np.sign() = 0
			self.majority_y = int(np.sign(np.sum(Y) + 0.1))
			self.left, self.right = -self.majority_y, self.majority_y
			self.fitted = True
			return self

		#since self.b does not classify anything, we turn self.fitted into False in order to fit X later
		if self.b[1]==-np.inf:
			self.fitted = False

		if not self.fitted:
			self.b, end = self.__find_b(X, Y)
			#print(self.b)
			if not end:
				result = self.g_func(X)
				self.left = DecisionTree().fit(X[result == -1], Y[result == -1], h - 1)
				if self.left.b[1] == -np.inf:
					self.left = self.left.majority_y
				self.right = DecisionTree().fit(X[result == 1], Y[result == 1], h - 1)
				if self.right.b[1] == -np.inf:
					self.right = self.right.majority_y
			self.fitted = True
		else:
			result = self.g_func(X)
			
			#keep fitting
			if isinstance(self.left, DecisionTree):
				self.left.fit(X[result == -1], Y[result == -1], h - 1)
			else:
				self.left = DecisionTree().fit(X[result == -1], Y[result == -1], h - 1)
				if self.left.b[1] == -np.inf:
					self.left = self.left.majority_y
			
			#keep fitting
			if isinstance(self.right, DecisionTree):
				self.right.fit(X[result == 1], Y[result == 1], h - 1)
			else:
				self.right = DecisionTree().fit(X[result == 1], Y[result == 1], h - 1)
				if self.right.b[1] == -np.inf:
					self.right = self.right.majority_y
		
		self.height = 1 + max(self.left.height if isinstance(self.left, DecisionTree) else 1, self.right.height if isinstance(self.right, DecisionTree) else 1)
		return self

	def predict(self, X):
		if not self.fitted:
			raise NotImplementedError('Not fitted yet')
		else:
			result = []
			for x in X:
				node = self
				while isinstance(node,DecisionTree):
					node = node.left if node.g_func(x.reshape(1, -1)) == -1 else node.right
				result.append(node)
			return np.array(result)
	
	def score(self, X, Y):
		return np.mean(self.predict(X) == Y)
		
