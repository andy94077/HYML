import numpy as np
class Adam():
	def __init__(self, gradient_func, eta=1e-3):
		self.eta = eta
		self.beta1 = 0.9
		self.beta2 = 0.999
		self.beta1_power = self.beta2_power = 1.0
		self.m = self.v = 0.0
		self.gradient_func = gradient_func

	def update(self, X, Y, w, *args):
		g = self.gradient_func(X, Y, w, *args)
		self.m = self.beta1 * self.m + (1 - self.beta1) * g
		self.v = self.beta2 * self.v + (1 - self.beta2) * g ** 2
		self.beta1_power *= self.beta1
		self.beta2_power *= self.beta2
		m_hat = self.m / (1 - self.beta1_power)
		v_hat = self.v / (1 - self.beta2_power)
		w -= self.eta * m_hat / (np.sqrt(v_hat) + 1e-8)
		
		
