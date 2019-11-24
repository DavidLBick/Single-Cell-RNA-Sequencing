import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

# implementation of orthogonal matching pursuit for logistic regression

class LogisticOMP:
	def __init__(n_nonzero_coefs, eps):
		self.n_nonzero_coefs = n_nonzero_coefs
		self.eps = eps
		self.G = None
		self.beta = None
		self.X = None
		self.y = None
		self.residual = None
		self.last_slctd_feature = None
		self.last_slctd_correlation = None
		self.feature_ranking = []
		self.clf = LogisticRegression(random_state=0, solver='saga', n_jobs=-1)
		
	def converged():
		# if enough non_zero_coefficients are found
		if np.sum(G) >= self.n_nonzero_coefs:
			return True
		elif 
		return False
	
	def g_inverse(x):
		return expit(x)
	
	def calc_residual():
		eta = np.matmul(self.X, self.beta)
		y_hat = self.g_inverse(eta)
		self.residual = y_hat - self.y
	
	def find_best_feature():
		# only calculate correlations for not-selected features
		all_correlations = np.matmul(np.transpose(X), self.residual)
		masked_correlations = np.multiply(all_correlations, np.invert(self.G))
		self.last_slctd_feature = np.argmax(masked_correlations)
		self.last_slctd_correlation = all_correlations[self.last_slctd_feature]
		self.feature_ranking.append(self.last_slctd_feature)
	
	def update_beta():
		slctd_X = self.X[:, self.G]
		self.clf.fit(slctd_X, self.y)
		learned_params = self.clf.coef_
		weights_idxs, _ = self.get_selected_feature_idxs()
		for logistic_idx, weight_idx in enumerate(weights_idxs):
			self.beta[weight_idx] = learned_params[logistic_idx]
				
	def fit(X, y):
		# expects y to be binary class labels
		self.X = X
		self.y = y
		
		self.G = np.zeros_like(X[0,:], dtype=bool)  # binary indx
		self.beta = np.zeros_like(X[0, :])
		while not self.converged():
			self.calc_residual()
			self.find_best_feature()
			if self.last_slctd_correlation < self.eps:
				return self
			self.G[last_slctd_feature] = True
			self.update_beta()
			
		return self
		
	def get_selected_feature_idxs():
		return np.nonzero(self.G), self.feature_ranking
	