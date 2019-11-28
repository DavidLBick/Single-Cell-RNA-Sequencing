import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

# implementation of orthogonal matching pursuit for logistic regression: http://proceedings.mlr.press/v15/lozano11a/lozano11a.pdf

class LogisticOMP:
	def __init__(self, n_nonzero_coefs, eps):
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
		self.clf = LogisticRegression(random_state=0, solver='saga', n_jobs=-1, max_iter=10000, class_weight='balanced', warm_start=True)
		self.n_features_selected = 0
		self.first_run = True
		self.previously_selected_features = None

	def converged(self):
		# if enough non_zero_coefficients are found
		if np.sum(self.G) >= self.n_nonzero_coefs:
			return True 
		return False
		
	def partial_converged(self):
		if self.n_features_selected >= self.n_nonzero_coefs:
			return True
		return False			
	
	def g_inverse(self, x):
		return expit(x)
	
	def calc_residual(self):
		eta = np.matmul(self.X, self.beta)
		y_hat = self.g_inverse(eta)
		self.residual = y_hat - self.y
	
	def find_best_feature(self):
		# only calculate correlations for not-selected features
		all_correlations = np.matmul(np.transpose(self.X), self.residual)
		correlations_of_previously_unselected_features = np.multiply(all_correlations, np.invert(self.previously_selected_features))
		masked_correlations = np.multiply(correlations_of_previously_unselected_features, np.invert(self.G))
		self.last_slctd_feature = np.argmax(masked_correlations)
		self.last_slctd_correlation = all_correlations[self.last_slctd_feature]
		print("correlation of selected feature", self.n_features_selected, '/', self.n_nonzero_coefs,':', self.last_slctd_correlation)
		self.feature_ranking.append(self.last_slctd_feature)
	
	def update_beta(self):
		slctd_X = self.X[:, self.G]
		# print('columns in logistic regression training:', slctd_X.shape[1])
		self.clf.fit(slctd_X, self.y)
		learned_params = self.clf.coef_[0]
		# print("learned_params", learned_params)
		weights_idxs, _ = self.get_selected_feature_idxs()
		for logistic_idx, weight_idx in enumerate(weights_idxs):
			# print("logistic index:", logistic_idx)
			# print("weight_idx:", weight_idx)
			self.beta[weight_idx] = learned_params[logistic_idx]
	
	def fit(self, X, y, previously_selected_features):
		#expects y to be binary class labels
		self.X = X
		self.y = y
		self.previously_selected_features = previously_selected_features  # binary mask
		
		self.G = np.zeros_like(self.X[0,:], dtype=bool)  # binary indx
		self.beta = np.zeros_like(self.X[0, :])
		while not self.converged():
			self.calc_residual()
			self.find_best_feature()
			if self.last_slctd_correlation < self.eps:
				return self
			self.G[self.last_slctd_feature] = True
			self.update_beta()
			self.n_features_selected += 1
			
		return self
	
	def partial_fit(self, X, y):
		self.X = X
		self.y = y
		
		if self.first_run:
			self.first_run = False
			self.G = np.zeros_like(self.X[0,:], dtype=bool)
			self.beta = np.zeros_like(self.X[0, :])
			
		while not self.partial_converged():
			self.calc_residual()
			self.find_best_feature()
			self.G[self.last_slctd_feature] = True
			self.update_beta()
			self.n_features_selected +=1
			
		self.n_features_selected = 0
		return self
			
		
	def get_selected_feature_idxs(self):	
		return np.nonzero(self.G)[0], np.array(self.feature_ranking)
	
	def get_binary_selected_feature_vector(self):
		return self.G
	
