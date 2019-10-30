"""
Ridge regression using scipy's minimize function and demonstrating the use of
sklearn's framework.

Author: David S. Rosenberg <david.davidr@gmail.com>
License: Creative Commons Attribution 4.0 International License
"""

from sklearn.base import BaseEstimator, RegressorMixin
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, make_scorer
import pandas as pd
from sklearn.metrics import confusion_matrix
import itertools
from setup_problem import load_problem
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


sns.set(style="white", palette="muted", color_codes=True)
class RidgeRegression(BaseEstimator, RegressorMixin):
	""" ridge regression"""

	def __init__(self, l2reg=1):
		if l2reg < 0:
			raise ValueError('Regularization penalty should be at least 0.')
		self.l2reg = l2reg

	def fit(self, X, y=None):
		n, num_ftrs = X.shape
		# convert y to 1-dim array, in case we're given a column vector
		y = y.reshape(-1)
		def ridge_obj(w):
			predictions = np.dot(X,w)
			residual = y - predictions
			empirical_risk = np.sum(residual**2) / n
			l2_norm_squared = np.sum(w**2)
			objective = empirical_risk + self.l2reg * l2_norm_squared
			return objective
		self.ridge_obj_ = ridge_obj

		w_0 = np.zeros(num_ftrs)
		self.w_ = minimize(ridge_obj, w_0).x
		return self

	def predict(self, X, y=None):
		try:
			getattr(self, "w_")
		except AttributeError:
			raise RuntimeError("You must train classifer before predicting data!")
		return np.dot(X, self.w_)

	def score(self, X, y):
		# Average square error
		try:
			getattr(self, "w_")
		except AttributeError:
			raise RuntimeError("You must train classifer before predicting data!")
		residuals = self.predict(X) - y
		return np.dot(residuals, residuals)/len(y)



def compare_our_ridge_with_sklearn(X_train, y_train, l2_reg=1):
	# First run sklearn ridge regression and extract the coefficients
	from sklearn.linear_model import Ridge
	# Fit with sklearn -- need to multiply l2_reg by sample size, since their
	# objective function has the total square loss, rather than average square
	# loss.
	n = X_train.shape[0]
	sklearn_ridge = Ridge(alpha=n*l2_reg, fit_intercept=False, normalize=False)
	sklearn_ridge.fit(X_train, y_train)
	sklearn_ridge_coefs = sklearn_ridge.coef_

	# Now run our ridge regression and compare the coefficients to sklearn's
	ridge_regression_estimator = RidgeRegression(l2reg=l2_reg)
	ridge_regression_estimator.fit(X_train, y_train)
	our_coefs = ridge_regression_estimator.w_

	print("Hoping this is very close to 0:{}".format(np.sum((our_coefs - sklearn_ridge_coefs)**2)))

def do_grid_search_ridge(X_train, y_train, X_val, y_val):
	# Now let's use sklearn to help us do hyperparameter tuning
	# GridSearchCv.fit by default splits the data into training and
	# validation itself; we want to use our own splits, so we need to stack our
	# training and validation sets together, and supply an index
	# (validation_fold) to specify which entries are train and which are
	# validation.
	X_train_val = np.vstack((X_train, X_val))
	y_train_val = np.concatenate((y_train, y_val))
	val_fold = [-1]*len(X_train) + [0]*len(X_val) #0 corresponds to validation

	# Now we set up and do the grid search over l2reg. The np.concatenate
	# command illustrates my search for the best hyperparameter. In each line,
	# I'm zooming in to a particular hyperparameter range that showed promise
	# in the previous grid. This approach works reasonably well when
	# performance is convex as a function of the hyperparameter, which it seems
	# to be here.
	param_grid = [{'l2reg':np.unique(np.concatenate((10.**np.arange(-6,1,1),
										   np.arange(1,3,.3)
											 ))) }]

	ridge_regression_estimator = RidgeRegression()
	grid = GridSearchCV(ridge_regression_estimator,
						param_grid,
						return_train_score=True,
						cv = PredefinedSplit(test_fold=val_fold),
						refit = True,
						scoring = make_scorer(mean_squared_error,
											  greater_is_better = False))
	grid.fit(X_train_val, y_train_val)

	df = pd.DataFrame(grid.cv_results_)
	# Flip sign of score back, because GridSearchCV likes to maximize,
	# so it flips the sign of the score if "greater_is_better=FALSE"
	df['mean_test_score'] = -df['mean_test_score']
	df['mean_train_score'] = -df['mean_train_score']
	cols_to_keep = ["param_l2reg", "mean_test_score","mean_train_score"]
	df_toshow = df[cols_to_keep].fillna('-')
	df_toshow = df_toshow.sort_values(by=["param_l2reg"])
	return grid, df_toshow

def compare_parameter_vectors(pred_fns):
	# Assumes pred_fns is a list of dicts, and each dict has a "name" key and a
	# "coefs" key
	fig, axs = plt.subplots(len(pred_fns),1, sharex=True,figsize=(10, 16))
	num_ftrs = len(pred_fns[0]["coefs"])
	for i in range(len(pred_fns)):
		title = pred_fns[i]["name"]
		coef_vals = pred_fns[i]["coefs"]
		axs[i].bar(range(num_ftrs), coef_vals)
		axs[i].set_xlabel('Feature Index')
		axs[i].set_ylabel('Parameter Value')
		axs[i].set_title(title)
	fig.subplots_adjust(hspace=0.3)

	return fig

def plot_prediction_functions(x, pred_fns, x_train, y_train, legend_loc="best"):
	# Assumes pred_fns is a list of dicts, and each dict has a "name" key and a
	# "preds" key. The value corresponding to the "preds" key is an array of
	# predictions corresponding to the input vector x. x_train and y_train are
	# the input and output values for the training data
	fig, ax = plt.subplots()
	ax.set_xlabel('Input Space: [0,1)')
	ax.set_ylabel('Action/Outcome Space')
	ax.set_title("Prediction Functions")
	plt.scatter(x_train, y_train, label='Training data')
	for i in range(len(pred_fns)):
		ax.plot(x, pred_fns[i]["preds"], label=pred_fns[i]["name"])
	legend = ax.legend(loc=legend_loc, shadow=True)
	return fig

def plot_confusion_matrix(cm, title, classes):      
	 plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)     
	 plt.title(title)       
	 plt.colorbar()     
	 tick_marks = np.arange(len(classes))       
	 plt.xticks(tick_marks, classes, rotation=45)       
	 plt.yticks(tick_marks, classes)        

	 thresh = cm.max() / 2.        
	 for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):     
		 plt.text(j, i, format(cm[i, j], 'd'),      
				  horizontalalignment="center",     
				  color="white" if cm[i, j] > thresh else "black")      

	 plt.tight_layout()        
	 plt.ylabel('True label')       
	 plt.xlabel('Predicted label')


# soft function follow the instructor
def soft_func(a,delta):
    sign = np.sign(a)
    if np.abs(a) - delta >=0:
        return np.dot(sign, np.abs(a) - delta)
    else:
        return 0
    
    
def shooting_algorithm(X, y, w, lambda_reg = 0.01, max_steps = 1000, tor = 1e-8):
    
    
    converge = False
    steps = 0
    d = X.shape[1]
    a = np.zeros(d)
    c = np.zeros(d)
    # loss function
    loss = np.dot(np.dot(X,w) - y, np.dot(X,w) - y) + lambda_reg*np.linalg.norm(w,ord = 1)
    while converge == False and steps <= max_steps:
        loss_prev = loss
        for i in range(d):
            a[i] = 2*np.dot(X.T[i],X.T[i])
            c[i] = 2*np.dot(X.T[i],y-np.dot(X,w)+np.dot(w[i],X.T[i]))
            if a[i] ==0 and c[i] ==0:
                w[i] = 0
            else:
                w[i] = soft_func(c[i]/a[i], lambda_reg/a[i])
        loss = np.dot(np.dot(X,w) - y, np.dot(X,w) - y) + lambda_reg*np.linalg.norm(w,ord = 1)
        change = loss_prev - loss
        if np.abs(change)>=tor:
            converge = False
        else:
            converge = True
        steps += 1
    return a,c,w

def random_shooting_algorithm(X, y, w, lambda_reg = 0.01, max_steps = 1000, tor = 1e-8):
    converge = False
    steps = 0
    d = X.shape[1]
    a = np.zeros(d)
    c = np.zeros(d)
    # loss function
    loss = np.dot(np.dot(X,w) - y, np.dot(X,w) - y) + lambda_reg*np.linalg.norm(w,ord = 1)
    while converge == False and steps <= max_steps:
        loss_prev = loss
        random = np.arange(X.shape[0])
        np.random.shuffle(random)
        X = X[random]
        y = y[random]
        for i in range(d):
            a[i] = 2*np.dot(X.T[i],X.T[i])
            c[i] = 2*np.dot(X.T[i],y-np.dot(X,w)+np.dot(w[i],X.T[i]))
            if a[i] ==0 and c[i] ==0:
                w[i] = 0
            else:
                w[i] = soft_func(c[i]/a[i], lambda_reg/a[i])
        loss = np.dot(np.dot(X,w) - y, np.dot(X,w) - y) + lambda_reg*np.linalg.norm(w,ord = 1)
        change = loss_prev - loss
        if np.abs(change)>=tor:
            converge = False
        else:
            converge = True
        steps += 1
    return a,c,w


def GD_lasso(X, y, alpha=0.01, lambda_reg =0.01, max_iter=1000 , tol=10e-8):
    ''' Implement normal gradient descent'''
    
    def compute_stochastic_gradient(X, y, theta_1, theta_2, lambda_reg):
        num_features = X.shape[1]
        predict = X.dot(theta_1 - theta_2)
        error = y - predict

        grad_1 =  - X.T.dot(error)  + 2*lambda_reg * np.ones(num_features)
        grad_2 =   X.T.dot(error)  + 2*lambda_reg * np.ones(num_features)
        return grad_1, grad_2

    def floor(array):
        for i, e in enumerate(array):
            array[i] = max(e, 0)
        return array
    from random import sample
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_1 = np.zeros(num_features) #Initialize theta+
    theta_2 = np.zeros(num_features) #Initialize theta-
    loss_hist =np.zeros(max_iter)
    
    for i in np.arange(max_iter):  
        theta_previous = theta_1 - theta_2
        grad_1, grad_2 = compute_stochastic_gradient(X, y, theta_1,theta_2,lambda_reg)
        theta_1 -= alpha * grad_1        
        theta_2 -= alpha * grad_2       
        theta_1 = floor(theta_1)
        theta_2 = floor(theta_2)            
        theta = theta_1 - theta_2            
        loss_hist[i] =  np.dot(np.dot(X,theta) - y,np.dot(X,theta) - y)/X.shape[0]       
       
        diff = np.linalg.norm(theta_previous - theta,ord=1)
        
        if diff < tol:
            print ('Converged after %d iteration' %i)
            break                    
    return theta, loss_hist

def SGD_lasso(X, y, alpha='1/t', lambda_reg =0.01, max_iter=1000 , tol=1e-8 ):   
    ''' Implement stochastic gradient descent '''
    
    def compute_stochastic_gradient(x_i, y_i, theta_1, theta_2, lambda_reg):
        num_features = len(x_i)
        predict = x_i.dot(theta_1 - theta_2)
        error = y_i - predict

        grad_1 =  x_i * -1 * error  + 2*lambda_reg * np.ones(num_features)
        grad_2 =  x_i * error  + 2*lambda_reg * np.ones(num_features)
        return grad_1, grad_2

    def floor(array):
        for i, e in enumerate(array):
            array[i] = max(e, 0)
        return array
    
    from random import sample
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_1 = np.zeros(num_features) #Initialize theta+
    theta_2 = np.zeros(num_features) #Initialize theta-    
    loss_hist =np.zeros(max_iter)
    
    for i in np.arange(max_iter): 
        theta_previous = theta_1 - theta_2
        index = np.random.permutation(num_instances)
        for it,ix in enumerate(index):
            x_i, y_i = X[ix],y[ix]

            if isinstance(alpha, float):
                step_size = alpha
            elif alpha == '1/t':
                step_size = 1.0/(num_instances*i+it+1)
            else:
                step_size = 1.0/np.sqrt(num_instances*i+it+1)
                               
            grad_1, grad_2 = compute_stochastic_gradient(x_i, y_i, theta_1,theta_2,lambda_reg)
            theta_1 -= step_size * grad_1        
            theta_2 -= step_size * grad_2       
            theta_1 = floor(theta_1)
            theta_2 = floor(theta_2)
           
        theta = theta_1 - theta_2            
        loss_hist[i] = np.dot(np.dot(X,theta) - y,np.dot(X,theta) - y)/X.shape[0]   
        
        diff =   diff = np.linalg.norm(theta_previous - theta,ord=1)
        
        if diff < tol:
            print ('Converged after %d iteration' %i)
            break
            
        
    return theta, loss_hist

def print_sparsity(Lambda, theta, theta_true, tol=10**-4, loss= False):
    ''' Compute the sparsity of theta_estimate'''
    
    df_comp= pd.DataFrame([theta,theta_true]).T
    df_comp['ans1']= df_comp[df_comp[1]==0][0].apply(lambda x : abs(x)<=tol) # true value ==0, estimate < tol
    df_comp['ans2']=df_comp[df_comp[1]!=0][0] !=0  # true value !=0, estimate !=0
    spar1 = len(df_comp[df_comp['ans1']==True])
    spar2 = len(df_comp[df_comp['ans2']==True])
    if loss:     
        print ('lambda = %f, %d , %d , loss=%f' %(Lambda, spar1,spar2,loss))
    else:
        print ('lambda = %f, %d , %d '%(Lambda, spar1,spar2))



def main():
    lasso_data_fname = "lasso_data.pickle"
	x_train, y_train, x_val, y_val, target_fn, coefs_true, featurize = load_problem(lasso_data_fname)

	# Generate features
	X_train = featurize(x_train)
	X_val = featurize(x_val)

	#Visualize training data
	fig, ax = plt.subplots()
	ax.imshow(X_train)
	ax.set_title("Design Matrix: Color is Feature Value")
	ax.set_xlabel("Feature Index")
	ax.set_ylabel("Example Number")
	plt.show(block=False)

	# Compare our RidgeRegression to sklearn's.
	compare_our_ridge_with_sklearn(X_train, y_train, l2_reg = 1.5)

	# Do hyperparameter tuning with our ridge regression
	grid, results = do_grid_search_ridge(X_train, y_train, X_val, y_val)
	print(results)

	# Plot validation performance vs regularization parameter
	fig, ax = plt.subplots()
    #ax.loglog(results["param_l2reg"], results["mean_test_score"])
	ax.semilogx(results["param_l2reg"], results["mean_test_score"])
	ax.grid()
	ax.set_title("Validation Performance vs L2 Regularization")
	ax.set_xlabel("L2-Penalty Regularization Parameter")
	ax.set_ylabel("Mean Squared Error")
	fig.show()

	# Let's plot prediction functions and compare coefficients for several fits
	# and the target function.
	pred_fns = []
	x = np.sort(np.concatenate([np.arange(0,1,.001), x_train]))
	name = "Target Parameter Values (i.e. Bayes Optimal)"
	pred_fns.append({"name":name, "coefs":coefs_true, "preds": target_fn(x) })

	l2regs = [0, grid.best_params_['l2reg'], 1]
	X = featurize(x)
	for l2reg in l2regs:
		ridge_regression_estimator = RidgeRegression(l2reg=l2reg)
		ridge_regression_estimator.fit(X_train, y_train)
		name = "Ridge with L2Reg="+str(l2reg)
		pred_fns.append({"name":name,
						 "coefs":ridge_regression_estimator.w_,
						 "preds": ridge_regression_estimator.predict(X) })

	f = plot_prediction_functions(x, pred_fns, x_train, y_train, legend_loc="best")
	f.show()

	f = compare_parameter_vectors(pred_fns)
	f.show()


	##Sample code for plotting a matrix
	## Note that this is a generic code for confusion matrix
	## You still have to make y_true and y_pred by thresholding as per the insturctions in the question.
	
    y_true = [1, 0, 1, 1, 0, 1]
	y_pred = [0, 0, 1, 1, 0, 1]
	eps = 1e-1;
	cnf_matrix = confusion_matrix(y_true, y_pred)
	plt.figure()
	plot_confusion_matrix(cnf_matrix, title="Confusion Matrix for $\epsilon = {}$".format(eps), classes=["Zero", "Non-Zero"])
	plt.show()
    
    
    """question 2.1"""
    
    l2reg = [1e-11, 1e-9, 1e-7,1e-5,1e-3,1e-2,0.05, 0.1, 0.5, 1.0]
    error_record = []
    for lambda_ in l2reg:
        regression_estimator = RidgeRegression(l2reg = lambda_)
        regression_estimator.fit(X_train, y_train)
        w_opt = regression_estimator.w_
        error = 1/len(y_val)*np.dot( np.dot(w_opt,X_val.T) - y_val, np.dot(w_opt,X_val.T) - y_val)
        error_record.append(error)
    
    error_df = pd.DataFrame(columns = ['lambda','empirical risk'])
    error_df['lambda'] = l2reg
    error_df['empirical risk'] = error_record
    print(error_df)
    #plot
    plt.figure()
    plt.plot(np.log10(error_df['lambda']),error_df['empirical risk'])
    plt.savefig("different lambda.jpg")
    
    """question 2.2"""
    print(np.argmax(np.abs(coefs_true)),np.max(np.abs(coefs_true)))
   
    """question 2.3"""
    regression_estimator = RidgeRegression(l2reg = 0.01)
    regression_estimator.fit(X_train, y_train)
    w_opt = regression_estimator.w_
    w = coefs_true
    w_ture = [(lambda i: 0 if i ==0 else 1) (i) for i in w]
    w_0 = [(lambda i: 0 if np.abs(i) < 1e-6 else 1) (i) for i in w_opt]
    w_1 = [(lambda i: 0 if np.abs(i) < 1e-3 else 1) (i) for i in w_opt]
    w_2 = [(lambda i: 0 if np.abs(i) < 1e-1 else 1) (i) for i in w_opt]
    w_3 = [(lambda i: 0 if np.abs(i) < 1e-2 else 1) (i) for i in w_opt]
	cnf_matrix = confusion_matrix(w_ture, w_0)
	plt.figure()
	plot_confusion_matrix(cnf_matrix, title="Confusion Matrix for $\epsilon = {}$".format(1e-6), classes=["Zero", "Non-Zero"])
	plt.show()
    cnf_matrix = confusion_matrix(w_ture, w_1)
	plt.figure()
	plot_confusion_matrix(cnf_matrix, title="Confusion Matrix for $\epsilon = {}$".format(1e-3), classes=["Zero", "Non-Zero"])
	plt.show()
    cnf_matrix = confusion_matrix(w_ture, w_2)
	plt.figure()
	plot_confusion_matrix(cnf_matrix, title="Confusion Matrix for $\epsilon = {}$".format(1e-1), classes=["Zero", "Non-Zero"])
	plt.show()
    cnf_matrix = confusion_matrix(w_ture, w_3)
    plt.figure()
	plot_confusion_matrix(cnf_matrix, title="Confusion Matrix for $\epsilon = {}$".format(1e-2), classes=["Zero", "Non-Zero"])
	plt.show()
    
    
    """
    question 3.2
    """
    #lamda starts from 0
    w = shooting_algorithm(X_train, y_train, w_opt, lambda_reg = 0, max_steps = 1000, tor = 1e-8)[2]
    # square loss
    loss_val = np.dot(np.dot(X_val,w) - y_val,np.dot(X_val,w) - y_val)/X.shape[0]
    print("Murphy loss value from cycle shooting algorithm with lambda equal to 0\t" + str(loss_val))
    w = shooting_algorithm(X_train, y_train, np.zeros(400), lambda_reg = 0, max_steps = 1000, tor = 1e-8)[2]
    # square loss
    loss_val = np.dot(np.dot(X_val,w) - y_val,np.dot(X_val,w) - y_val)/X.shape[0]
    print("from zero loss value from cycle shooting algorithm with lambda equal to 0\t" + str(loss_val))
    w = random_shooting_algorithm(X_train, y_train, w_opt, lambda_reg = 0, max_steps = 1000, tor = 1e-8)[2]
    # square loss
    loss_val = np.dot(np.dot(X_val,w) - y_val,np.dot(X_val,w) - y_val)/X.shape[0]
    print("Murphy loss value from random shooting algorithm with lambda equal to 0\t" + str(loss_val))
    w = random_shooting_algorithm(X_train, y_train, np.zeros(400), lambda_reg = 0, max_steps = 1000, tor = 1e-8)[2]
    # square loss
    loss_val = np.dot(np.dot(X_val,w) - y_val,np.dot(X_val,w) - y_val)/X.shape[0]
    print("from zero loss value from random shooting algorithm with lambda equal to 0\t" + str(loss_val))
    
    
    
    #lambda equal to 0.01 which is recommend
    w = shooting_algorithm(X_train, y_train, w_opt, lambda_reg = 0.01, max_steps = 1000, tor = 1e-8)[2]
    # square loss
    loss_val = np.dot(np.dot(X_val,w) - y_val,np.dot(X_val,w) - y_val)/X.shape[0]
    print("Murphy loss value from cycle shooting algorithm with lambda equal to 0.01\t" + str(loss_val))
    w = shooting_algorithm(X_train, y_train, np.zeros(400), lambda_reg = 0.01, max_steps = 1000, tor = 1e-8)[2]
    # square loss
    loss_val = np.dot(np.dot(X_val,w) - y_val,np.dot(X_val,w) - y_val)/X.shape[0]
    print("from zero loss value from cycle shooting algorithm with lambda equal to 0\t" + str(loss_val))
    w = random_shooting_algorithm(X_train, y_train, w_opt, lambda_reg = 0.01, max_steps = 1000, tor = 1e-8)[2]
    # square loss
    loss_val = np.dot(np.dot(X_val,w) - y_val,np.dot(X_val,w) - y_val)/X.shape[0]
    print("Murphy loss value from random shooting algorithm with lambda equal to 0\t" + str(loss_val))
    w = random_shooting_algorithm(X_train, y_train, np.zeros(400), lambda_reg = 0.01, max_steps = 1000, tor = 1e-8)[2]
    # square loss
    loss_val = np.dot(np.dot(X_val,w) - y_val,np.dot(X_val,w) - y_val)/X.shape[0]
    print("from zero loss value from random shooting algorithm with lambda equal to 0.01\t" + str(loss_val))
    
    #lambda equal to 0.1
    w = shooting_algorithm(X_train, y_train, w_opt, lambda_reg = 0.1, max_steps = 1000, tor = 1e-8)[2]
    # square loss
    loss_val = np.dot(np.dot(X_val,w) - y_val,np.dot(X_val,w) - y_val)/X.shape[0]
    print("Murphy loss value from cycle shooting algorithm with lambda equal to 0.1\t" + str(loss_val))
    w = shooting_algorithm(X_train, y_train, np.zeros(400), lambda_reg = 0.1, max_steps = 1000, tor = 1e-8)[2]
    # square loss
    loss_val = np.dot(np.dot(X_val,w) - y_val,np.dot(X_val,w) - y_val)/X.shape[0]
    print("from zero loss value from cycle shooting algorithm with lambda equal to 0\t" + str(loss_val))
    w = random_shooting_algorithm(X_train, y_train, w_opt, lambda_reg = 0.1, max_steps = 1000, tor = 1e-8)[2]
    # square loss
    loss_val = np.dot(np.dot(X_val,w) - y_val,np.dot(X_val,w) - y_val)/X.shape[0]
    print("Murphy loss value from random shooting algorithm with lambda equal to 0.1\t" + str(loss_val))
    w = random_shooting_algorithm(X_train, y_train, np.zeros(400), lambda_reg = 0.1, max_steps = 1000, tor = 1e-8)[2]
    # square loss
    loss_val = np.dot(np.dot(X_val,w) - y_val,np.dot(X_val,w) - y_val)/X.shape[0]
    print("from zero loss value from random shooting algorithm with lambda equal to 0.01\t" + str(loss_val))
    """
    question 3.3
    """
    l1reg = [1e-11, 1e-9, 1e-7,1e-5,1e-3,1e-2,0.05, 0.1, 0.5, 1.0, 10, 100]
    error_record = []
    for lambda_ in l1reg:
        w = random_shooting_algorithm(X_train, y_train, w_opt, lambda_reg = lambda_, max_steps = 1000, tor = 1e-8)[2]
        error = np.dot(np.dot(X_val,w) - y_val,np.dot(X_val,w) - y_val)/X.shape[0]
        error_record.append(error)
        
    error_df = pd.DataFrame(columns = ['lambda','empirical risk'])
    error_df['lambda'] = l1reg
    error_df['empirical risk'] = error_record
    print(error_df)
    #plot
    plt.figure()
    plt.plot(np.log10(error_df['lambda']),error_df['empirical risk'])
    plt.savefig("different lambda.jpg")
    """
    question 3.4
    """
    lambda_max = max(2*np.abs(X_train.T.dot(y_train)))
    lambda_lasso= [lambda_max*0.8**i for i in range(30)]

    w=np.zeros((30,400))
    loss=np.zeros(30)

    for i in range(30):
        w[i]= shooting_algorithm(X_train,y_train,w[i-1],lambda_lasso[i],max_steps = 1000,tor = 1e-8)[2]
        loss[i] = (1/X_train.shape[0])*np.dot(np.dot(X_val,w[i])-y_val,np.dot(X_val,w[i])-y_val)
        
    print(loss)
    error_df = pd.DataFrame(columns = ['lambda','empirical risk'])
    error_df['lambda'] = lambda_lasso
    error_df['empirical risk'] = loss
    plt.figure()
    plt.plot(error_df['lambda'],error_df['empirical risk'])
    plt.xlabel("lambda")
    plt.ylabel("average loss")
    plt.title("loss shooting algorithm")
    plt.savefig("question3_4.jpg")
    
    
    error_record = []
    for Lambda in l1reg:
        theta, loss_hist= SGD_lasso(X_train, y_train,alpha =0.0001,lambda_reg=Lambda,max_iter=1000)
        loss = np.dot(np.dot(X_val,theta) - y_val,np.dot(X_val,theta) - y_val)/X.shape[0]
        error_record.append(loss)
        print(loss)
    error_df = pd.DataFrame(columns = ['lambda','empirical risk'])
    error_df['lambda'] = l1reg
    error_df['empirical risk'] = error_record
    print(error_df)
    #plot
    plt.figure()
    plt.plot(np.log10(error_df['lambda']),error_df['empirical risk'])
    plt.xlabel("lambda with log")
    plt.ylabel("empriical risk")
    plt.savefig("SGD_empirical.png")
    
    error_record = []
    for Lambda in l1reg:
        theta, loss_hist= GD_lasso(X_train, y_train, alpha=0.0001, lambda_reg = Lambda, max_iter=1000 , tol=1000 )
        loss = np.dot(np.dot(X_val,theta) - y_val,np.dot(X_val,theta) - y_val)/X.shape[0]
        error_record.append(loss)
        print(loss)
    error_df = pd.DataFrame(columns = ['lambda','empirical risk'])
    error_df['lambda'] = l1reg
    error_df['empirical risk'] = error_record
    print(error_df)
    #plot
    plt.figure()
    plt.plot(np.log10(error_df['lambda']),error_df['empirical risk'])
    plt.xlabel("lambda with log")
    plt.ylabel("empriical risk")
    plt.savefig("GD_empirical.png")

    result_SGD =[]
    l2reg = [1e-11, 1e-9, 1e-7,1e-5,1e-3,1e-2,0.05, 0.1, 0.5, 1.0]
    theta, loss_hist= SGD_lasso(X_train, y_train,alpha =0.0001,lambda_reg=1e-11,max_iter=1000)
    loss = np.dot(np.dot(X_val,theta) - y_val,np.dot(X_val,theta) - y_val)/X.shape[0]
    result_SGD.append(loss)
    print(loss)
    theta, loss_hist= SGD_lasso(X_train, y_train,alpha =0.0001,lambda_reg=1e-9,max_iter=1000)
    loss = np.dot(np.dot(X_val,theta) - y_val,np.dot(X_val,theta) - y_val)/X.shape[0]
    result_SGD.append(loss)
    print(loss)
    theta, loss_hist= SGD_lasso(X_train, y_train,alpha =0.0001,lambda_reg=1e-7,max_iter=1000)
    loss = np.dot(np.dot(X_val,theta) - y_val,np.dot(X_val,theta) - y_val)/X.shape[0]
    result_SGD.append(loss)
    print(loss)
    theta, loss_hist= SGD_lasso(X_train, y_train,alpha =0.0001,lambda_reg=1e-5,max_iter=1000)
    loss = np.dot(np.dot(X_val,theta) - y_val,np.dot(X_val,theta) - y_val)/X.shape[0]
    result_SGD.append(loss)
    print(loss)
    theta, loss_hist= SGD_lasso(X_train, y_train,alpha =0.0001,lambda_reg=1e-3,max_iter=1000)
    loss = np.dot(np.dot(X_val,theta) - y_val,np.dot(X_val,theta) - y_val)/X.shape[0]
    result_SGD.append(loss)
    print(loss)
    theta, loss_hist= SGD_lasso(X_train, y_train,alpha =0.0001,lambda_reg=1e-2,max_iter=1000)
    loss = np.dot(np.dot(X_val,theta) - y_val,np.dot(X_val,theta) - y_val)/X.shape[0]
    result_SGD.append(loss)
    print(loss)
    theta, loss_hist= SGD_lasso(X_train, y_train,alpha =0.0001,lambda_reg=0.05,max_iter=1000)
    loss = np.dot(np.dot(X_val,theta) - y_val,np.dot(X_val,theta) - y_val)/X.shape[0]
    result_SGD.append(loss)
    print(loss)
    theta, loss_hist= SGD_lasso(X_train, y_train,alpha =0.0001,lambda_reg=0.1,max_iter=1000)
    loss = np.dot(np.dot(X_val,theta) - y_val,np.dot(X_val,theta) - y_val)/X.shape[0]
    result_SGD.append(loss)
    print(loss)
    theta, loss_hist= SGD_lasso(X_train, y_train,alpha =0.0001,lambda_reg=0.5,max_iter=1000)
    loss = np.dot(np.dot(X_val,theta) - y_val,np.dot(X_val,theta) - y_val)/X.shape[0]
    result_SGD.append(loss)
    print(loss)
    theta, loss_hist= SGD_lasso(X_train, y_train,alpha =0.0001,lambda_reg=1,max_iter=1000)
    loss = np.dot(np.dot(X_val,theta) - y_val,np.dot(X_val,theta) - y_val)/X.shape[0]
    result_SGD.append(loss)
    print(loss)
    
    
    
    error_df = pd.DataFrame(columns = ['lambda','empirical risk'])
    error_df['lambda'] = l2reg
    error_df['empirical risk'] = result_SGD
    print(error_df)
    #plot
    plt.figure()
    plt.plot(np.log10(error_df['lambda']),error_df['empirical risk'])
    plt.savefig("SGD.jpg")
    
   
   
   
   
   
   
   
   
def find_positive(x): 
    result = max(x, 0) 
    return result
def positive_project(x): 
    result = np.array(list(map(find_positive, x))) 
    return result
def projection_SGD_split(X, y, theta_positive_0, theta_negative_0, lambda_reg = 0.01, alpha = 0.01, num_iter = 1000): 
    m, n = X.shape 
    theta_positive = np.zeros(n) 
    theta_negative = np.zeros(n) 
    theta_positive[0:n] = theta_positive_0 
    theta_negative[0:n] = theta_negative_0 
    times = 0 
    theta = theta_positive - theta_negative
    loss = np.dot(np.dot(X_val,theta) - y_val,np.dot(X_val,theta) - y_val)/X.shape[0]
    loss_change = 1. 
    while (loss_change>1e-6) and (times<num_iter): 
        loss_old = loss 
        for i in range(m): 
            X_sample = X[i, :] 
            y_sample = y[i] 
            var_1 = np.dot(X_sample, theta.T) 
            var_2 = var_1 - y_sample 
            var_3 = 2*var_2*X_sample 
            grad_positive = var_3 + lambda_reg 
            grad_negative = (-1.)*var_3 + lambda_reg 
            theta_positive = theta_positive - alpha*grad_positive 
            theta_negative = theta_negative - alpha*grad_negative 
            
            theta_positive = positive_project(theta_positive) 
            theta_negative = positive_project(theta_negative) 
            
            theta = theta_positive - theta_negative
            loss = np.dot(np.dot(X_val,theta) - y_val,np.dot(X_val,theta) - y_val)/X.shape[0]
            loss_change = abs(loss - loss_old) 
            times += 1 
    print(times)
    return theta

if __name__ == '__main__':
  main()
