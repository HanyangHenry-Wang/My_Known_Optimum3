import numpy as np
import GPy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import randrange, uniform
from scipy.stats import norm
import math


def Objective_Function3(variance=10,lengthscale=1.2, seed=1):

  np.random.seed(seed) 
  temp = float(np.random.uniform(low=-2, high=2, size=1))
  
  bounds=np.asarray([[0.,5.5],[0.,5.5],[0.,5.5]])
  dim = bounds.shape[0]
  
  X_total = []
  
  for a in range(12):
    for b in range(12):
      for c in range (12):
        X_temp = np.array([a*0.5,b*0.5,c*0.5])
        X_total.append(X_temp)
      
  X_total = np.array(X_total)


  X_sample=np.array([[5.1,3.8,1.1]])
  Y_sample=np.array([[temp]])

  kernel = GPy.kern.RBF(input_dim=3,variance=variance,lengthscale=lengthscale)
  m = GPy.models.GPRegression(X_sample,Y_sample,kernel)
  m.Gaussian_noise.variance.fix(0.0)

  Y_total = m.posterior_samples_f(X_total,size=1).reshape(-1,1)

  return X_total, Y_total


def init(X_total, Y_total,seed=1):
  
  np.random.seed(seed)

  X_sample = []
  Y_sample = []
  
  
  x_index_holder = [3,20,100,210,300,450,600,700,900,1100,1410,1550] # 12 initial points


  for idx in x_index_holder:
    X_temp = X_total[idx]           #int(np.random.randint(40, size=1))
    Y_temp =  Y_total[idx]           #int(np.random.randint(40, size=1))

    X_sample.append(X_temp)
    Y_sample.append(Y_temp)

  return np.array(X_sample),np.array(Y_sample)


def EI(mean,var,y_max):

  z = (mean - y_max)/np.sqrt(var)        
  out=(mean - y_max) * norm.cdf(z) + np.sqrt(var) * norm.pdf(z)

  return out 

def findmax(mean,var,fstar):
   pdf_fmax = 1/(np.sqrt(2*np.pi*var))*np.exp(-(fstar-mean)**2/(2*var))

   return pdf_fmax
 

def MSE(mean,var,fstar):
    
  gamma = (fstar-mean)/np.sqrt(var)  
  
  out = (gamma*norm.pdf(gamma))/(2*norm.cdf(gamma))-np.log(norm.cdf(gamma))

  return out 

def over_max(mean,var,fstar):
  z = (fstar - mean)/np.sqrt(var)   
  below_max = norm.cdf(z)
  out = 1 - below_max
  
  return out 




def normal_BO(X_total,Y_total,acq,fstar, seed = 1): #the shape of X_total is (2500,2), the shape of Y_total is (50,50)

  fstar_true = np.max(Y_total)
  
  total_round = 20
  Y_max_holder = []

  X_sample, Y_sample = init(X_total, Y_total,seed)

  Y_max= np.max(Y_sample)
  Y_max_holder.append(Y_max)

  for n in range(total_round):
    #train the GP model for X and centrailised Y
    kernel = GPy.kern.RBF(input_dim=3,variance=10,lengthscale=1.2)
    m = GPy.models.GPRegression(X_sample,Y_sample.reshape(-1,1),kernel)
    m.Gaussian_noise.variance.fix(0.0)

    #find the X that can maximize the acqusition function:
    mean,var = m.predict(X_total,include_likelihood=False)

    #print(mean[mean>fstar])

    if acq == 'ei':
      acq_value = EI(mean,var,Y_max_holder[-1])
      
    elif acq=='find_max':
      acq_value = findmax(mean,var,fstar)
    
    elif acq == 'mes':
      acq_value = MSE(mean,var,fstar)
      
    elif acq == 'over_max':
      acq_value = over_max(mean,var,fstar)
      
    elif acq == 'tei':
      part1 = EI(mean,var,Y_max_holder[-1])
      part2 = EI(mean,var,fstar)
      acq_value = part1-part2

    index = np.argmax(acq_value)
    X_chosen = X_total[index]
    Y_chosen = Y_total[index]

    X_sample = np.concatenate((X_sample, X_chosen.reshape(-1,3)), axis=0)
    Y_sample = np.concatenate((Y_sample, np.array([Y_chosen])), axis=0)

    Y_max= np.max(Y_sample)
    Y_max_holder.append(Y_max)


  Y_max_holder = np.array(Y_max_holder)
  regret_holder = fstar_true - Y_max_holder

  return regret_holder




def PO_BO(X_total,Y_total, fstar,seed = 1): #the shape of X_total is (2500,2), the shape of Y_total is (50,50)

  fstar_true = np.max(Y_total)
    
  total_round = 20
  Y_max_holder = []

  X_sample, Y_sample = init(X_total, Y_total,seed=seed)

  Y_max= np.max(Y_sample)
  Y_max_holder.append(Y_max)

  for n in range(total_round):
    print(n)
    #train the GP model for X and centrailised Y
    kernel = GPy.kern.RBF(input_dim=3,variance=10,lengthscale=1.2)
    m = GPy.models.GPRegression(X_sample,Y_sample.reshape(-1,1),kernel)
    m.Gaussian_noise.variance.fix(0.0)

    #find the X that can maximize the acqusition function:
    mean,var = m.predict(X_total,include_likelihood=False)

    part1_total = findmax(mean,var,fstar)
    part1_total = part1_total.reshape(-1,)

    part2_total = np.zeros(X_total.shape[0])

    for i in range(X_total.shape[0]):
        X_sample_temp = np.concatenate((X_sample, np.array([X_total[i]])), axis=0)
        Y_sample_temp = np.concatenate((Y_sample, np.array([[fstar]])), axis=0)

        X_current_0 = X_total[i][0]
        X0_lower = max(0.0,X_current_0-4*0.5)
        X0_upper = min(5.85,X_current_0+4*0.5)
        X0_range = np.arange(X0_lower,X0_upper,0.5)

        X_current_1 = X_total[i][1]
        X1_lower = max(0.0,X_current_1-4*0.5)
        X1_upper = min(5.85,X_current_1+4*0.5)
        X1_range = np.arange(X1_lower,X1_upper,0.5)
        
        X_current_2 = X_total[i][1]
        X2_lower = max(0.0,X_current_2-4*0.5)
        X2_upper = min(5.85,X_current_2+4*0.5)
        X2_range = np.arange(X2_lower,X2_upper,0.5)
        
        

        X_near = []

        for x0 in X0_range:
          for x1 in X1_range:
            for x2 in X2_range:
              X_near.append([x0,x1,x2])
        X_near = np.array(X_near)



        kernel_temp = GPy.kern.RBF(input_dim=3,variance=10,lengthscale=1.2)
        m_temp = GPy.models.GPRegression(X_sample_temp.reshape(-1,3),Y_sample_temp.reshape(-1,1),kernel_temp)
        m_temp.Gaussian_noise.variance.fix(0.0)

        mean_temp,var_temp = m_temp.predict(X_near,include_likelihood=False)
        z = (fstar-mean_temp)/np.sqrt(var_temp)
        PnI = norm.cdf(z)
        part2 = np.min(PnI)
        part2_total[i] = part2

    acq_value = part1_total*part2_total
    index = np.argmax(acq_value)
    X_chosen = X_total[index]
    Y_chosen = Y_total[index]

    X_sample = np.concatenate((X_sample, X_chosen.reshape(-1,3)), axis=0)
    Y_sample = np.concatenate((Y_sample, np.array([Y_chosen])), axis=0)

    Y_max= np.max(Y_sample)
    Y_max_holder.append(Y_max)


  Y_max_holder = np.array(Y_max_holder)
  regret_holder = fstar_true - Y_max_holder

  return regret_holder