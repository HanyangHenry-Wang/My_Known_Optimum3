import known_optimum
from known_optimum import Objective_Function3
from known_optimum import normal_BO, PO_BO
import numpy as np


regret_holder1 = []
regret_holder2 = []
regret_holder3 = []
regret_holder4 = []
regret_holder5 = []

N = 150

for i in range(N):
  print(i)
  X_total, Y_total = Objective_Function3(variance=10,lengthscale=1.2,seed=i+50)
  fstar = np.max(Y_total)
  print('optimal: ',fstar)
  
  regret_temp1 = normal_BO(X_total,Y_total,'ei',fstar,seed = i+50)
  regret_holder1.append(regret_temp1)

  regret_temp2 = normal_BO(X_total,Y_total,'tei',fstar,seed = i+50)
  regret_holder2.append(regret_temp2)
  
  regret_temp3 = normal_BO(X_total,Y_total,'mes',fstar,seed = i+50)
  regret_holder3.append(regret_temp3)
  
  regret_temp4 = normal_BO(X_total,Y_total,'find_max',fstar,seed = i+50)
  regret_holder4.append(regret_temp4)
  
  regret_temp5 = PO_BO(X_total,Y_total,fstar,seed = i+50)
  regret_holder5.append(regret_temp5)
  
  
  
np.savetxt('GP3d_ei', regret_holder1, delimiter=',')
np.savetxt('GP3d_tei', regret_holder2, delimiter=',')
np.savetxt('GP3d_mes', regret_holder3, delimiter=',')
np.savetxt('GP3d_findamx', regret_holder4, delimiter=',')
np.savetxt('GP3d_po', regret_holder5, delimiter=',')