#!/usr/bin/env python
# coding: utf-8

# ## Notebook for Discrete Network Design Problem (DNDP)
# 
# This self-contained notebook contains all the code to reproduce the results for the DNDP.  For DNDP, the implementation is more straightforward as a single model is trained for the unchanged network. In contrast, KP, CNP, and DRP use trained models across instances with variable parameters.  This assumption for DNDP allows simpler models to be used, i.e., feed-forward networks and gradient-boosted trees, and simpler surrogate models. his assumption for DNDP allow simplier models to be used, i.e., feed-forward networks and gradient boosted trees, and simplier surrogate models.  

# ### DNDP info
# 
# $t(x_a)=T_a\Bigg(1+B_a\bigg(\dfrac{x}{c_a}\bigg)^4\Bigg)$
# 
# Follower objective: $T_a x_a + \frac{T_a B_a}{5c_a^4}x^5$
# 
# $B_a$ is alpha
# 
# $e_a$ is beta
# 
# $T_a$ is fftt
# 
# How are new link endpoints, capacities, and costs chosen?
# 
# How are leader budgets chosen? Sum of costs x  25/50/75%

# In[ ]:


fig_path = 'figs/'
data_path = 'data/'
result_path = 'results/'
instance_path = 'Instances/'


# In[ ]:


import gurobipy as gp  

import time
import collections
import numpy as np
import random
import pandas as pd
import pickle as pkl
import itertools
from scipy import stats
import math
import matplotlib.pyplot as plt

import pyomo.environ as pyo
from pyomo.environ import SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition

from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error

import gurobipy as gp
from gurobi_ml import add_predictor_constr
from gurobi_ml.sklearn import add_gradient_boosting_regressor_constr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# ## Functions for Instance Reading + Data Collection

# In[ ]:


def read_instance(net,NDP,B_prop,m,scal_time,scal_flow,timelimit,my_edges=None,my_data=None):

    #---read network and instance data in extended TNTP format
    nodes,links,capa,fftt,alpha,beta,links2,cost = read_network_data(net,NDP)

    #---read trip data in TNTP format
    OD,orig,dest = read_trip_data(net)

    N = list(nodes)
    A1 = list(links)
    if my_edges is None:
        A2 = list(links2)
    else:
        A2 = my_edges
        fftt = my_data['fftt']
        alpha = my_data['alpha_raw']
        beta = my_data['beta']
        e = my_data['exp']
        capa = my_data['capa']
        cost = my_data['cost']

    A = A1+A2
    O = list(orig)
    D = list(dest)

    #---create node-destination demand matrix
    TD = 0
    d = {(i,s):0 for i in N for s in D}
    for r in O:
        for s in D:
            d[r,s] = OD[r,s]
            TD += d[r,s]
    for s in D:
        d[s,s] = - sum(d[j,s] for j in O)

    #---create link delay function parameters from TNTP data
    #---link delay functional form: t[i,j] = T[i,j] + c[i,j]*(x**exp[i,j])
    T = {(i,j):fftt[i,j] for (i,j) in A}
    c = {(i,j):fftt[i,j]*alpha[i,j]/(capa[i,j]**beta[i,j]) for (i,j) in A}
    e = {(i,j):beta[i,j] for (i,j) in A}

    #---create link cost matrix and budget
    g = {(i,j):cost[i,j] for (i,j) in A2}
    TC = sum(g[i,j] for (i,j) in A2)
    B = B_prop*TC

    #---link delay function linear approximation using m uniform segments
    #---maximum link flow is instance-specific: value is calibrated for Sioux Falls under base demand
    Mflow = 1e5*scal_flow
    V = set([i for i in range(0,m+1)])
    a = {(i,j,v):float() for (i,j) in A for v in V}
    for (i,j) in A:
        cnt = 0
        step = Mflow/(len(V)-1)
        for v in V:
            a[i,j,v] = cnt*step
            cnt += 1

    #---time and flow scaling
    #---big-M coefficient is instance-specific: value is calibrated for Sioux Falls under base demand
    Mtt = 1e3*scal_time
    for (i,j) in A:
        T[i,j] = T[i,j]*scal_time
        c[i,j] = c[i,j]*scal_time/(scal_flow**e[i,j])
    for i in N:
        for s in D:
            d[i,s] = d[i,s]*scal_flow

    print('Instance',)
    print('Total scaled demand',TD*scal_flow)
    print('Total cost',TC,'Budget',B)

    data = {'nodes':N,'links1':A1,'links2':A2,'links':A,'orig':O,'dest':D,'fftt':T,'coef':c,'exp':e,
            'approx':V,'alpha':a,'cost':g,'demand':d,'budget':B,'Mflow':Mflow,'Mtt':Mtt,'timelimit':timelimit,
            'capa':capa, 'alpha_raw':alpha, 'beta':beta}
    return data

#---read network and instance data
#---nodes and links are sets; links is a set of tuples
#---cap, fftt, alpha, beta and cost are dictionaries where the keys are links

def read_network_data(net,NDP):
    network_data = open(instance_path +'SiouxFalls_DNDP_instances/' + net+NDP+'.txt','r')
    lines_net = network_data.readlines()
    network_data.close()
    nb_nodes = int(lines_net[1].split("\t")[0].split(" ")[3])
    nb_links = int(lines_net[3].split("\t")[0].split(" ")[3])
    nb_links2 = int(lines_net[4].split("\t")[0].split(" ")[4])
    cap = {};fftt = {};alpha = {};beta = {};cost = {};
    nodes = set();links = set();links2 = set();

    #---offset is network-specific
    if net == 'SF':
        offset = 9
    for i in range(offset,offset+nb_links):
        a = int(lines_net[i].split("\t")[1])
        b = int(lines_net[i].split("\t")[2])
        cap[(a,b)] = float(lines_net[i].split("\t")[3])
        fftt[(a,b)] = float(lines_net[i].split("\t")[5])
        alpha[(a,b)] = float(lines_net[i].split("\t")[6])
        beta[(a,b)] = float(lines_net[i].split("\t")[7])
        nodes.add(a)
        nodes.add(b)
        links.add((a,b))
    for i in range(offset+nb_links,offset+nb_links+nb_links2):
        a = int(lines_net[i].split("\t")[1])
        b = int(lines_net[i].split("\t")[2])
        cap[(a,b)] = float(lines_net[i].split("\t")[3])
        fftt[(a,b)] = float(lines_net[i].split("\t")[5])
        alpha[(a,b)] = float(lines_net[i].split("\t")[6])
        beta[(a,b)] = float(lines_net[i].split("\t")[7])
        cost[(a,b)] = float(lines_net[i].split("\t")[11])
        nodes.add(a)
        nodes.add(b)
        links2.add((a,b))
    return nodes,links,cap,fftt,alpha,beta,links2,cost


#---read trip data
#---OD_demand is a dictionary where the keys are OD pairs and the values are the demands
# instance_path = '/content/drive/My Drive/Bilevel_data/'
def read_trip_data(net):
    trip_data = open(instance_path + 'SiouxFalls_TNTP/' + net+'_trips.txt','r')
    trip_lines = trip_data.readlines()
    trip_data.close()
    nb_zones = int(trip_lines[0].split("\t")[0].split(" ")[3])
    total_flow = float(trip_lines[1].split("\t")[0].split(" ")[3])
    dest = 0
    line_nb = 0
    OD_demand = {}
    O = set()
    D = set()
    for line in trip_lines:
        if line_nb>=5:
            if line.split(" ")[0]=="Origin":
                orig = int(line.split(" ")[1])
                orig_flow = 0
            elif len(line.split())>0:
                k=0
                bouh = int(len(line.split())/3)
                for i in range(bouh):
                    dest = int(line.split()[k])
                    demand = float(line.split()[k+2].split(";")[0])
                    orig_flow += demand
                    k += 3
                    OD_demand[(orig,dest)] = demand
                    O.add(orig)
                    D.add(dest)
        line_nb += 1
    return OD_demand,O,D



# In[ ]:


#---solve TAP as a convex NLP using Pyomo and IPOPT for a given y-vector
def TAP_cvx(data,yopt):
    time0 = time.time()
    N = data['nodes']
    A = data['links']
    A1 = data['links1']
    A2 = data['links2']
    D = data['dest']
    T = data['fftt']
    c = data['coef']
    e = data['exp']
    d = data['demand']
    Mflow = data['Mflow']
    Mtt = data['Mtt']

    TAP = pyo.ConcreteModel()
    TAP.x = pyo.Var([(i,j) for (i,j) in A],domain=pyo.NonNegativeReals)
    TAP.xc = pyo.Var([(i,j) for (i,j) in A],[s for s in D],domain=pyo.NonNegativeReals)

    TAP.cons = pyo.ConstraintList()
    for i in N:
        for s in D:
            TAP.cons.add(sum(TAP.xc[i,j,s] for j in N if (i,j) in A) - sum(TAP.xc[j,i,s] for j in N if (j,i) in A) == d[i,s])

    TAP.flow = pyo.ConstraintList()
    for (i,j) in A:
        TAP.flow.add(sum(TAP.xc[i,j,s] for s in D) == TAP.x[i,j])

    TAP.dsgn = pyo.ConstraintList()
    for (i,j) in A2:
        TAP.dsgn.add(TAP.x[i,j] <= yopt[i,j]*Mflow)

    TAP.obj = pyo.Objective(expr=(sum(TAP.x[i,j]*T[i,j] + c[i,j]/(e[i,j]+1)*((TAP.x[i,j])**(e[i,j]+1)) for (i,j) in A)))

    opt = SolverFactory('ipopt')#, executable='ipopt/')
    results = opt.solve(TAP)
    
    time_cvx = time.time() - time0

    if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
        x_cvx = {}
        t_cvx = {}
        TSTT_cvx = 0
        for (i,j) in A:
            x_cvx[i,j] = pyo.value(TAP.x[i,j])
            if (i,j) in A2 and yopt[i,j] < 1e-4:
                t_cvx[i,j] = Mtt
            else:
                t_cvx[i,j] = T[i,j] + c[i,j]*(x_cvx[i,j]**(e[i,j]))
            TSTT_cvx += x_cvx[i,j]*T[i,j] + c[i,j]*(x_cvx[i,j]**(e[i,j]+1))
        LL_obj = pyo.value(TAP.obj)
        return TSTT_cvx, LL_obj, time_cvx, x_cvx

    elif (results.solver.termination_condition == TerminationCondition.infeasible):
        print('infeasible',results.solver.status,results.solver.termination_condition)
        return -1,-1,time_cvx,{}
    else:
        print('Solver Status',results.solver.status,results.solver.termination_condition)
        return -1,-1,time_cvx,{}


# In[ ]:


def get_projects():

    net = 'SF'
    attributes = ['fftt','alpha_raw','beta','capa','exp','cost']
    all_data = {attribute:{} for attribute in attributes}
    all_projects = set()
    # iteration over the 10 20-link instances only
    for i in range(2,3):
        for j in range(1,11):
            NDP = '_DNDP_'+str(i*10)+'_'+str(j)
            print('instance',NDP)
            
            #---read instance data
            data = read_instance(net,NDP,0,100,1e-0,1e-3,600)
            
            A2 = data['links2']
            all_projects.update(A2)
            for attribute in attributes:
                all_data[attribute].update(data[attribute])

    all_projects_list = list(all_projects)
    project_mapping = {all_projects_list[i]:i for i in range(len(all_projects_list))}

    return all_data, project_mapping


# ## MKKT Baseline Implementation

# In[ ]:


def model_MKKT_gurobi(data, timelimit_grb=60):
    print('model_MKKT_gurobi timelimit', timelimit_grb)

    MKKT = gp.Model()
    N = data['nodes']
    A = data['links']
    A1 = data['links1']
    A2 = data['links2']
    D = data['dest']
    V = data['approx']
    T = data['fftt']
    c = data['coef']
    e = data['exp']
    a = data['alpha']
    g = data['cost']
    B = data['budget']
    d = data['demand']
    Mflow = data['Mflow']
    Mtt = data['Mtt']
    timelimit = data['timelimit']
    Vp = V.difference({0})

    #---primal follower variables
    A_lists = list(zip(*A))
    A2_lists = list(zip(*A2))
    indices_AD = [(i,j,s) for (i,j) in A for s in D]
    indices_AV = [(i,j,v) for (i,j) in A for v in V]
    indices_ND = [(i,s) for i in N for s in D]
    indices_A = list(zip(A_lists[0], A_lists[1]))
    indices_A2 = list(zip(A2_lists[0], A2_lists[1]))

    xc = MKKT.addVars(indices_AD)
    ll = MKKT.addVars(indices_AV)
    lr = MKKT.addVars(indices_AV)

    #---dual follower variables
    pi = MKKT.addVars(indices_ND)
    beta = MKKT.addVars(indices_A)
    gamma = MKKT.addVars(indices_A)
    mu = MKKT.addVars(indices_A2)

    #---leader and linearization variables
    y = MKKT.addVars(indices_A2, vtype=gp.GRB.BINARY)
    phi = MKKT.addVars(indices_A2)

    MKKT.addConstr(sum(y[i,j]*g[i,j] for (i,j) in A2) <= B)

    for i in N:
        for s in D:
            MKKT.addConstr(sum(xc[i,j,s] for j in N if (i,j) in A)
                               - sum(xc[j,i,s] for j in N if (j,i) in A) == d[i,s])
    for (i,j) in A:
        MKKT.addConstr(sum(xc[i,j,s] for s in D) == sum(ll[i,j,v]*a[i,j,v-1] + lr[i,j,v]*a[i,j,v] for v in Vp))
        MKKT.addConstr(sum(ll[i,j,v] + lr[i,j,v] for v in V) == 1)

    for (i,j) in A2:
        for s in D:
            MKKT.addConstr(xc[i,j,s] <= y[i,j]*Mflow)

    for (i,j) in A1:
        for s in D:
            MKKT.addConstr(pi[i,s] - pi[j,s] + beta[i,j] >= - T[i,j])
    for (i,j) in A2:
        for s in D:
            MKKT.addConstr(pi[i,s] - pi[j,s] + beta[i,j] + mu[i,j] >= - T[i,j])
    for (i,j) in A:
        for v in Vp:
            MKKT.addConstr(- beta[i,j]*a[i,j,v] + gamma[i,j] >= - (c[i,j]/(e[i,j]+1))*(a[i,j,v]**(e[i,j]+1)))
        MKKT.addConstr(gamma[i,j] >= 0)

    for (i,j) in A2:
        MKKT.addConstr(phi[i,j] <= mu[i,j])
        MKKT.addConstr(phi[i,j] >= mu[i,j] - (1 - y[i,j])*Mtt)
        MKKT.addConstr(phi[i,j] <= y[i,j]*Mtt)
        MKKT.addConstr(phi[i,j] >= 0)

    # primal-dual constraint
    MKKT.addConstr(sum(T[i,j]*sum(xc[i,j,s] for s in D)
                           + (c[i,j]/(e[i,j]+1))*sum(ll[i,j,v]*(a[i,j,v-1]**(e[i,j]+1)) + lr[i,j,v]*(a[i,j,v]**(e[i,j]+1))
                                                     for v in Vp) for (i,j) in A)
                       <= -(sum(pi[i,s]*d[i,s] for i in N for s in D)
                           + sum(gamma[i,j] for (i,j) in A) + sum(phi[i,j]*Mflow for (i,j) in A2)))

    for (i,j) in A:
        for s in D:
            MKKT.addConstr(xc[i,j,s] >= 0)
        for v in V:
            MKKT.addConstr(ll[i,j,v] >= 0)
            MKKT.addConstr(lr[i,j,v] >= 0)
    for (i,j) in A2:
        MKKT.addConstr(mu[i,j] >= 0)

    MKKT.setObjective(sum(T[i,j]*sum(xc[i,j,s] for s in D) for (i,j) in A)
                 + sum(c[i,j]*sum(ll[i,j,v]*(a[i,j,v-1]**(e[i,j]+1)) + lr[i,j,v]*(a[i,j,v]**(e[i,j]+1))
                                  for v in Vp) for (i,j) in A))

    MKKT.Params.MIPFocus = 1
    MKKT.Params.TimeLimit = timelimit_grb
    MKKT.Params.OutputFlag = 0

    MKKT.optimize()

    time_MKKT = MKKT.Runtime
    if MKKT.Status == 'INFEASIBLE' or MKKT.SolCount == 0:
        print('status\t%s' % MKKT.Status)
        return -1,-1,time_MKKT,{},{}

    UB_MKKT = MKKT.getObjective().getValue()
    gap_MKKT = MKKT.MIPGap
    print('\n---MKKT-------------------------------------')
    print('status\t%s' % MKKT.Status)
    print('time\t%.2f' % time_MKKT)
    print('OPT\t%.3f, GAP\t%.3f' % (UB_MKKT, gap_MKKT))
    yopt = {(i,j):y[i,j].X for (i,j) in A2}
    # xcopt = {(i,j,s):xc[i,j,s].X for (i,j) in A for s in D}
    # xopt = {(i,j):sum(xcopt[i,j,s] for s in D) for (i,j) in A}
    # xopt = {(i,j):sum(xcopt[i,j,s] for s in D) for (i,j) in A}
    return UB_MKKT,gap_MKKT,time_MKKT,yopt,None


# ## Data Collection

# In[ ]:


all_data, project_mapping = get_projects()


# In[ ]:


# definte network
net = 'SF'

# number of samples for train/test sets
num_samples = {'train' : 1000, 'test' : 100}

x_all = {}
y_ul_all = {}
y_ll_all = {}


# In[ ]:


time_data = time.time()

for mode in ['train', 'test']:
    x_all[mode] = np.zeros((num_samples[mode], len(project_mapping)))
    y_ul_all[mode] = np.zeros(num_samples[mode])
    y_ll_all[mode] = np.zeros(num_samples[mode])
    
    for i in range(num_samples[mode]):
        NDP = '_DNDP_10_1'
        
        num_edges_added = random.randint(1,20)
        
        my_edges = random.sample(list(project_mapping.keys()), num_edges_added)
        print(my_edges)
        
        #---read instance data
        data = read_instance(net,NDP,0,100,1e-0,1e-3,600, my_edges=my_edges, my_data=all_data)
        
        #---get total cost
        TC = sum(data['cost'][i,j] for (i,j) in data['cost'])
        
        #--- 1 level of budget: 25% of TC
        for k in range(1,2):
            data['budget'] = TC*k/4
            print('>>> NDP',i,100*k/4)
            # UB_MKKT,gap_MKKT,time_MKKT,y_MKKT,x_MKKT = model_MKKT_gurobi(data)
        
        #---determine true TSTT using convex local solver
        y_MKKT = {(i,j):1 for (i,j) in my_edges}
        TSTT_cvx, LL_obj, time_cvx, x_cvx = TAP_cvx(data,y_MKKT)
        
        x_in = np.zeros(len(project_mapping))
        x_in[[project_mapping[edge] for edge in my_edges]] = 1
        y_out = TSTT_cvx
        
        x_all[mode][i] = x_in
        y_ul_all[mode][i] = y_out
        y_ll_all[mode][i] = LL_obj

time_data = time.time() - time_data


# In[ ]:


print("Data collection time:", time_data)


# In[ ]:


# save data
data = {'x' : x_all, 'y_ul_all' : y_ul_all, 'y_ll_all' : y_ll_all, 'time' : time_data}

with open('data/ml_data.pkl', 'wb') as p:
    pkl.dump(data, p)


# In[ ]:


# # load data
# with open(f'{data_path}ml_data.pkl', 'rb') as p:
#     data = pkl.load(p)

# x_all = data['x']
# y_ul_all = data['y_ul_all']
# y_ll_all = data['y_ll_all']


# In[ ]:


plt.hist(data['y_ul_all']['train'], bins=20)
plt.title("Upper-Level Objectives")
plt.xlabel("Objective")
plt.ylabel("Freq")
plt.savefig(f"{fig_path}/label_dist_upper.png")


# In[ ]:


plt.hist(data['y_ll_all']['train'], bins=20)

plt.title("Lower-Level Objectives")
plt.xlabel("Objective")
plt.ylabel("Freq")
# plt.show()
plt.savefig(f"{fig_path}/label_dist_lower.png")


# ## Preprocess Data

# In[ ]:


seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# In[ ]:


min_max_scaler_ul = preprocessing.MinMaxScaler()
min_max_scaler_ul.fit(y_ul_all['train'].reshape(-1, 1))
y_ul_all['train'] = min_max_scaler_ul.transform(y_ul_all['train'].reshape(-1, 1))
y_ul_all['test'] = min_max_scaler_ul.transform(y_ul_all['test'].reshape(-1, 1))

min_max_scaler_ll = preprocessing.MinMaxScaler()
min_max_scaler_ll.fit(y_ll_all['train'].reshape(-1, 1))
y_ll_all['train'] = min_max_scaler_ll.transform(y_ll_all['train'].reshape(-1, 1))
y_ll_all['test'] = min_max_scaler_ll.transform(y_ll_all['test'].reshape(-1, 1))


print("Upper-level:", min_max_scaler_ul.data_min_, min_max_scaler_ul.data_max_)
print("Lower-level:", min_max_scaler_ll.data_min_, min_max_scaler_ll.data_max_)


# ## Train GBT

# ### Lower-Level

# In[ ]:


time_gbt_ul = time.time()
gbt_ul = GradientBoostingRegressor().fit(x_all['train'], y_ul_all['train'])
time_gbt_ul = time.time() - time_gbt_ul


# In[ ]:


# evaluate model
pred_tr = gbt_ul.predict(x_all['train'])
pred_te = gbt_ul.predict(x_all['test'])

print('GBT train MSE:', mean_squared_error(pred_tr, y_ul_all['train']))
print('GBT test MSE: ', mean_squared_error(pred_te, y_ul_all['test']))
print("Time GBT:", time_gbt_ul)


# In[ ]:


gbt_ul._time = time_gbt_ul


# In[ ]:


with open(f"{data_path}gbt_upper.pkl", "wb") as p:
    pkl.dump(gbt_ul, p)


# ### Upper-Level

# In[ ]:


time_gbt_ll = time.time()
gbt_ll = GradientBoostingRegressor().fit(x_all['train'], y_ll_all['train'])
time_gbt_ll = time.time() - time_gbt_ll


# In[ ]:


# evaluate model
pred_tr = gbt_ll.predict(x_all['train'])
pred_te = gbt_ll.predict(x_all['test'])

print('GBT train MSE:', mean_squared_error(pred_tr, y_ll_all['train']))
print('GBT test MSE: ', mean_squared_error(pred_te, y_ll_all['test']))
print("Time GBT:", time_gbt_ll)


# In[ ]:


gbt_ll._time = time_gbt_ll


# In[ ]:


with open(f"{data_path}gbt_lower.pkl", "wb") as p:
    pkl.dump(gbt_ll, p)


# ## Train NN

# In[ ]:


from sklearn.metrics import r2_score

def test_model_predictions(a_dataset, device=torch.device("cuda"), print_predictions=False, get_ranking=False, mae=False, verbose=True):
    test_loader = DataLoader(a_dataset, shuffle=False, batch_size=len(a_dataset))
    err = 0
    err_max_over = -1
    err_max_under = -1
    counter = 0
    
    labels_instance = []
    outputs_instance = []
    
    res_kendall_all = []
    
    with torch.no_grad():
      for i,(features,labels_all) in enumerate(test_loader,0):
          features = features.reshape(-1,input_size).to(device)
          outputs_all = nn_ul(features).to(device)
          if print_predictions:
              print(labels_all, outputs_all)
          labels_all = labels_all.cpu()
          outputs_all = outputs_all.cpu()
    r2_nn = r2_score(outputs_all.numpy(), labels_all.numpy())
    print('r2nn =', r2_nn)
    
    for i in range(len(labels_all)):
      outputs = outputs_all[i]
      labels = labels_all[i]
      err_cur = np.abs(labels-outputs)/labels if not mae else np.abs(labels-outputs)
      err += err_cur
      err_max_over = np.max([err_cur, err_max_over]) if outputs - labels > 0 else err_max_over
      err_max_under = np.max([err_cur, err_max_under]) if outputs - labels < 0 else err_max_under
    
      labels_instance += [labels.item()]
      outputs_instance += [outputs.item()]
    
      if get_ranking and (i % num_sample_perinst) == 0 and i > 0:
        res_kendall = stats.kendalltau(labels_instance, outputs_instance)
        res_kendall_all += [res_kendall.statistic]
        
        labels_instance = []
        outputs_instance = []
    
      counter += 1
        
    try:
        mape = err/counter
    except:
        mape = err[0,0].item()/counter
    if verbose:
        print("Mean Percentage Error =", mape)
        print("max over/underestimate \%:", err_max_over, err_max_under)
    if get_ranking:
        print("Kendall:", np.min(res_kendall_all), np.median(res_kendall_all), np.max(res_kendall_all))
    
    print("-------------------------------------")
    return mape, err_max_over, err_max_under


# In[ ]:


class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, num_layers, output_relu=False):
        super(FeedForward, self).__init__()
        layers = collections.OrderedDict()
        for i in range(num_layers+1):
          in_size = input_size if i == 0 else hidden_size
          out_size = embedding_size if i == num_layers else hidden_size
          layers[str(i)] = nn.Linear(in_size, out_size, bias=True)
          if i < num_layers or output_relu:
            layers["relu" + str(i)] = nn.ReLU()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.sequential = torch.nn.Sequential(layers)

    def forward(self,x):
        out = self.sequential(x)
        return out


# ### Upper-Level

# In[ ]:


# initialize network
nn_ul = FeedForward(input_size=len(project_mapping), hidden_size=16, embedding_size=1, num_layers=1)


# In[ ]:


# initialize dataset
dataset = TensorDataset(torch.from_numpy(x_all['train']).float(), torch.from_numpy(y_ul_all['train']).float())
dataset_test = TensorDataset(torch.from_numpy(x_all['test']).float(), torch.from_numpy(y_ul_all['test']).float())

training_size = len(dataset)
input_size = len(dataset[0][0])
print(training_size, input_size)

batch_size = 64
loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

total_size = len(loader)


# In[ ]:


# initialize optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(nn_ul.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, cooldown=100, verbose=True)


# In[ ]:


# device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
nn_ul.to(device)


# In[ ]:


# Train the model
nn_ul_time = time.time()

num_epochs = 100
loss_epoch = []
val_mape_min = math.inf
loss_epoch_min_idx = 0
epoch = 0
while epoch < (num_epochs):
    loss_epoch += [0]
    for i,(features, labels) in enumerate(loader,0):
        features = features.reshape(-1,input_size).to(device)
        labels = labels.to(device)

        # forward
        outputs = nn_ul(features).to(device)
        loss = criterion(outputs,labels)
        loss_epoch[-1] += loss.item()/training_size

        optimizer.zero_grad()

        # backpropagation
        loss.backward()
        optimizer.step()
   
    epoch += 1

nn_ul_time = time.time() - nn_ul_time
print("Training time: ", nn_ul_time)


# In[ ]:


print("MSE final epoch:", loss_epoch[-1])


# In[ ]:


plt.plot(loss_epoch)
plt.title('NN lower-level loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.savefig(f"{fig_path}/loss_nn_lower.png")


# In[ ]:


nn_ul._time = nn_ul_time


# In[ ]:


path_nn = f'{data_path}nn_upper.pt'
torch.save(nn_ul, path_nn)


# In[ ]:


nn_ul.cpu()


# ### Lower-Level

# In[ ]:


# initialize network
nn_ll = FeedForward(input_size=len(project_mapping), hidden_size=8, embedding_size=1, num_layers=1)


# In[ ]:


# initialize dataset
dataset = TensorDataset(torch.from_numpy(x_all['train']).float(), torch.from_numpy(y_ll_all['train']).float())
dataset_test = TensorDataset(torch.from_numpy(x_all['test']).float(), torch.from_numpy(y_ll_all['test']).float())

training_size = len(dataset)
input_size = len(dataset[0][0])
print(training_size, input_size)

batch_size = 64
loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

total_size = len(loader)


# In[ ]:


# initialize optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(nn_ll.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, cooldown=100, verbose=True)


# In[ ]:


# device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
nn_ll.to(device)


# In[ ]:


# Train the model
nn_ll_time = time.time()

num_epochs = 100
loss_epoch = []
val_mape_min = math.inf
loss_epoch_min_idx = 0
epoch = 0
while epoch < (num_epochs):
    loss_epoch += [0]
    for i,(features, labels) in enumerate(loader,0):
        features = features.reshape(-1,input_size).to(device)
        labels = labels.to(device)

        # forward
        outputs = nn_ll(features).to(device)
        loss = criterion(outputs,labels)
        loss_epoch[-1] += loss.item()/training_size

        optimizer.zero_grad()

        # backpropagation
        loss.backward()
        optimizer.step()
   
    epoch += 1

nn_ll_time = time.time() - nn_ll_time

print("Training time: ", nn_ll_time)


# In[ ]:


print("MSE final epoch:", loss_epoch[-1])


# In[ ]:


plt.plot(loss_epoch)
plt.title('NN upper-level loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.savefig(f"{fig_path}/loss_nn_upper.png")


# In[ ]:


nn_ll._time = nn_ll_time


# In[ ]:


path_nn = f'{data_path}nn_lower.pt'
torch.save(nn_ll, path_nn)


# In[ ]:


nn_ll.cpu()


# # Evaluate

# ### Surrogate Models

# In[ ]:


def get_gurobi_model(data, pred_model, model_type, approx_type, project_mapping, nonconvex=-1, timelimit=60):
    grb_model = gp.Model()
    
    N = data['nodes']
    A = data['links']
    A1 = data['links1']
    A2 = data['links2']
    D = data['dest']
    V = data['approx']
    T = data['fftt']
    c = data['coef']
    e = data['exp']
    a = data['alpha']
    g = data['cost']
    B = data['budget']
    d = data['demand']
    Mflow = data['Mflow']
    Mtt = data['Mtt']
    Vp = V.difference({0})
    
    #---primal follower variables
    A_lists = list(zip(*A))
    A2_lists = list(zip(*A2))
    indices_AD = [(i,j,s) for (i,j) in A for s in D]
    indices_AV = [(i,j,v) for (i,j) in A for v in V]
    indices_ND = [(i,s) for i in N for s in D]
    indices_A = list(zip(A_lists[0], A_lists[1]))
    indices_A2 = list(zip(A2_lists[0], A2_lists[1]))
    
    xc = grb_model.addVars(indices_AD, name="xc")
    
    #---leader
    y = grb_model.addVars(project_mapping.keys(), vtype=gp.GRB.BINARY, name="y")
    # leader variable fixed to zero if not candidate edge
    for edge, edge_id in project_mapping.items():
        if edge not in A2:
            grb_model.addConstr(y[edge] == 0)

    # leader budget constraint
    grb_model.addConstr(gp.quicksum(y[edge] * g[edge] for edge in A2) <= B) # y @ edge_costs <= B)

    # add prediction for network/gbt
    pred_var = grb_model.addMVar((1,), lb=-gp.GRB.INFINITY, name="pred")
    if model_type == "nn":
        pred_constr = add_predictor_constr(grb_model, pred_model.sequential, y, pred_var)
    elif model_type == "gbt":
        pred_constr = add_gradient_boosting_regressor_constr(grb_model, pred_model, y, pred_var)
    else:
        raise Exception(f"model_type ({model_type}) not implemented")

    # add constraints/objective for upper/lower specific approximations
    if approx_type == "lower":
        # follower flow variables, original and to the power 5
        x = grb_model.addVars(indices_A, name="x")
        x5p = grb_model.addVars(indices_A, name="x5p")

        grb_model._x = x
        grb_model._x5p = x5p

        # neural network slack variable
        slack = grb_model.addMVar((1,), name="slack")
        
        # follower variable to the power 5
        for edge in A:
            grb_model.addGenConstrPoly(x[edge], x5p[edge], [1, 0, 0, 0, 0, 0], name=f"poly_{edge}")
        
        # follower flow balance constraint
        for i in N:
            for s in D:
                grb_model.addConstr(gp.quicksum(xc[i,j,s] for j in N if (i,j) in A) -
                                    gp.quicksum(xc[j,i,s] for j in N if (j,i) in A) == d[i,s])
        
        # follower edge flow = sum of all flows through edge to all destination nodes
        for (i,j) in A:
            grb_model.addConstr(gp.quicksum(xc[i,j,s] for s in D) == x[i,j])
        
        # follower can use edge only if it is chosen by leader
        for (i,j) in A2:
            grb_model.addConstr(x[i,j] <= y[i,j]*Mflow)

        # upper/lower objectives
        ul_obj = gp.quicksum(T[i,j]*x[i,j] + c[i,j] * x5p[i,j] for (i,j) in A)
        ll_obj = gp.quicksum(T[i,j]*x[i,j] + (c[i,j] / (e[i,j] + 1)) * x5p[i,j] for (i,j) in A)

        # constraint to track upper-level objective
        ul_obj_var = grb_model.addMVar((1,), name="ul_obj")
        grb_model.addConstr(ul_obj_var == ul_obj)
        
        # scaled prediction
        scaler = min_max_scaler_ll
        pred_sc = pred_var * (scaler.data_max_ - scaler.data_min_) + scaler.data_min_
        
        grb_model.setObjective(ul_obj + slack_obj_coef * slack, gp.GRB.MINIMIZE)
        grb_model.addConstr(ll_obj <= pred_sc + slack) # justin
      
    elif approx_type == "upper":
        grb_model.setObjective(pred_var, gp.GRB.MINIMIZE)

    else:
        raise Exception(f"approx_type ({approx_type}) not implemented")
        
    grb_model.Params.OutputFlag = 0
    grb_model.Params.MIPFocus = 1
    grb_model.Params.TimeLimit = timelimit
    grb_model.Params.FuncNonlinear = nonconvex
    
    grb_model.optimize()
    
    yopt = {(i,j):y[i,j].X for (i,j) in A2}
    
    return yopt, grb_model.Runtime, grb_model


# # Experiements with $\lambda=1$ 

# In[ ]:


ml_timelimit = 5
slack_obj_coef = 1.0


# In[ ]:


results = []

#---iterate over 10- and 20-link instance sets
for i in range(1,3):

    #---iterate over 10 instances
    for j in range(1,11):

        NDP = '_DNDP_'+str(i*10)+'_'+str(j)
        instance_name = 'SF' + NDP
        print('instance', instance_name)

        #---read instance data
        data = read_instance(net,NDP,0,100,1e-0,1e-3,600)

        #---get total cost
        TC = sum(data['cost'][i,j] for (i,j) in data['cost'])

        #---iterate over 3 levels of budget: 25%, 50% and 75% of TC
        for k in range(1,4):
            data['budget'] = TC*k/4
            print('>>> NDP',i*10,j,100*k/4)

            mlvariants = [['nn', 'upper', 0],
                          ['nn', 'lower', 0],
                          ['gbt', 'upper', 0],
                          ['gbt', 'lower', 0]]

            for mlvariant in mlvariants:
                print('mlvariant = ', mlvariant)
                if mlvariant[0] == 'nn' and mlvariant[1] == 'upper':
                    pred_model = nn_ul
                if mlvariant[0] == 'nn' and mlvariant[1] == 'lower':
                    pred_model = nn_ll
                if mlvariant[0] == 'gbt' and mlvariant[1] == 'upper':
                    pred_model = gbt_ul
                if mlvariant[0] == 'gbt' and mlvariant[1] == 'lower':
                    pred_model = gbt_ll

                # solve surrogate ML model
                y_ml, time_ml, grb_model = get_gurobi_model(
                    data=data, 
                    pred_model=pred_model, 
                    model_type=mlvariant[0],
                    approx_type=mlvariant[1], 
                    project_mapping=project_mapping, 
                    nonconvex=mlvariant[2], 
                    timelimit=ml_timelimit)
                
                #---determine true TSTT using convex local solver
                trueval_ml, lower_obj_ml, time_cvx_ml, x_cvx_ml = TAP_cvx(data, y_ml)
                
                print("  Upper-level obj: ", trueval_ml)
                print("  Lower-level obj: ", lower_obj_ml)
                
                if mlvariant[1] == 'upper':
                    scaler = min_max_scaler_ul
                    obj_sc = grb_model.objVal * (scaler.data_max_ - scaler.data_min_) + scaler.data_min_
                    print("  Upper-level pred:", obj_sc[0])
                    print("  Pred Gap:        ", 100 * np.abs(trueval_ml - obj_sc[0]) / trueval_ml)

                if mlvariant[1] == 'lower':
                    pred = grb_model.getVarByName("pred[0]").x
                    slack = grb_model.getVarByName("slack[0]").x
                    ul_obj_surr = grb_model.getVarByName("ul_obj[0]").x
                    
                    scaler = min_max_scaler_ll
                    pred_sc = pred * (scaler.data_max_ - scaler.data_min_) + scaler.data_min_
                    print("  Lower-level pred:", pred_sc[0])
                    print("  Upper-level surr:", ul_obj_surr)
                    print("  Pred Gap:        ", 100 * np.abs(lower_obj_ml - pred_sc) / lower_obj_ml)
                    print("  Surr Gap:        ", 100 * np.abs(trueval_ml - ul_obj_surr) / trueval_ml)
                    print("  slack:           ", slack)

                print("  Time:   ", time_ml)
                
                results += [[
                  instance_name,
                  k/4.0,
                  i*10,
                  '_'.join(str(att) for att in mlvariant),
                  trueval_ml,
                  lower_obj_ml,
                  time_ml
                ]]


# In[ ]:


df = pd.DataFrame(results)
df.columns = ['instance_name', 'budget', 'n_edges', 'method', 'obj', 'lower_obj', 'time']
summary = df.groupby(['budget', 'n_edges', 'method']).agg({'obj': ['mean'], 'time': ['mean']})


# In[ ]:


df.to_csv(f'{result_path}ml_results_tml-{ml_timelimit}_skl-{slack_obj_coef}.csv')


# # Experiements with $\lambda=0.1$ 

# In[ ]:


ml_timelimit = 5
slack_obj_coef = 0.1


# In[ ]:


results = []

#---iterate over 10- and 20-link instance sets
for i in range(1,3):

    #---iterate over 10 instances
    for j in range(1,11):

        NDP = '_DNDP_'+str(i*10)+'_'+str(j)
        instance_name = 'SF' + NDP
        print('instance', instance_name)

        #---read instance data
        data = read_instance(net,NDP,0,100,1e-0,1e-3,600)

        #---get total cost
        TC = sum(data['cost'][i,j] for (i,j) in data['cost'])

        #---iterate over 3 levels of budget: 25%, 50% and 75% of TC
        for k in range(1,4):
            data['budget'] = TC*k/4
            print('>>> NDP',i*10,j,100*k/4)

            mlvariants = [['nn', 'lower', 0],
                          ['gbt', 'lower', 0]]

            for mlvariant in mlvariants:
                print('mlvariant = ', mlvariant)
                if mlvariant[0] == 'nn' and mlvariant[1] == 'upper':
                    pred_model = nn_ul
                if mlvariant[0] == 'nn' and mlvariant[1] == 'lower':
                    pred_model = nn_ll
                if mlvariant[0] == 'gbt' and mlvariant[1] == 'upper':
                    pred_model = gbt_ul
                if mlvariant[0] == 'gbt' and mlvariant[1] == 'lower':
                    pred_model = gbt_ll

                # solve surrogate ML model
                y_ml, time_ml, grb_model = get_gurobi_model(
                    data=data, 
                    pred_model=pred_model, 
                    model_type=mlvariant[0],
                    approx_type=mlvariant[1], 
                    project_mapping=project_mapping, 
                    nonconvex=mlvariant[2], 
                    timelimit=ml_timelimit)

                #---determine true TSTT using convex local solver
                trueval_ml, lower_obj_ml, time_cvx_ml, x_cvx_ml = TAP_cvx(data, y_ml)
                
                print("  Upper-level obj: ", trueval_ml)
                print("  Lower-level obj: ", lower_obj_ml)
                
                if mlvariant[1] == 'upper':
                    scaler = min_max_scaler_ul
                    obj_sc = grb_model.objVal * (scaler.data_max_ - scaler.data_min_) + scaler.data_min_
                    print("  Upper-level pred:", obj_sc[0])
                    print("  Pred Gap:        ", 100 * np.abs(trueval_ml - obj_sc[0]) / trueval_ml)

                if mlvariant[1] == 'lower':
                    pred = grb_model.getVarByName("pred[0]").x
                    slack = grb_model.getVarByName("slack[0]").x
                    ul_obj_surr = grb_model.getVarByName("ul_obj[0]").x
                    
                    scaler = min_max_scaler_ll
                    pred_sc = pred * (scaler.data_max_ - scaler.data_min_) + scaler.data_min_
                    print("  Lower-level pred:", pred_sc[0])
                    print("  Upper-level surr:", ul_obj_surr)
                    print("  Pred Gap:        ", 100 * np.abs(lower_obj_ml - pred_sc) / lower_obj_ml)
                    print("  Surr Gap:        ", 100 * np.abs(trueval_ml - ul_obj_surr) / trueval_ml)
                    print("  slack:           ", slack)

                print("  Time:   ", time_ml)
                
                results += [[
                  instance_name,
                  k/4.0,
                  i*10,
                  '_'.join(str(att) for att in mlvariant),
                  trueval_ml,
                  lower_obj_ml,
                  time_ml
                ]]


# In[ ]:


df = pd.DataFrame(results)
df.columns = ['instance_name', 'budget', 'n_edges', 'method', 'obj', 'lower_obj', 'time']
summary = df.groupby(['budget', 'n_edges', 'method']).agg({'obj': ['mean'], 'time': ['mean']})


# In[ ]:


df.to_csv(f'{result_path}ml_results_tml-{ml_timelimit}_skl-{slack_obj_coef}.csv')


# ## MKKT Baseline

# In[ ]:


results = []

#---iterate over 10- and 20-link instance sets
for i in range(1,3):

    #---iterate over 10 instances
    for j in range(1,11):

        NDP = '_DNDP_'+str(i*10)+'_'+str(j)
        instance_name = 'SF' + NDP
        print('instance', instance_name)

        #---read instance data
        data = read_instance(net,NDP,0,100,1e-0,1e-3,600)

        #---get total cost
        TC = sum(data['cost'][i,j] for (i,j) in data['cost'])

        #---iterate over 3 levels of budget: 25%, 50% and 75% of TC
        for k in range(1,4):
            data['budget'] = TC*k/4
            print('>>> NDP',i*10,j,100*k/4)
                
            timelimits = [5, 10, 30]
            for timelimit in timelimits:
                UB_MKKT,gap_MKKT,time_MKKT,y_MKKT,x_MKKT = model_MKKT_gurobi(data, timelimit_grb=timelimit)
                
                TSTT_cvx, lower_obj_mkkt, time_cvx, x_cvx = TAP_cvx(data,y_MKKT)
                
                if UB_MKKT == -1:
                    TSTT_cvx = -1
                else:
                    #---determine true TSTT using convex local solver
                    TSTT_cvx, lower_obj_cvx, time_cvx, x_cvx = TAP_cvx(data,y_MKKT)

                print("Timelimit:", timelimit)
                print("  Upper-level obj: ", TSTT_cvx)
                print("  Lower-level obj: ", lower_obj_cvx)
                print('  Time (mkkt):     ', time_MKKT )
                print('  Time (check):    ', time_cvx)
                
                results += [[
                    instance_name,
                    k/4.0,
                    i*10,
                    'mkkt_%d' % timelimit,
                    TSTT_cvx,
                    lower_obj_cvx,
                    time_MKKT
                ]]


# In[ ]:


df = pd.DataFrame(results)
df.columns = ['instance_name', 'budget', 'n_edges', 'method', 'obj', 'lower_obj', 'time']
summary = df.groupby(['budget', 'n_edges', 'method']).agg({'obj': ['mean'], 'time': ['mean']})


# In[ ]:


df.to_csv(f'{result_path}baseline_results.csv')


# In[ ]:





# In[ ]:




