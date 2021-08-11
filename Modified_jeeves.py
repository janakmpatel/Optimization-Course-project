#!/usr/bin/env python3
# -*- Modified HV by Team 12 -*-
import time
initial=time.time()
import numpy as np
import matplotlib.pyplot as plt
#function
def func(x):
    fx = 100*((x[1]-x[0]**2)**2)+((1-x[0])**2)
    return fx
#FV method
def HV(func,x_init,max_iter,epsilon,delta,alpha):
    k=0
    it =0
    xk=x_init
    x_vec_save=[]
    f_vec_save=[]
    x_vec_save.append(x_init)             #To create array of all x at different iteration
    f_vec_save.append(func(x_init))
    n=x_init.shape[0]
    e=np.eye(n)
    while delta>epsilon and k<max_iter:
        for i in range(n):
            xi = xk
            he=delta*(np.vstack(e[:,i]))
            x_p=xk+he
            x_n=xk-he
            f=func(xk)
            f1=func(x_p)
            f2=func(x_n)
            if f1<f:
                xk_1=x_p
            elif f2<f:
                xk_1=x_n
            else:
                xk_1=xk
            xk=xk_1
          
        if (xk_1).all==(xi).all:
            delta=delta/alpha
        else:
            
            sk = xk_1-xi
            for i in range (max_iter):
                
                xk_2 = xk_1+sk
                f_new = func(xk_2)
                f_old = func(xk_1)
                if f_new<f_old:
                    x_converged=xk_2
                    xk_1 = xk_2
                else:
                    x_converged = xk_1
                    break
                i+=1
            
        k=k+1
        it=k
        xk = x_converged
        x_vec_save= np.append(x_vec_save,x_converged)
        f_save = func(x_converged)
        f_vec_save= np.append(f_vec_save,f_save)
    return(x_converged,x_vec_save,f_vec_save,it)
#PARAmeters     
delta=1
alpha=2
epsilon=10**(-6)
#Evaluating
max_iter=15000
x_init=np.vstack([1.5,1.5])
(x_converged,x_vec_save,f_vec_save,it)=HV(func,x_init,max_iter,epsilon,delta,alpha)

x1 = []
x2 = []
for i in range(2*it):
 if (i%2==0):  
  x1_vec = x_vec_save[i]
  x1 = np.append(x1,x1_vec)
 else:
     x2_vec = x_vec_save[i]
     x2 = np.append(x2,x2_vec)
print(x_converged)
print(it)
##plotting
plt.figure(1)
k = np.linspace(0,it-1,it) 
plt.plot(k,x1,label = 'x1',marker = 'x', markersize = '2')
plt.plot(k,x2,label ='x2', marker = 'o', markersize = '2')
plt.xlabel('iteration number')
plt.ylabel('x')
plt.legend()

plt.figure(2)
n = np.linspace(0,it,it+1)
plt.plot(n,f_vec_save, marker = '*')
plt.xlabel('iteration number')
plt.ylabel('function value')
plt.show()
final=time.time()
print(final-initial)