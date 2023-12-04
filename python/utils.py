import sys
import os
# Get the absolute path of the directory containing the .so file
dir_path = os.path.dirname(os.path.abspath(__file__)) + "/../cpp"
# Add the directory to the Python path
sys.path.append(dir_path)

import numpy as np
from scipy.optimize import line_search
from autograd import grad
import autograd.numpy as au
from _cgpy import CG
from _cgpy.Matrix import Naive_Matrix

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def generate_pos_def(n):
    A = np.random.rand(n, n)
    return A.dot(A.T)

def generate_symmetric(n):
    A = np.random.rand(n, n)
    return (A + A.T)/2

def custom_linear_CG(x, a, b, epsilon = 5e-7, epoch=10000000):
    mat_a = Naive_Matrix(a)
    mat_b = Naive_Matrix(b)
    mat_x = Naive_Matrix(x)
    linear_cg_model = CG.linear_CG(mat_a, mat_b, mat_x, epsilon, epoch)
    mat_x_min = linear_cg_model.solve()
    return np.array(mat_x_min.tolist())

def np_linear_CG(x, A, b, epsilon, epoch=10000000):
    res = A.dot(x) - b
    delta = -res 
    count = 0    
    while True:
        
        if (np.linalg.norm(res) <= epsilon) or (count >= epoch):
            return x
        
        D = A.dot(delta)
        beta = -(res.dot(delta))/(delta.dot(D)) 
        x = x + beta*delta

        res = A.dot(x) - b 
        chi = res.dot(D)/(delta.dot(D)) 
        delta = chi*delta -  res 

        count += 1
        if(count % 10000 == 0):
            print("res", res)
            print("norm", np.linalg.norm(res))
            print("count", count)
            print("------------------")
        #print("A", A)
        #print("B", b)
        #print("x", x)
        #print("res", res)
        #print("delta", delta)
        #print("D", D)
        #print("beta", beta)
        #print("chi", chi)
        #print("norm", np.linalg.norm(res))
        #print("------------------")

def nonlinear_func_1(x): # Objective function
    return x[0]**4 - 2*x[0]**2*x[1] + x[0]**2 + x[1]**2 - 2*x[0] + 1

def nonlinear_func_2(x): # Objective function
    x = np.array(x)
    return (np.sum(x**4))**0.5

def nonlinear_CG(X, tol, alpha_1, alpha_2, f, Df, method = "Fletcher_Reeves"):
    method_dict = {
                "Fletcher_Reeves": Fletcher_Reeves_next_iteration,\
                "Polak_Ribiere": Polak_Ribiere_next_iteration,\
                "Hager-Zhang": Hager_Zhang_next_iteration,\
                "Dai-Yuan": Dai_Yuan_next_iteration,\
    }

    NORM = np.linalg.norm
    next_Df = Df(X)
    delta = - next_Df 
    
    while True:
        start_point = X
        beta = line_search(f=f, myfprime=Df, xk=start_point, pk=delta, c1=alpha_1, c2=alpha_2)[0] 
        if beta!=None:
            next_X = X+ beta*delta 
        else:
            return X, f(X)

        if NORM(Df(next_X)) < tol:
            return next_X, f(next_X)

        else:
            X = next_X
            cur_Df = next_Df
            next_Df = Df(X)
            if method in method_dict:
                delta = method_dict[method](cur_Df, next_Df, delta)
            else:
                raise AssertionError("method not supported")
            
def Fletcher_Reeves_next_iteration(cur_Df, next_Df, delta):
    chi = np.linalg.norm(next_Df)**2/np.linalg.norm(cur_Df)**2
    delta = -next_Df + chi*delta
    return delta

def Polak_Ribiere_next_iteration(cur_Df, next_Df, delta):
    chi = (next_Df-cur_Df).dot(next_Df)/np.linalg.norm(cur_Df)**2 
    chi = max(0, chi) 
    delta = -next_Df + chi*delta 
    return delta

def Hager_Zhang_next_iteration(cur_Df, next_Df, delta):
    Q = next_Df - cur_Df
    M = Q - 2*delta*NORM(Q)**2/(delta.dot(Q))
    N = next_Df/(delta.dot(Q))
    chi = M.dot(N)
    delta = -next_Df + chi*delta
    return delta

def Dai_Yuan_next_iteration(cur_Df, next_Df, delta):
    chi = np.linalg.norm(next_Df)**2/delta.dot(next_Df - cur_Df)
    delta = -next_Df + chi*delta
    return delta

if __name__ == '__main__':
    x =np.random.uniform(low=0.5, high=13.3, size=(1000,))
    print(x)
    y = nonlinear_func_2(x)
    print(y)