import numpy as np
from scipy.optimize import line_search
from autograd import grad
import autograd.numpy as au

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def generate_pos_def(n):
    A = np.random.rand(n, n)
    return A.dot(A.T)

def generate_symmetric(n):
    A = np.random.rand(n, n)
    return (A + A.T)/2

def linear_CG(x, A, b, epsilon, epoch=10000000):
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

def nonlinear_CG(Xj, tol, alpha_1, alpha_2, f, Df, method = "Fletcher_Reeves"):
    NORM = np.linalg.norm
    D = Df(Xj)
    delta = -D 
    
    while True:
        start_point = Xj 
        beta = line_search(f=f, myfprime=Df, xk=start_point, pk=delta, c1=alpha_1, c2=alpha_2)[0] 
        if beta!=None:
            X = Xj+ beta*delta 
        else:
            return Xj, f(Xj)

        if NORM(Df(X)) < tol:
            return X, f(X)

        else:
            Xj = X
            d = D
            D = Df(Xj)
            if(method == "Fletcher_Reeves"):
                delta = Fletcher_Reeves_next_iteration(d, D, delta)
            elif(method == "Polak_Ribiere"):
                delta = Polak_Ribiere_next_iteration(d, D, delta)
            elif(method == "Hager-Zhang"):
                delta = Hager_Zhang_next_iteration(d, D, delta)
            elif(method == "Dai-Yuan"):
                delta = Dai_Yuan_next_iteration(d, D, delta)
            else:
                raise AssertionError("method not supported")

def Fletcher_Reeves_next_iteration(D, D_next, delta):
    chi = np.linalg.norm(D_next)**2/np.linalg.norm(D)**2
    delta = -D_next + chi*delta
    return delta

def Polak_Ribiere_next_iteration(d, D, delta):
    chi = (D-d).dot(D)/np.linalg.norm(d)**2 
    chi = max(0, chi) 
    delta = -D + chi*delta 
    return delta

def Hager_Zhang_next_iteration(d, D, delta):
    M = Q - 2*delta*NORM(Q)**2/(delta.dot(Q))
    N = D/(delta.dot(Q))
    chi = M.dot(N) # See line (19)
    delta = -D + chi*delta
    return delta

def Dai_Yuan_next_iteration(d, D, delta):
    chi = np.linalg.norm(D)**2/delta.dot(D - d) # See line (16)
    delta = -D + chi*delta
    return delta

if __name__ == '__main__':
    x =np.random.uniform(low=0.5, high=13.3, size=(1000,))
    print(x)
    y = nonlinear_func_2(x)
    print(y)