import sys
import os
# Get the absolute path of the directory containing the .so file
dir_path = os.path.dirname(os.path.abspath(__file__)) + "/../cpp"
# Add the directory to the Python path
sys.path.append(dir_path)

import numpy as np
import random
from autograd import grad
import autograd.numpy as au
from _cgpy import CG
from _cgpy.Matrix import Naive_Matrix
from _cgpy.Matrix import Accelerated_Matrix

def is_pos_def(x): 
    ## Check if a matrix is positive definite
    return np.all(np.linalg.eigvals(x) > 0)

def generate_pos_def(n):
    ## Generate a random positive definite matrix
    A = np.random.rand(n, n)
    return A.dot(A.T)

def generate_symmetric(n):
    ## Generate a random symmetric matrix
    A = np.random.rand(n, n)
    return (A + A.T)/2

def custom_linear_CG(x, a, b, epsilon = 5e-7, epoch=10000000, num_threads = -1):
    ## Solve the linear system Ax = b using conjugate gradient method by calling the C++ library
    mat_x_min = None
    if(num_threads == 1):
        mat_a = Naive_Matrix(a)
        mat_b = Naive_Matrix(b)
        mat_x = Naive_Matrix(x)
        linear_cg_model = CG.linear_CG(epsilon, epoch)
        mat_x_min = linear_cg_model.solve_by_Naive_Matrix(mat_a, mat_b, mat_x)

    else:
        mat_a = Accelerated_Matrix(a)
        mat_b = Accelerated_Matrix(b)
        mat_x = Accelerated_Matrix(x)
        if(num_threads == -1):
            linear_cg_model = CG.linear_CG(epsilon, epoch)
        else:
            linear_cg_model = CG.linear_CG(epsilon, epoch, num_threads)
        mat_x_min = linear_cg_model.solve_by_Accelerated_Matrix(mat_a, mat_b, mat_x)
    
    return np.array(mat_x_min.tolist())

def np_linear_CG(x, A, b, epsilon, epoch=10000000):
    ## Solve the linear system Ax = b using conjugate gradient method by calling the numpy library
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

def nonlinear_func_1(x):
    ## The function is f(x) = x1^4 - 2*x1^2*x2 + x1^2 + x2^2 - 2*x1 + 1
    return x[0]**4 - 2*x[0]**2*x[1] + x[0]**2 + x[1]**2 - 2*x[0] + 1

def nonlinear_func_2(x):
    ## The function is f(x) = x1^4 + x2^4 + ... + xn^4
    x = np.array(x)
    return (np.sum(x**4))**0.5

def custom_naive_line_search(f, df, x, d, alpha=5e-4, beta=0.8):
    """
    Functionality: Perform a backtracking line search to find the step size.
    Parameters:
    f: The objective function.
    df: The gradient of the objective function.
    x: The current point.
    d: The search direction.
    alpha: The fraction of decrease in f we expect.
    beta: The fraction by which we decrease t if the previous t doesn't work.
    """
    Mat_df_x = Naive_Matrix(df(x)) #df(x)
    Mat_d = Naive_Matrix(d) #d
    Mat_df_x_dot_d_mul_alpha = (Mat_df_x @ Mat_d)[0, 0] * alpha #df(x).dot(d) * alpha
    f_x = f(x) #f(x)

    t = 1.0
    Mat_x_p_t_d = Naive_Matrix(x) + Mat_d * t #x + t * d

    while f(Mat_x_p_t_d.tolist()) > (Mat_df_x_dot_d_mul_alpha * t + f_x):
        t *= beta
        Mat_x_p_t_d = Naive_Matrix(x) + Mat_d * t
    return t

def custom_accelerated_line_search(f, df, x, d, alpha=5e-4, beta=0.8, num_threads = -1):
    """
    Functionality: Perform a backtracking line search to find the step size.
    Parameters:
    f: The objective function.
    df: The gradient of the objective function.
    x: The current point.
    d: The search direction.
    alpha: The fraction of decrease in f we expect.
    beta: The fraction by which we decrease t if the previous t doesn't work.
    """
    Mat_df_x = Accelerated_Matrix(df(x)) #df(x)
    Mat_d = Accelerated_Matrix(d) #d
    if(num_threads != -1):
        Mat_df_x.set_number_of_threads(num_threads)
        Mat_d.set_number_of_threads(num_threads)
    Mat_df_x_dot_d_mul_alpha = (Mat_df_x @ Mat_d)[0, 0] * alpha #df(x).dot(d) * alpha
    f_x = f(x) #f(x)

    t = 1.0
    Mat_x_p_t_d = Accelerated_Matrix(x) + Mat_d * t #x + t * d
    if(num_threads != -1):
        Mat_x_p_t_d.set_number_of_threads(num_threads)

    while f(Mat_x_p_t_d.tolist()) > (Mat_df_x_dot_d_mul_alpha * t + f_x):
        t *= beta
        Mat_x_p_t_d = Accelerated_Matrix(x) + Mat_d * t
    return t

def np_line_search(f, df, x, d, alpha=5e-4, beta=0.8):
    """
    Functionality: Perform a backtracking line search to find the step size.
    Parameters:
    f: The objective function.
    df: The gradient of the objective function.
    x: The current point.
    d: The search direction.
    alpha: The fraction of decrease in f we expect.
    beta: The fraction by which we decrease t if the previous t doesn't work.
    """
    t = 1.0
    while f(x + t * d) > f(x) + alpha * t * np.dot(df(x), d):
        t *= beta
    return t



def custom_nonlinear_CG(X, tol, alpha, beta, f, Df, method = "Fletcher_Reeves", num_threads = -1):
    ## Solve the nonlinear system using conjugate gradient method by calling the c++ library
    if(num_threads == 1):
        return custom_naive_nonlinear_CG(X, tol, alpha, beta, f, Df, method = "Fletcher_Reeves")
    
    else:
        return custom_accelerated_nonlinear_CG(X, tol, alpha, beta, f, Df, method = "Fletcher_Reeves", num_threads = -1)
                
def custom_naive_nonlinear_CG(X, tol, alpha, beta, f, Df, method = "Fletcher_Reeves"):
    ## Solve the nonlinear system using conjugate gradient method by calling the c++ library
    method_dict = {}
    method_dict = {
                "Fletcher_Reeves": CG.nonlinear_CG.Naive_Fletcher_Reeves_next_iteration,\
                "Hager-Zhang": CG.nonlinear_CG.Naive_Hager_Zhang_next_iteration,\
                "Dai-Yuan": CG.nonlinear_CG.Naive_Dai_Yuan_next_iteration,\
    }

    if method in method_dict: 
        method_func = method_dict[method]
    else:
        raise AssertionError("method not supported")
    
    NORM = np.linalg.norm
    next_Df = Df(X)
    delta = - next_Df 

    while True:
        start_point = X
        step = custom_naive_line_search(f = f, df = Df, x = start_point, d = delta, alpha=alpha, beta=beta)
        if step!=None:
            next_X = X+ step*delta
        elif step==NAN:
            raise AssertionError("It diverges, please try another start point or another hyperparameter.")
        else:
            return X, f(X)
        if NORM(Df(next_X)) < tol:
            return next_X, f(next_X)

        else:
            X = next_X
            cur_Df = next_Df
            next_Df = Df(X)
            Mat_cur_Df = Naive_Matrix(cur_Df)
            Mat_next_Df = Naive_Matrix(next_Df)
            Mat_delta = Naive_Matrix(delta)
            Mat_delta = method_func(Mat_cur_Df, Mat_next_Df, Mat_delta)
            delta = np.array(Mat_delta.tolist())

def custom_accelerated_nonlinear_CG(X, tol, alpha, beta, f, Df, method = "Fletcher_Reeves", num_threads = -1):
    ## Solve the nonlinear system using conjugate gradient method by calling the c++ library
    method_dict = {}
    method_dict = {
                "Fletcher_Reeves": CG.nonlinear_CG.Accelerated_Fletcher_Reeves_next_iteration,\
                "Hager-Zhang": CG.nonlinear_CG.Accelerated_Hager_Zhang_next_iteration,\
                "Dai-Yuan": CG.nonlinear_CG.Accelerated_Dai_Yuan_next_iteration,\
    }

    if method in method_dict: 
        method_func = method_dict[method]
    else:
        raise AssertionError("method not supported")
    
    NORM = np.linalg.norm
    next_Df = Df(X)
    delta = - next_Df 

    while True:
        start_point = X
        step = custom_accelerated_line_search(f = f, df = Df, x = start_point, d = delta, alpha=alpha, beta=beta, num_threads = num_threads)
        if step!=None:
            next_X = X+ step*delta 
        elif step==NAN:
            raise AssertionError("It diverges, please try another start point or another hyperparameter.")
        else:
            return X, f(X)

        if NORM(Df(next_X)) < tol:
            return next_X, f(next_X)

        else:
            X = next_X
            cur_Df = next_Df
            next_Df = Df(X)
            Mat_cur_Df = Accelerated_Matrix(cur_Df)
            Mat_next_Df = Accelerated_Matrix(next_Df)
            Mat_delta = Accelerated_Matrix(delta)
            Mat_delta = method_func(Mat_cur_Df, Mat_next_Df, Mat_delta, num_threads)
            delta = np.array(Mat_delta.tolist())

def np_nonlinear_CG(X, tol, alpha, beta, f, Df, method = "Fletcher_Reeves"):
    ## Solve the nonlinear system using conjugate gradient method by calling the numpy library
    method_dict = {
                "Fletcher_Reeves": Fletcher_Reeves_next_iteration,\
                "Hager-Zhang": Hager_Zhang_next_iteration,\
                "Dai-Yuan": Dai_Yuan_next_iteration,\
    }

    NORM = np.linalg.norm
    next_Df = Df(X)
    delta = - next_Df 

    if method in method_dict: 
        method_func = method_dict[method]
    else:
        raise AssertionError("method not supported")

    while True:
        start_point = X
        step = np_line_search(f = f, df = Df, x = start_point, d = delta, alpha=alpha, beta=beta)
        if step!=None:
            next_X = X+ step*delta 
        elif step==NAN:
            raise AssertionError("It diverges, please try another start point or another hyperparameter.")
        else:
            return X, f(X)

        if NORM(Df(next_X)) < tol:
            return next_X, f(next_X)

        else:
            X = next_X
            cur_Df = next_Df
            next_Df = Df(X)
            delta = method_func(cur_Df, next_Df, delta)
            delta = np.array(delta.tolist())
            
def Fletcher_Reeves_next_iteration(cur_Df, next_Df, delta):
    chi = np.linalg.norm(next_Df)**2/np.linalg.norm(cur_Df)**2
    delta = -next_Df + chi*delta
    return delta

def Hager_Zhang_next_iteration(cur_Df, next_Df, delta):
    Q = next_Df - cur_Df
    M = Q - delta * (np.linalg.norm(Q)**2) * 2/(delta.dot(Q))
    N = next_Df/(delta.dot(Q))
    chi = M.dot(N)
    delta = -next_Df + chi*delta
    return delta

def Dai_Yuan_next_iteration(cur_Df, next_Df, delta):
    chi = np.linalg.norm(next_Df)**2/delta.dot(next_Df - cur_Df)
    delta = -next_Df + chi*delta
    return delta