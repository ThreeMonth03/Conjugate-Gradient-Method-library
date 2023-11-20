import numpy as np

def linear_CG(x, A, b, epsilon):
    res = A.dot(x) - b # Initialize the residual
    delta = -res # Initialize the descent direction
    
    while True:
        
        if np.linalg.norm(res) <= epsilon:
            return x # Return the minimizer x* and the function value f(x*)
        
        D = A.dot(delta)
        beta = -(res.dot(delta))/(delta.dot(D)) # Line (11) in the algorithm
        x = x + beta*delta # Generate the new iterate

        res = A.dot(x) - b # generate the new residual
        chi = res.dot(D)/(delta.dot(D)) # Line (14) in the algorithm 
        delta = chi*delta -  res # Generate the new descent direction

if __name__ == '__main__':
    a = np.array([[2,1],[1,4]])
    b = np.array([3,0])
    x = linear_CG(np.zeros(2), a, b, 1e-10)
    print(x)