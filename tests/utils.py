import numpy as np
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def generate_pos_def(n):
    A = np.random.rand(n, n)
    return A.dot(A.T)

def generate_symmetric(n):
    A = np.random.rand(n, n)
    return (A + A.T)/2

def linear_CG(x, A, b, epsilon, epoch=10000000):
    res = A.dot(x) - b # Initialize the residual
    delta = -res # Initialize the descent direction
    count = 0    
    while True:
        
        if (np.linalg.norm(res) <= epsilon) or (count >= epoch):
            return x # Return the minimizer x* and the function value f(x*)
        
        D = A.dot(delta)
        beta = -(res.dot(delta))/(delta.dot(D)) # Line (11) in the algorithm
        x = x + beta*delta # Generate the new iterate

        res = A.dot(x) - b # generate the new residual
        chi = res.dot(D)/(delta.dot(D)) # Line (14) in the algorithm 
        delta = chi*delta -  res # Generate the new descent direction

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

if __name__ == '__main__':
    a = np.array([[2,1],[1,4]])
    b = np.array([3,0])
    x = linear_CG(np.zeros(2), a, b, 1e-10)
    x_min = np.linalg.solve(a, b) # Differentiate to find the minimizer
    print('x',x)
    print('x_min', x_min)