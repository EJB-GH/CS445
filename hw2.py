import random
'''
13. Write a short program to execute the Gradient Descent (GD) algorithm as described in class. Recall that
the key steps in GD are as follows:

Apply GD to approximately solve for the global minimum of the function 𝑓(𝑥, 𝑦) = 52 − 4𝑥 + 2𝑥^2 + 24𝑦 + 3𝑦^2.

You will run (3) sets of experiments using different values for η: (i) η = .1, (ii) η =.01, and (iii) η =.001. 

Run GD for 500 steps for each experiment; in each case initialize 𝑥" ∈ [−10,10] × [−10,10].
R𝑒𝑝𝑜𝑟𝑡 𝑡ℎ𝑒 𝑏𝑒𝑠𝑡 𝑝𝑒𝑟𝑓𝑜𝑟𝑚𝑎𝑛𝑐𝑒 𝑜𝑢𝑡 𝑜𝑓 10 trials for each of the different η value cases. Provide some comments and analysis about your results.

init 𝑥0 ∈ [−10,10] × [−10,10]

Key Steps GD -> x0 = random
                xt = xt-1 - lr* grad(f(xt-1))            
'''

#learning rates
lr, lr2, lr3 = 0.1, 0.01, 0.001

#how many steps for gd
steps = 1000

def gradient(x,y):
    #get the derivatives for the variables
    dx = (-4 + (4 * x))
    dy = (24 + (6 * y))
    return dx, dy

#get the value of the function
def function(x,y):
    return 52 - (4 * x) + (2 * (x ** 2)) + (24 * y) + (3 * (y ** 2))

def experiments():
    lr_r = []
    lr_r2 = []
    lr_r3 = []

    #run descent on each rate
    lr_r = descent(lr)
    lr_r2 = descent(lr2)
    lr_r3 = descent(lr3)

    print(f"Min Value of LR = 0.1: {min(lr_r)}")
    print(f"Min Value of LR = 0.01: {min(lr_r2):.2f}")
    print(f"Min Value of LR = 0.001: {min(lr_r3):.2f}")

    #display used to test
    '''
    display = [lr_r, lr_r2, lr_r3]
    for rate in display:
        for i in range(len(rate)):
            print(f"Run {i+1}: Result: {rate[i]:.2f}")
        
        print("----------------------------------------------")
    '''

def descent(lr):
    #keep a local to return
    result = []

    #best of 10
    for i in range(10):
        #init the values
        x_i = random.uniform(-10,10)
        y_i = random.uniform(-10,10)

        #500 runs
        for step in range(steps):
            #get the gradients
            dx, dy = gradient(x_i, y_i)

            x_i = x_i - (lr * dx)
            y_i = y_i - (lr * dy)

        result.append(function(x_i, y_i))
    return result      

experiments()
