
## For Students to Fill in
## use get_one_x_y_sample()

def estimate_risk(f, n_try = int(1e5) ):
    sum = 0;
    for i in range(n_try):
        x, y = get_one_x_y_sample()
        sum += (f(x) - y)**2
    return sum/n_try