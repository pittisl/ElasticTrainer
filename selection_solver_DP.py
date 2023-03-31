import numpy as np
import time


def selection_DP(t_dy, t_dw, I, rho=0.3):
    """
    Solving layer selection problem via dynamic programming

    Args:
        t_dy (np.array int16): downscaled t_dy [N,]
        t_dw (np.array int16): downscaled t_dw [N,]
        I (np.array float32): per-layer contribution to loss drop [N,]
        rho (float32): backprop timesaving ratio
    """
    
    # Initialize the memo tables of subproblems
    N = t_dw.shape[0] # number of NN layers
    T = np.sum(t_dw + t_dy) # maximally possible BP time
    T_limit = int(rho * T)
    t_dy_cumsum = 0
    for k in range(N):
        t_dy_cumsum += t_dy[k]
        if t_dy_cumsum > T_limit:
            break
    N_limit = k
    # Infinite importance
    MINIMAL_IMPORTANCE = -99999.0
    # L[k, t] - maximum cumulative importance when:
    # 1. selectively training within last k layers,
    # 2. achieving BP time at most t
    L_memo = np.zeros(shape=(N_limit + 1, T_limit + 1), dtype=np.float32)
    L_memo[0, 0] = 0
    #L_memo[0, 1:] = MINIMAL_IMPORTANCE
        
    # M[k, t, :] - solution to subproblem L[k, t]
    M_memo = np.zeros(shape=(N_limit + 1, T_limit + 1, N), dtype=np.uint8)
    
    S_memo = np.zeros(shape=(N_limit + 1, T_limit + 1), dtype=np.uint8)
    S_memo[0, 0] = 1
    S_memo[1:, 0] = 1
    S_memo[0, 1:] = 1
    
    max_importance = MINIMAL_IMPORTANCE
    k_final, t_final = 0, 0
    # Solving all the subproblems recursively
    for k in range(1, N_limit + 1):
        for t in range(0, T_limit + 1):
            # Subproblem 1:
            # If layer k-1 is NOT selected
            # --> no BP time increase 
            # --> no importance increase
            l_skip_curr_layer = L_memo[k - 1, t]
            
            # Subproblem 2:
            # If layer k-1 is selected
            # --> BP time increases dt = t_dw[k - 1] + sum(t_dy[k-2 : n])
            opt_k = -1
            opt_t = -1
            l_max = l_skip_curr_layer
            t_p = t - t_dw[k - 1]
            # traverse from layer k-1 to the beginning
            for k_p in range(k - 1, -1, -1):
                t_p -= t_dy[k_p]
                if t_p >= 0 and S_memo[k_p, t_p] == 1:
                    l_candidate = L_memo[k_p, t_p] + I[k - 1]
                    if l_candidate > l_max:
                        opt_k = k_p
                        opt_t = t_p
                        l_max = l_candidate
                        
            # make sure valid solution found by checking integer variable
            if opt_k >= 0:
                L_memo[k, t] = l_max
                M_memo[k, t, :(k - 1)] = M_memo[opt_k, opt_t, :(k - 1)]
                M_memo[k, t, k - 1] = 1
                S_memo[k, t] = 1
            # no valid solution from backtrace or no larger than not selecting
            else:
                L_memo[k, t] = l_skip_curr_layer
                M_memo[k, t, :(k - 1)] = M_memo[k - 1, t, :(k - 1)]
                M_memo[k, t, k - 1] = 0
                S_memo[k, t] = 0
            
            if l_max > max_importance:
                max_importance = L_memo[k, t]
                k_final, t_final = k, t
    
    M_sol = M_memo[k_final, t_final, :]
    return max_importance, M_sol

def downscale_t_dy_and_t_dw(t_dy, t_dw, Tq=1e3):
    T = np.sum(t_dw + t_dy)
    scale = Tq / T
    t_dy_q = np.floor(t_dy * scale).astype(np.int16)
    t_dw_q = np.floor(t_dw * scale).astype(np.int16)
    disco = 1.0 * np.sum(t_dy_q + t_dw_q) / Tq
    return t_dy_q, t_dw_q, disco

def simple_test():
    t_dy = np.array([0, 2, 1, 4, 0])
    t_dw = np.array([5, 1, 7, 3, 1])
    I = np.array([1., 3., 10., 5., 3.])
    max_importance, M_sol = selection_DP(t_dy, t_dw, I, rho=0.6)
    print(max_importance)
    print(M_sol)

def main():
    I = np.loadtxt('importance.out')
    # print(I)
    t_dw = np.loadtxt('t_dw.out')
    t_dy = np.loadtxt('t_dy.out')
    t_dy_q, t_dw_q, disco = downscale_t_dy_and_t_dw(t_dy, t_dw, Tq=1e3)
    print('t_dy_q:', t_dy_q)
    print('t_dw_q:', t_dw_q)
    t_dy_q = np.flip(t_dy_q)
    t_dw_q = np.flip(t_dw_q)
    I = np.flip(I)
    t1 = time.time()
    max_importance, M_sol = selection_DP(t_dy_q, t_dw_q, I, rho=0.3*disco)
    t2 = time.time()
    print("t:", t2-t1)
    M_sol = np.flip(M_sol)
    print('max_I:', max_importance)
    print('m:', M_sol)
    print('%T_sel:', 100 * np.sum(np.maximum.accumulate(M_sol) * t_dy + M_sol * t_dw) / np.sum(t_dy + t_dw))

if __name__ == '__main__':
    main()