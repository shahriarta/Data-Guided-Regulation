"""
Created on June 2 2020

@Manuscript: Online Regulation of Unstable LTI Systems from a Single Trajectory
@authors: Shahriar Talebi, Siavash Alemzadeh, Niyousha Rahimi, Mehran Mesbahi
"""
# optimizer
import cvxpy as cvx
import mosek
import cvxopt

import control as ctrl


import numpy as np
from numpy import linalg as la
import control
import matplotlib.pyplot as plt
import scipy.special
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import (InsetPosition, mark_inset)
# adjusting the font for plotting
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

linestyle_str = [
     ('solid', 'solid'),      # Same as (0, ()) or '-'
     ('dotted', 'dotted'),    # Same as (0, (1, 1)) or ':'
     ('dashed', 'dashed'),    # Same as '--'
     ('dashdot', 'dashdot')]  # Same as '-.'

linestyle_tuple = {
     'loosely dotted':        (0, (1, 10)),
     'dotted':                (0, (1, 1)),
     'densely dotted':        (0, (1, 1)),

     'loosely dashed':        (0, (5, 10)),
     'dashed':                (0, (5, 5)),
     'densely dashed':        (0, (5, 1)),

     'loosely dashdotted':    (0, (3, 10, 1, 10)),
     'dashdotted':            (0, (3, 5, 1, 5)),
     'densely dashdotted':    (0, (3, 1, 1, 1)),

     'dashdotdotted':         (0, (3, 5, 1, 5, 1, 5)),
     'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
     'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))}


def data2hankel(data,L):
    # the input is is a list of column vectors
    # the ouput is the Henkel defined in DeePC paper
    T = data.__len__()
    if T < L:
        raise Exception('Insufficient data for required length----data2hankel routine')

    H_L = np.concatenate(data[0:T-L+1], axis=1)
    for iter in range(1, L):
        H_L = np.concatenate([H_L, np.concatenate(data[iter:iter+T-L+1], axis=1)], axis=0)

    return H_L

def deepc(traj_u, traj_y, data_u, data_y, T_ini, N, q, r, lam_sigma, lam_g):
    size_g = data_u.__len__() - T_ini - N + 1
    U_p, U_N = np.split(data2hankel(data_u, T_ini + N), [m * T_ini], axis=0)
    Y_p, Y_N = np.split(data2hankel(data_y, T_ini + N), [n * T_ini], axis=0)

    if la.matrix_rank(data2hankel(data_u, T_ini + N)) < T_ini + N:
        print('Warning: data is not persistently exciting')

    # building the optimizer
    y = cvx.Variable((n*N, 1))
    u = cvx.Variable((m*N, 1))
    g = cvx.Variable((size_g, 1))
    sigma = cvx.Variable((T_ini*n, 1))

    u_ini = np.concatenate(traj_u[-T_ini:], axis=0)
    y_ini = np.concatenate(traj_y[-T_ini:], axis=0)

    constr = []
    constr += [U_p @ g == u_ini]
    constr += [Y_p @ g == y_ini + sigma]
    constr += [U_N @ g == u]
    constr += [Y_N @ g == y]

    cost = q*cvx.sum_squares(y) + r*cvx.sum_squares(u) + lam_sigma*cvx.norm(sigma, 2) + lam_g *cvx.norm(g, 2)


    # sums problem objectives and concatenates constraints.
    problem = cvx.Problem(cvx.Minimize(cost), constr)
    # problem.solve(solver=cvx.ECOS, verbose=True, feastol=1e-5)

    problem.solve(solver=cvx.MOSEK, mosek_params = {mosek.dparam.optimizer_max_time:  100.0,
                                                    # mosek.dparam.intpnt_co_tol_infeas: 1e-1,
                                                    # mosek.iparam.intpnt_solve_form:mosek.solveform.dual
                                                    })

    # problem.solve(solver=cvx.CVXOPT, verbose=True, feastol=1e-5)
    if problem.status == 'infeasible':
        raise Exception('The optimization problem for DeePC is infeasible')

    return u[0:m].value


def mpc(x_0, T):
    x = cvx.Variable((n, T + 1))
    u = cvx.Variable((m, T))
    sigma = cvx.Variable((n,T+1))

    cost = 0
    constr = []
    for t in range(T):
        cost += cvx.sum_squares(x[:, t + 1]) + 0.001*cvx.sum_squares(u[:, t]) + 1000*cvx.sum_squares(sigma[:, t])
        constr += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + sigma[:, t]]
    # sums problem objectives and concatenates constraints.
    constr += [cvx.norm(x[:, T], 2) <= 1, x[:, 0] == x_0.reshape(-1,)]
    problem = cvx.Problem(cvx.Minimize(cost), constr)
    problem.solve(solver=cvx.MOSEK)

    return u[:, 0].value.reshape(-1, 1)


def Trajectory_generator(A, B, K_old, x0, noise, alpha, number_of_iteration, control_mode):
    ''' generate the trajectory and upper-bound based on DGR paper and compare it to DeePC paper '''


    # System dimensions
    n = np.shape(A)[0]
    m = np.shape(B)[1]


    mean_w = 0

    ############# Implementing DeePC for comparison ###################

    # Parameters for DeePC algorithm (for comparison)
    if control_mode == 0 or control_mode == 2:
        T_ini = 1  # 8
        N_deepc = 4  # 30
    else:
        T_ini = 1
        N_deepc = 5
    # L_deepc = T_ini + N_deepc
    T_deepc = (m+1)*(T_ini+N_deepc + n)



    # solver parameters
    q = 80
    r = 0.000000001
    lam_sigma = 800000
    lam_g = 1


    X_deepc = [x0]
    U_deepc = []

    # offline data
    data_u = []
    data_x = [x0]

    # offline data generation
    for t in range(T_deepc):
        data_u.append(-K_old@data_x[-1] + np.array(np.random.normal(mean_w, 2, size=(m, 1))))
        data_x.append(A @ data_x[-1] + B @ data_u[-1])

    print('DeePC started')
    # DeePC implementation
    for t in range(number_of_iteration):

        if noise:
            if control_mode == 0 or control_mode == 2:
                w = np.array(np.random.normal(mean_w, 0.001, size=(n, 1)))
            else:
                w = np.array(np.random.normal(mean_w, 0.01, size=(n, 1)))
        else:
            w = np.zeros((n, 1))

        if t <= T_ini:
            U_deepc.append(np.array(np.random.normal(mean_w, 0.1, size=(m, 1))))
            X_deepc.append(A @ X_deepc[-1] + B @ U_deepc[-1] + w)
        else:
            if la.norm(X_deepc[-1]) > 1000:
                print('DeePC is blown up at')
                print(t)
                break

            try:
                u_N_deepc = deepc(U_deepc, X_deepc[:-1], data_u, data_x[:-1], T_ini, N_deepc, q, r, lam_sigma, lam_g)
            except:
                print('Deepc is stopped at iteration')
                print(t)
                break
            U_deepc.append(u_N_deepc)
            X_deepc.append(A@X_deepc[-1] + B @ U_deepc[-1] + w)

    print('DeePC ended')

    ###################### Algorithm 1 implementation ####################
    # Parameters for Algorithm 1
    X = [x0]
    Y = []
    K = [np.zeros((m, n))]
    u = []

    G_alpha = la.pinv(alpha * np.eye(m) + B.T @ B) @ B.T

    X_old = [x0]
    X_new = [x0]

    print('DGR started')
    for t in range(number_of_iteration):
        
        if noise:
            if control_mode == 0 or control_mode == 2:
                w = np.array(np.random.normal(mean_w, 0.001, size=(n, 1)))
            else:
                w = np.array(np.random.normal(mean_w, 0.01, size=(n, 1)))
        else:
            w = np.zeros((n, 1))

        u.append(-K[-1] @ X[-1])
        X.append(A @ X[-1] + B @ u[-1] + w)
        Y.append(X[-1] - B @ u[-1] )

        K.append(G_alpha @ np.hstack(Y) @ la.pinv(np.hstack(X[:-1])))
        
        if t < T_deepc:
            X_new.append(X[-1])            
        elif t == T_deepc:
            xposition = t
            
            Q = np.eye(np.shape(A)[0])
            if control_mode == 0 or control_mode ==2:
                R = 0.0000001*np.eye(np.shape(B)[1])
            else:
                R = 0.0000001*np.eye(np.shape(B)[1])
            A_hat = np.hstack(Y) @ la.pinv(np.hstack(X[:-1]))

            (_,_,K1) = control.dare(A_hat, B, Q, R)
            X_new.append((A-B@K1) @ X_new[-1] + w)
        else:
            X_new.append((A-B@K1) @ X_new[-1] + w)

        if la.norm(X_old[-1]) < 200:
            X_old.append((A - B @ K_old) @ X_old[-1] + w)

    print('DGR ended')

    # Find upper-bound based on Lemma 2
    z     = [X[0]]
    z_bar = [X[0] / la.norm(X[0])]
    w_bar = [X[0] / la.norm(X[0]) - z_bar[0]]
    
    for t in range(1, number_of_iteration+1):
        
        X_subspace = np.hstack(X[:t])
        l = len(X_subspace)
        Ux, _ , _ = la.svd(X_subspace, 0)
        z.append((np.eye(l) - Ux @ Ux.T) @ X[t])

        if la.norm(z[-1]) > 10e-12:
            z_bar.append(z[-1] / la.norm(z[-1]))
            w_bar.append(X[t]/la.norm(X[t]) - z_bar[-1])
        else:
            z_bar.append(np.zeros((n, 1)))
            w_bar.append(X[t] / la.norm(X[t]))

    Ur, _, _ = la.linalg.svd(B, 0)

    tilde_A = (np.eye(n) - Ur @ Ur.T) @ A
    tilde_B = Ur @ Ur.T @ A
    a_t     = [la.norm(A)]

    for t in range(1, number_of_iteration + 1):
        a_t.append(la.norm(la.matrix_power(tilde_A, t) @ A, 2))

    L  = [0]
    UB = [la.norm(X[0])]

    L.append(la.norm(A @ z_bar[0]))
    UB.append(L[-1] * la.norm(X[0]))
    Delta = B @ (la.pinv(B)-G_alpha) @ A
    
    for t in range(2, number_of_iteration + 1):
        sum_bl = 0
        for r in range(1, t):
            sum_bl += np.sqrt( la.norm( la.matrix_power(tilde_A, t-1-r) @ tilde_B @ z_bar[r])**2
                        + la.norm( la.matrix_power(tilde_A, t-1-r) @ Delta @ w_bar[r])**2 ) * L[r]
        L.append( a_t[t-1] + sum_bl )
        UB.append( L[-1] * la.norm(X[0]) )


    # ###################### DPC for a system after DGR is done ###############

    # online data generation for a system with DGR in the closed loop
    data_u = []
    data_x = [x0]

    # Parameters for DGR
    Y = []
    u_dgr = []
    K_dgr = [np.zeros((m, n))]
    G_alpha = la.pinv(alpha * np.eye(m) + B.T @ B) @ B.T

    for t in range(T_deepc):
        if noise:
            if control_mode == 0 or control_mode == 2:
                w = np.array(np.random.normal(mean_w, 0.001, size=(n, 1)))
            else:
                w = np.array(np.random.normal(mean_w, 0.01, size=(n, 1)))
        else:
            w = np.zeros((n, 1))

        data_u.append(np.array(np.random.normal(mean_w, 1, size=(m, 1))))
        u_dgr.append(-K_dgr[-1] @ data_x[-1] + data_u[-1])

        data_x.append(A @ data_x[-1] + B@u_dgr[-1] + w)
        Y.append(data_x[-1] - B @ u_dgr[-1])
        K_dgr.append(G_alpha @ np.hstack(Y) @ la.pinv(np.hstack(data_x[:-1])))

    print('DGR+DeePC started')
    # DeePC implementation

    X_deepc_dgr = []
    U_deepc_dgr = []
    X_deepc_dgr[0:T_deepc] = data_x
    U_deepc_dgr[0:T_deepc] = data_u

    # solver parameters
    if control_mode == 0 or control_mode == 2:
        q = 400
        r = 0.05
        lam_sigma = 10000
        lam_g = 1
    elif control_mode == 1 or control_mode == 3:
        q = 400
        r = 0.05
        lam_sigma = 10000
        lam_g = 1

    for t in range(T_deepc, number_of_iteration):

        if noise:
            if control_mode == 0 or control_mode == 2:
                w = np.array(np.random.normal(mean_w, 0.001, size=(n, 1)))
            else:
                w = np.array(np.random.normal(mean_w, 0.01, size=(n, 1)))
        else:
            w = np.zeros((n, 1))

        if la.norm(X_deepc_dgr[-1]) > 1000:
            print('DeePC is blown up at')
            print(t)
            break

        try:
            u_N_deepc_dgr = deepc(U_deepc_dgr, X_deepc_dgr[:-1], data_u, data_x[:-1], T_ini, N_deepc, q, r, lam_sigma,
                                  lam_g)
        except:
            print('DeePC is stopped at iteration')
            print(t)
            break
        U_deepc_dgr.append(u_N_deepc_dgr)
        X_deepc_dgr.append(A @ X_deepc_dgr[-1] + B @(-K_dgr[-1]@ X_deepc_dgr[-1] + U_deepc_dgr[-1]) + w)

    print('DGR+DeePC ended')


    return X, X_old, UB, X_new, xposition, K, X_deepc, X_deepc_dgr
    

def dynamics():
    
    ''' Defines the dynamics of the X-29 aircraft '''
    
    # Longitudinal control
    A1 = [[ -0.4272e-01 , -0.8541e+01 , -0.4451   , -0.3216e+02 ],
          [ -0.7881e-03 , -0.5291     ,  0.9896   ,  0.1639e-09 ],
          [  0.4010e-03 ,  0.3542e+01 , -0.2228   ,  0.6150e-08 ],
          [  0.0        ,  0.0        ,  0.10e+01 ,  0.0        ]]
    
    B1 = [[ -0.3385e-01 , -0.9386e-01 ,  0.4888e-02 ],
          [ -0.1028e-02 , -0.1297e-02 , -0.4054e-03 ],
          [  0.2718e-01 , -0.5744e-02 , -0.1351e-01 ],
          [  0.0        ,  0.0        ,  0.0        ]]
    
    # Lateral-directional control
    A2 = [[ -0.1817     ,  0.1496     , -0.9825 ,  0.1119     ],
          [ -0.3569e+01 , -0.1704e+01 ,  0.9045 , -0.5531e-06 ],
          [  0.1218e+01 , -0.8208e-01 , -0.1826 , -0.4630e-07 ],
          [  0.0        ,  0.1000e+01 ,  0.1513 ,  0.0        ]]
    
    B2 = [[ -0.4327e-03 ,  0.3901e-03 ],
          [  0.3713     ,  0.5486e-01 ],
          [  0.2648e-01 , -0.1353e-01 ],
          [  0.0        ,  0.0        ]]
    
    #################### Table 13-14 ND-UA ####################
    
    # Longitudinal control
    A3 = [[ -0.1170e-01 , -0.6050e+01 , -0.3139   , -0.3211e+02 ],
          [ -0.1400e-03 , -0.8167     ,  0.9940   ,  0.2505e-10 ],
          [  0.3213e-03 ,  0.1214e+02 , -0.4136   ,  0.3347e-08 ],
          [  0.0        ,  0.0        ,  0.10e+01 ,  0.0        ]]
    
    B3 = [[ -0.6054e-01 , -0.1580     ,  0.1338e-01 ],
          [ -0.8881e-03 , -0.3604e-02 , -0.5869e-03 ],
          [  0.1345     , -0.8383e-01 , -0.4689e-01 ],
          [  0.0        ,  0.0        ,  0.0        ]]
    
    # Lateral-directional control
    A4 = [[ -0.1596     ,  0.7150e-01 , -0.9974     , 0.4413e-01 ],
          [ -0.1520e+02 , -0.2602e+01 ,  0.1106e+01 , 0.0        ],
          [  0.6840e+01 , -0.1026     , -0.6375e-01 , 0.0        ],
          [  0.0        ,  0.10e+01   ,  0.7168e-01 , 0.0        ]]
    
    B4 = [[ -0.5980e-03 ,  0.6718e-03 ],
          [  0.1343e+01 ,  0.2345     ],
          [  0.8974e-01 , -0.7097e-01 ],
          [  0.0        ,  0.0        ]]
        
    return [A1, A2, A3, A4], [B1, B2, B3, B4]


def plots_traj(XX, XF, X_update, XX_deepc, XX_deepc_dgr, up_a, xposition, K, mode, mode_title):
    
    ''' Setting for the graphical visualizations '''
        
    font_size = 18
    font = {'size' : font_size}
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')
    matplotlib.rc('font', **font)
    
    plt.rc('xtick', labelsize=30)
    plt.rc('ytick', labelsize=30)
    
    plt.figure(figsize=(13, 10))
    ax = plt.subplot(111)
    
    if mode == 'Longitudinal':
        lb = ['$u$', '$w$', '$q$', '$\\theta$']
    elif mode == 'Lateral_directional':
        lb = ['$v$',  '$p$', '$r$', '$\phi$']
    
    line1 = []
    for i in range(n):
        line1.append(0)
        line1[i], = ax.plot(XX[i,:], alpha=0.35, label=lb[i])
    
    l4, = ax.plot(list(range(xposition, X_update.shape[1])), la.norm(X_update[:, xposition:], axis=0), color='b', linewidth=2.5, label='$\\||x_t\\||$ LQR for $\hat{A}$', linestyle='--')
    l1, = ax.plot(la.norm(XX, axis=0), color='k', linewidth=2.5, label='$\\||x_t\\||$ DGR ON')
    l2, = ax.plot(la.norm(XF, axis=0), color='r', linewidth=2.5, label='$\\||x_t\\||$ DGR OFF', linestyle=':')
    l3, = ax.plot(up_a, color='g', linewidth=2.5, label='Upper Bound', linestyle=linestyle_tuple['densely dashdotdotted'])
    l7, = ax.plot(list(range(xposition, XX_deepc_dgr.shape[1])), la.norm(XX_deepc_dgr[:, xposition:], axis=0), color='orange', linewidth=2.5, label='DGR+DeePC', linestyle='-.')
    l5, = ax.plot(la.norm(XX_deepc, axis=0), color='c', linewidth=2.5, label='Offline data+DeePC', linestyle=linestyle_tuple['densely dashdotted'])

    box = ax.get_position()
    
    first_legend = plt.legend(handles=[l1, l2, l3, l4, l7, l5], loc='upper right', prop={'size': 24})
    leg = plt.gca().add_artist(first_legend)
    
    second_legend = plt.legend(handles=line1, prop={'size': 24}, loc='lower right', bbox_to_anchor=(0.98, 0.2), borderaxespad=0)
    leg = plt.gca().add_artist(second_legend)
    
    ax.axvline(x=xposition, color='gray', linestyle='--', linewidth=3.0)

    # plt.xlim(0, 130)
    plt.ylim(-17, 70)
    plt.xlabel('Iteration $t$', fontsize=30)
    plt.grid()
    plt.show()
    
    # ############# Inner Plot
    ax2 = plt.axes([0, 0, 1, 1])
    ip = InsetPosition(ax, [0.32, 0.3, 0.5, 0.2])
    ax2.set_axes_locator(ip)
    mark_inset(ax, ax2, loc1=3, loc2=4, fc="none", ec='y', linewidth=2.0)
    
    ax2.set_title('zoom in', size=25)
    ax2.set(ylim=([-0.5, 4]), xlim=([xposition-5, 80]))
    ax2.plot(up_a, color='g', linewidth=2.5, label='Upper Bound', linestyle=linestyle_tuple['densely dashdotdotted'])
    ax2.plot(list(range(xposition, X_update.shape[1])), la.norm(X_update[:, xposition:], axis=0), color='b', linewidth=2.5, linestyle='--')
    ax2.plot(la.norm(XX, axis=0), color='k', linewidth=2.5)
    ax2.plot(list(range(xposition, XX_deepc_dgr.shape[1])), la.norm(XX_deepc_dgr[:, xposition:], axis=0), color='orange', linewidth=2.5, label='DGR+DeePC', linestyle='-.')
    
    line1 = []
    lb = ['$u$', '$w$', '$q$', '$\\theta$',   '$v$',  '$p$', '$r$', '$\\phi$']
    for i in range(n):
        line1.append(0)
        line1[i], = ax2.plot(XX[i,:], alpha=0.35, label=lb[i])


def plot_controller(K_list, ideal_K):
    font_size = 18
    font = {'size': font_size}
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')
    matplotlib.rc('font', **font)

    plt.rc('xtick', labelsize=30)
    plt.rc('ytick', labelsize=30)

    plt.figure(figsize=(13, 10))

    stylelist = ['-', '-.', '--', ':']
    for i in range(K_list.__len__()):
        plt.plot(list(map(lambda s: la.norm(s - ideal_K[i], np.inf)/la.norm(K_list[i][0] - ideal_K[i], np.inf), K_list[i])), linestyle=stylelist[i])
    leg = ['Longitudinal-ND-PA', 'Lateral_directional-ND-PA','Longitudinal-ND-UA', 'Lateral_directional-ND-UA']
    plt.legend(leg, prop={'size': 24})
    plt.yscale('log')
    plt.grid()
    plt.xlabel('Iteration $t$', fontsize=30)
    plt.ylabel('$\\frac{||K_t - G_\\alpha A||_{\\infty}}{||K_0 - G_\\alpha A||_{\\infty}}$', fontsize=40)
    plt.tight_layout(pad=0.1)
        
################################ Main ################################

if __name__ == "__main__":

    dynamics_A, dynamics_B = dynamics()
    final_K = []
    ideal_K = []
    
    mode = ['Longitudinal', 'Lateral_directional', 'Longitudinal', 'Lateral_directional']
    mode_title = ['The state trajectory of X-29 in ND-PA mode with and without DGR \n for Longitudinal control', 
                  'The state trajectory of X-29 in ND-PA mode with and without DGR \n for Lateral-directional control',
                  'The state trajectory of X-29 in ND-UA mode with and without DGR \n for Longitudinal control',
                  'The state trajectory of X-29 in ND-UA mode with and without DGR \n for Lateral-directional control']
    
    for control_mode in range(len(mode)):
        print('control mode:')
        print(control_mode)
        A = dynamics_A[control_mode]
        B = dynamics_B[control_mode]
    
        n = np.shape(A)[0]
        m = np.shape(B)[1]
        
        C = np.zeros((n, n))
        D = np.zeros((n, m))
        
        # Discretizing the continuous-time system        
        dt = 0.05
        sys1 = control.StateSpace(A, B, C, D)
        sysd = sys1.sample(dt)
        A = np.asarray(sysd.A)
        B = np.asarray(sysd.B)
        
        # LQR control for the original system
        Q = np.eye(np.shape(A)[0])
        R = np.eye(np.shape(B)[1])
        (P1, L1, K1) = control.dare(A, B, Q, R)
            
        # Adding perturbation dA 
        Ur, _ , _ = la.linalg.svd(B, 0)
        iterator = 0
        while 1:
            np.random.seed(iterator)
            dA = np.zeros((n, n))
            dA[0:n, 0:n] = np.random.normal(0, .05, (n, n))
            tilde_A = (np.eye(n) - Ur @ Ur.T) @ (A + dA)
            if np.max(np.abs(la.eigvals(tilde_A))) < 0.94:
                break
            iterator += 1
        
        A = A + dA

        # Generating the trajectory 
        x0 = 10 * np.random.randn(n, 1)

        alpha = 0.0
        if control_mode == 0 or control_mode == 2:
            alpha = 1e-9
        else:
            alpha = 5e-7

        number_of_iteration = 20*n  # or set it to 40 * n

        x_a, x_fre, up_a, x_update, xposition, K, x_deepc, x_deepc_dgr = Trajectory_generator(A, B, K1, x0, True, alpha,
                                                                                 number_of_iteration, control_mode)


        final_K.append(K)
        ideal_K.append(la.pinv(alpha * np.eye(m) + B.T @ B) @ B.T@A)
        
        plots_traj(np.hstack(x_a), np.hstack(x_fre), np.hstack(x_update), np.hstack(x_deepc), np.hstack(x_deepc_dgr), up_a, xposition, K, mode[control_mode], mode_title[control_mode])

    plot_controller(final_K, ideal_K)
    print('The parameters for DGR+DeePC must be adjusted for each specific instance of perturbation.')