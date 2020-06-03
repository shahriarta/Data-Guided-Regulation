"""
Created on June 2 2020

@Manuscript: Online Regulation of Unstable LTI Systems from a Single Trajectory
@authors: Shahriar Talebi, Siavash Alemzadeh, Niyousha Rahimi, Mehran Mesbahi
"""

import numpy as np
from numpy import linalg as la
import control
import matplotlib.pyplot as plt
import scipy.special
import matplotlib


def Trajectory_generator(A, B, K_old, x0, noise, alpha):
    
    ''' generate the trajectory and upper-bound based on DGR '''
    
    # System dimensions
    n = np.shape(A)[0]
    m = np.shape(B)[1]
    
    # Parameters for Algorithm 1
    X = [x0]
    Y = []     
    K = [np.zeros((m,n))]
    u = []

    G_alpha = la.pinv(alpha * np.eye(m) + B.T @ B) @ B.T

    X_old = [x0]
    X_new = [x0]

    number_of_iteration = 40 * n
    mean_w = 0
    
    # Algorithm 1 implementation
    for t in range(number_of_iteration):
        
        if noise:
            w = np.array(np.random.normal(mean_w, 0.01, size=(n, 1)))
        else:
            w = np.zeros((n, 1))

        u.append(-K[-1] @ X[-1])
        
        X.append(A @ X[-1] + B @ u[-1] + w)
        Y.append(X[-1] - B @ u[-1] )

        A_hat = np.hstack(Y) @ la.pinv(np.hstack(X[:-1]))
        K.append(G_alpha @ A_hat)
        
        if t < 30:
            X_new.append(X[-1])            
        elif t == 30:            
            xposition = t
            Y_ = np.hstack(Y)
            X_ = np.hstack(X[:-1])
            A1_hat = Y_[0:4,:]@la.pinv(X_[0:4,:])
            A2_hat = Y_[4:8,:]@la.pinv(X_[4:8,:])
            A_hat = 0*A_hat
            A_hat[0:4, 0:4] = A1_hat[0:4, 0:4]
            A_hat[4:8, 4:8] = A2_hat[0:4, 0:4]
            
            Q = np.eye(np.shape(A)[0])
            R = np.eye(np.shape(B)[1])
            
            (P1,L1,K1) = control.dare(A_hat,B,Q,R)
            X_new.append((A-B@K1) @ X_new[-1] + w)
        else:
            X_new.append((A-B@K1) @ X_new[-1] + w)
            
        X_old.append((A-B@K_old) @ X_old[-1] + w)

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

    Ur, _ , _ = la.linalg.svd(B, 0)

    tilde_A = (np.eye(n) - Ur @ Ur.T) @ A
    tilde_B = Ur @ Ur.T @ A
    a_t     = [la.norm(A)]

    for t in range(1, number_of_iteration + 1):
        a_t.append( la.norm(la.matrix_power(tilde_A, t) @ A, 2) )

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

    return X, X_old, UB, X_new, xposition
    

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


def plots(XX, XF, X_update, up_a, xposition, mode, mode_title):
    
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
    for i in range(4):
        line1.append(0)
        line1[i], = ax.plot(XX[i,:], alpha=0.35, label=lb[i])
    
    l4, = ax.plot(la.norm(X_update, axis=0), color='b', linewidth=2.5, label='$\|x_t\|$ LQR for $\hat{A}$')
    l1, = ax.plot(la.norm(XX, axis=0), color='k', linewidth=2.5, label='$\|x_t\|$ DGR ON')
    l2, = ax.plot(la.norm(XF, axis=0), color='r', linewidth=2.5, label='$\|x_t\|$ DGR OFF')
    l3, = ax.plot(up_a, color='g', linewidth=2.5, label='Upper Bound')
    
    box = ax.get_position()
    
    first_legend = plt.legend(handles=[l1, l2, l3, l4], loc='upper right',prop={'size': 24})
    leg = plt.gca().add_artist(first_legend)
    
    second_legend = plt.legend(handles=line1, prop={'size': 24}, loc='lower right', bbox_to_anchor=(0.98, 0.39), borderaxespad=0)
    leg = plt.gca().add_artist(second_legend)
    
    ax.axvline(x=xposition, color='gray', linestyle='--', linewidth=3.0)
    
    plt.xlim(0, 130)
    plt.ylim(-10,75)
    plt.xlabel('Iteration $t$', fontsize=30)
    plt.title(mode_title, fontsize=30)
    plt.grid()
    plt.show()
    
    # Inner Plot
    ax2 = plt.axes([0, 0, 1, 1])
    ip = InsetPosition(ax, [0.32, 0.43, 0.5, 0.2])
    ax2.set_axes_locator(ip)
    mark_inset(ax, ax2, loc1=3, loc2=4, fc="none", ec='y', linewidth=2.0)
    
    ax2.set_title('zoom in', size=25)
    ax2.set(ylim=([-0.5, 3]), xlim=([xposition, 130]))
    ax2.plot(la.norm(X_update, axis=0), color='b', linewidth=2.5)
    ax2.plot(la.norm(XX, axis=0), color='k', linewidth=2.5)
    ax2.plot(up_a, color='g', linewidth=2.5, label='Upper Bound')
    
    line1 = []
    lb = ['$u$', '$w$', '$q$', '$\\theta$',   '$v$',  '$p$', '$r$', '$\phi$']
    for i in range(4):
        line1.append(0)
        line1[i], = ax2.plot(XX[i,:], alpha=0.35, label=lb[i])
        
################################ Main ################################

if __name__ == "__main__":

    dynamics_A, dynamics_B = dynamics() 
    
    mode = ['Longitudinal', 'Lateral_directional','Longitudinal', 'Lateral_directional']
    mode_title = ['The state trajectory of X-29 in ND-PA mode with and without DGR \n for Longitudinal control', 
                  'The state trajectory of X-29 in ND-PA mode with and without DGR \n for Lateral-directional control',
                  'The state trajectory of X-29 in ND-UA mode with and without DGR \n for Longitudinal control',
                  'The state trajectory of X-29 in ND-UA mode with and without DGR \n for Lateral-directional control']
    
    for control_mode in range(len(mode)):
        
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
            dA = np.zeros((4, 4))
            dA[0:4, 0:4] = np.random.normal(0, .05, (4, 4))  
            tilde_A = (np.eye(n) - Ur @ Ur.T) @ (A + dA)
            if np.max(np.abs(la.eigvals(tilde_A))) < 1.:
                break
            iterator += 1
        
        A = A + dA
    
        # Generating the trajectory 
        x0 = 10 * np.random.randn(n, 1)
    
        if control_mode == 0:
            alpha = 0.0
        elif control_mode == 1:
            alpha = 0.0
        elif control_mode == 2:
            alpha = 0.0
        elif control_mode == 3:
            alpha = 0.0
    
        x_a, x_fre, up_a, x_update, xposition = Trajectory_generator(A, B, K1, x0, True, alpha)
        
        XX = np.zeros((4, 161))
        XF = np.zeros((4, 161))
        X_update = np.zeros((4, 161))
        
        XX[0:4, :] = np.hstack(x_a)
        XF[0:4, :] = np.hstack(x_fre)
        X_update[0:4, :] = np.hstack(x_update)
        
        plots(XX, XF, X_update, up_a, xposition, mode[control_mode], mode_title[control_mode])
    