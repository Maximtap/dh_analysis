import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl

use_pgf = False
if use_pgf: 
         mpl.use('pgf') 
         mpl.rcParams.update({ 
             'pgf.texsystem': 'pdflatex', 
             'font.family': 'serif', 
             'text.usetex': True, 
             'pgf.rcfonts': False, 
             # 'bbox_inches': 'tight', 
         })


#gamma = 0.00001

# gelenkwinkel
beta1 = 0
beta2 = 1

#stoerung der dh parameter
a_error = 0 #0.1
d_error = 0#.0001

def dh_transform(theta, d, a, alpha):
    """
    Calculates the DH transformation matrix for the given DH parameters
    
    Parameters:
        alpha (float): The twist angle (in radians)
        a (float): The link length
        d (float): The link offset
        theta (float): The joint angle (in radians)
    
    Returns:
        A 4x4 numpy array representing the DH transformation matrix
    """
    
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    T = np.array([
        [ct, -st*ca, st*sa, a*ct],
        [st, ct*ca, -ct*sa, a*st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ])
    
    return T

def error(gamma:float, d_amp = 1):
    echte_posi = np.array([np.cos(gamma)*np.cos(beta2) + 1, -np.sin(beta2), np.sin(gamma)*np.cos(beta2)])

    gamma = 1/2*np.pi - gamma
    erste = dh_transform(beta1 - 1/2 * np.pi, np.tan(gamma)*d_amp + d_error, 0, 1/2 * np.pi - gamma)
    zweite = dh_transform(-beta2 + 1/2 * np.pi, -math.pow(np.cos(gamma), -1), 1 + a_error, 0)
    gamma = 1/2*np.pi - gamma

    ergebnis = erste @ zweite
    pos_error = np.sum(np.abs(ergebnis[0:3,3] - echte_posi))
    return pos_error

def error_naiv(gamma: float, d_amp = 1):
    echte_posi = np.array([np.cos(gamma)*np.cos(beta2) + 1, -np.sin(beta2), np.sin(gamma)*np.cos(beta2)])
    erste = dh_transform(0, 0, 1, 0)
    zweite = dh_transform(-beta2, 0, 1, 0)
    ergebnis = erste @ zweite
    pos_error = np.sum(np.abs(ergebnis[0:3,3] - echte_posi))
    return pos_error

def rotation_matrix_4x4_y(alpha):
    """Returns a 4x4 rotation matrix about the y-axis with angle alpha in radians"""
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)

    rotation_matrix = np.array([
        [cos_alpha, 0, sin_alpha, 0],
        [0, 1, 0, 0],
        [-sin_alpha, 0, cos_alpha, 0],
        [0, 0, 0, 1]
    ])

    return rotation_matrix

def rotation_matrix_4x4_z(alpha):
    """Returns a 4x4 rotation matrix about the z-axis with angle alpha in radians"""
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)

    rotation_matrix = np.array([
        [cos_alpha, -sin_alpha, 0, 0],
        [sin_alpha, cos_alpha, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    return rotation_matrix

def translation_matrix_4x4(t):
    """Returns a 4x4 translation matrix that translates by vector t"""
    translation_matrix = np.identity(4)
    translation_matrix[:3, 3] = t

    return translation_matrix



def error_hayati(gamma, d_amp = 1):
    echte_posi = np.array([np.cos(gamma)*np.cos(beta2) + 1, -np.sin(beta2), np.sin(gamma)*np.cos(beta2)])

   # gamma = 1/2*np.pi - gamma
    hay_mat = translation_matrix_4x4([(1+np.tan(gamma))*d_amp + d_error,0,0]) @ rotation_matrix_4x4_y(-gamma) @ rotation_matrix_4x4_z(-beta2) @ translation_matrix_4x4([1,0,0]) @ translation_matrix_4x4([0,0,1/np.cos(gamma)])
    # auf das gleich koord syst wie e_posi bringen um vergl zu können
    hay_mat[2,3] -= 1
    
    pos_error = np.sum(np.abs(hay_mat[0:3,3] - echte_posi))
    #print(hay_mat)
    return pos_error


# erster Graf
xlog = np.logspace(0, -5, 100)
xlin = np.linspace(1e-0, 1e-03, 1000)
x = xlog

y = [error(i) for i in  x]
yhayati = [error_hayati(i) for i in x]
ynaiv = [error_naiv(i) for i in x]
# # Error in position

fig, ax = plt.subplots()

# Plot the data on a logarithmic x-axis
ax.loglog(x, ynaiv, label='DH-naiv')
ax.loglog(x, y, label='DH')
ax.loglog(x, yhayati, label='HM')



ax.set_xlim(max(x), min(x))

#plt.axvline(x= 0.78)
# Set the title and axis labels
ax.legend(loc='upper right')
ax.set_title("Logarithmisch")
ax.set_xlabel("Gamma in Rad")
ax.set_ylabel("Fehler")

def save_plt(name = 'plot'):
    if use_pgf: 
            fig.set_size_inches(2.5, 2.5)
            plt.savefig(name + '.pgf', format='pgf', bbox_inches='tight')
    else:
         plt.show()

save_plt('noD_log')


# gleicher Plot linear
fig, ax = plt.subplots()

x = xlog

y = [error(i) for i in  x]
yhayati = [error_hayati(i) for i in x]
ynaiv = [error_naiv(i) for i in x]

ax.plot(x, ynaiv, label='DH-naiv')
ax.plot(x, y, label='DH')
ax.plot(x, yhayati, label='HM')

ax.set_xlim(max(x), min(x))

ax.legend(loc='upper right')
ax.set_title("Linear")
ax.set_xlabel("Gamma in Rad")
ax.set_ylabel("Fehler")

save_plt('noD_lin')

#Plot mit fehler log
fig, ax = plt.subplots()

x = xlog

r = 1e-06
d_amp = r + 1

y = [error(i, d_amp) for i in  x]
yhayati = [error_hayati(i, d_amp) for i in x]
ynaiv = [error_naiv(i, d_amp) for i in x]

ax.loglog(x, ynaiv, label='DH-naiv')
ax.loglog(x, y, label='DH')
ax.loglog(x, yhayati, label='HM')

ax.set_xlim(max(x), min(x))

ax.legend(loc='upper right')
ax.set_title("Logarithmisch")
ax.set_xlabel("Gamma in Rad")
ax.set_ylabel("Fehler")

save_plt('D_log')


#Plot mit fehler lin

fig, ax = plt.subplots()

x = np.logspace(-2.6, -5, 300)

y = [error(i, d_amp) for i in  x]
yhayati = [error_hayati(i, d_amp) for i in x]
ynaiv = [error_naiv(i, d_amp) for i in x]

ax.plot(x, ynaiv, label='DH-naiv')
ax.plot(x, y, label='DH')
ax.plot(x, yhayati, label='HM')

ax.set_xlim(max(x), min(x))

ax.legend(loc='upper right')
ax.set_title("Linear")
ax.set_xlabel("Gamma in Rad")
ax.set_ylabel("Fehler")

save_plt('D_lin')

###
### PLots in Abhängigkeit von delta d
###

fig, ax = plt.subplots()

x = np.logspace(-1, -6, 300)

gam = 0.01

y = [error(gam, 1+i) for i in  x]
yhayati = [error_hayati(gam, 1+i) for i in x]
ynaiv = [error_naiv(gam, 1+i) for i in x]

ax.loglog(x, ynaiv, label='DH-naiv')
ax.loglog(x, y, label='DH')
ax.loglog(x, yhayati, label='HM')

# ax.set_xlim(max(x), min(x))

ax.legend(loc='upper left')
ax.set_title("Logarithmisch")
ax.set_xlabel("r")
ax.set_ylabel("Fehler")

save_plt('r_log')


# linear

fig, ax = plt.subplots()
x = np.logspace(-3, -6, 300)

y = [error(gam, 1+i) for i in  x]
yhayati = [error_hayati(gam, 1+i) for i in x]
ynaiv = [error_naiv(gam, 1+i) for i in x]

ax.plot(x, ynaiv, label='DH-naiv')
ax.plot(x, y, label='DH')
ax.plot(x, yhayati, label='HM')

# ax.set_xlim(max(x), min(x))

ax.legend(loc='upper left')
ax.set_title("Linear")
ax.set_xlabel("r")
ax.set_ylabel("Fehler")

save_plt('r_lin')