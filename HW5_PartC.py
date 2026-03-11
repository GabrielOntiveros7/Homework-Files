# region imports
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# endregion

# region functions
def ode_system(t, X, *params):
    '''
    The ode system is defined in terms of state variables.
    I have as unknowns:
    x: position of the piston (This is not strictly needed unless I want to know x(t))
    xdot: velocity of the piston
    p1: pressure on right of piston
    p2: pressure on left of the piston
    For initial conditions, we see: x=x0=0, xdot=0, p1=p1_0=p_a, p2=p2_0=p_a
    :param X: The list of state variables.
    :param t: The time for this instance of the function.
    :param params: the list of physical constants for the system.
    :return: The list of derivatives of the state variables.
    '''
    # unpack the parameters
    A, Cd, ps, pa, V, beta, rho, Kvalve, m, y = params

    # state variables
    x = X[0]
    xdot = X[1]
    p1 = X[2]
    p2 = X[3]

    # use my equations from the assignment
    xddot = (p1 - p2) * A / m
    p1dot = (y * Kvalve * (ps - p1) - rho * A * xdot) * beta / (V * rho)
    p2dot = -(y * Kvalve * (p2 - pa) - rho * A * xdot) * beta / (rho * V)

    # return the list of derivatives of the state variables
    return [xdot, xddot, p1dot, p2dot]


def main():
    # After some trial and error, I found all the action seems to happen in the first 0.02 seconds
    t = np.linspace(0, 0.02, 200)
    # myargs=(A, Cd, Ps, Pa, V, beta, rho, Kvalve, m, y)
    myargs = (4.909E-4, 0.6, 1.4E7, 1.0E5, 1.473E-4, 2.0E9, 850.0, 2.0E-5, 30, 0.002)
    pa = myargs[3]
    ic = [0, 0, pa, pa]

    sln = solve_ivp(ode_system, (t[0], t[-1]), ic, t_eval=t, args=myargs)

    # unpack result into meaningful names
    xvals = sln.y[0]
    xdot = sln.y[1]
    p1 = sln.y[2]
    p2 = sln.y[3]

    # plot xdot vs time
    plt.figure()
    plt.plot(t, xdot, 'b-')
    plt.title('Piston Velocity vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel(r'$\dot{x}$ (m/s)')
    plt.grid(True)

    # plot p1 and p2 vs time
    plt.figure()
    plt.plot(t, p1, 'b-', label='$P_1$')
    plt.plot(t, p2, 'r-', label='$P_2$')
    plt.title('Pressures vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure (Pa)')
    plt.legend()
    plt.grid(True)

    plt.show()
# endregion

# region function calls
if __name__ == "__main__":
    main()
# endregion