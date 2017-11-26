"""
Author ： g faia
Email : gutianfeigtf@163.com
"""
import matplotlib.pyplot as plt
import numpy as np


def draw_pic():
    """
    draw the function of ackley.
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x, y = np.mgrid[-10:10:100j, -10:10:100j]
    z = -20*np.exp(-0.2*np.sqrt(1/2*(x**2+y**2)))-\
        np.exp(1/2*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y)))+20+np.exp(1)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, alpha=0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


if __name__ == "__main__":
    draw_pic()
