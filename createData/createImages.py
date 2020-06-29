import matplotlib.pyplot as plt 
import matplotlib
import numpy as np



def create_equation(equation, filename, fontsize):
    fig = plt.figure()
    ax = plt.axes([0, 0, 1, 1])  
    rend = fig.canvas.get_renderer()
    my_text = ax.text(0.5, 0.5, equation, fontsize = fontsize, horizontalalignment = 'center',verticalalignment = 'center', usetex = True)
    box = my_text.get_window_extent(renderer=rend)
    width = box.width / fig.dpi
    height = np.ceil(box.height / fig.dpi)
    my_padding = 0.5
    fig.set_size_inches((my_padding + width, my_padding + height))
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    ax.grid(False)
    ax.set_axis_off()
    plt.savefig(filename)
    plt.cla()
    plt.close('all')
    fig.clf()
    
    return
