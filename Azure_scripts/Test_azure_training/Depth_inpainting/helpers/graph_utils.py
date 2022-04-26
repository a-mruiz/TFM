import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

def smooth(scalars, weight=0.01):  # Weight between 0 and 1
    """Smooth the data points in scalars by given weight

    Returns:
        list: smoothed points
    """
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed

def make_graph(data,labels, x_axis ,title="", subtitle="", x_label="",y_label="",x_ticker=20, y_ticker=2.5, x_lim_low=0, x_lim_high=350, grid_y=True, legend=True, show=True, output_dir="output/"):
    """Create graph for given parameters

    Args:
        data (listoflists): datapoints
        labels (list): labels for each datapoint series
        x_axis (list): x_axis values
        title (str, optional): Title. Defaults to "".
        x_label (str, optional): Label for x axis. Defaults to "".
        y_label (str, optional): Label for y axis. Defaults to "".
        x_ticker (int, optional): Show ticks in x each.... Defaults to 20.
        y_ticker (float, optional): Show ticks in y each.... Defaults to 2.5.
        x_lim_low (int, optional): Low limit for x axis. Defaults to 0.
        x_lim_high (int, optional): High limit for x axis. Defaults to 350.
        grid_y (bool, optional): Apply grid on y axis or not. Defaults to True.
        legend (bool, optional): Show legend. Defaults to True.
    """
    plt.clf()
    for d,l in zip(data, labels):
        plt.plot(x_axis, smooth(d), label=l)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    loc = plticker.MultipleLocator(base=x_ticker) # this locator puts ticks at regular intervals
    plt.gca().xaxis.set_major_locator(loc)
    loc = plticker.MultipleLocator(base=y_ticker) # this locator puts ticks at regular intervals
    plt.gca().yaxis.set_major_locator(loc)
    plt.xlim(x_lim_low,x_lim_high)
    plt.title(subtitle,fontsize=16)
    plt.suptitle(title,fontsize=24, y=1)
    if grid_y:
        plt.grid(axis = 'y')
    if legend:
        plt.legend()
    plt.savefig(output_dir+'/'+title+'.png')
    if show:
        plt.show()
    
def make_barplot(data, labels, ranges, time_labels, title="",x_label="",y_label=""):
    plt.clf()
    r_master=[]
    data_master=[]
    for r,l,d in zip(ranges, labels, data):
        plt.bar(r,d,width=0.22,label=l,edgecolor='white')
        r_master.extend(r)
        data_master.extend(d)
    plt.title(title)
    # Add xticks on the middle of the group bars
    plt.xlabel(x_label)
    plt.xticks([r + 0.25 for r in range(len(data[0]))], ['1', '2', '4', '8', '16'])
    plt.ylabel(y_label)
    plt.grid(axis = 'y')
    
    print(len(time_labels))
    print(len(r_master))
    
    
    for i in range(len(r_master)):
        splitted_time=time_labels[i].split('.')[0].split(':')
        time=splitted_time[0]+':'+splitted_time[1]
        plt.text(x = r_master[i]-0.075 , y = data_master[i]+0.1, s = time, size = 6)
    
    plt.legend(loc='lower right')
    plt.savefig("saved_imgs/"+title+'.png')
    
    plt.show()