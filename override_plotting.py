import matplotlib.pyplot as plt
import os
import datetime

plot_counter = 0 
def get_plot_name():
    global plot_counter
    plot_counter += 1
    return f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-{plot_counter}.png"

# save to ./figs/temp/yyyy-mm-dd-hh-mm-ss-{8 bit random b64 string}.png
save_path = os.path.join(
    "figs",
    "temp",
)
# create dir if not exists
os.makedirs(save_path, exist_ok=True)

def show_wrapper():
    # save the plot
    plt.savefig(os.path.join(save_path, get_plot_name()))

    # show the plot
    plt._original_show()

# Store the original plt.show() function
plt._original_show = plt.show

# Override plt.show() globally
plt.show = show_wrapper
