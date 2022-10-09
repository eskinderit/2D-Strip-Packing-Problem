import re

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


class Rectangle:
    """
    Implementation of a Rectangle structure

    x: the x coordinate of the LEFT BOTTOM corner
    y: the y coordinate of the LEFT BOTTOM corner
    width: the width of the rectangles
    height: the weight of the rectangles
    """

    def __init__(self, width, height, x=None, y=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def is_square(self):
        return self.height == self.width


def plot_rectangles(rectangles, title="", indexes=True):
    """
    This function plots the given input rectangles and computes the placing margins of those

    rectangles: the rectangles to be plotted
    title: the title to be assigned to the plot
    indexes: True to plot the indexes of the input rectangles, ordered by list index
    """

    title = re.split("/", title)
    title = title[-1].replace(".txt", "")

    fig, ax = plt.subplots()

    max_height = max([rectangles[i].y + rectangles[i].height] for i in range(len(rectangles)))[0]
    max_width = max([rectangles[i].x + rectangles[i].width] for i in range(len(rectangles)))[0]

    for i in range(0, len(rectangles)):
        np.random.seed(i)
        rect_draw = patches.Rectangle((rectangles[i].x, rectangles[i].y), rectangles[i].width, rectangles[i].height,
                                      facecolor=np.random.rand(3, ), edgecolor='k', label="ciao")
        ax.add_patch(rect_draw)

        if indexes:
            cx = rectangles[i].x + rectangles[i].width / 2.0
            cy = rectangles[i].y + rectangles[i].height / 2.0
            ax.annotate(i, (cx, cy), color='k',
                        fontsize=9, ha='center', va='center')

    ax.set_title("Instance: {}, Width: {}, Height: {}".format(title, max_width, max_height))
    ax.spines['top'].set_visible(False)
    ax.set_xlim((0, max_width))
    ax.set_ylim((0, max_height))
    # ax.set_aspect('equal')
    ax.autoscale_view(tight=True)
    ax.set_axisbelow(True)
    ax.grid()
    plt.show()


def write_log(path, instance, add_text=""):
    out_text = str(instance.W) + " " + str(instance.H) + "\n"
    out_text += str(len(instance.rectangles)) + "\n"
    for r in instance.rectangles:
        out_text += str(r.width) + " " + str(r.height) + " " + str(r.x) + " " + str(r.y) + "\n"

    out_text += add_text

    with open(path, "w") as file:
        file.writelines(out_text)
        file.close()
