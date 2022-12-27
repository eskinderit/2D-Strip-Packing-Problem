import re
import gc
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
        self.rotated = None

    def is_square(self):
        return self.height == self.width

    def get_width(self):
        return self.width


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
        rotation_log = False
        if r.rotated:
            rotation_log = "rotated"
        else:
            rotation_log = "not-rotated"
        out_text += str(r.width) + " " + str(r.height) + " " + str(r.x) + " " + str(r.y) + " " + rotation_log + "\n"

    out_text += add_text

    with open(path, "w") as file:
        file.writelines(out_text)
        file.close()


def plot_result(path):
    rectangles = []

    with open(path, 'r') as file:
        file = file.readlines()

        for j in range(2, int(file[1]) + 2):
            width, height, x, y, _ = file[j].split()
            rectangles.append(Rectangle(int(width), int(height), int(x), int(y)))

        time = file[int(file[1]) + 3]
        timeout_reached = file[int(file[1]) + 4]
        bound = file[int(file[1]) + 5]
        plot_rectangles(rectangles, title=path + f" time: {time} timeout reached: {timeout_reached}  bound obtained: {bound}")

def read_reached_bounds(folder_path, start, end):
    """
    Dataframe of the given folder upper bounds.
    folder : the folder containing all the solved instances to be included on the
    The instances included in the dataframe are from ins-{start}.txt to ins-{end+1}.txt
    """

    heights = []
    indexes = []

    for i in range(start, end+1):
        path = folder_path + f"/ins-{i}.txt"
        with open(path, 'r') as file:
            file = file.readlines()
            _, height = file[0].split()
            indexes.append(i)
            kind_of_bound = file[int(file[1]) + 5]
            if kind_of_bound == "OPTIMAL":
                heights.append(height + " BOLD")
            elif kind_of_bound == "NOT_OPTIMAL":
                heights.append(height)
            elif kind_of_bound == "UPPER_BOUND":
                heights.append(f" - ({height}**)")
            else:
                raise Exception("ïnstance not legal")
    return pd.DataFrame(heights, index=indexes, columns=['bound_reached'])


class VLSI_Instance:
    """
    In this implementation, VLSI_Instance is the class representing the parsed instance to be solved (with a
    fixed width of the container and with a defined amount of rectangles to be placed inside that one)

    path: the path from which the instances are taken

    """

    def __init__(self, path, order_by_width = True):

        rectangles = []

        with open(path, 'r') as file:
            file = file.readlines()

            for j in range(2, int(file[1]) + 2):
                width, height = file[j].split()
                rectangles.append(Rectangle(int(width), int(height)))

        self.name = path
        self.W = int(file[0])
        self.H = None
        self.n_instances = int(file[1])

        # sort by width, area, height (in this order)
        if order_by_width:
            rectangles = sorted(rectangles, key=lambda x: (x.width, x.width*x.height, x.height), reverse=True)

        self.rectangles = rectangles
        for j in range(0, len(self.rectangles)):
            if self.rectangles[j].width > self.W:
                raise Exception(f"The width of the rectangle n.{j} is over the container width W = {self.W}")


    def H_LB(self):
        """
        *Height Lower Bound*
        In this implementation, the lower bound is computed using as best case the one in which no
        blank spaces are left, so H = Atot/W
        """

        area = 0

        for rectangle in self.rectangles:
            area += (rectangle.height * rectangle.width)
        return int(np.ceil(area / self.W))

    def H_UB_BL(self, plot=False):
        '''
        *Bottom-Left-Justified Height Upper Bound*:
        In this implementation, the rectangles are placed one by one in the first available
        spot from left to right.
        Although its good Upper Bound estimate, it's slower than other methods, therefore it is
        useful when the model is very sensible to the Upper Bound setting.
        '''

        W = self.W
        occupied_height = np.full((self.H_UB_naive(), W), False)

        for r in self.rectangles:

            i = -1
            hole_found = False
            while not hole_found:
                i += 1
                j = -1
                while (not hole_found) and (j < occupied_height.shape[1]-r.width):
                    j += 1
                    if np.all(occupied_height[i:i + r.height, j:j + r.width] == False):
                        occupied_height[i:i + r.height, j:j + r.width] = True
                        hole_found = True
                        r.x = j
                        r.y = i

        if plot:
            plot_rectangles(self.rectangles, title=self.name)

        del occupied_height
        gc.collect()

        self.H = max([(r.height + r.y) for r in self.rectangles])

        return self.H

    def H_UB_naive(self, plot=False):
        """
        *Naive Height Upper Bound (sum of all heights)*
        In this implementation, the upper bound is computed as the sum of all heights.
        This method gives the quickest approximation, however the estimate is very raw.
        Not recommended for models that are very sensible to Upper Bound values setting.
        """

        height = 0

        for rectangle in self.rectangles:
            height += rectangle.height
            
            
        if plot:
            h = 0
            for rectangle in self.rectangles:
                rectangle.x = 0
                rectangle.y = h
                h += rectangle.height
        
            plot_rectangles(self.rectangles, title=self.name)
            
        return height

    def H_UB_rotation(self, plot=False):
        """
        *Native Height Upper Bound (sum of all the shortest sides)*
        Same as H_UB_naive, but considers just the smaller side of the rectangle
        
        """
        
        height = 0

        for rectangle in self.rectangles:
            height += min(rectangle.height, rectangle.width)
            
        if plot:
            h = 0
            for rectangle in self.rectangles:
                
                min_dim = min(rectangle.height, rectangle.width)
                
                if min_dim == rectangle.width:
                    rectangle.width = rectangle.height
                    rectangle.height = min_dim
                
                rectangle.x = 0
                rectangle.y = h
                h += min_dim
        
            plot_rectangles(self.rectangles, title=self.name)
            
        return min(height, self.H_UB_BL())

    def biggest_rectangle_index(self):

        biggest_rectangle_index = 0
        smaller_rectangles = []

        area = 0
        for j in range(len(self.rectangles)):
            a = self.rectangles[j].width * self.rectangles[j].height
            if a > area:
                area = a
                biggest_rectangle_index = j + 1

        for j in range(len(self.rectangles)):
            if j != biggest_rectangle_index-1 and self.rectangles[j].width > (self.W - self.rectangles[biggest_rectangle_index-1].width) // 2:
                smaller_rectangles.append(j + 1)

        return biggest_rectangle_index, smaller_rectangles

    def get_width_height(self):
        widths = []
        heights = []

        for rect in self.rectangles:
            widths.append(rect.width)
            heights.append(rect.height)

        return widths, heights

    def get_large_rectangles_index(self):
        """
        returns the indexes of the rectangles whose width sum
        gets over the width of the main plate,
        using the first element as if it had index 1
        """
        large_rectangles = []
        for i in range(0, len(self.rectangles) - 1):
            for j in range(i + 1, len(self.rectangles)):
                if self.rectangles[i].width + self.rectangles[j].width > self.W:
                    large_rectangles.append([i + 1, j + 1])

        return large_rectangles

    def get_same_dim_rectangles_index(self):
        """
        returns the indexes of the rectangles having same dimension,
        using the first element as if it had index 1
        """
        same_dim_rectangles = []
        for i in range(0, len(self.rectangles) - 1):
            for j in range(i + 1, len(self.rectangles)):
                if self.rectangles[i].width == self.rectangles[j].width and self.rectangles[i].height == \
                        self.rectangles[j].height:
                    same_dim_rectangles.append([i + 1, j + 1])

        return same_dim_rectangles

    def get_squares_index(self):
        """
        returns the squares indexes, using the first element as
        if it had index 1
        """
        squares = []
        for i in range(0, len(self.rectangles)):
            if self.rectangles[i].is_square():
                squares.append(i + 1)
        return squares