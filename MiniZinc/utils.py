import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re
import numpy as np
import os

class Rectangle:
  '''
  Implementation of a Rectangle structure

  x: the x coordinate of the LEFT BOTTOM corner
  y: the y coordinate of the LEFT BOTTOM corner
  width: the width of the rectangles
  height: the weight of the rectangles 
  '''
  def __init__ (self, width, height, x=None, y=None):
    self.x = x
    self.y = y
    self.width = width
    self.height = height

def plot_rectangles(rectangles, title="", indexes=True):

  title = re.split("/", title)
  title = title[-1].replace(".txt","")
  
  fig, ax = plt.subplots()

  max_height = max([rectangles[i].y + rectangles[i].height] for i in range(len(rectangles)))[0]
  max_width = max([rectangles[i].x + rectangles[i].width] for i in range(len(rectangles)))[0]
  
  for i in range(0,len(rectangles)):
    np.random.seed(i)
    rect_draw = patches.Rectangle( (rectangles[i].x, rectangles[i].y) , rectangles[i].width , rectangles[i].height, facecolor = np.random.rand(3,), edgecolor='k', label="ciao")
    ax.add_patch(rect_draw)

    if indexes:
      cx = rectangles[i].x + rectangles[i].width/2.0
      cy = rectangles[i].y + rectangles[i].height/2.0
      ax.annotate(i, (cx, cy), color='k', 
                  fontsize=9, ha='center', va='center')
    
  ax.set_title("Instance: {}, Width: {}, Height: {}".format(title, max_width, max_height))
  ax.spines['top'].set_visible(False)
  ax.set_xlim((0, max_width))
  ax.set_ylim((0, max_height))
  #ax.set_aspect('equal')
  ax.autoscale_view(tight=True)
  ax.set_axisbelow(True)
  ax.grid()
  plt.show()

def txt2dzn():
    os.chdir(os.path.dirname(__file__))
    for j in range(40):
        x_dim=[]
        y_dim=[]    
        file_name="ins-"+str(j+1)

        with open(file_name+".txt") as file:
            lines = file.readlines()
            fixed_width=int(lines[0])
            n_components=int(lines[1])
            for i in range(2,len(lines)):
                a,b=lines[i].split()
                x_dim.append(int(a))
                y_dim.append(int(b))

        with open(file_name+".dzn", 'w') as f:
            f.write('n_components = {};\n'.format(n_components))
            f.write('widths = '+str(x_dim)+';\n')
            f.write('heights = '+str(y_dim)+';\n')
            f.write("fixed_width = "+str(fixed_width)+';\n')

# txt2dzn()

for i in range(1):
    n_instance=str(i+1)
    os.chdir(r"C:\Folders\Università\MAGISTRALE\1° Anno magistrale\CDMO")
    stream = os.popen("minizinc Strip_Packing.mzn instances/ins-"+str(n_instance)+".dzn --solver-time-limit 300000 -f --solver chuffed -o "+"out"+str(n_instance)+".txt")
    print("instance "+n_instance)
