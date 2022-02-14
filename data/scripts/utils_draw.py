import numpy as np
import pandas as pd
import pyqtgraph.opengl as gl
  
# # # # # # # # # # # #
# draw the human pose #
# # # # # # # # # # # #
def draw_pose(pose, center, w, color, link_ids):
    pose = np.array(pose)
    center = np.array(center)
    pose += center
    pose[:,-1] -= np.min(pose[:,-1]) # make sure foot is touching the ground
    
    for a,b in link_ids:
        x = pose[a] 
        y = pose[b]
        line = np.array([x,y])
        line_renderer = gl.GLLinePlotItem(pos=line, width=5, antialias=False, color=color)
        line_renderer.setGLOptions('opaque')
        w.addItem(line_renderer)
        
    return w