import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
  
# # # # # # # # # # # # #
# loads the human pose  #
# # # # # # # # # # # # #

def load_pose(filename):
    pose = pd.read_csv(filename, sep=" ")
    pose = pose[["worldLinkFramePosition_x","worldLinkFramePosition_y","worldLinkFramePosition_z"]].values.astype(np.float32)
    return pose
    
# # # # # # # # # # # # # # # # # # # #
# get the object noun given its name  #
# # # # # # # # # # # # # # # # # # # # 

def get_noun(name):
    for x in OBJECT_NAMES:
        if x in name:
            return x
            
def get_colour(name, colours):
    for k,v in colours.items():
        if k in name:
            return v
    
    if "bowl" in name:
        return colours["white"]
        
    if "jug" in name:
        return colours["dark_green"]

# # # # # # # # # # # # # # # # # # #   
# loads the object cloud and color  #
# # # # # # # # # # # # # # # # # # #

def load_cloud(filename,size=0.1,color=[1.0,0.0,0.0,0.5]):
    cloud_pos   = pd.read_csv(filename).values
    cloud_size  = np.ones(shape=cloud_pos.shape[0])*size
    cloud_color = np.ones(shape=(cloud_pos.shape[0],4))*color
    return [cloud_pos, cloud_size, cloud_color]
 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# rotates then translates the object wrt to world coordinates #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def transform_object(vertices, translation, rotation):
       
    translation=np.squeeze(translation)
    rotation=np.squeeze(rotation)
          
    # get rotation matrix
    rotation = R.from_quat(rotation).as_matrix()
        
    # center
    #center = np.mean(vertices,axis=0)
        
    # rotate about origin first then translate
    #vertices = np.matmul(rotation,(vertices-center).T).T + center + translation
    vertices = np.matmul(rotation,(vertices).T).T + translation
    return vertices