import random
import argparse
import numpy as np
import pandas as pd
import math
import json
import sys
import os
import cv2
import skimage.graph

import pyqtgraph as pg
import pyqtgraph.opengl as gl

from PyQt5.QtCore import QBuffer
from PyQt5 import QtCore, QtGui
from pyqtgraph.Qt import QtWidgets
from scipy.spatial.transform import Rotation as R

sys.path.append("..")
from utils_data import *
from utils_draw import *
from utils_processing import *

pg.setConfigOption('background', 'white')
       
# get the point closest to the grid
def get_closest_grid_index(grid, point):

    distances = abs(grid - point)
    distances = np.sum(distances,axis=1)
    grid_id = np.argmin(distances)
    #start_point = int(np.floor(start_point/30)), start_point%30
    return grid_id
       
if __name__ == '__main__':
    
    # python draw-pred.py --frame 4618 --draw_pred_pose 1
    # python draw-pred.py --frame 26777 --draw_pred_pose 1
        
    parser = argparse.ArgumentParser()   
    parser.add_argument('--root', default=os.path.join("..","..","..","results"), type=str)
    parser.add_argument('--result_root', default="ICASSP2022-vanilla", type=str)
    parser.add_argument('--result_name', default="200k", type=str)
    parser.add_argument('--sequence', default="p1_1", type=str)
    parser.add_argument('--frame', default=4618, type=int)
    
    parser.add_argument('--default_gaze_color', choices=[0,1], default=1, type=int)
    parser.add_argument('--default_object_color', choices=[0,1], default=1, type=int)
    parser.add_argument('--default_key_object_color', choices=[0,1], default=1, type=int)
    
    parser.add_argument('--draw_true_pose', choices=[0,1], default=0, type=int)
    parser.add_argument('--draw_pred_pose', choices=[0,1], default=0, type=int)
    
    parser.add_argument('--obstacle_pad_size', nargs="*", default=[0.2,0.2], type=float)
    
    args = parser.parse_args()
    
    # initialize screen
    app = QtWidgets.QApplication(sys.argv)
    w = gl.GLViewWidget()
        
    sequence = args.sequence
    filename = int(args.frame) # 0000002555 0000002600 4447 4376
    filename = str(filename).zfill(10) 
    results_dir = os.path.join(args.root,args.result_root,args.result_name,sequence)   
    pose_dir    = os.path.join("../../xyz-poses/",sequence)
    object_dir  = os.path.join("../../object-positions-orientations/",sequence)
    
    #print(os.path.join(pose_dir,filename))
    print(os.path.join(object_dir,filename))
                
    # # # # # # #
    # Load data #
    # # # # # # #
    
    result = json.load(open(os.path.join(results_dir,filename+".json"),"r"))
    all_objects = pd.read_csv(os.path.join(object_dir,str(result["key_frame"]).zfill(10)+".txt"),sep=" ")
        
    # if no prediction for the object scores were made
    if "pred_object_scores" in result:
        pred_object_scores = result["pred_object_scores"]
    else:
        pred_object_scores = [0,0,0,0,0,0,0,0,0,0]
    pred_key_object_idx = np.argmax(pred_object_scores)            
    true_key_object_idx = result["object_scores"]
        
    # # # # # # #
    # Draw Pose #
    # # # # # # #
    
    w = draw_pose(result["inp_pose"], result["inp_center"], w, (1,1,1,1), link_ids)
    
    # if no prediction for the object scores were made
    if args.draw_pred_pose:
        if "pred_object_scores" in result:
            w = draw_pose(result["pred_key_pose"][pred_key_object_idx], result["inp_center"], w, (0,0,1,1), link_ids)
        else:
            w = draw_pose(result["pred_key_pose"], result["inp_center"], w, (0,0,1,1), link_ids)
            
        if args.draw_true_pose:
            w = draw_pose(result["true_key_pose"], result["inp_center"], w, (1,0,0,1), link_ids)

    # # # # # # # # # # # # # # #
    # Draw Grid and Trajectory  #
    # # # # # # # # # # # # # # #
    
    offset = np.array([5,5])
    
    # get furnitures
    furniture_bboxes,padded_furniture_bboxes = {},{}
    furnitures = all_objects[all_objects['name'].str.contains("shelf") | all_objects['name'].str.contains("table") | all_objects['name'].str.contains("chair")]
    for index,row in furnitures.iterrows():
        
        # furniture name
        furniture_name = row["name"]
        
        # furniture bounding box at origin
        with open(os.path.join("..","..","meshes","json",furniture_name+".json"), 'r') as fp:
            data = json.load(fp)
        vertices = np.array(data["vertices"])
        min_x, min_y, min_z = np.min(vertices[:,0]), np.min(vertices[:,1]), np.min(vertices[:,2])
        max_x, max_y, max_z = np.max(vertices[:,0]), np.max(vertices[:,1]), np.max(vertices[:,2])
                        
        if "chair" not in furniture_name:
        
            # pad obstacles
            width  = abs(max_x - min_x)
            height = abs(max_y - min_y)
            padded_min_x = min_x - width*args.obstacle_pad_size[0]
            padded_max_x = max_x + width*args.obstacle_pad_size[0]
            padded_min_y = min_y - width*args.obstacle_pad_size[1]
            padded_max_y = max_y + width*args.obstacle_pad_size[1]                    
            
            tl = np.array([[min_x,max_y,min_z]])
            tr = np.array([[max_x,max_y,min_z]])
            bl = np.array([[min_x,min_y,min_z]])
            br = np.array([[max_x,min_y,min_z]])
            
            padded_tl = np.array([[padded_min_x,padded_max_y,min_z]])
            padded_tr = np.array([[padded_max_x,padded_max_y,min_z]])
            padded_bl = np.array([[padded_min_x,padded_min_y,min_z]])
            padded_br = np.array([[padded_max_x,padded_min_y,min_z]])
            
        else:
        
            # pad obstacles
            width  = abs(max_x - min_x)
            height = abs(max_z - min_z)
            padded_min_x = min_x - width*args.obstacle_pad_size[0]
            padded_max_x = max_x + width*args.obstacle_pad_size[0]
            padded_min_z = min_z - width*args.obstacle_pad_size[1]
            padded_max_z = max_z + width*args.obstacle_pad_size[1]
        
            tl = np.array([[min_x,max_y,min_z]])
            tr = np.array([[max_x,max_y,min_z]])
            bl = np.array([[min_x,max_y,max_z]])
            br = np.array([[max_x,max_y,max_z]])
            
            padded_tl = np.array([[padded_min_x,max_y,padded_min_z]])
            padded_tr = np.array([[padded_max_x,max_y,padded_min_z]])
            padded_bl = np.array([[padded_min_x,max_y,padded_max_z]])
            padded_br = np.array([[padded_max_x,max_y,padded_max_z]])
           
        corners = np.concatenate((tl,tr,br,bl))
        padded_corners = np.concatenate((padded_tl,padded_tr,padded_br,padded_bl))
                                
        # furniture transformation parameters
        translation = row[["x","y","z"]].values# - [1,0,0]
        rotation = row[["a","b","c","d"]].values
        
        # transform corners
        corners = (transform_object(corners, translation=translation, rotation=rotation)[:,:2]+offset)*100
        corners = corners.astype(int)
        furniture_bboxes[furniture_name] = corners
        
        # transform corners
        padded_corners = (transform_object(padded_corners, translation=translation, rotation=rotation)[:,:2]+offset)*100
        padded_corners = padded_corners.astype(int)
        padded_furniture_bboxes[furniture_name] = padded_corners
    
    padded_top_down_map = np.ones((1000,1000,3), np.uint8)*255    
    # draw furnitures on 
    for k,v in padded_furniture_bboxes.items():
        if "chair" in k:
            cv2.fillPoly(padded_top_down_map, [v], 0)
        else:
            cv2.fillPoly(padded_top_down_map, [v], 0)
                    
    num_points = 12
    x = np.linspace(-1.30, 1.10, num_points)
    y = np.linspace(-0.40, 2.65, num_points)

    # create the 2D grid
    # row major i.e. grid[0:num_points] draws the first row
    X, Y = np.meshgrid(x, y)
    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))
    grid = np.stack((X,Y),axis=1)
    grid2d = (grid+offset)*100
    grid3d = np.concatenate((grid,np.zeros((grid.shape[0],1),dtype=grid.dtype)),axis=1)
    
    # # # # # # # # #
    # A* algorithm  #
    # # # # # # # # #
    
    # get start point
    #start_world_coordinate = (load_pose(os.path.join(pose_folder,str(start).zfill(10)+".txt"))[11][:2]+offset)*100
    start_world_coordinate = ((np.array(result["inp_pose"]) + np.array(result["inp_center"]))[11][:2]+offset)*100
    start_grid_index       = get_closest_grid_index(grid=grid2d, point=start_world_coordinate)
    start_grid_coordinate = int(np.floor(start_grid_index/num_points)), start_grid_index%num_points
    
    # get end point
    key_object = np.array(result["key_object"]) + np.array(result["inp_center"]) #all_objects[all_objects["name"] == key_object_name]
    key_object = np.squeeze(key_object)
    key_object = (key_object[:2] + offset) * 100 #(key_object[["x","y","z"]].values[0][:2]+offset)*100
    key_object = key_object.astype(int)
    end_world_coordinate = key_object
    end_grid_index       = get_closest_grid_index(grid=grid2d, point=end_world_coordinate)
    end_grid_coordinate  = int(np.floor(end_grid_index/num_points)), end_grid_index%num_points
                
    # set distances that lie in obstacle to inf
    grid_distances = np.sum(abs(grid2d - end_world_coordinate),axis=1)
    for k,point in enumerate(grid2d):
        point = point.astype(int)
        color = padded_top_down_map[point[1],point[0]]
        if (color == np.array([0,0,0])).all():
            grid_distances[k] = 10000
    
    # compute trajectory
    grid_distances = np.reshape(grid_distances,[num_points,num_points])
    path, cost = skimage.graph.route_through_array(grid_distances, start=(start_grid_coordinate[0],start_grid_coordinate[1]), end=(end_grid_coordinate[0],end_grid_coordinate[1]), fully_connected=True)
    path = [xy[0]*num_points+xy[1] for xy in path]
    
    grid3dspace = [grid3d[i] for i in range(len(grid)) if i not in path]
    grid3dspace = np.array(grid3dspace)
    scatter = gl.GLScatterPlotItem(pos=grid3dspace, color=(1,1,0,0.75), size=5)
    w.addItem(scatter)
    
    grid3dpath = [grid3d[i] for i in path]
    grid3dpath = np.array(grid3dpath)
    scatter = gl.GLScatterPlotItem(pos=grid3dpath, color=(1,1,0,1), size=15)
    w.addItem(scatter)
    
    # # # # # # # # # #
    # Draw Furnitures #
    # # # # # # # # # #
    furnitures = all_objects[all_objects['name'].str.contains("shelf") | all_objects['name'].str.contains("table") | all_objects['name'].str.contains("chair")]
    for idx,furniture in furnitures.iterrows(): 
            
        furniture_cloud_name = furniture["name"]
      
        # load furniture meshfile
        with open(os.path.join("../../meshes/json",furniture_cloud_name+".json"), 'r') as fp:
            data = json.load(fp)
        faces = np.array(data["faces"])
        vertices = np.array(data["vertices"])[:,:3]

        # furniture color
        colors = np.repeat(np.array([[0.7, 0.7, 0.7, 0.85]]), faces.shape[0], axis=0)
                            
        # transform furniture
        translation = furniture[["x","y","z"]].values        
        rotation = furniture[["a","b","c","d"]].values
        vertices = transform_object(vertices, translation=translation, rotation=rotation)
        
        # Mesh item will automatically compute face normals.
        object_renderer = gl.GLMeshItem(vertexes=vertices, faces=faces, faceColors=colors, smooth=False)
        object_renderer.setGLOptions('opaque')
        w.addItem(object_renderer)
    
    # # # # # # # # # # # # # # # #
    # Draw Object and Key-Objects #
    # # # # # # # # # # # # # # # #
    
    # load key object and objects 
    objects = all_objects[~all_objects['name'].str.contains("shelf") & ~all_objects['name'].str.contains("table") & ~all_objects['name'].str.contains("chair")]
    true_key_object_idx = result["object_scores"]
    #print(objects)
    for (idx,object),pred_object_score in zip(objects.iterrows(),pred_object_scores): 
            
        object_cloud_name = object["name"]
        #print("object", object_cloud_name, "object score", pred_object_score)
      
        # load object meshfile
        with open(os.path.join("../../meshes/json",object_cloud_name+".json"), 'r') as fp:
            data = json.load(fp)
        faces = np.array(data["faces"])
        vertices = np.array(data["vertices"])[:,:3]

        # color for key-object
        if idx == true_key_object_idx:
            print("key object", object_cloud_name)
            colors = np.repeat(np.array([[0, 0, 1, 1]]), faces.shape[0], axis=0) if args.default_key_object_color else np.repeat(np.array([[0, 1, 0, pred_object_score*2]]), faces.shape[0], axis=0)
        
        # colors for non key-objects
        elif idx != true_key_object_idx:
        
            # color for object stands
            if "shelf" in object_cloud_name or "table" in object_cloud_name or "chair" in object_cloud_name:
                colors = np.repeat(np.array([[0.7, 0.7, 0.7, 0.85]]), faces.shape[0], axis=0)
            
            # color objects
            else:
                pred_object_score = 1 if args.default_object_color else pred_object_score*2
                colors = np.repeat(np.array([[0, 1, 0, pred_object_score]]), faces.shape[0], axis=0)
                
        # transform object
        translation = object[["x","y","z"]].values        
        rotation = object[["a","b","c","d"]].values
        vertices = transform_object(vertices, translation=translation, rotation=rotation)
        
        # Mesh item will automatically compute face normals.
        object_renderer = gl.GLMeshItem(vertexes=vertices, faces=faces, faceColors=colors, smooth=False)
        object_renderer.setGLOptions('opaque')
        w.addItem(object_renderer)

    # # # # # # # # #
    # Draw Eye Gaze #
    # # # # # # # # #

    if "gaze_scores" in result:
        gaze_scores = np.array(result["gaze_scores"]) # [num objects (10), gaze length, 1]
        gaze_scores = gaze_scores[np.argmax(pred_object_scores)]
        for i in range(len(result["gaze_vector"])):
            w = draw_gaze_np(result["gaze_vector"][i], result["center"], w, (1,0,1,1), width=2) if args.default_gaze_color else draw_gaze_np(result["gaze_vector"][i], result["center"], w, (1,0,1,gaze_scores[i]*5), width=2)
    
    from pyqtgraph.Qt import QtCore, QtGui
    #g = gl.GLGridItem(size=QtGui.QVector3D(100,100,1),color=(255, 255, 0, 100))
    #w.addItem(g)   

    w.orbit(-45,0)
    
    w.show()
    app.exec()