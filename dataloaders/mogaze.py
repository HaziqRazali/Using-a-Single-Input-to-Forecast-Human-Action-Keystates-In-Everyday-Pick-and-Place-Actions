import os
import cv2
import time
import math
import h5py
import json
import torch 
import random
import numpy as np
import pandas as pd
import torchvision.transforms.functional as transforms
import skimage.graph

from tqdm import tqdm
from PIL import Image
from glob import glob
from random import randint
from scipy.spatial.transform import Rotation as R

def delete(data, substring):
    return [x for x in data if substring not in x]

def load_pose(filename):
    pose = pd.read_csv(filename, sep=" ")
    pose = pose[["worldLinkFramePosition_x","worldLinkFramePosition_y","worldLinkFramePosition_z"]].values.astype(np.float32)
    return pose

train = ["p4_1","p5_1","p6_1","p6_2","p7_3"]
val = ["p1_1","p1_2","p2_1","p3_1"]
test = ["p1_1","p1_2","p2_1","p3_1"]
class dataloader(torch.utils.data.Dataset):
    def __init__(self, args, dtype):

        for key, value in args.__dict__.items():
            setattr(self, key, value)
        self.dtype = dtype

        """
        Read human, object and segmentation folders given the dtype
        """
        pose_list,object_list,metadata_list = [],[],[]
        pose_folders     = sorted(glob(os.path.join(self.dataset_root,"xyz-poses","*")))        # human pose
        object_folders   = sorted(glob(os.path.join(self.dataset_root,"object-positions-orientations","*")))  # position of every object
        metadata_folders = sorted(glob(os.path.join(self.dataset_root,"segmentations-processed","*")))   # sequence name, inp frame, key frame, key-object name, 
        segment_files    = sorted(glob(os.path.join(self.dataset_root,"segmentations","*")))                 # segments of every sequence
        null_segments    = sorted(glob(os.path.join(self.dataset_root,"null-segments","*")))                      # null segments of every sequence
                
        pose_folders     = [f for f in pose_folders for g in eval(dtype) if g in f]
        object_folders   = [f for f in object_folders for g in eval(dtype) if g in f]
        metadata_folders = [f for f in metadata_folders for g in eval(dtype) if g in f]
        segment_files    = [f for f in segment_files for g in eval(dtype) if g in f]
        null_segments    = [f for f in null_segments for g in eval(dtype) if g in f]
        
        print("Reading from", self.dataset_root, flush=True)
        
        """
        Read human, object and segmentation data
        - the entire sequence gets processed if dtype is neither train nor val i.e. synthetic
        """
        for pose_folder,object_folder,metadata_folder,segment_file,null_segment in zip(pose_folders,object_folders,metadata_folders,segment_files,null_segments):
            print("Reading sequence " + pose_folder.split("/")[-1])
            
            # read in the list of text files
            pose_files_all     = sorted(glob(os.path.join(pose_folder,"*")))
            object_files_all   = sorted(glob(os.path.join(object_folder,"*")))
            metadata_files_all = sorted(glob(os.path.join(metadata_folder,"*")))
            
            assert len(pose_files_all) == len(object_files_all)
            assert len(object_files_all) == len(metadata_files_all)

            # # # # # # # # # # # # # # # # # # # # # # #
            # remove the start and end of each segment  #
            # # # # # # # # # # # # # # # # # # # # # # #

            non_null_entries = []
            segments = h5py.File(segment_file, 'r')['segments']
            for segment in segments:
                non_null_entries.extend(range(segment[0]+1,segment[1]-1))

            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # remove first self.inp_length*self.time_step_size frames   #
            # remove last self.out_length*self.time_step_size frames    #
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

            after_first_N_frames = [i for i in range(self.inp_length*self.time_step_size, len(pose_files_all) - self.time_step_size*self.out_length)]
            
            # # # # # # # # # # # # #
            # remove idle segments  #
            # # # # # # # # # # # # #

            # get the null segments where the person is awaiting instructions
            inactive_frames = []
            f = open(null_segment, "r")
            for x in f:
              x = x.split("-")
              l = int(x[0])
              r = int(x[1])
              inactive_frames.extend(range(l,r+1))
            # get the non null segments
            active_frames = range(inactive_frames[-1]+1)
            active_frames = list(set(active_frames) - set(inactive_frames))
            
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # and remove the first and last portions of each segment                      #
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            
            valid_action_frames = []
             # read the segments for the current sequence
            segments = h5py.File(segment_file, 'r')['segments']
            for segment_x,segment_y in zip(segments[:-1], segments[1:]):
                
                # get the current segment
                x_label = segment_x["label"].decode("utf-8")
                y_label = segment_y["label"].decode("utf-8")
                
                valid_action_frames.extend(range(segment_x[0] + self.inp_length*self.time_step_size,segment_x[1] - self.out_length*self.time_step_size))
            
            # valid frames
            valid_frames = sorted(list(set(valid_action_frames) & set(active_frames) & set(after_first_N_frames) & set(non_null_entries)))
                        
            # collect the valid frames
            pose_files     = [pose_files_all[i] for i in valid_frames]
            object_files   = [object_files_all[i] for i in valid_frames]
            metadata_files = [metadata_files_all[i] for i in valid_frames]     
            
            pose_list.extend(pose_files)
            object_list.extend(object_files)
            metadata_list.extend(metadata_files)
            print(dtype, "samples", len(pose_files), len(object_files), len(metadata_files))
            
        print(dtype, "samples", len(pose_list), len(object_list), len(metadata_list))
                
        self.data = pd.DataFrame({"pose": pose_list, "object": object_list, "metadata": metadata_list})
        #self.data = pd.DataFrame({"pose": pose_list[-1000:], "object": object_list[-1000:], "metadata": metadata_list[-1000:]})   
        self.args = args
                                            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        #data_loader_start_time = time.time()
        
        # filename
        data = self.data.iloc[idx] 
        
        # get metadata
        metadata    = data["metadata"]
        metadata    = pd.read_csv(metadata)
        frame           = metadata["frame"].values[0]
        sequence        = metadata["sequence"].values[0]
        inp_frame       = metadata["frame"].values[0]
        key_frame       = metadata["key_frame"].values[0]
        key_object_name = metadata["key_object"].values[0]
        
        # # # # # # #
        # Pose Data #
        # # # # # # #
        
        # get inp pose
        inp_pose_filename = os.path.join(self.dataset_root,"xyz-poses",sequence,str(inp_frame).zfill(10)+".txt")
        inp_pose = load_pose(inp_pose_filename)
        inp_center = inp_pose[11]
        
        # get key pose
        key_pose_filename = os.path.join(self.dataset_root,"xyz-poses",sequence,str(key_frame).zfill(10)+".txt")
        key_pose = load_pose(key_pose_filename)
        key_center = key_pose[11]
        
        # # # # # # # # # # # # 
        # Generate trajectory #
        # # # # # # # # # # # #
        
        trajectory = self.generate_trajectory(sequence, inp_frame, key_frame, key_object_name) # [trajectory length, 2]
        trajectory = np.concatenate((trajectory,np.zeros((trajectory.shape[0],1),dtype=trajectory.dtype)),axis=1)
        
        # compute total distance travelled
        relative_distances = np.zeros(trajectory.shape, dtype=trajectory.dtype)
        relative_distances[1:] = trajectory[1:] - trajectory[:-1]
        total_distance = np.sum(np.abs(relative_distances))
        
        # # # # # # # #
        # Object Data #
        # # # # # # # #
        
        # get objects and key object
        # make sure to use the key_frame since the key-object moves if the person is holding on to it
        objects_filename = os.path.join(self.dataset_root,"object-positions-orientations",sequence,str(key_frame).zfill(10)+".txt")
        objects_df       = pd.read_csv(objects_filename, sep=" ")
        object_scores = objects_df.index[objects_df["name"] == key_object_name].values[0]
        key_object    = objects_df[objects_df["name"] == key_object_name]
        key_object    = key_object[["x","y","z"]].values
        
        # get approach_angle
        #key_center[:,2] = 0
        #approach_angle = key_object - key_center
        
        # normalize along x,y but not height
        if len(trajectory) == 1:
            approach_angle = key_object - trajectory[-1]
            
        if len(trajectory) == 2:
            approach_angle = key_object - (trajectory[-1] + trajectory[-2]) / 2
            
        if len(trajectory) >= 3:
            
            # set height of trajectory to the shoulder (can be linnerShoulder/rinnerShoulder/lShoulder/rShoulder
            shoulder_trajectory = np.copy(trajectory)       # [trajectory length, 3]
            shoulder_trajectory[:,-1] += inp_pose[8,-1]     # [trajectory length, 3]
            
            # compute the closest point to the arm of the shoulder
            distances = np.sqrt(np.sum((shoulder_trajectory - key_object)**2,axis=-1)) # [trajectory length]
            closest_point = np.abs(distances - 0.1528039)
            closest_point_idx = np.argmin(closest_point)
            
            # compute the approach vector from the closest point
            approach_angle = key_object - trajectory[closest_point_idx]
        
        # compute the normalized x,y approach angle
        approach_angle = np.squeeze(approach_angle)                             # [3]
        approach_angle_xy = approach_angle[:2]
        approach_angle_xy = approach_angle_xy / np.sqrt(np.sum(approach_angle_xy**2))
        approach_angle[:2] = approach_angle_xy
        approach_angle = approach_angle.astype(np.float32)
        
        # # # # # # # # # #       
        # preprocess data #
        # # # # # # # # # #
        
        # subtract the x,y,z coordinates of all by the x,y,z coordinates of the inp hip
        key_center = (key_center - inp_center).astype(np.float32)
        inp_pose = (inp_pose - inp_center).astype(np.float32)
        key_pose = (key_pose - inp_center).astype(np.float32)
        key_object = (key_object - inp_center).astype(np.float32)
        dtime = (key_frame/120 - frame/120).astype(np.float32)
        
        return {"sequence":sequence, "inp_frame":inp_frame, "key_frame":key_frame,
                "inp_center":inp_center, "inp_pose":inp_pose, 
                "key_center":key_center, "key_pose":key_pose, 
                "approach_angle":approach_angle, "key_object":key_object, "key_object_name":key_object_name, "object_scores":object_scores, 
                "time": dtime, "total_distance":total_distance,
                "filename":os.path.join(sequence,str(frame).zfill(10))}
     
    def generate_trajectory(self, sequence, inp_frame, key_frame, key_object_name):
        
        # offset in meters
        offset = np.array([5,5])
        
        # top down view of the scene
        top_down_view = np.ones((1000,1000), np.uint8)*255
        
        # load objects
        objects_filename = os.path.join(self.dataset_root,"object-positions-orientations",sequence,str(key_frame).zfill(10)+".txt")
        all_objects = pd.read_csv(objects_filename, sep=" ")
            
        # # # # # # # # # # # # # # # #
        # draw furnitures (obstacles) #
        # # # # # # # # # # # # # # # #
        
        # get furnitures
        furniture_bboxes = {}
        furnitures = all_objects[all_objects['name'].str.contains("shelf") | all_objects['name'].str.contains("table") | all_objects['name'].str.contains("chair")]
        for index,row in furnitures.iterrows():
            
            # furniture name
            furniture_name = row["name"]
            
            # furniture bounding box at origin
            #with open(os.path.join(mesh_foldername,furniture_name+".json"),"r") as fp:
            with open(os.path.join(self.dataset_root,"meshes","json",furniture_name+".json"), 'r') as fp:
                data = json.load(fp)
            vertices = np.array(data["vertices"])
            min_x, min_y, min_z = np.min(vertices[:,0]), np.min(vertices[:,1]), np.min(vertices[:,2])
            max_x, max_y, max_z = np.max(vertices[:,0]), np.max(vertices[:,1]), np.max(vertices[:,2])
                            
            if "chair" not in furniture_name:
            
                # pad obstacles
                width  = abs(max_x - min_x)
                height = abs(max_y - min_y)
                min_x = min_x - width*self.obstacle_pad_size[0]
                max_x = max_x + width*self.obstacle_pad_size[0]
                min_y = min_y - width*self.obstacle_pad_size[1]
                max_y = max_y + width*self.obstacle_pad_size[1]                    
                
                tl = np.array([[min_x,max_y,min_z]])
                tr = np.array([[max_x,max_y,min_z]])
                bl = np.array([[min_x,min_y,min_z]])
                br = np.array([[max_x,min_y,min_z]])
            else:
            
                # pad obstacles
                width  = abs(max_x - min_x)
                height = abs(max_z - min_z)
                min_x = min_x - width*self.obstacle_pad_size[0]
                max_x = max_x + width*self.obstacle_pad_size[0]
                min_z = min_z - width*self.obstacle_pad_size[1]
                max_z = max_z + width*self.obstacle_pad_size[1]
            
                tl = np.array([[min_x,max_y,min_z]])
                tr = np.array([[max_x,max_y,min_z]])
                bl = np.array([[min_x,max_y,max_z]])
                br = np.array([[max_x,max_y,max_z]])
            corners = np.concatenate((tl,tr,br,bl))
                                    
            # furniture transformation parameters
            translation = row[["x","y","z"]].values
            rotation = row[["a","b","c","d"]].values
            
            # transform corners
            corners = (transform_object(corners, translation=translation, rotation=rotation)[:,:2]+offset)*100
            corners = corners.astype(int)
                                    
            # append
            furniture_bboxes[furniture_name] = corners
     
        # draw furnitures
        for k,v in furniture_bboxes.items():
            if "chair" in k:
                cv2.fillPoly(top_down_view, [v], 0)
            else:
                cv2.fillPoly(top_down_view, [v], 0)
     
        # # # # # # #
        # draw grid # 
        # # # # # # #
        
        num_points = 20
        x = np.linspace(-1.30, 1.10, num_points)
        y = np.linspace(-0.40, 2.65, num_points)

        # create the 2D grid
        # row major i.e. grid[0:num_points] draws the first row
        X, Y = np.meshgrid(x, y)
        X = X.reshape((np.prod(X.shape),))
        Y = Y.reshape((np.prod(Y.shape),))
        grid = np.stack((X,Y),axis=1)
        grid = (grid+offset)*100
        
        for k,point in enumerate(grid):
            point = point.astype(int)
            color = top_down_view[point[1],point[0]]
            if (color == np.array([255,255,255])).all():
                cv2.circle(top_down_view, (point[0],point[1]), 2, (0,0,255), -1)
                
        # # # # # # # # #
        # A* algorithm  #
        # # # # # # # # #
        
        # get start point
        inp_pose_filename = os.path.join(self.dataset_root,"xyz-poses",sequence,str(inp_frame).zfill(10)+".txt")
        start_world_coordinate = (load_pose(inp_pose_filename)[11][:2]+offset)*100
        start_grid_index       = get_closest_grid_index(grid=grid, point=start_world_coordinate)
        start_grid_coordinate = int(np.floor(start_grid_index/num_points)), start_grid_index%num_points
        
        # get end point
        key_object = all_objects[all_objects["name"] == key_object_name]
        key_object = (key_object[["x","y","z"]].values[0][:2]+offset)*100
        key_object = key_object.astype(int)
        end_world_coordinate = key_object
        end_grid_index       = get_closest_grid_index(grid=grid, point=end_world_coordinate)
        end_grid_coordinate  = int(np.floor(end_grid_index/num_points)), end_grid_index%num_points
                    
        # set distances that lie in obstacle to inf
        grid_distances = np.sum(abs(grid - end_world_coordinate),axis=1)
        for k,point in enumerate(grid):
            point = point.astype(int)
            color = top_down_view[point[1],point[0]]
            if (color == np.array([0,0,0])).all():
                grid_distances[k] = 1000
        
        # compute trajectory
        grid_distances = np.reshape(grid_distances,[num_points,num_points])
        path, cost = skimage.graph.route_through_array(grid_distances, start=(start_grid_coordinate[0],start_grid_coordinate[1]), end=(end_grid_coordinate[0],end_grid_coordinate[1]), fully_connected=True)
        path = [xy[0]*num_points+xy[1] for xy in path]
     
        # get coordinates of trajectory
        trajectory = []
        for p in path:
            trajectory.append(grid[p].astype(int))
        trajectory = np.array(trajectory)/100-offset
        return trajectory
        
# get the point closest to the grid
def get_closest_grid_index(grid, point):

    distances = abs(grid - point)
    distances = np.sum(distances,axis=1)
    grid_id = np.argmin(distances)
    #start_point = int(np.floor(start_point/30)), start_point%30
    return grid_id
    
def transform_object(vertices, translation, rotation):
     
    """
    # input shape should be as follows
    # print(vertices.shape, translation.shape, rotation.shape)
    # (num_vertices, 3) (3,) (4,)
    """
     
    translation=np.squeeze(translation)
    rotation=np.squeeze(rotation)
        
    # get rotation matrix
    rotation = r = R.from_quat(rotation).as_matrix()
    
    # rotate about origin first then translate
    vertices = np.matmul(rotation,vertices.T).T + translation
    
    return vertices 
