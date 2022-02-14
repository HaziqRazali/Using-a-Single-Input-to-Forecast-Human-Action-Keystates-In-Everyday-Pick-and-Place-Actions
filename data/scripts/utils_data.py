import numpy as np

OBJECT_NAMES = ["table","cup","shelf","plate","jug","plate","chair","bowl"]
OBJECT_COLOURS = {"red":np.array([1,0,0,1]), "green":np.array([0,1,0,1]), "dark_green":np.array([0,0.75,0,1]), "blue":np.array([0,0,1,1]), "pink":np.array([1,0.752,0.796,1]), "white":np.array([1,1,1,1])}

# # # # # # # # # # # #
# joint names and ids #
# # # # # # # # # # # #

joint_names = ["head","neck","torso","linnerShoulder","lShoulder","lElbow","lWrist","rinnerShoulder","rShoulder","rElbow","rWrist","pelvis","base","lHip","lKnee","lAnkle","lToe","rHip","rKnee","rAnkle","rToe"]
joint_ids = {name: idx for idx, name in enumerate(joint_names)}

# # # # # # # # # # # #
# link names and ids  #
# # # # # # # # # # # #

link_names = [["head","neck"],["neck","torso"],
["torso","linnerShoulder"],["linnerShoulder","lShoulder"],["lShoulder","lElbow"],["lElbow","lWrist"],
["torso","rinnerShoulder"],["rinnerShoulder","rShoulder"],["rShoulder","rElbow"],["rElbow","rWrist"],
["torso","pelvis"],["pelvis","base"],
["base","lHip"],["lHip","lKnee"],["lKnee","lAnkle"],["lAnkle","lToe"],
["base","rHip"],["rHip","rKnee"],["rKnee","rAnkle"],["rAnkle","rToe"]]

link_ids = [[0,1],[1,2],
[2,3],[3,4],[4,5],[5,6],
[2,7],[7,8],[8,9],[9,10],
[2,11],[11,12],
[12,13],[13,14],[14,15],[15,16],
[12,17],[17,18],[18,19],[19,20]]