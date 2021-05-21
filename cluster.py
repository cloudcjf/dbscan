#coding:utf-8  
import numpy as np
import struct
from numpy.core.fromnumeric import sort
from sklearn import cluster, datasets, mixture
import matplotlib.pyplot as plt
from pandas import DataFrame
# from pyntcloud import PyntCloud
import math
import random
from collections import defaultdict
from sklearn.cluster import DBSCAN
# import DBSCAN_fast as dbs
import os
import datetime


DATASET_FOLDER = "/media/cloud/Data/benchmark_datasets/oxford/2014-05-19-13-20-57/pointcloud_20m_10overlap" 
# matplotlib显示点云函数
def Point_Cloud_Show(points):
    fig = plt.figure(figsize=(150,150))
    plt.title('Point Cloud')
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(points)):
        ax.scatter(points[i][:, 0], points[i][:, 1], points[i][:, 2], cmap='spectral', s=10, linewidths=0, alpha=1, marker=".")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    plt.show()
 
# 功能：从kitti的.bin格式点云文件中读取点云
# 输入：
#     path: 文件路径
# 输出：
#     点云数组
def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)
 
def load_pc_file(filename):
    # returns Nx3 matrix
    pc = np.fromfile(os.path.join(DATASET_FOLDER, filename), dtype=np.float64)

    if(pc.shape[0] != 4096*3):
        print("Error in pointcloud shape")
        return np.array([])

    pc = np.reshape(pc,(pc.shape[0]//3, 3))
    return pc
 
# 功能：从点云中提取聚类
# 输入：
#     data: 点云（滤除地面后的点云）
# 输出：
#     clusters_index： 一维数组，存储的是点云中每个点所属的聚类编号（参考上一章内容容易理解）
def dbscan_clustering(data):
    #使用sklearn dbscan 库
    #eps 两个样本的最大距离，即扫描半径； min_samples 作为核心点的话邻域(即以其为圆心，eps为半径的圆，含圆上的点)中的最小样本数(包括点本身);n_jobs ：使用CPU格式，-1代表全开
    # startT = datetime.datetime.now()
    cluster_index = DBSCAN(eps=0.05,min_samples=20,n_jobs=-1).fit_predict(data)
    # cluster_set = set(cluster_index)  # get unique values in list
    min_value = min(cluster_index)
    max_value = max(cluster_index)
    print(min_value,max_value)
    # endT = datetime.datetime.now()
    # computation_time = (endT - startT).microseconds
    # print(len(cluster_index))
    return cluster_index,min_value,max_value
 
 
def main():

    filename = '1400505894395159.bin'         #数据集路径
    print('clustering pointcloud file:', filename)
 
    origin_points = load_pc_file(filename)    # 读取数据点
    # Point_Cloud_Show(origin_points)           
    clusters = []
    cluster_index,min_value,max_value = dbscan_clustering(origin_points)
    for i in range(max_value+1):
        cluster = origin_points[np.where(cluster_index == i)]
        clusters.append(cluster)

    print("clusters size:",len(clusters))
    Point_Cloud_Show(clusters)

 
if __name__ == '__main__':
    main()