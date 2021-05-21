#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/progressive_morphological_filter.h>
#include <pcl/common/common.h>
#include <jsoncpp/json/json.h>
#include <time.h>
#include <iostream>

#include "DBSCAN_simple.h"
#include "DBSCAN_precomp.h"
#include "DBSCAN_kdtree.h"

// Visualization, [The CloudViewer](https://pcl.readthedocs.io/projects/tutorials/en/latest/cloud_viewer.html#cloud-viewer)
template <typename PointCloudPtrType>
void show_point_cloud(PointCloudPtrType cloud, std::string display_name) {
  pcl::visualization::CloudViewer viewer(display_name);
  viewer.showCloud(cloud);
  while (!viewer.wasStopped())
  {

  }
}


pcl::PointCloud<pcl::PointXYZ>::Ptr readBinData(std::string &in_file) {

    std::fstream input(in_file.c_str(), std::ios::in|std::ios::binary);
    if (!input.good()) {
        exit(EXIT_FAILURE);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>);

    for (int i = 0; input.good() && !input.eof(); i++) {
        double point[3];
        pcl::PointXYZ newpoint;
        // double temp;
        input.read((char *) &point, 3*sizeof(double)); /// point.x||point.y||point.z的地址是连续的,所以可以这样赋值
        // input.read((char *) &temp, sizeof(double));
        newpoint.x = point[0];
        newpoint.y = point[1];
        newpoint.z = point[2];
        // newpoint.intensity = temp;
        // std::cout << newpoint.x << "\n";
        pc->push_back(newpoint);
    }
    input.close();
    return pc;
}

int main(int argc, char** argv) {
    std::ifstream json_file("../param.json");
    if(!json_file.is_open())
    {
        std::cerr << "Open json file failed." << "\n";
        exit(1);
    }
    Json::Reader reader;
    Json::Value value;
    if(!reader.parse(json_file, value)) exit(1);
    std::string pc_dir = value["pc_path"].asString();
    std::string pc_id = value["filename"].asString();
    int MinClusterSize = value["setMinClusterSize"].asInt();
    int MaxClusterSize = value["setMaxClusterSize"].asInt();
    float ClusterTolerance = value["setClusterTolerance"].asFloat();
    // std::string kp_dir = value["kp_path"].asString();
    // int pc_size = value["pc_size"].asInt();
    // int kp_size = value["kp_size"].asInt();
    // int segments_size = value["segments_size"].asInt();
    std::stringstream ss;
    ss << pc_id << ".bin";
    std::string filename;
    filename = pc_dir + ss.str();
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud = readBinData(filename);
    show_point_cloud(cloud, "colored of raw point cloud");
    std::cout << "Point cloud size: " << cloud->size() << std::endl;
    // KdTree, for more information, please ref [How to use a KdTree to search](https://pcl.readthedocs.io/projects/tutorials/en/latest/kdtree_search.html#kdtree-search)
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);
    // Segmentation, [Euclidean Cluster Extraction](https://pcl.readthedocs.io/projects/tutorials/en/latest/cluster_extraction.html#cluster-extraction)
    std::vector<pcl::PointIndices> cluster_indices;
    clock_t start_ms = clock();
    
    // test 1. uncomment the following two lines to test the simple dbscan
    // DBSCANSimpleCluster<pcl::PointXYZ> ec;
    // ec.setCorePointMinPts(20);

    // test 2. uncomment the following two lines to test the precomputed dbscan
    // DBSCANPrecompCluster<pcl::PointXYZ>  ec;
    // ec.setCorePointMinPts(20);

    // test 3. uncomment the following two lines to test the dbscan with Kdtree for accelerating
    DBSCANKdtreeCluster<pcl::PointXYZ> ec;
    ec.setCorePointMinPts(20);

    // test 4. uncomment the following line to test the EuclideanClusterExtraction
    // pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;

    ec.setClusterTolerance(ClusterTolerance);
    ec.setMinClusterSize(MinClusterSize);
    ec.setMaxClusterSize(MaxClusterSize);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);
    
    clock_t end_ms = clock();
    std::cout << "cluster time cost:" << double(end_ms - start_ms) / CLOCKS_PER_SEC << " s" << std::endl;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_clustered(new pcl::PointCloud<pcl::PointXYZI>);
    int j = 0;
    // visualization, use indensity to show different color for each cluster.
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); it++, j++) {
        for(std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit) {
            pcl::PointXYZI tmp;
            tmp.x = cloud->points[*pit].x;
            tmp.y = cloud->points[*pit].y;
            tmp.z = cloud->points[*pit].z;
            tmp.intensity = j%8;
            cloud_clustered->points.push_back(tmp);
        }
    }
    std::cout << "segments size: " << cloud_clustered->size() << std::endl;
    // cloud_clustered->width = cloud_clustered->points.size();
    // cloud_clustered->height = 1;
    show_point_cloud(cloud_clustered, "colored clusters of point cloud");
    // IO, [Writing Point Cloud data to PCD files](https://pcl.readthedocs.io/projects/tutorials/en/latest/writing_pcd.html#writing-pcd)
    // pcl::PCDWriter writer;
    // writer.write<pcl::PointXYZI>("cloud_clustered.pcd", *cloud_clustered, false);

    return 0;
}