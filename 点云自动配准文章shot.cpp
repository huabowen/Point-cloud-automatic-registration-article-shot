#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <vector>
#include <string>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/fpfh_OMP.h>
#include <pcl/features/pfh.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/shot.h>
#include <pcl/visualization/pcl_plotter.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/common/transforms.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/features/board.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/registration/correspondence_rejection_one_to_one.h>
#include <pcl/keypoints/uniform_sampling.h>
using namespace std;  // 可以加入 std 的命名空间


pcl::PointCloud<pcl::Normal>::Ptr normal_estimation_OMP(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float leaf_size, float radius = 5) {
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
	ne.setNumberOfThreads(10);
	ne.setInputCloud(cloud);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	ne.setSearchMethod(tree);
	ne.setRadiusSearch(leaf_size);
	//ne.setKSearch(k);
	ne.compute(*normals);
	return normals;
}

void show_key_point(pcl::PointCloud<pcl::PointXYZ>::Ptr Acloud, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_keypoint) {
	// 初始化点云可视化对象
	pcl::visualization::PCLVisualizer viewer_final("3D Viewer");
	viewer_final.setBackgroundColor(255, 255, 255);  //白色背景

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_cloud(Acloud, 0, 255, 0);//蓝色点云
	viewer_final.addPointCloud<pcl::PointXYZ>(Acloud, color_cloud, "1");

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_key(cloud_keypoint, 255, 0, 0);//关键点
	viewer_final.addPointCloud<pcl::PointXYZ>(cloud_keypoint, color_key, "2");
	viewer_final.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "2");

	// 等待直到可视化窗口关闭
	while (!viewer_final.wasStopped())
	{
		viewer_final.spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

void show_point_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
	// 初始化点云可视化对象
	pcl::visualization::PCLVisualizer viewer_final("3D Viewer");

	viewer_final.setBackgroundColor(255, 255, 255); //白色背景
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_cloud(cloud, 0, 255, 0);//蓝色点云
	viewer_final.addPointCloud<pcl::PointXYZ>(cloud, color_cloud, "1");

	// 等待直到可视化窗口关闭
	while (!viewer_final.wasStopped())
	{
		viewer_final.spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

void show_point_clouds(vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds) {
	// 初始化点云可视化对象
	pcl::visualization::PCLVisualizer viewer_final("3D Viewer");
	viewer_final.setBackgroundColor(255, 255, 255);   //白色背景
	for (int i = 0; i < clouds.size(); i++) {
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color(clouds[i], rand() % 255, rand() % 255, rand() % 255);
		viewer_final.addPointCloud<pcl::PointXYZ>(clouds[i], color, to_string(i));

	}

	// 等待直到可视化窗口关闭
	while (!viewer_final.wasStopped())
	{
		viewer_final.spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

float com_avg_curvature(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals, int i, float radius, pcl::search::KdTree<pcl::PointXYZ>::Ptr tree) {
	float avg_curvature = 0;
	vector<int> point_ind;
	vector<float> point_dist;
	tree->radiusSearch(cloud->points[i], radius, point_ind, point_dist);

	//tree->nearestKSearch(cloud->points[i], num, point_ind, point_dist);
	for (int i = 0; i < point_ind.size(); i++) {
		avg_curvature += normals->points[point_ind[i]].curvature;
	}
	avg_curvature = avg_curvature / float(point_ind.size());
	return avg_curvature;
}

bool is_max_avg_curvature(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, vector<float> avg_curvatures, int point, vector<bool>& possible_key, vector<bool> possible_key_possible, pcl::search::KdTree<pcl::PointXYZ>::Ptr tree, float radius = 4) {

	vector<int> point_ind;
	vector<float> point_dis;
	tree->radiusSearch(cloud->points[point], radius, point_ind, point_dis);//此处半径为计算曲率和的半径

	if (point_ind.size() < 5)
		return false;

	for (int i = 1; i < point_ind.size(); i++) {
		if (possible_key_possible[point_ind[i]]) {
			if (avg_curvatures[point_ind[0]] > avg_curvatures[point_ind[i]])
				possible_key[point_ind[i]] = false;
			else if (avg_curvatures[point_ind[0]] < avg_curvatures[point_ind[i]])
				possible_key[point_ind[0]] = false;

		}

	}
	return possible_key[point_ind[0]];
}

float cosa(float nx, float ny, float nz, float cx, float cy, float cz) {
	if ((cx == 0 && cy == 0 && cz == 0) || isnan(nx) || isnan(ny) || isnan(nz))
		return 0;
	float angle = 0;
	angle = (nx*cx + ny * cy + nz * cz) / (sqrtf(pow(nx, 2) + pow(ny, 2) + pow(nz, 2))*sqrtf(pow(cx, 2) + pow(cy, 2) + pow(cz, 2)));
	return angle;
}

float dist(float nx, float ny, float nz, float cx, float cy, float cz) {
	float distance = 0;
	distance = sqrtf(pow(nx - cx, 2) + pow(ny - cy, 2) + pow(nz - cz, 2));
	return distance;

}

pcl::PFHSignature125 com_feature(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals, pcl::search::KdTree<pcl::PointXYZ>::Ptr tree, int point, float radius = 3, int num_angle = 25, int num_distance = 8) {
	vector<int> sup_ind;
	vector<float> sup_dis;
	tree->radiusSearch(cloud->points[point], radius, sup_ind, sup_dis);

	pcl::PointCloud<pcl::PointXYZ>::Ptr ksearch_keypoint(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::Normal>::Ptr ksearch_normals(new pcl::PointCloud<pcl::Normal>());
	pcl::copyPointCloud(*cloud, sup_ind, *ksearch_keypoint);
	pcl::copyPointCloud(*normals, sup_ind, *ksearch_normals);
	pcl::PFHSignature125 feature;
	for (int i = 0; i < feature.descriptorSize(); i++) {
		feature.histogram[i] = 0;
	}
	if (ksearch_keypoint->size() == 0)
		return feature;

	Eigen::Vector4f centroid;  //质心 
	pcl::compute3DCentroid(*ksearch_keypoint, centroid); //估计质心的坐标
	//cout << centroid.x() << " " << centroid.y() << " " << centroid.z() << endl;

	vector<float> vec_distance;
	for (int i = 0; i < ksearch_keypoint->size(); i++) {
		float distance = 0;
		distance = dist(ksearch_keypoint->points[i].x, ksearch_keypoint->points[i].y, ksearch_keypoint->points[i].z, centroid.x(), centroid.y(), centroid.z());
		vec_distance.push_back(distance);
	}
	float max_diatance = *std::max_element(std::begin(vec_distance), std::end(vec_distance));
	float min_diatance = *std::min_element(std::begin(vec_distance), std::end(vec_distance));
	float res_distance = (max_diatance - min_diatance) / num_distance;
	//cout << "max_diatance:  " << max_diatance << endl;
	//cout << "min_diatance:  " << min_diatance << endl;
	//cout << "res_distance:  " << res_distance << endl;

	for (int i = 0; i < ksearch_keypoint->size(); i++) {


		float angle = 0;
		angle = cosa(ksearch_normals->points[i].normal_x, ksearch_normals->points[i].normal_y, ksearch_normals->points[i].normal_z, ksearch_keypoint->points[i].x - centroid.x(), ksearch_keypoint->points[i].y - centroid.y(), ksearch_keypoint->points[i].z - centroid.z());
		angle += 1;
		int bin_angle = int(angle / (2.0 / num_angle));
		int bin_distance = 0;
		if (res_distance != 0) {
			bin_distance = int((vec_distance[i] - min_diatance) / res_distance);
		}

		if (bin_distance > num_distance - 1) bin_distance = num_distance - 1;
		if (bin_angle > num_angle - 1) bin_angle = num_angle - 1;
		//feature.histogram[bin_distance] += 1;
		feature.histogram[num_angle*bin_distance + bin_angle] += 1;
	}
	for (int i = 0; i < feature.descriptorSize(); i++) {
		feature.histogram[i] = feature.histogram[i] / (float)(ksearch_keypoint->size());
	}

	return feature;

}


pcl::PointCloud<pcl::PFHSignature125>::Ptr com_pfh_feature(pcl::PointCloud<pcl::PointXYZ>::Ptr search_cloud, pcl::PointCloud<pcl::Normal>::Ptr normal, pcl::PointCloud<pcl::PointXYZ>::Ptr key, float leaf_size) {
	pcl::PointCloud<pcl::PFHSignature125>::Ptr features(new pcl::PointCloud<pcl::PFHSignature125>());
	pcl::PFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::PFHSignature125> pfh;
	pfh.setInputCloud(key);
	pfh.setInputNormals(normal);
	pfh.setSearchSurface(search_cloud);
	pfh.setRadiusSearch(leaf_size);
	pfh.compute(*features);
	return features;
}

pcl::PointCloud<pcl::FPFHSignature33>::Ptr com_fpfh_feature(pcl::PointCloud<pcl::PointXYZ>::Ptr search_cloud, pcl::PointCloud<pcl::Normal>::Ptr normal, pcl::PointCloud<pcl::PointXYZ>::Ptr key, float leaf_size) {
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr features(new pcl::PointCloud<pcl::FPFHSignature33>());
	pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
	fpfh.setInputCloud(key);
	fpfh.setInputNormals(normal);
	fpfh.setSearchSurface(search_cloud);
	fpfh.setRadiusSearch(leaf_size);
	fpfh.compute(*features);
	return features;
}

pcl::PointCloud<pcl::SHOT352>::Ptr com_shot_feature(pcl::PointCloud<pcl::PointXYZ>::Ptr search_cloud, pcl::PointCloud<pcl::Normal>::Ptr normal, pcl::PointCloud<pcl::PointXYZ>::Ptr key, float leaf_size) {
	pcl::PointCloud<pcl::SHOT352>::Ptr features(new pcl::PointCloud<pcl::SHOT352>());
	pcl::SHOTEstimation<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> fpfh;
	fpfh.setInputCloud(key);
	fpfh.setInputNormals(normal);
	fpfh.setSearchSurface(search_cloud);
	fpfh.setRadiusSearch(leaf_size);
	fpfh.compute(*features);
	return features;
}


pcl::PointCloud<pcl::PointXYZ>::Ptr key_detect_u(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float leaf_size) {
	pcl::PointCloud<pcl::PointXYZ>::Ptr key(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::UniformSampling<pcl::PointXYZ> filter;
	filter.setInputCloud(cloud);
	filter.setRadiusSearch(leaf_size);
	filter.filter(*key);
	return key;
}

pcl::PointCloud<pcl::PFHSignature125>::Ptr com_features(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals, pcl::search::KdTree<pcl::PointXYZ>::Ptr tree,
	pcl::PointCloud<pcl::PointXYZ>::Ptr key, float radius = 4, int num_angle = 12, int num_distance = 10) {

	pcl::PointCloud<pcl::PFHSignature125>::Ptr f(new pcl::PointCloud<pcl::PFHSignature125>());
	for (int k = 0; k < key->size(); k++) {
		vector<int> sup_ind;
		vector<float> sup_dis;
		tree->radiusSearch(key->points[k], radius, sup_ind, sup_dis);
		pcl::PFHSignature125 features;
		for (int i = 0; i < features.descriptorSize(); i++) {
			features.histogram[i] = 0;
		}
		for (int i = 0; i < sup_ind.size(); i++) {
			pcl::PFHSignature125 feature = com_feature(cloud, normals, tree, sup_ind[i], radius, num_angle, num_distance);
			for (int j = 0; j < features.descriptorSize(); j++) {
				features.histogram[j] += feature.histogram[j];
			}
		}
		for (int i = 0; i < features.descriptorSize(); i++) {
			features.histogram[i] = features.histogram[i] / (float)(sup_ind.size());
		}
		f->push_back(features);
	}

	return f;
}

pcl::PointCloud<pcl::PFHSignature125>::Ptr com_features2(pcl::PointCloud<pcl::PointXYZ>::Ptr key, pcl::PointCloud<pcl::PFHSignature125>::Ptr features, vector<float>& dis) {
	pcl::PointCloud<pcl::PFHSignature125>::Ptr new2_features(new pcl::PointCloud<pcl::PFHSignature125>);
	pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree_key(new pcl::KdTreeFLANN<pcl::PointXYZ>);
	kdtree_key->setInputCloud(key);
	for (int i = 0; i < key->size(); i++) {
		std::vector<int> neigh_indices(2);   //设置最近邻点的索引
		std::vector<float> neigh_sqr_dists(2); //申明最近邻平方距离值
		kdtree_key->nearestKSearch(key->at(i), 2, neigh_indices, neigh_sqr_dists);
		pcl::PFHSignature125 feature;
		for (int j = 0; j < feature.descriptorSize(); j++) {
			feature.histogram[j] = 0.5f*(features->points[i].histogram[j] + features->points[neigh_indices[1]].histogram[j]);
		}
		dis.push_back(sqrt(neigh_sqr_dists[1]));
		new2_features->push_back(feature);
	}
	return new2_features;
}

pcl::PointCloud<pcl::FPFHSignature33>::Ptr com_features2(pcl::PointCloud<pcl::PointXYZ>::Ptr key, pcl::PointCloud<pcl::FPFHSignature33>::Ptr features, vector<float>& dis) {
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr new2_features(new pcl::PointCloud<pcl::FPFHSignature33>);
	pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree_key(new pcl::KdTreeFLANN<pcl::PointXYZ>);
	kdtree_key->setInputCloud(key);
	for (int i = 0; i < key->size(); i++) {
		std::vector<int> neigh_indices(2);   //设置最近邻点的索引
		std::vector<float> neigh_sqr_dists(2); //申明最近邻平方距离值
		kdtree_key->nearestKSearch(key->at(i), 2, neigh_indices, neigh_sqr_dists);
		pcl::FPFHSignature33 feature;
		for (int j = 0; j < feature.descriptorSize(); j++) {
			feature.histogram[j] = 0.5f*(features->points[i].histogram[j] + features->points[neigh_indices[1]].histogram[j]);
		}
		dis.push_back(sqrt(neigh_sqr_dists[1]));
		new2_features->push_back(feature);
	}
	return new2_features;
}



pcl::PointCloud<pcl::SHOT352>::Ptr com_features2(pcl::PointCloud<pcl::PointXYZ>::Ptr key, pcl::PointCloud<pcl::SHOT352>::Ptr features, vector<float>& dis) {
	pcl::PointCloud<pcl::SHOT352>::Ptr new2_features(new pcl::PointCloud<pcl::SHOT352>);
	pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree_key(new pcl::KdTreeFLANN<pcl::PointXYZ>);
	kdtree_key->setInputCloud(key);
	for (int i = 0; i < key->size(); i++) {
		std::vector<int> neigh_indices(2);   //设置最近邻点的索引
		std::vector<float> neigh_sqr_dists(2); //申明最近邻平方距离值
		kdtree_key->nearestKSearch(key->at(i), 2, neigh_indices, neigh_sqr_dists);
		pcl::SHOT352 feature;
		for (int j = 0; j < feature.descriptorSize(); j++) {
			feature.descriptor[j] = 0.5f*(features->points[i].descriptor[j] + features->points[neigh_indices[1]].descriptor[j]);
		}
		dis.push_back(sqrt(neigh_sqr_dists[1]));
		new2_features->push_back(feature);
	}
	return new2_features;
}


pcl::PointCloud<pcl::PointXYZ>::Ptr key_detect(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::Normal>::Ptr normals, pcl::search::KdTree<pcl::PointXYZ>::Ptr tree,
	float radius_curvature = 5) {

	vector<bool> possible_key(cloud->size(), false);
	//int nums = 0;
	for (int i = 0; i < cloud->size(); i++) {
		if (normals->points[i].curvature > 0.01) {
			possible_key[i] = true;
			//nums += 1;
		}
	}
	vector<bool> possible_key_possible(possible_key);

	vector<float> avg_curvatures;
	for (int i = 0; i < cloud->size(); i++) {
		if (possible_key[i])
			avg_curvatures.push_back(com_avg_curvature(cloud, normals, i, radius_curvature, tree));
		else
			avg_curvatures.push_back(0);
	}
	//pcl::PointCloud<pcl::PFHSignature125>::Ptr features(new pcl::PointCloud<pcl::PFHSignature125>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr key(new pcl::PointCloud<pcl::PointXYZ>);
	for (int i = 0; i < cloud->size(); i++) {
		if (possible_key[i]) {
			if (is_max_avg_curvature(cloud, avg_curvatures, i, possible_key, possible_key_possible, tree, radius_curvature)) {//此处半径为计算曲率和的半径
				key->push_back(cloud->points[i]);
				//pcl::PFHSignature125 feature = com_features(cloud, normals1, tree, i, radius_curvature*1.5f, 12, 10);//此处半径为计算关键点邻域的半径
				//features->push_back(feature);
			}
		}
	}
	return key;
}

pcl::CorrespondencesPtr com_correspondence(pcl::PointCloud<pcl::PFHSignature125>::Ptr feature_source, pcl::PointCloud<pcl::PFHSignature125>::Ptr feature_target, float dis) {
	//  使用Kdtree找出 Model-Scene 匹配点
	pcl::CorrespondencesPtr model_scene_corrs(new pcl::Correspondences());

	pcl::KdTreeFLANN<pcl::PFHSignature125> match_search;   //设置配准的方法
	match_search.setInputCloud(feature_target);  //输入模板点云的描述子

  //每一个场景的关键点描述子都要找到模板中匹配的关键点描述子并将其添加到对应的匹配向量中。
	for (size_t i = 0; i < feature_source->size(); ++i)
	{
		std::vector<int> neigh_indices(1);   //设置最近邻点的索引
		std::vector<float> neigh_sqr_dists(1); //申明最近邻平方距离值
		int found_neighs = match_search.nearestKSearch(feature_source->at(i), 1, neigh_indices, neigh_sqr_dists);
		//scene_descriptors->at (i)是给定点云 1是临近点个数 ，neigh_indices临近点的索引  neigh_sqr_dists是与临近点的索引

		if (found_neighs == 1 && sqrt(neigh_sqr_dists[0]) < dis) // 仅当描述子与临近点的平方距离小于0.25（描述子与临近的距离在一般在0到1之间）才添加匹配
		{
			//neigh_indices[0]给定点，  i  是配准数  neigh_sqr_dists[0]与临近点的平方距离
			pcl::Correspondence corr(i, neigh_indices[0], sqrt(neigh_sqr_dists[0]));
			model_scene_corrs->push_back(corr);   //把配准的点存储在容器中
		}
	}
	return model_scene_corrs;
}

pcl::CorrespondencesPtr com_correspondence(pcl::PointCloud<pcl::FPFHSignature33>::Ptr feature_source, pcl::PointCloud<pcl::FPFHSignature33>::Ptr feature_target, float dis) {
	//  使用Kdtree找出 Model-Scene 匹配点
	pcl::CorrespondencesPtr model_scene_corrs(new pcl::Correspondences());

	pcl::KdTreeFLANN<pcl::FPFHSignature33> match_search;   //设置配准的方法
	match_search.setInputCloud(feature_target);  //输入模板点云的描述子

  //每一个场景的关键点描述子都要找到模板中匹配的关键点描述子并将其添加到对应的匹配向量中。
	for (size_t i = 0; i < feature_source->size(); ++i)
	{
		std::vector<int> neigh_indices(1);   //设置最近邻点的索引
		std::vector<float> neigh_sqr_dists(1); //申明最近邻平方距离值
		int found_neighs = match_search.nearestKSearch(feature_source->at(i), 1, neigh_indices, neigh_sqr_dists);
		//scene_descriptors->at (i)是给定点云 1是临近点个数 ，neigh_indices临近点的索引  neigh_sqr_dists是与临近点的索引

		if (found_neighs == 1 && sqrt(neigh_sqr_dists[0]) < dis) // 仅当描述子与临近点的平方距离小于0.25（描述子与临近的距离在一般在0到1之间）才添加匹配
		{
			//neigh_indices[0]给定点，  i  是配准数  neigh_sqr_dists[0]与临近点的平方距离
			pcl::Correspondence corr(i, neigh_indices[0], sqrt(neigh_sqr_dists[0]));
			model_scene_corrs->push_back(corr);   //把配准的点存储在容器中
		}
	}
	return model_scene_corrs;
}

pcl::CorrespondencesPtr com_correspondence(pcl::PointCloud<pcl::SHOT352>::Ptr feature_source, pcl::PointCloud<pcl::SHOT352>::Ptr feature_target, float dis) {
	//  使用Kdtree找出 Model-Scene 匹配点
	pcl::CorrespondencesPtr model_scene_corrs(new pcl::Correspondences());

	pcl::KdTreeFLANN<pcl::SHOT352> match_search;   //设置配准的方法
	match_search.setInputCloud(feature_target);  //输入模板点云的描述子

  //每一个场景的关键点描述子都要找到模板中匹配的关键点描述子并将其添加到对应的匹配向量中。
	for (size_t i = 0; i < feature_source->size(); ++i)
	{
		if (!pcl_isfinite(feature_source->at(i).descriptor[0])) //忽略 NaNs点
		{
			continue;
		}
		std::vector<int> neigh_indices(1);   //设置最近邻点的索引
		std::vector<float> neigh_sqr_dists(1); //申明最近邻平方距离值
		int found_neighs = match_search.nearestKSearch(feature_source->at(i), 1, neigh_indices, neigh_sqr_dists);
		//scene_descriptors->at (i)是给定点云 1是临近点个数 ，neigh_indices临近点的索引  neigh_sqr_dists是与临近点的索引

		if (found_neighs == 1 && sqrt(neigh_sqr_dists[0]) < dis) // 仅当描述子与临近点的平方距离小于0.25（描述子与临近的距离在一般在0到1之间）才添加匹配
		{
			//neigh_indices[0]给定点，  i  是配准数  neigh_sqr_dists[0]与临近点的平方距离
			pcl::Correspondence corr(i, neigh_indices[0], sqrt(neigh_sqr_dists[0]));
			model_scene_corrs->push_back(corr);   //把配准的点存储在容器中
		}
	}
	return model_scene_corrs;
}


pcl::CorrespondencesPtr com_correspondence2(pcl::PointCloud<pcl::PFHSignature125>::Ptr feature_source, vector<float> dis_source, pcl::PointCloud<pcl::PFHSignature125>::Ptr feature_target, vector<float> dis_target, float dis, float leaf_size) {
	//  使用Kdtree找出 Model-Scene 匹配点
	pcl::CorrespondencesPtr model_scene_corrs(new pcl::Correspondences());

	pcl::KdTreeFLANN<pcl::PFHSignature125> match_search;   //设置配准的方法
	match_search.setInputCloud(feature_target);  //输入模板点云的描述子
  //每一个场景的关键点描述子都要找到模板中匹配的关键点描述子并将其添加到对应的匹配向量中。
	for (size_t i = 0; i < feature_source->size(); i++) {
		float dis_feature = 0.0f;
		for (int j = 0; j < feature_source->points[i].descriptorSize(); j++) {
			dis_feature += pow(feature_source->points[i].histogram[j] - feature_target->points[i].histogram[j], 2.0f);
		}
		dis_feature = sqrt(dis_feature);
		if (dis_feature < dis && abs(dis_source[i] - dis_target[i]) < 10.0f*leaf_size) // 仅当描述子与临近点的平方距离小于0.25（描述子与临近的距离在一般在0到1之间）才添加匹配
		{
			//neigh_indices[0]给定点，  i  是配准数  neigh_sqr_dists[0]与临近点的平方距离
			pcl::Correspondence corr(i, i, dis_feature);
			model_scene_corrs->push_back(corr);//把配准的点存储在容器中
		}
	}
	return model_scene_corrs;
}


pcl::CorrespondencesPtr com_correspondence2(pcl::PointCloud<pcl::FPFHSignature33>::Ptr feature_source, vector<float> dis_source, pcl::PointCloud<pcl::FPFHSignature33>::Ptr feature_target, vector<float> dis_target, float dis, float leaf_size) {
	//  使用Kdtree找出 Model-Scene 匹配点
	pcl::CorrespondencesPtr model_scene_corrs(new pcl::Correspondences());

	pcl::KdTreeFLANN<pcl::FPFHSignature33> match_search;   //设置配准的方法
	match_search.setInputCloud(feature_target);  //输入模板点云的描述子
  //每一个场景的关键点描述子都要找到模板中匹配的关键点描述子并将其添加到对应的匹配向量中。
	for (size_t i = 0; i < feature_source->size(); i++) {
		float dis_feature = 0.0f;
		for (int j = 0; j < feature_source->points[i].descriptorSize(); j++) {
			dis_feature += pow(feature_source->points[i].histogram[j] - feature_target->points[i].histogram[j], 2.0f);
		}
		dis_feature = sqrt(dis_feature);
		if (dis_feature < dis && abs(dis_source[i] - dis_target[i]) < 10.0f*leaf_size) // 仅当描述子与临近点的平方距离小于0.25（描述子与临近的距离在一般在0到1之间）才添加匹配
		{
			//neigh_indices[0]给定点，  i  是配准数  neigh_sqr_dists[0]与临近点的平方距离
			pcl::Correspondence corr(i, i, dis_feature);
			model_scene_corrs->push_back(corr);//把配准的点存储在容器中
		}
	}
	return model_scene_corrs;
}

pcl::CorrespondencesPtr com_correspondence2(pcl::PointCloud<pcl::SHOT352>::Ptr feature_source, vector<float> dis_source, pcl::PointCloud<pcl::SHOT352>::Ptr feature_target, vector<float> dis_target, float dis, float leaf_size) {
	//  使用Kdtree找出 Model-Scene 匹配点
	pcl::CorrespondencesPtr model_scene_corrs(new pcl::Correspondences());

	pcl::KdTreeFLANN<pcl::SHOT352> match_search;   //设置配准的方法
	match_search.setInputCloud(feature_target);  //输入模板点云的描述子
  //每一个场景的关键点描述子都要找到模板中匹配的关键点描述子并将其添加到对应的匹配向量中。
	for (size_t i = 0; i < feature_source->size(); i++) {
		float dis_feature = 0.0f;
		for (int j = 0; j < feature_source->points[i].descriptorSize(); j++) {
			dis_feature += pow(feature_source->points[i].descriptor[j] - feature_target->points[i].descriptor[j], 2.0f);
		}
		dis_feature = sqrt(dis_feature);
		if (dis_feature < dis && abs(dis_source[i] - dis_target[i]) < 10.0f*leaf_size) // 仅当描述子与临近点的平方距离小于0.25（描述子与临近的距离在一般在0到1之间）才添加匹配
		{
			//neigh_indices[0]给定点，  i  是配准数  neigh_sqr_dists[0]与临近点的平方距离
			pcl::Correspondence corr(i, i, dis_feature);
			model_scene_corrs->push_back(corr);//把配准的点存储在容器中
		}
	}
	return model_scene_corrs;
}

//void show_coor(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_model, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_scenes, pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_model, pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_scenes, pcl::PointCloud<pcl::PFHSignature125>::Ptr features_model, pcl::PointCloud<pcl::PFHSignature125>::Ptr features_scenes, pcl::CorrespondencesPtr corr) {
//	for (int i = 0; i < corr->size(); i++) {
//		cout << corr->at(i).index_query << "---" << corr->at(i).index_match << "---" << corr->at(i).distance << endl;
//
//		//pcl::visualization::PCLPlotter plotter;
//		//plotter.addFeatureHistogram<pcl::PFHSignature125>(*features_model, "pfh", corr->at(i).index_query);
//		//plotter.addFeatureHistogram<pcl::PFHSignature125>(*features_scenes, "pfh", corr->at(i).index_match);
//		pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_ptr_model(new pcl::PointCloud<pcl::PointXYZ>());
//		pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_ptr_scenes(new pcl::PointCloud<pcl::PointXYZ>());
//		keypoints_ptr_model->push_back(keypoints_model->points[corr->at(i).index_query]);
//		keypoints_ptr_scenes->push_back(keypoints_scenes->points[corr->at(i).index_match]);
//		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
//		int v1(0);
//		viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);  //4个参数分别是X轴的最小值，最大值，Y轴的最小值，最大值，取值0-1，v1是标识
//		viewer->setBackgroundColor(255, 255, 255, v1);    //设置视口的背景颜色
//		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_key_model(keypoints_ptr_model, 0, 0, 0);
//		viewer->addPointCloud<pcl::PointXYZ>(keypoints_ptr_model, color_key_model, "color_key_model", v1);
//		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 15, "color_key_model");
//
//		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_cloud_model(cloud_model, 0, 0, 0);
//		viewer->addPointCloud<pcl::PointXYZ>(cloud_model, color_cloud_model, "cloud_model", v1);
//
//		int v2(0);
//		viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);  //4个参数分别是X轴的最小值，最大值，Y轴的最小值，最大值，取值0-1，v1是标识
//		viewer->setBackgroundColor(255, 255, 255, v2);    //设置视口的背景颜色
//		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_key_scenes(keypoints_ptr_scenes, 0, 0, 0);
//		viewer->addPointCloud<pcl::PointXYZ>(keypoints_ptr_scenes, color_key_scenes, "color_key_scenes", v2);
//		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 15, "color_key_scenes");
//
//		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_cloud_scenes(cloud_scenes, 0, 0, 0);
//		viewer->addPointCloud<pcl::PointXYZ>(cloud_scenes, color_cloud_scenes, "cloud_scenes", v2);
//
//		//plotter.plot();
//		// 等待直到可视化窗口关闭
//		while (!viewer->wasStopped())
//		{
//			viewer->spinOnce(100);
//			//boost::this_thread::sleep(boost::posix_time::microseconds(100000));
//		}
//	}
//
//}

void show_coor(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_model, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_scenes, pcl::PointCloud<pcl::PointXYZ> keypoints_model, pcl::PointCloud<pcl::PointXYZ> keypoints_scenes, pcl::PointCloud<pcl::PFHSignature125>::Ptr features_model, pcl::PointCloud<pcl::PFHSignature125>::Ptr features_scenes, pcl::CorrespondencesPtr corr) {
	for (int i = 0; i < corr->size(); i++) {
		cout << corr->at(i).index_query << "---" << corr->at(i).index_match << "---" << corr->at(i).distance << endl;
		//pcl::visualization::PCLPlotter plotter;
		//plotter.addFeatureHistogram<pcl::PFHSignature125>(*features_model, "pfh", corr->at(i).index_query);
		//plotter.addFeatureHistogram<pcl::PFHSignature125>(*features_scenes, "pfh", corr->at(i).index_match);
		cout << features_model->points[corr->at(i).index_query] << endl;
		cout << features_scenes->points[corr->at(i).index_match] << endl;
		pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_ptr_model(new pcl::PointCloud<pcl::PointXYZ>());
		pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_ptr_scenes(new pcl::PointCloud<pcl::PointXYZ>());
		keypoints_ptr_model->push_back(keypoints_model.points[corr->at(i).index_query]);
		keypoints_ptr_scenes->push_back(keypoints_scenes.points[corr->at(i).index_match]);
		boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));

		int v1(0);
		viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);  //4个参数分别是X轴的最小值，最大值，Y轴的最小值，最大值，取值0-1，v1是标识
		viewer->setBackgroundColor(255, 255, 255, v1);    //设置视口的背景颜色
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_key_model(keypoints_ptr_model, 255, 0, 0);
		viewer->addPointCloud<pcl::PointXYZ>(keypoints_ptr_model, color_key_model, "color_key_model", v1);
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "color_key_model");

		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_cloud_model(cloud_model, 0, 0, 255);
		viewer->addPointCloud<pcl::PointXYZ>(cloud_model, color_cloud_model, "cloud_model", v1);

		int v2(0);
		viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);  //4个参数分别是X轴的最小值，最大值，Y轴的最小值，最大值，取值0-1，v1是标识
		viewer->setBackgroundColor(255, 255, 255, v2);    //设置视口的背景颜色
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_key_scenes(keypoints_ptr_scenes, 255, 0, 0);
		viewer->addPointCloud<pcl::PointXYZ>(keypoints_ptr_scenes, color_key_scenes, "color_key_scenes", v2);
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "color_key_scenes");

		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_cloud_scenes(cloud_scenes, 0, 0, 255);
		viewer->addPointCloud<pcl::PointXYZ>(cloud_scenes, color_cloud_scenes, "cloud_scenes", v2);

		//plotter.plot();
		// 等待直到可视化窗口关闭
		while (!viewer->wasStopped())
		{
			viewer->spinOnce(100);
			//boost::this_thread::sleep(boost::posix_time::microseconds(100000));
		}
	}

}

float get_leaf_size(pcl::PointCloud<pcl::PointXYZ>::Ptr Acloud, int nums = 2000) {
	//建立A点云KD-TREE
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(Acloud);//Acloud在Bcloud中进行搜索
	//进行1邻域点搜索
	int K = 2;
	std::vector<int> pointIdxNKNSearch(K);//最近点索引
	std::vector<float> pointNKNSquaredDistance(K);//最近点距离

	//在B点云中计算点与最近邻的平均距离
	float avgdistance = 0;
	for (int i = 0; i < Acloud->size(); i++) {
		kdtree.nearestKSearch(Acloud->points[i], K, pointIdxNKNSearch, pointNKNSquaredDistance);
		avgdistance = avgdistance + sqrt(pointNKNSquaredDistance[1]);
		pointIdxNKNSearch.clear();
		pointNKNSquaredDistance.clear();
	}
	avgdistance = (float)avgdistance / (float)(Acloud->size());
	cout << "平均距离：" << avgdistance << endl;
	float num;
	num = pow((float)Acloud->points.size() / (float)nums, 1.0 / 3.0);
	avgdistance = avgdistance * num;
	return avgdistance;

}

pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_grid(pcl::PointCloud<pcl::PointXYZ>::Ptr Acloud, float leaf_size) {
	//体素滤波
	pcl::PointCloud<pcl::PointXYZ>::Ptr Acloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::VoxelGrid<pcl::PointXYZ> sor;  //创建滤波对象
	sor.setInputCloud(Acloud);            //设置需要过滤的点云给滤波对象
	sor.setLeafSize(leaf_size, leaf_size, leaf_size);  //设置滤波时创建的体素体积
	sor.filter(*Acloud_filtered);           //执行滤波处理，存储输出	
	return Acloud_filtered;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_to_num(pcl::PointCloud<pcl::PointXYZ>::Ptr Acloud, float& leaf_size, int point_num = 20000) {
	while (Acloud->points.size() >= point_num + point_num * 0.2f) {
		leaf_size = get_leaf_size(Acloud, point_num);
		*Acloud = *voxel_grid(Acloud, leaf_size);
		cout << "点云点数：" << Acloud->size() << endl;
	}
	return Acloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr approximate_voxel_grid(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float size = 1) {
	cout << "before filter: " << cloud->size() << endl;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::ApproximateVoxelGrid<pcl::PointXYZ> avg;
	avg.setInputCloud(cloud);
	avg.setLeafSize(size, size, size);
	avg.filter(*cloud_filtered);
	cout << "after filter: " << cloud_filtered->size() << endl;
	return cloud_filtered;
}

void show_line(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target, pcl::PointCloud<pcl::PointXYZ>::Ptr key_source, pcl::PointCloud<pcl::PointXYZ>::Ptr key_target, pcl::CorrespondencesPtr corr, float leaf_size) {

	pcl::PointCloud<pcl::PointXYZ>::Ptr new_key_source(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr new_key_target(new pcl::PointCloud<pcl::PointXYZ>());


	for (int i = 0; i < corr->size(); i++) {

		new_key_source->push_back(key_source->points[corr->at(i).index_query]);
		new_key_target->push_back(key_target->points[corr->at(i).index_match]);

	}

	pcl::PointCloud<pcl::PointXYZ>::Ptr new_cloud_source(new pcl::PointCloud<pcl::PointXYZ>);
	*new_cloud_source = *cloud_source;
	for (int i = 0; i < cloud_source->size(); i++) {
		new_cloud_source->points[i].y += 300.0f* leaf_size;
	}

	for (int i = 0; i < new_key_source->size(); i++) {
		new_key_source->points[i].y += 300.0f* leaf_size;
	}

	pcl::visualization::PCLVisualizer line("line");
	line.setBackgroundColor(255, 255, 255);
	line.addPointCloud(cloud_target, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_target, 0.0, 0, 255.0), "cloud_target");
	line.addPointCloud(new_cloud_source, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(new_cloud_source, 0.0, 255, 0), "cloud_source");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>color_new_key_target(new_key_target, 255, 0, 0);//红色关键点
	line.addPointCloud<pcl::PointXYZ>(new_key_target, color_new_key_target, "new_key_target");
	line.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "new_key_target");

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_new_key_source(new_key_source, 255, 0, 0);//红色关键点
	line.addPointCloud<pcl::PointXYZ>(new_key_source, color_new_key_source, "new_key_source");
	line.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "new_key_source");


	for (int i = 0; i < new_key_source->size(); i++)
	{

		pcl::PointXYZ source_point = new_key_source->points[i];
		pcl::PointXYZ target_point = new_key_target->points[i];
		line.addLine(source_point, target_point, 255, 0, 255, to_string(i));
		line.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, to_string(i));
	}
	line.spin();
}

std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > show_line(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_model, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_scenes, pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_ptr_model, pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_ptr_scenes, pcl::PointCloud<pcl::Normal>::Ptr normals1_model, pcl::PointCloud<pcl::Normal>::Ptr normals1_scenes, pcl::CorrespondencesPtr corr, float radius) {
	std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
	std::vector<pcl::Correspondences> clustered_corrs;

	pcl::PointCloud<pcl::ReferenceFrame>::Ptr model_rf(new pcl::PointCloud<pcl::ReferenceFrame>());
	pcl::PointCloud<pcl::ReferenceFrame>::Ptr scene_rf(new pcl::PointCloud<pcl::ReferenceFrame>());

	pcl::BOARDLocalReferenceFrameEstimation<pcl::PointXYZ, pcl::Normal, pcl::ReferenceFrame> rf_est;
	rf_est.setFindHoles(true);
	rf_est.setRadiusSearch(radius);

	rf_est.setInputCloud(keypoints_ptr_model);
	rf_est.setInputNormals(normals1_model);
	rf_est.setSearchSurface(cloud_model);
	rf_est.compute(*model_rf);

	rf_est.setInputCloud(keypoints_ptr_scenes);
	rf_est.setInputNormals(normals1_scenes);
	rf_est.setSearchSurface(cloud_scenes);
	rf_est.compute(*scene_rf);

	//  Clustering
	pcl::Hough3DGrouping<pcl::PointXYZ, pcl::PointXYZ, pcl::ReferenceFrame, pcl::ReferenceFrame> clusterer;
	clusterer.setHoughBinSize(radius);
	clusterer.setHoughThreshold(radius);
	clusterer.setUseInterpolation(true);
	clusterer.setUseDistanceWeight(false);

	clusterer.setInputCloud(keypoints_ptr_model);
	clusterer.setInputRf(model_rf);
	clusterer.setSceneCloud(keypoints_ptr_scenes);
	clusterer.setSceneRf(scene_rf);
	clusterer.setModelSceneCorrespondences(corr);

	//clusterer.cluster (clustered_corrs);
	clusterer.recognize(rototranslations, clustered_corrs);

	std::cout << "Model instances found: " << rototranslations.size() << std::endl;
	for (size_t i = 0; i < rototranslations.size(); ++i)
	{
		std::cout << "\n    Instance " << i + 1 << ":" << std::endl;
		std::cout << "        Correspondences belonging to this instance: " << clustered_corrs[i].size() << std::endl;

		// Print the rotation matrix and translation vector
		Eigen::Matrix3f rotation = rototranslations[i].block<3, 3>(0, 0);
		Eigen::Vector3f translation = rototranslations[i].block<3, 1>(0, 3);

		printf("\n");
		printf("            | %6.3f %6.3f %6.3f | \n", rotation(0, 0), rotation(0, 1), rotation(0, 2));
		printf("        R = | %6.3f %6.3f %6.3f | \n", rotation(1, 0), rotation(1, 1), rotation(1, 2));
		printf("            | %6.3f %6.3f %6.3f | \n", rotation(2, 0), rotation(2, 1), rotation(2, 2));
		printf("\n");
		printf("        t = < %0.3f, %0.3f, %0.3f >\n", translation(0), translation(1), translation(2));
	}
	pcl::visualization::PCLVisualizer viewer("Correspondence Grouping");
	viewer.setBackgroundColor(255, 255, 255);

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_cloud_scenes(cloud_scenes, 0, 0, 255);
	viewer.addPointCloud(cloud_scenes, color_cloud_scenes, "cloud_scenes");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud_scenes");

	pcl::PointCloud<pcl::PointXYZ>::Ptr off_cloud_model(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr off_keypoints_model(new pcl::PointCloud<pcl::PointXYZ>());


	//  We are translating the model so that it doesn't end in the middle of the scene representation
	pcl::transformPointCloud(*cloud_model, *off_cloud_model, Eigen::Vector3f(-200, 0, 0), Eigen::Quaternionf(1, 0, 0, 0));
	pcl::transformPointCloud(*keypoints_ptr_model, *off_keypoints_model, Eigen::Vector3f(-200, 0, 0), Eigen::Quaternionf(1, 0, 0, 0));

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_off_model(off_cloud_model, 0, 255, 0);
	viewer.addPointCloud(off_cloud_model, color_off_model, "off_cloud_model");



	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_keypoints_ptr_scenes(keypoints_ptr_scenes, 255, 0, 0);
	viewer.addPointCloud(keypoints_ptr_scenes, color_keypoints_ptr_scenes, "keypoints_ptr_scenes");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "keypoints_ptr_scenes");

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_off_keypoints_model(off_keypoints_model, 255, 0, 0);
	viewer.addPointCloud(off_keypoints_model, color_off_keypoints_model, "off_keypoints_model");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "off_keypoints_model");


	for (size_t i = 0; i < rototranslations.size(); ++i)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr rotated_model(new pcl::PointCloud<pcl::PointXYZ>());
		pcl::transformPointCloud(*cloud_model, *rotated_model, rototranslations[i]);

		std::stringstream ss_cloud;
		ss_cloud << "instance" << i;

		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> rotated_model_color_handler(rotated_model, 0, 255, 0);
		viewer.addPointCloud(rotated_model, rotated_model_color_handler, ss_cloud.str());


		for (size_t j = 0; j < clustered_corrs[i].size(); ++j)
		{
			std::stringstream ss_line;
			ss_line << "correspondence_line" << i << "_" << j;
			pcl::PointXYZ& model_point = off_keypoints_model->at(clustered_corrs[i][j].index_query);
			pcl::PointXYZ& scene_point = keypoints_ptr_scenes->at(clustered_corrs[i][j].index_match);

			//  We are drawing a line for each pair of clustered correspondences found between the model and the scene
			viewer.addLine<pcl::PointXYZ, pcl::PointXYZ>(model_point, scene_point, 255, 0, 255, ss_line.str());
		}

	}

	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
	return rototranslations;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr add_gaussian_noise(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float m) {
	float leaf_size = 0;
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(cloud);//Acloud在Bcloud中进行搜索
	//进行1邻域点搜索
	int K = 2;
	std::vector<int> pointIdxNKNSearch(K);//最近点索引
	std::vector<float> pointNKNSquaredDistance(K);//最近点距离
	//在B点云中计算点与最近邻的平均距离
	double avgdistance = 0;
	for (int i = 0; i < cloud->size(); i++) {
		kdtree.nearestKSearch(cloud->points[i], K, pointIdxNKNSearch, pointNKNSquaredDistance);
		leaf_size = leaf_size + sqrt(pointNKNSquaredDistance[1]);
		pointIdxNKNSearch.clear();
		pointNKNSquaredDistance.clear();
	}
	leaf_size = (float)leaf_size / (float)(cloud->size());
	//添加高斯噪声
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudfiltered(new pcl::PointCloud<pcl::PointXYZ>());
	cloudfiltered->points.resize(cloud->points.size());//将点云的cloud的size赋值给噪声
	cloudfiltered->header = cloud->header;
	cloudfiltered->width = cloud->width;
	cloudfiltered->height = cloud->height;
	boost::mt19937 rng;
	rng.seed(static_cast<unsigned int>(time(0)));
	boost::normal_distribution<> nd(0, m*leaf_size);
	boost::variate_generator<boost::mt19937&, boost::normal_distribution<>> var_nor(rng, nd);
	//添加噪声
	for (size_t point_i = 0; point_i < cloud->points.size(); ++point_i)
	{
		cloudfiltered->points[point_i].x = cloud->points[point_i].x + static_cast<float> (var_nor());
		cloudfiltered->points[point_i].y = cloud->points[point_i].y + static_cast<float> (var_nor());
		cloudfiltered->points[point_i].z = cloud->points[point_i].z + static_cast<float> (var_nor());
	}
	return cloudfiltered;
}

float com_leaf(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(cloud);//Acloud在Bcloud中进行搜索
	//进行1邻域点搜索
	int K = 2;
	std::vector<int> pointIdxNKNSearch(K);//最近点索引
	std::vector<float> pointNKNSquaredDistance(K);//最近点距离
	//在B点云中计算点与最近邻的平均距离
	float leaf_size = 0;
	for (int i = 0; i < cloud->size(); i++) {
		kdtree.nearestKSearch(cloud->points[i], K, pointIdxNKNSearch, pointNKNSquaredDistance);
		leaf_size = leaf_size + sqrt(pointNKNSquaredDistance[1]);
		pointIdxNKNSearch.clear();
		pointNKNSquaredDistance.clear();
	}
	leaf_size = (float)leaf_size / (float)(cloud->size());
	//cout << "平均距离：" << leaf_size << endl;
	return leaf_size;
}

int main(int argc, char** argv) {

	double start = 0;
	double end = 0;


	string road = "D:/code/PCD/自建配准点云/scene+rt/";
	vector<string> names = { "a1", "a2","b1","b2","c1","c2","c3","c4","c5","c6","d1","d2","g1","g2","h1","h2" };

	//////////////////////读取点云//////////////////////////////////////////////
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPLYFile("D:/code/PCD/自建配准点云/scene+rt/d1.ply", *cloud_source);
	cout << "滤波前源点云点数：" << cloud_source->size() << endl;
	//pcl::PointCloud<pcl::PointXYZ>::Ptr old_cloud_source(new pcl::PointCloud<pcl::PointXYZ>);
	//*old_cloud_source = *cloud_source;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPLYFile("D:/code/PCD/自建配准点云/scene+rt/d2.ply", *cloud_target);
	cout << "滤波前目标点云点数：" << cloud_target->size() << endl;

	//pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_source(new pcl::PointCloud<pcl::PointXYZ>);
	//pcl::io::loadPLYFile("D:/code/PCD/识别点云/scene/filter/cheff.ply", *cloud_source);
	//cout << "滤波前源点云点数：" << cloud_source->size() << endl;

	//pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_target(new pcl::PointCloud<pcl::PointXYZ>);
	//pcl::io::loadPLYFile("D:/code/PCD/识别点云/model/filter/cheff.ply", *cloud_target);
	//cout << "滤波前目标点云点数：" << cloud_target->size() << endl;


	//pcl::visualization::PCLVisualizer visu0("before");
	//visu0.setBackgroundColor(255,255,255);
	//visu0.addPointCloud(cloud_target, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_target, 0.0, 0, 255.0), "scen1e2");
	//visu0.addPointCloud(cloud_source, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_source, 0.0, 255, 0), "object1_aligned2");
	//visu0.spin();
	//*cloud_source = *add_gaussian_noise(cloud_source, 1);
	//*cloud_target = *add_gaussian_noise(cloud_target, 1);

	//////////////////////循环体素滤波//////////////////////////////////////////////////////////////////////
	start = GetTickCount();
	float leaf_size = 0;
	*cloud_source = *point_cloud_to_num(cloud_source, leaf_size, 20000);
	end = GetTickCount();
	leaf_size = com_leaf(cloud_source);
	end = GetTickCount();
	cout << "源点云分辨率：" << leaf_size << endl;
	cout << "源点云点数：" << cloud_source->size() << endl;
	cout << "源点云循环体素滤波：" << end - start << "ms" << endl;
	//pcl::io::savePLYFile("e:/boy1.ply", *cloud_source);



	/////////////////源点云特征估计///////////////////////////////////////////////////////////////////////////
	pcl::PointCloud<pcl::Normal>::Ptr normals_source(new pcl::PointCloud<pcl::Normal>());
	*normals_source = *normal_estimation_OMP(cloud_source, leaf_size*5.0f);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_source(new pcl::search::KdTree<pcl::PointXYZ>());
	tree_source->setInputCloud(cloud_source);

	pcl::PointCloud<pcl::SHOT352>::Ptr features_source(new pcl::PointCloud<pcl::SHOT352>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr key_source(new pcl::PointCloud<pcl::PointXYZ>);


	start = GetTickCount();

	*key_source = *key_detect_u(cloud_source, 20.0*leaf_size);//对比实验
	*features_source = *com_shot_feature(cloud_source, normals_source, key_source, 30.0 *leaf_size);//对比实验



	end = GetTickCount();
	cout << "源点云关键点数目：" << key_source->size() << endl;
	cout << "源点云特征估计：" << end - start << "ms" << endl;
	//show_key_point(cloud_source, key_source);


	//////////////////////循环体素滤波//////////////////////////////////////////////////////////////////////
	start = GetTickCount();
	*cloud_target = *voxel_grid(cloud_target, leaf_size);
	*cloud_target = *point_cloud_to_num(cloud_target, leaf_size, 20000);
	leaf_size = com_leaf(cloud_target);
	end = GetTickCount();
	cout << "目标点云分辨率：" << leaf_size << endl;
	cout << "目标点云点数：" << cloud_target->size() << endl;
	cout << "目标点云循环体素滤波：" << end - start << "ms" << endl;
	//pcl::io::savePLYFile("e:/boy2.ply", *cloud_target);

	//////////////////目标点云特征估计//////////////////////////////////////////////////////////////////////////
	pcl::PointCloud<pcl::Normal>::Ptr normals_target(new pcl::PointCloud<pcl::Normal>());
	*normals_target = *normal_estimation_OMP(cloud_target, leaf_size*5.0f);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_target(new pcl::search::KdTree<pcl::PointXYZ>());
	tree_target->setInputCloud(cloud_target);

	pcl::PointCloud<pcl::SHOT352>::Ptr features_target(new pcl::PointCloud<pcl::SHOT352>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr key_target(new pcl::PointCloud<pcl::PointXYZ>);

	start = GetTickCount();

	*key_target = *key_detect_u(cloud_target, 20.0*leaf_size);//对比实验
	*features_target = *com_shot_feature(cloud_target, normals_target, key_target, 30.0 *leaf_size);//对比实验

	end = GetTickCount();
	cout << "目标点云关键点数目：" << key_target->size() << endl;
	cout << "目标点云特征估计：" << end - start << "ms" << endl;
	//show_key_point(cloud_target, key_target);


	////////////////////初始对应关系估计////////////////////////////////////////////////////////////////////////
	start = GetTickCount();
	pcl::CorrespondencesPtr corr(new pcl::Correspondences());
	float dis = 0.5;

	*corr = *com_correspondence(features_source, features_target, dis);
	end = GetTickCount();
	cout << "初始对应关系数目：" << corr->size() << endl;
	cout << "初始对应关系估计：" << end - start << "ms" << endl;
	//show_coor(cloud_source, cloud_target, *key_source, *key_target, features_source, features_target, corr);
	show_line(cloud_source, cloud_target, key_source, key_target, corr, leaf_size);

	pcl::registration::CorrespondenceRejectorOneToOne coo;
	coo.setInputCorrespondences(corr);
	coo.getRemainingCorrespondences(*corr, *corr);

	/////////////////////提取初始对应关系关键点和特征///////////////////////////////////////////////////////////////////////
	pcl::PointCloud<pcl::PointXYZ>::Ptr new_key_source(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr new_key_target(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::SHOT352>::Ptr new_features_source(new pcl::PointCloud<pcl::SHOT352>);
	pcl::PointCloud<pcl::SHOT352>::Ptr new_features_target(new pcl::PointCloud<pcl::SHOT352>);

	for (int i = 0; i < corr->size(); i++) {

		new_key_source->push_back(key_source->points[corr->at(i).index_query]);
		new_key_target->push_back(key_target->points[corr->at(i).index_match]);
		new_features_source->push_back(features_source->points[corr->at(i).index_query]);
		new_features_target->push_back(features_target->points[corr->at(i).index_match]);
	}

	//////////////////去除错误对应关系////////////////////////////////////////////////////////////////////////////////
	pcl::PointCloud<pcl::SHOT352>::Ptr new2_features_source(new pcl::PointCloud<pcl::SHOT352>);
	vector<float> dis_source;
	pcl::PointCloud<pcl::SHOT352>::Ptr new2_features_target(new pcl::PointCloud<pcl::SHOT352>);
	vector<float> dis_target;
	start = GetTickCount();
	*new2_features_source = *com_features2(new_key_source, new_features_source, dis_source);
	*new2_features_target = *com_features2(new_key_target, new_features_target, dis_target);

	pcl::CorrespondencesPtr corr2(new pcl::Correspondences());
	float dis2 = 1;

	*corr2 = *com_correspondence2(new2_features_source, dis_source, new2_features_target, dis_target, dis2, leaf_size);
	end = GetTickCount();
	cout << "对应关系数目：" << corr2->size() << endl;
	cout << "对应关系：" << end - start << "ms" << endl;

	/////////////////////提取对应关系关键点和特征///////////////////////////////////////////////////////////////////////
	pcl::PointCloud<pcl::PointXYZ>::Ptr new3_key_source(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr new3_key_target(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::SHOT352>::Ptr new3_features_source(new pcl::PointCloud<pcl::SHOT352>);
	pcl::PointCloud<pcl::SHOT352>::Ptr new3_features_target(new pcl::PointCloud<pcl::SHOT352>);
	for (int i = 0; i < corr2->size(); i++) {
		new3_key_source->push_back(new_key_source->points[corr2->at(i).index_query]);
		new3_key_target->push_back(new_key_target->points[corr2->at(i).index_match]);
		new3_features_source->push_back(new2_features_source->points[corr2->at(i).index_query]);
		new3_features_target->push_back(new2_features_target->points[corr2->at(i).index_match]);
	}
	//show_coor(cloud_source, cloud_target, *new_key_source, *new_key_target, new2_features_source, new2_features_target, corr2);
	show_line(cloud_source, cloud_target, new_key_source, new_key_target, corr2, leaf_size);
	////////////////////SVD////////////////////////////////////////////////////////////////////////
	start = GetTickCount();
	pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ, float> svd;
	Eigen::Matrix4f trans;
	svd.estimateRigidTransformation(*new3_key_source, *new3_key_target, trans);
	pcl::transformPointCloud(*new3_key_source, *new3_key_source, trans);
	pcl::transformPointCloud(*cloud_source, *cloud_source, trans);
	end = GetTickCount();
	cout << "SVD：" << end - start << "ms" << endl;
	//pcl::visualization::PCLVisualizer visu1("Alignment1");
	//visu1.setBackgroundColor(255, 255, 255);
	//visu1.addPointCloud(cloud_target, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_target, 0.0, 0, 0.0), "scene");
	//visu1.addPointCloud(cloud_source, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_source, 0.0, 0.0, 0), "object_aligned");
	//visu1.spin();


	//////////////////////随机采样一致性//////////////////////////////////////////////////////////////////////
	start = GetTickCount();
	pcl::SampleConsensusPrerejective<pcl::PointXYZ, pcl::PointXYZ, pcl::SHOT352> align;
	align.setInputSource(new3_key_source);
	align.setSourceFeatures(new3_features_source);
	align.setInputTarget(new3_key_target);
	align.setTargetFeatures(new3_features_target);
	align.setMaximumIterations(10000); // Number of RANSAC iterations
	align.setNumberOfSamples(3); // Number of points to sample for generating/prerejecting a pose
	align.setCorrespondenceRandomness(3); // Number of nearest features to use
	align.setSimilarityThreshold(0.9f); // Polygonal edge length similarity threshold
	align.setMaxCorrespondenceDistance(2.0f*leaf_size); // Inlier threshold
	//align.setRANSACOutlierRejectionThreshold(5.0f * leaf_size);
	align.setInlierFraction(0.25f); // Required inlier fraction for accepting a pose hypothesis
	align.align(*new3_key_source);

	end = GetTickCount();
	cout << "随机采样一致性：" << end - start << "ms" << endl;
	cout << "分数： " << align.getFitnessScore(5.0f*leaf_size) << endl;
	Eigen::Matrix4f transformation = align.getFinalTransformation();
	pcl::console::print_info("Inliers: %i/%i\n", align.getInliers().size(), new3_key_source->size());
	pcl::transformPointCloud(*cloud_source, *cloud_source, transformation);
	//Show alignment
	//pcl::visualization::PCLVisualizer visu2("Alignment2");
	//visu2.setBackgroundColor(255, 255, 255);
	//visu2.addPointCloud(cloud_source, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_source, 0.0, 255.0, 0.0), "scene1");
	//visu2.addPointCloud(cloud_target, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_target, 0.0, 0.0, 255.0), "object_aligned1");
	//visu2.spin();


	vector<int> indice;
	indice = align.getInliers();
	//////////////////将点云平移，方便显示////////////////////////////////////////////////////////////////////////
	pcl::PointCloud<pcl::PointXYZ>::Ptr new4_key_source(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr new4_key_target(new pcl::PointCloud<pcl::PointXYZ>);
	for (int i = 0; i < indice.size(); i++) {
		new4_key_source->push_back(new3_key_source->points[indice[i]]);
		new4_key_target->push_back(new3_key_target->points[indice[i]]);
	}
	pcl::PointCloud<pcl::PointXYZ>::Ptr new4_cloud_source(new pcl::PointCloud<pcl::PointXYZ>);
	*new4_cloud_source = *cloud_source;
	for (int i = 0; i < new4_cloud_source->size(); i++) {
		new4_cloud_source->points[i].x += 300.0f* leaf_size;
		new4_cloud_source->points[i].y += 300.0f* leaf_size;
	}
	for (int i = 0; i < new4_key_source->size(); i++) {
		new4_key_source->points[i].x += 300.0f* leaf_size;
		new4_key_source->points[i].y += 300.0f* leaf_size;
	}
	////////////////////显示对应点连线//////////////////////////////////////////////////////////////////////
	pcl::visualization::PCLVisualizer line("line");
	line.setBackgroundColor(255, 255, 255);
	line.addPointCloud(new4_cloud_source, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(new4_cloud_source, 0.0, 0, 0), "new4_cloud_source");
	line.addPointCloud(cloud_target, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_target, 0.0, 0.0, 0), "cloud_target");

	line.addPointCloud<pcl::PointXYZ>(new4_key_source, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(new4_key_source, 0, 0, 0), "new4_key_source");
	line.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "new4_key_source");

	line.addPointCloud<pcl::PointXYZ>(new4_key_target, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(new4_key_target, 0, 0, 0), "new4_key_target");
	line.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "new4_key_target");

	for (int i = 0; i < new4_key_source->size(); i++)
	{
		line.addLine(new4_key_source->points[i], new4_key_target->points[i], 0, 0, 0, to_string(i));
		line.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, to_string(i));
	}
	line.spin();

	////////////////////ICP////////////////////////////////////////////////////////////////////////
	start = GetTickCount();
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	icp.setInputSource(cloud_source);
	icp.setInputTarget(cloud_target);
	icp.setTransformationEpsilon(5.0f*leaf_size);
	icp.setMaxCorrespondenceDistance(5.0f * leaf_size);
	icp.setMaximumIterations(3000);
	icp.align(*cloud_source);
	end = GetTickCount();
	cout << "ICP：" << end - start << "ms" << endl;
	std::cout << "has converged:" << icp.hasConverged() << " score: " <<
		icp.getFitnessScore(5.0f*leaf_size) << std::endl;
	std::cout << icp.getFinalTransformation() << std::endl;
	pcl::visualization::PCLVisualizer visu("Alignment");
	visu.setBackgroundColor(255, 255, 255);
	pcl::transformPointCloud(*cloud_source, *cloud_source, icp.getFinalTransformation());
	visu.addPointCloud(cloud_source, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_source, 0.0, 255.0, 0.0), "scene1");
	visu.addPointCloud(cloud_target, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud_target, 0.0, 0.0, 255.0), "object_aligned1");
	visu.spin();
	system("pause");
	return 0;
}
