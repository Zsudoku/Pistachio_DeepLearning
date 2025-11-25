#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <ros/ros.h>
#include <ros/console.h>
#include <thread>

#include <filesystem>
#include <nlohmann/json.hpp>
#include "read_configs.h"
#include "dataset.h"
#include "map_user.h"
#include <httplib.h>
namespace fs = std::filesystem;

using namespace httplib;
using json = nlohmann::json;
using namespace cv;
using namespace std;
  
  // 创建上传目录（如果不存在）
void create_upload_dir() {
      fs::path upload_dir = "uploads";
      if (!fs::exists(upload_dir)) {
          fs::create_directory(upload_dir);
      }
  }

std::vector<double> SaveData(const std::vector<std::pair<std::string, Eigen::Matrix4d>>& trajectory) {
  std::vector<double> buffer = {0, 0, 0, 0, 0, 0, 0}; // 位置(3) + 四元数(4)
  if (trajectory.empty()) return buffer;
  const auto& pose_pair = trajectory.back(); // 取最新位姿
  Eigen::Vector3d t = pose_pair.second.block<3, 1>(0, 3);
  Eigen::Quaterniond q(pose_pair.second.block<3, 3>(0, 0));

  double sample[7] = {t(0), t(1), t(2), q.x(), q.y(), q.z(), q.w()};
  memcpy(buffer.data(), sample, sizeof(sample));

  return buffer;
}

//检查jpeg
bool CheckJpeg(string file)
{
	if (file.empty())
	{
		return false;
	}
	ifstream in(file.c_str(), ios::in | ios::binary);
	if (!in.is_open())
	{
		cout << "Error opening file!" << endl;
		return false;
	}

	int start;
	in.read((char*)&start, 4);
	short int lstart = start << 16 >> 16;

	//cout << hex << lstart << "  ";
	in.seekg(-4, ios::end);
	int end;
	in.read((char*)&end, 4);
	short int lend = end >> 16;
	//cout << hex << lend << endl;

	in.close();
	if ((lstart != -9985) || (lend != -9729)) //0xd8ff 0xd9ff
	{
		return true;
	}
	return false;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "air_slam");
  ros::NodeHandle nh;

  std::string config_path, model_dir, map_root, voc_path, traj_path;
  ros::param::get("~config_path", config_path);
  ros::param::get("~model_dir", model_dir);
  ros::param::get("~map_root", map_root);
  ros::param::get("~voc_path", voc_path);
  ros::param::get("~traj_path", traj_path);

  RelocalizationConfigs configs(config_path, model_dir);
  ros::param::get("~dataroot", configs.dataroot);
  ros::param::get("~camera_config_path", configs.camera_config_path);
  std::ofstream outFile;

  outFile.open("/my_workspace/CamTrans.txt");
  outFile.close();

  MapUser map_user(configs, nh);
  map_user.LoadMap(map_root);
  map_user.LoadVocabulary(voc_path);
  double threshold_ratio = 0.3;

  Server svr;
  
  // 文件上传接口
  svr.Post("/upload", [&](const Request& req, Response& res) {
    create_upload_dir();
    json result;
    // std::cout<<"start Get"<<std::endl;
    std::unordered_set<std::string> processed_images;
    std::vector<std::pair<std::string, Eigen::Matrix4d>> trajectory;
    try {
        // 处理multipart表单
        if (!req.has_file("file") || !req.has_file("name")) {
            result["code"] = 1003;
            result["msg"] = "Missing file or name field";
            result["data"] = "";
            res.set_content(result.dump(), "application/json");
            std::cout << "Missing file or name field" << std::endl;
            return;
        }

        // 保存文件
        const auto& file = req.get_file_value("file");
        fs::path save_path = fs::path("uploads") / file.filename;
        std::ofstream ofs(save_path, std::ios::binary);
        ofs << file.content;
        ofs.close();
        //std::cout << "图像保存成功！" << std::endl;
        //std::this_thread::sleep_for(std::chrono::milliseconds(10));
        // slam start
        std::string image_path = "/home/jg/.ros/uploads/testImg.jpg";
        bool img_flag = CheckJpeg(image_path);
        //std::cout << "图像读取成功！" << std::endl;
        if (img_flag){
          result["code"] = 1002;
          result["msg"] = "图片存在问题";
          result["data"] = "";
          ROS_INFO("There are issues with the image");
          //std::cout << "图片存在问题" << std::endl;
          res.set_content(result.dump(), "application/json");
          return;
        }
        cv::Mat image = cv::imread(image_path, 0);
        Mat mask;
        inRange(image, Scalar(128), Scalar(128), mask); // 创建128像素的掩码
        int bad_pixels = countNonZero(mask);
      
        double ratio = static_cast<double>(bad_pixels) / image.total();
  
          // 判断是否需要跳过
        if (ratio > threshold_ratio) {
          ROS_INFO("Detected abnormal pixel ratio");
          //std::cout << "检测到异常像素比例: " << ratio * 100 << "%，跳过本帧" << std::endl;
          result["code"] = 1002;
          result["msg"] = "检测到异常像素比例";
          result["data"] = "";
          res.set_content(result.dump(), "application/json");
          return;
        }
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        std::string image_idx = " ";

        if (map_user.Relocalization(image_path, image, pose)) {
          trajectory.emplace_back(std::make_pair(image_idx, pose));
          auto data = SaveData(trajectory);
          result["code"] = 200;
          result["msg"] = "图片解析成功";
          result["data"] = data;
        } else {
          result["code"] = 1002;
          result["msg"] = "图片解析失败";
          result["data"] = "";
        }
      }catch (const std::exception& e) {
      result["code"] = 1001;
      result["msg"] = std::string("图片保存失败: ") + e.what();
      ROS_INFO("Image save failed");
      //std::cout << "图片保存失败" << std::endl;
    } 
    res.set_content(result.dump(), "application/json");
    });

  std::cout << "Server started at http://192.168.0.230:8004\n";
  svr.listen("192.168.0.230", 8004);
  svr.stop();
  map_user.StopVisualization();
  ros::shutdown();

  return 0;
}