#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <ros/ros.h>
#include <ros/console.h>
#include <thread>
#include <atomic>
#include <csignal>

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

std::atomic<bool> server_running(true);
std::atomic<bool> ros_running(true);

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
bool CheckJpeg(string file) {
    if (file.empty()) {
        return false;
    }
    ifstream in(file.c_str(), ios::in | ios::binary);
    if (!in.is_open()) {
        cout << "Error opening file!" << endl;
        return false;
    }

    int start;
    in.read((char*)&start, 4);
    short int lstart = start << 16 >> 16;

    in.seekg(-4, ios::end);
    int end;
    in.read((char*)&end, 4);
    short int lend = end >> 16;

    in.close();
    if ((lstart != -9985) || (lend != -9729)) { //0xd8ff 0xd9ff
        return true;
    }
    return false;
}

void signalHandler(int signum) {
    ROS_INFO("Interrupt signal received. Shutting down...");
    server_running = false;
    ros_running = false;
}

void run_http_server(MapUser& map_user, double threshold_ratio) {
    Server svr;

    // 文件上传接口
    svr.Post("/upload", [&](const Request& req, Response& res) {
        create_upload_dir();
        json result;
        std::unordered_set<std::string> processed_images;
        std::vector<std::pair<std::string, Eigen::Matrix4d>> trajectory;
        try {
            // 处理multipart表单
            if (!req.has_file("file") || !req.has_file("name")) {
                result["code"] = 1003;
                result["msg"] = "Missing file or name field";
                result["data"] = "error";
                res.set_content(result.dump(), "application/json");
                ROS_WARN("Missing file or name field");
                return;
            }

            // 保存文件
            const auto& file = req.get_file_value("file");
            fs::path save_path = fs::path("uploads") / file.filename;
            std::ofstream ofs(save_path, std::ios::binary);
            ofs << file.content;
            ofs.close();

            std::string image_path = save_path.string();
            bool img_flag = CheckJpeg(image_path);
            if (img_flag) {
                result["code"] = 1002;
                result["msg"] = "图片存在问题";
                result["data"] = "error";
                ROS_WARN("There are issues with the image");
                res.set_content(result.dump(), "application/json");
                return;
            }
            
            cv::Mat image = cv::imread(image_path, 0);
            if (image.empty()) {
                result["code"] = 1002;
                result["msg"] = "无法读取图片";
                result["data"] = "";
                ROS_WARN("Failed to read image: %s", image_path.c_str());
                res.set_content(result.dump(), "application/json");
                return;
            }
            
            Mat mask;
            inRange(image, Scalar(128), Scalar(128), mask);
            int bad_pixels = countNonZero(mask);
            double ratio = static_cast<double>(bad_pixels) / image.total();

            if (ratio > threshold_ratio) {
                ROS_WARN("Detected abnormal pixel ratio: %.2f%%", ratio * 100);
                result["code"] = 1002;
                result["msg"] = "检测到异常像素比例";
                result["data"] = "error";
                res.set_content(result.dump(), "application/json");
                return;
            }
            
            Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
            std::string image_idx = " ";

            if (map_user.Relocalization(image, pose)) {
                trajectory.emplace_back(std::make_pair(image_idx, pose));
                auto data = SaveData(trajectory);
                result["code"] = 200;
                result["msg"] = "图片解析成功";
                result["data"] = data;
                //ROS_INFO("Relocalization succeeded");
            } else {
                result["code"] = 1002;
                result["msg"] = "图片解析失败";
                result["data"] = "no data";
                //ROS_WARN("Relocalization failed");
            }
        } catch (const std::exception& e) {
            result["code"] = 1001;
            result["msg"] = std::string("处理失败: ") + e.what();
            ROS_ERROR("Processing failed: %s", e.what());
        } 
        res.set_content(result.dump(), "application/json");
    });

    ROS_INFO("HTTP server started at http://192.168.0.230:8004");
    svr.listen("192.168.0.230", 8004);
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "air_slam");
    ros::NodeHandle nh;

    // 注册信号处理函数
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    std::string config_path, model_dir, map_root, voc_path, traj_path;
    ros::param::get("~config_path", config_path);
    ros::param::get("~model_dir", model_dir);
    ros::param::get("~map_root", map_root);
    ros::param::get("~voc_path", voc_path);
    ros::param::get("~traj_path", traj_path);

    RelocalizationConfigs configs(config_path, model_dir);
    ros::param::get("~dataroot", configs.dataroot);
    ros::param::get("~camera_config_path", configs.camera_config_path);
    
    std::ofstream outFile("/my_workspace/CamTrans.txt");
    outFile.close();

    MapUser map_user(configs, nh);
    map_user.LoadMap(map_root);
    map_user.LoadVocabulary(voc_path);
    double threshold_ratio = 0.3;

    // 启动HTTP服务器线程
    std::thread http_thread(run_http_server, std::ref(map_user), threshold_ratio);

    // ROS主循环
    ros::Rate rate(10); // 10Hz
    while (ros_running && ros::ok()) {
        ros::spinOnce();
        rate.sleep();
    }

    // 清理工作
    ROS_INFO("Shutting down...");
    server_running = false; // 通知HTTP服务器停止
    
    // 等待HTTP线程结束
    if (http_thread.joinable()) {
        http_thread.join();
    }

    map_user.StopVisualization();
    ros::shutdown();

    return 0;
}