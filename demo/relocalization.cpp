#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <ros/ros.h>
#include <ros/console.h>
#include <thread>
#include <atomic>
#include <csignal>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <memory>

#include <filesystem>
#include <nlohmann/json.hpp>
#include "read_configs.h"
#include "dataset.h"
#include "map_user.h"
#include <httplib.h>
#include "base64.h"
namespace fs = std::filesystem;

using namespace httplib;
using json = nlohmann::json;
using namespace cv;
using namespace std;

std::atomic<bool> server_running(true);
std::atomic<bool> ros_running(true);

// 请求队列和同步机制
std::mutex queue_mutex;
std::condition_variable queue_cv;

// 使用智能指针管理请求，避免内存管理问题
struct RequestData {
    std::string base64_str;
    std::shared_ptr<Response> response;
    
    RequestData(const std::string& base64, std::shared_ptr<Response> res) 
        : base64_str(base64), response(res) {}
};

std::queue<std::shared_ptr<RequestData>> request_queue;
std::atomic<bool> processing_request(false);

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
    // 通知所有等待的线程
    queue_cv.notify_all();
}

// 安全的Base64解码函数
std::string safe_base64_decode(const std::string& base64_str) {
    try {
        return base64_decode(base64_str);
    } catch (const std::exception& e) {
        ROS_ERROR("Base64 decoding failed: %s", e.what());
        throw;
    }
}

// 安全的图像解码函数
cv::Mat safe_imdecode(const std::vector<uchar>& buffer) {
    try {
        cv::Mat image = cv::imdecode(buffer, cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            throw std::runtime_error("Decoded image is empty");
        }
        return image;
    } catch (const cv::Exception& e) {
        ROS_ERROR("OpenCV imdecode failed: %s", e.what());
        throw;
    }
}

// 图像处理工作线程
void image_processing_worker(MapUser& map_user, double threshold_ratio) {
    while (server_running) {
        std::shared_ptr<RequestData> request;
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            
            // 等待队列中有请求或停止信号
            queue_cv.wait(lock, [] {
                return !request_queue.empty() || !server_running;
            });
            
            if (!server_running && request_queue.empty()) {
                break;
            }
            
            // 取出队列中的请求
            request = request_queue.front();
            request_queue.pop();
        }
        
        processing_request = true;
        
        // 处理图像
        json result;
        std::vector<std::pair<std::string, Eigen::Matrix4d>> trajectory;
        
        try {
            ROS_INFO("Processing image...");
            
            // Base64解码
            std::string image_data = safe_base64_decode(request->base64_str);
            
            // 检查解码后的数据大小
            if (image_data.empty()) {
                throw std::runtime_error("Decoded image data is empty");
            }
            
            // 创建缓冲区并解码图像
            std::vector<uchar> buffer(image_data.begin(), image_data.end());
            cv::Mat image = safe_imdecode(buffer);
            
            // 检查图像质量
            cv::Mat mask;
            cv::inRange(image, cv::Scalar(128), cv::Scalar(128), mask);
            int bad_pixels = cv::countNonZero(mask);
            double ratio = static_cast<double>(bad_pixels) / image.total();

            if (ratio > threshold_ratio) {
                ROS_WARN("Detected abnormal pixel ratio: %.2f%%", ratio * 100);
                result["code"] = 1002;
                result["msg"] = "检测到异常像素比例";
                result["data"] = json::array();
            } else {
                // 重定位处理
                Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
                std::string image_idx = " ";

                if (map_user.Relocalization(image, pose)) {
                    trajectory.emplace_back(std::make_pair(image_idx, pose));
                    auto data = SaveData(trajectory);
                    result["code"] = 200;
                    result["msg"] = "图片解析成功";
                    result["data"] = data;
                    ROS_INFO("Image processed successfully");
                } else {
                    result["code"] = 1002;
                    result["msg"] = "图片解析失败";
                    result["data"] = json::array();
                    ROS_WARN("Image processing failed");
                }
            }
        } catch (const std::exception& e) {
            result["code"] = 1001;
            result["msg"] = std::string("处理失败: ") + e.what();
            result["data"] = json::array();
            ROS_ERROR("Processing failed: %s", e.what());
        }
        
        // 设置响应内容
        if (request->response) {
            request->response->set_content(result.dump(), "application/json");
        }
        
        processing_request = false;
        
        // 通知可能等待的线程
        queue_cv.notify_one();
        
        // 在处理完每个请求后添加小的延迟，减少内存压力
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void run_http_server(MapUser& map_user, double threshold_ratio) {
    // 设置httplib服务器参数
    Server svr;
    
    // 限制并发连接数，减轻内存压力
    svr.set_keep_alive_max_count(10);
    
    // 设置请求体大小限制
    svr.set_payload_max_length(50 * 1024 * 1024); // 50MB

    // 启动图像处理工作线程
    std::thread worker_thread(image_processing_worker, std::ref(map_user), threshold_ratio);

    // 文件上传接口改为Base64图像接口
    svr.Post("/upload", [&](const Request& req, Response& res) {
        // 使用智能指针管理Response
        auto response_ptr = std::make_shared<Response>();
        
        try {
            // 解析JSON请求体
            auto json_body = json::parse(req.body);
            
            // 检查必需字段
            if (!json_body.contains("image_base64")) {
                json error_result;
                error_result["code"] = 1003;
                error_result["msg"] = "Missing image_base64 field";
                error_result["data"] = json::array();
                res.set_content(error_result.dump(), "application/json");
                ROS_WARN("Missing image_base64 field");
                return;
            }
            
            // 获取Base64字符串
            std::string base64_str = json_body["image_base64"];
            std::string img_timestamp = json_body["timestamp"];
            ROS_INFO("Received image with timestamp: %s", img_timestamp.c_str());
            // 验证Base64字符串长度
            if (base64_str.empty() || base64_str.length() < 100) {
                json error_result;
                error_result["code"] = 1003;
                error_result["msg"] = "Invalid image data";
                error_result["data"] = json::array();
                res.set_content(error_result.dump(), "application/json");
                ROS_WARN("Invalid image data received");
                return;
            }
            
            // 将请求加入队列
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                auto request_data = std::make_shared<RequestData>(base64_str, response_ptr);
                request_queue.push(request_data);
            }
            
            // 通知工作线程有新请求
            queue_cv.notify_one();
            
            // 等待处理完成（设置超时）
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (queue_cv.wait_for(lock, std::chrono::seconds(1), [&]() {
                return !processing_request && request_queue.empty();
            })) {
                // 处理完成，设置响应
                res.set_content(response_ptr->body, "application/json");
            } else {
                // 超时
                json timeout_result;
                timeout_result["code"] = 1004;
                timeout_result["msg"] = "Processing timeout";
                timeout_result["data"] = json::array();
                res.set_content(timeout_result.dump(), "application/json");
                ROS_WARN("Request processing timeout");
            }
            
        } catch (const std::exception& e) {
            json error_result;
            error_result["code"] = 1001;
            error_result["msg"] = std::string("处理失败: ") + e.what();
            error_result["data"] = json::array();
            res.set_content(error_result.dump(), "application/json");
            ROS_ERROR("Request processing failed: %s", e.what());
        }
    });

    // 添加状态查询接口
    svr.Get("/status", [](const Request& req, Response& res) {
        json result;
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            result["queue_size"] = request_queue.size();
        }
        result["processing"] = processing_request.load();
        result["server_running"] = server_running.load();
        res.set_content(result.dump(), "application/json");
    });

    // 添加健康检查接口
    svr.Get("/health", [](const Request& req, Response& res) {
        res.set_content("OK", "text/plain");
    });

    ROS_INFO("HTTP server started at http://192.168.0.230:8004");
    
    try {
        svr.listen("192.168.0.230", 8004);
    } catch (const std::exception& e) {
        ROS_ERROR("HTTP server failed: %s", e.what());
    }
    
    // 等待工作线程结束
    if (worker_thread.joinable()) {
        worker_thread.join();
    }
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
    double threshold_ratio = 0.2;

    // 启动HTTP服务器线程
    std::thread http_thread(run_http_server, std::ref(map_user), threshold_ratio);

    // ROS主循环
    ros::Rate rate(100); // 10Hz
    while (ros_running && ros::ok()) {
        ros::spinOnce();
        rate.sleep();
    }

    // 清理工作
    ROS_INFO("Shutting down...");
    server_running = false; // 通知HTTP服务器停止
    
    // 通知所有等待的线程
    queue_cv.notify_all();
    
    // 等待HTTP线程结束
    if (http_thread.joinable()) {
        http_thread.join();
    }

    map_user.StopVisualization();
    ros::shutdown();

    return 0;
}