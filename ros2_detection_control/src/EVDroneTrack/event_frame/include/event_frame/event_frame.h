#include <ros/ros.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>


#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>

namespace event_frame{

    class accumulate_frame{
        public:
            accumulate_frame(ros::NodeHandle & nh, ros::NodeHandle nh_private);
            virtual ~accumulate_frame();
        private:
            ros::NodeHandle nh_;
            ros::Subscriber event_sub_;
            // ros::Subscriber camera_info_sub_;

            image_transport::Publisher image_pub_;
            image_transport::Publisher event_pub_;

            image_transport::Subscriber image_sub_;
            // void cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& msg);
            void eventsCallback(const dvs_msgs::EventArray::ConstPtr& msg);
            void imageCallback(const sensor_msgs::Image::ConstPtr& msg);
            void get_frame_positive(const dvs_msgs::Event& event,cv::Mat& pos_img);
            void get_frame_negetive(const dvs_msgs::Event& event, cv::Mat& neg_img);
            void get_frame_time(const dvs_msgs::Event& event,cv::Mat& time_img,const ros::Time& start_time);

            struct ImgData 
            {
                cv::Mat img;
                ros::Time t;
                std::vector<cv::Point2f> points;
                // cv::Mat homography; //homography from previous image to this
                // cv::Mat homography_accumulated; //homography from previous image to the second last
                // cv::Point2f translation; //mean translation, previous to this
            };
            std::vector<dvs_msgs::Event> events_;
            // std::vector<ImgData> images_;
            bool color_image_ = false;
            
    };

}
