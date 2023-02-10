#include "event_frame/event_frame.h"
#include <std_msgs/Float32.h>
#include <iostream>

namespace event_frame{

    accumulate_frame::accumulate_frame(ros::NodeHandle & nh, ros::NodeHandle nh_private) : nh_(nh){
        // got_camera_info_ = false;
        // used_last_image_ = false;

        // setup subscribers and publishers
        event_sub_ = nh_.subscribe("events", 1, &accumulate_frame::eventsCallback, this);
        // camera_info_sub_ = nh_.subscribe("camera_info", 1, &accumulate_frame::cameraInfoCallback, this);

        image_transport::ImageTransport it_(nh_);
        image_sub_ = it_.subscribe("image", 1, &accumulate_frame::imageCallback, this);
        image_pub_ = it_.advertise("raw_image", 1);
        event_pub_ = it_.advertise("event_frame", 1);

        // for (int i = 0; i < 2; ++i)
        //     for (int k = 0; k < 2; ++k)
        //     event_stats_[i].events_counter_[k] = 0;
        // event_stats_[0].dt = 1;
        // event_stats_[0].events_mean_lasttime_ = 0;
        // event_stats_[0].events_mean_[0] = nh_.advertise<std_msgs::Float32>("events_on_mean_1", 1);
        // event_stats_[0].events_mean_[1] = nh_.advertise<std_msgs::Float32>("events_off_mean_1", 1);
        // event_stats_[1].dt = 5;
        // event_stats_[1].events_mean_lasttime_ = 0;
        // event_stats_[1].events_mean_[0] = nh_.advertise<std_msgs::Float32>("events_on_mean_5", 1);
        // event_stats_[1].events_mean_[1] = nh_.advertise<std_msgs::Float32>("events_off_mean_5", 1);
    }

    accumulate_frame::~accumulate_frame()
    {
        image_pub_.shutdown();
        event_pub_.shutdown();

    }

    void accumulate_frame::eventsCallback(const dvs_msgs::EventArray::ConstPtr& msg){
        for (const auto& event : msg->events)
            events_.push_back(event);
    }

    void accumulate_frame::imageCallback(const sensor_msgs::Image::ConstPtr& msg){
    
        //store the image data
        if (events_.size()<5)
        {
            // std::cout <<"event less than 5\n";
            return;
        }
        ros::Time start_time = events_[0].ts;
        ros::Time end_time = events_[events_.size()-1].ts;
        ImgData img_data;
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
        if (msg->encoding == "rgb8") cv::cvtColor(cv_ptr->image, img_data.img, CV_RGB2BGR);
        if (msg->encoding == "mono8")
        {
            if (color_image_)
            {
            cv::cvtColor(cv_ptr->image, img_data.img, CV_BayerBG2BGR);
            }
            else
            {
            cv::cvtColor(cv_ptr->image, img_data.img, CV_GRAY2BGR);
            }
        }
        img_data.t = msg->header.stamp;

        cv::Size img_size = img_data.img.size();
        cv::Mat pos_img = cv::Mat::zeros(img_size, CV_16UC1);
        cv::Mat neg_img = cv::Mat::zeros(img_size, CV_16UC1);
        cv::Mat time_img = cv::Mat::zeros(img_size, CV_32FC1);
        // cv::Mat out_img = cv::Mat::zeros(img_size, CV_16UC3);
        for (const auto& event : events_)
        {
            

            get_frame_positive( event, pos_img);
            get_frame_negetive( event, neg_img);
            get_frame_time( event, time_img, start_time);
                   

        }
        
        
        
        cv::normalize(pos_img,pos_img,0,255,cv::NORM_MINMAX);
        cv::normalize(neg_img,neg_img,0,255,cv::NORM_MINMAX);
        pos_img.convertTo(pos_img,CV_8UC1);
        neg_img.convertTo(neg_img,CV_8UC1);

        // std::cout<<"start time"<< start_time.toSec()<<std::endl;
        // std::cout<<"end time"<< end_time.toSec()<<std::endl;
        // std::cout<<"time diff is:"<< (end_time.toSec() - start_time.toSec())<<std::endl;
        time_img = time_img/(end_time.toSec() - start_time.toSec())*255;
        time_img.convertTo(time_img,CV_8UC1);
        cv::normalize(time_img,time_img,0,255,cv::NORM_MINMAX);
        
        cv::Mat img_tmp[3] = {pos_img,neg_img,time_img}; 
        

        cv_bridge::CvImage cv_image;
        cv_image.encoding = "bgr8";
        cv::merge(img_tmp,3,cv_image.image);


        event_pub_.publish(cv_image.toImageMsg());
        
        events_.clear();
        // // cv_bridge::CvImage cv_image;
        img_data.img.copyTo(cv_image.image);
        cv_image.encoding = "bgr8";
        image_pub_.publish(cv_image.toImageMsg());

    }
    void accumulate_frame::get_frame_positive(const dvs_msgs::Event& event,  cv::Mat& pos_img){
        if (event.polarity == true ){
            // std::cout<<"the position is: "<<event.x<<" "<<event.y<<std::endl;
            ++pos_img.at<uint16_t>(cv::Point(event.x, event.y));
    
        }
    }
    
    void accumulate_frame::get_frame_negetive(const dvs_msgs::Event& event,  cv::Mat& neg_img){
        if (event.polarity == false ){

            int x = cvRound(event.x), y = cvRound(event.y);
            ++neg_img.at<uint16_t>(cv::Point(x, y));
    
        }
    }

    void accumulate_frame::get_frame_time(const dvs_msgs::Event& event,  cv::Mat& time_img,const ros::Time& start_time){

        int x = cvRound(event.x), y = cvRound(event.y);
        if ( time_img.at<float>(cv::Point(x, y)) > 0)
        {
            time_img.at<float>(cv::Point(x, y)) = (time_img.at<float>(cv::Point(x, y)) + (event.ts.toSec()-start_time.toSec()))/2;
        }
        else 
        {
            time_img.at<float>(cv::Point(x, y)) += (event.ts.toSec()-start_time.toSec());
        }
            
    }

}