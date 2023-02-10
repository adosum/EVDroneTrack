#include "event_frame/event_frame.h"

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "event_frame");

  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");

  event_frame::accumulate_frame frame(nh, nh_private);

  ros::spin();

  return 0;
}