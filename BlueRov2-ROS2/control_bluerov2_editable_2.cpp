#include <ros/ros.h>
#include <std_msgs/UInt16.h>
#include <std_msgs/Float32.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/BatteryState.h>
#include <geometry_msgs/Quaternion.h>
#include <tf/tf.h>      
#include <math.h> // for trigonometry and pow() 

using namespace std;

// Global variables
float posx, posy, posz, posyaw; // Position and orientation from odometry/IMU
float z_baro, zbf;              // Barometric pressure and filtered value
float bat;                      // Battery voltage

// Control variables
float xDes = 1.0, yDes = 1.0, zDes, yawDes; // Desired positions and yaw
float ex, ey, ez, eyaw;                     // Errors
float Kp_z, Ki_z, Kd_z;                     // Z-axis PID gains
float Kp_yaw, Ki_yaw, Kd_yaw;               // Yaw-axis PID gains
float ux, uy, uz, uyaw;                     // Control signals
int verticalc, lateralc, fforwardc, yawc;

// Joystick variables
int axis_vrtl, axis_yaw; 

// Flags
bool controles = false;

// Functions
void posCallback(const sensor_msgs::Imu::ConstPtr &msg) {
    // Extract orientation
    posyaw = tf::getYaw(msg->orientation); // Use tf::getYaw for efficiency
}

void odomCallback(const nav_msgs::Odometry::ConstPtr &msg) {
    posx = msg->pose.pose.position.x;
    posy = msg->pose.pose.position.y;
    posz = msg->pose.pose.position.z;
}

void presCallback(const sensor_msgs::FluidPressure::ConstPtr &msg) {
    z_baro = msg->fluid_pressure - 802.6;  
    zbf = 0.1061 * z_baro + 0.8939 * zbf; // Update filtered value
}

void joystickCallback(const sensor_msgs::Joy::ConstPtr &msg) { 
    axis_vrtl = 1500 - (msg->axes[1] * 200) * 0.5;   // Vertical axis
    axis_yaw = 1500 + (msg->axes[3] * 50);         // Yaw axis

    controles = msg->buttons[1]; // Control flag activated by button 1

    // Other button mappings for arming, disarming, etc. can be added here
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "controller");
    ros::NodeHandle nh;

    // Subscribers
    ros::Subscriber subPos = nh.subscribe("/BlueRov2/imu/data", 1, posCallback);
    ros::Subscriber odomPos = nh.subscribe("/BlueRov2/odometry", 1, odomCallback);
    ros::Subscriber subPres = nh.subscribe("topic_name", 1, presCallback); // Replace 'topic_name'
    ros::Subscriber subJoy = nh.subscribe("joy", 1, joystickCallback);

    // Publishers
    ros::Publisher pubVertical = nh.advertise<std_msgs::UInt16>("BlueRov2/rc_channel3/set_pwm", 1);
    ros::Publisher pubYaw = nh.advertise<std_msgs::UInt16>("BlueRov2/rc_channel4/set_pwm", 1); 
    // Publishers for lateral, forward, roll, and pitch channels (if needed)

    ros::Rate loop_rate(20);

    // Control parameters - adjust as needed
    Kp_z = 2.5;
    Ki_z = 0.0;
    Kd_z = 1.0;

    Kp_yaw = 0.5;
    Ki_yaw = 0.01;
    Kd_yaw = 0.25;

    while (ros::ok()) {
        if (controles) {
            // Calculate errors
            ex = xDes - posx;
            ey = yDes - posy;
            ez = zDes - zbf;
            eyaw = yawDes - posyaw;

            // ... (rest of the control calculations)
        } else {
            // Manual mode using joystick values
            verticalc = axis_vrtl;
            yawc = axis_yaw;
        }

        // Publish control signals
        std_msgs::UInt16 vertical_msg, yaw_msg; 
        vertical_msg.data = verticalc;
        yaw_msg.data = yawc;
        pubVertical.publish(vertical_msg);
        pubYaw.publish(yaw_msg);

        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
