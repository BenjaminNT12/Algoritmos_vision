import tf
from math import pi
import rospy
from std_msgs.msg import Float32, Bool, UInt16, String
from sensor_msgs.msg import Joy
import math
import time

# Variables globales para posCallback()
posax = 0
posay = 0
posaz = 0
posaw = 0
posroll = 0
pospitch = 0
posyaw = 0

# Variables globales para odomCallback()
posx = 0
posy = 0
posz = 0

# Variables globales para presCallback()
fluid_press = 0
diff_press = 0
z_baro = 0
zbf = 0
zbf_a = 0

# Variables globales para dvextCallback()
pos_x = 0
pos_y = 0

# Variables globales para chatterCallback()
dato_anterior = None
dvl1 = ''
tiempo = 0
myfile2 = None
d2 = 0
d1 = 0


def posCallback(msg):
    global posax, posay, posaz, posaw
    global posroll, pospitch, posyaw

    posax = msg.orientation.x
    posay = msg.orientation.y
    posaz = msg.orientation.z
    posaw = msg.orientation.w

    posroll_p = msg.angular_velocity.x
    pospitch_p = msg.angular_velocity.y
    posyaw_p = msg.angular_velocity.z

    veax = msg.angular_velocity.x

    q = tf.Quaternion(posax, posay, posaz, posaw)
    m = tf.Matrix3x3(q)
    posrollrad, pospitchrad, posyawrad = m.getRPY()

    posroll = posrollrad * (180 / pi)
    pospitch = -1 * (pospitchrad * (180 / pi))

    posyaw = posyawrad * (180 / pi)
    if posyaw > 0:
        posyaw = posyaw - 360
        
def odomCallback(msg):
    global posx, posy, posz

    posx = msg.pose.pose.position.x
    posy = msg.pose.pose.position.y
    posz = msg.pose.pose.position.z
    
def presCallback(msg):
    global fluid_press, diff_press, z_baro, zbf, zbf_a

    fluid_press = msg.fluid_pressure
    diff_press = msg.variance

    z_baro = fluid_press - 802.6

    a_z = 0.1061

    zbf = a_z * z_baro + (1 - a_z) * zbf_a
    zbf_a = zbf
    
def dvextCallback(msg):
    global pos_x, pos_y

    data = msg.data.split()

    if len(data) >= 2:
        pos_x = float(data[0])
        pos_y = float(data[1])
        
def chatterCallback(msg):
    global dato_anterior, dvl1, tiempo, myfile2, d1, d2

    dato = msg.data

    if dato != dato_anterior:
        if d2 == 1:
            dvl1 = dvl1 + dato
            d1 = 1
            d2 = 0
        else:
            pos1 = dato.find("$DVPDL")
            if pos1 > 0:
                gr = dato
                dvl1 = dato
                pos2 = dvl1.find("")
                pos3 = dvl1.find("", pos2 + 1)
                if pos3 > 0:
                    d1 = 1
                else:
                    d2 = 1

        if d1 == 1:
            resultant = str(tiempo)
            tt = resultant + "," + dvl1

            myfile2.write(tt + "\t" + dvl1 + "\n")

            pos4 = dvl1.find("DVPDL") + 6
            tr = [0.0] * 8
            for i in range(8):
                posf = dvl1.find(',', pos4)
                try:
                    tr[i] = float(dvl1[pos4:posf])
                except ValueError as e:
                    rospy.logerr("Error: %s", e)
                pos4 = posf + 1
            d1 = 0

            dato_anterior = dato

def saturate(a):
    if a > 1650:
        a = 1650
    elif a < 1350:
        a = 1350
    return a

def saturate_yaw(c):
    if c > 1900:
        c = 1900
    elif c < 1100:
        c = 1100
    return c

def saturate_pos_x(c):
    if c > 1900:
        c = 1900
    elif c < 1100:
        c = 1100
    return c

def saturate_pos_y(c):
    if c > 1900:
        c = 1900
    elif c < 1100:
        c = 1100
    return c

def saturate_ob(c):
    if c > 100:
        c = 100
    elif c < -100:
        c = -100
    return c

def sat_k(h, b, k, m):
    out = 0
    d = b / k
    if abs(h) > d:
        out = b * (abs(h) ** (m - 1))
    elif abs(h) <= d:
        out = b * (d ** (m - 1))
    return int(out)

def sgn(in_val):
    if in_val > 0:
        return 1
    elif in_val < 0:
        return -1
    return 0

def sat(S, varsigma):
    if S < -varsigma:
        return -1
    elif -varsigma <= S <= varsigma:
        return int(S / varsigma)
    elif S > varsigma:
        return 1
    return 0

def main():
    rospy.init_node('dvl_subscriber')
    rospy.Subscriber('dvl_out', Float32, dvl_callback)

    rospy.init_node('publisher')
    number1_pub = rospy.Publisher('Xdeseada', Float32, queue_size=10)
    number2_pub = rospy.Publisher('Ydeseada', Float32, queue_size=10)
    number3_pub = rospy.Publisher('Xactual', Float32, queue_size=10)
    number4_pub = rospy.Publisher('Yactual', Float32, queue_size=10)

    rospy.init_node('octosub_node')
    sub_pres = rospy.Subscriber('/BlueRov2/pressure', Float32, posCallback)
    sub_bat = rospy.Subscriber('/BlueRov2/battery', Float32, odomCallback)
    pub_arm = rospy.Publisher('/BlueRov2/arm', Bool, queue_size=1)
    pub_mod = rospy.Publisher('/BlueRov2/mode/set', String, queue_size=1)
    pub_vertical = rospy.Publisher('/BlueRov2/rc_channel3/set_pwm', UInt16, queue_size=1)
    pub_lateral = rospy.Publisher('/BlueRov2/rc_channel6/set_pwm', UInt16, queue_size=1)
    pub_forward = rospy.Publisher('/BlueRov2/rc_channel5/set_pwm', UInt16, queue_size=1)
    pub_yaw = rospy.Publisher('/BlueRov2/rc_channel4/set_pwm', UInt16, queue_size=1)
    pub_roll = rospy.Publisher('/BlueRov2/rc_channel2/set_pwm', UInt16, queue_size=1)
    pub_pitch = rospy.Publisher('/BlueRov2/rc_channel1/set_pwm', UInt16, queue_size=1)
    
    rospy.init_node('joystick_node')
    joystick_sub = rospy.Subscriber('joy', Joy, joystick_callback)

    archivo = 0
    last_file = open('/home/cesar/BlueROV2_ws/src/reading_dvl/src/Datos-DVL/lastlog.txt', 'r')
    archivo = int(last_file.read())
    last_file.close()

    str1 = "/home/cesar/BlueROV2_ws/src/reading_dvl/src/Datos-DVL/data_dvl{}.txt".format(archivo)
    last_file1 = open('/home/cesar/BlueROV2_ws/src/reading_dvl/src/Datos-DVL/lastlog.txt', 'w')
    archivo += 1
    last_file1.write(str(archivo))
    last_file1.close()

    myfile = open(str1, 'w')

    archivo2 = 0
    last_file2 = open('/home/cesar/BlueROV2_ws/src/reading_dvl/src/Datos-IMU/lastlog.txt', 'r')
    archivo2 = int(last_file2.read())
    last_file2.close()

    str3 = "/home/cesar/BlueROV2_ws/src/reading_dvl/src/Datos-IMU/data_imu{}.txt".format(archivo2)
    last_file3 = open('/home/cesar/BlueROV2_ws/src/reading_dvl/src/Datos-IMU/lastlog.txt', 'w')
    archivo2 += 1
    last_file3.write(str(archivo2))
    last_file3.close()

    myfile2 = open(str3, 'w')

    while not rospy.is_shutdown():
        rospy.spin()

    myfile.close()
    myfile2.close()
    
if __name__ == '__main__':
    main()
