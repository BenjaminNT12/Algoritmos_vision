# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image

# class BlueROV2VideoNode(Node):

#     def __init__(self):
#         super().__init__("bluerov2_video_node")

#         # Create a subscriber to the "/bluerov2/camera/image_raw" topic.
#         self.image_subscriber = self.create_subscription(Image, "/bluerov2/camera/image_raw", self.image_callback, 10)

#     def image_callback(self, image_msg):
#         # Get the image data from the message.
#         image_data = image_msg.data

#         # Save the image data to a file.
#         with open("bluerov2_video.jpg", "wb") as f:
#             f.write(image_data)

# if __name__ == "__main__":
#     rclpy.init()

#     # Create a BlueROV2VideoNode object.
#     node = BlueROV2VideoNode()

#     # Spin the node.
#     rclpy.spin(node)

#     rclpy.shutdown()
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv2 import imshow, waitKey, destroyAllWindows

class BlueROV2VideoNode(Node):

    def __init__(self):
        super().__init__("bluerov2_video_node")

        # Create a subscriber to the "/bluerov2/camera/image_raw" topic.
        self.image_subscriber = self.create_subscription(Image, "/bluerov2/camera/image_raw", self.image_callback, 10)

    def image_callback(self, image_msg):
        # Get the image data from the message.
        image_data = image_msg.data

        # Convert the image data to a NumPy array.
        image_array = np.frombuffer(image_data, dtype=np.uint8)

        # Decode the image from the NumPy array.
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Show the image on the screen.
        imshow("BlueROV2 Video", image)

        # Wait for the user to press a key.
        key = waitKey(1)

        # If the user presses the ESC key, close the window.
        if key == 27:
            destroyAllWindows()

if __name__ == "__main__":
    rclpy.init()

    # Create a BlueROV2VideoNode object.
    node = BlueROV2VideoNode()

    # Spin the node.
    rclpy.spin(node)

    rclpy.shutdown()
    