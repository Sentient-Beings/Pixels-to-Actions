import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import redis
import cv2
import numpy as np
import threading

class VisionToAction(Node):
    def __init__(self):
        super().__init__('vision_to_action')
        
        self.bridge = CvBridge()
        self.image_publisher = self.create_publisher(Image, '/camera/image_raw', 10)
        
        self.redis_client = redis.Redis(host='localhost', port=6379)
        self.pubsub = self.redis_client.pubsub()
        self.pubsub.subscribe('image_channel')
        
        self.running = True
        self.subscriber_thread = threading.Thread(target=self.redis_subscriber_loop)
        self.subscriber_thread.daemon = True
        self.subscriber_thread.start()
        
        self.get_logger().info('Vision to Action node started')
    
    def redis_subscriber_loop(self):
        while self.running:
            try:
                message = self.pubsub.get_message()
                if message and message['type'] == 'message':
                    data = message['data']
                    
                    # Split metadata and image data
                    metadata_end = data.find(b'|')
                    if metadata_end == -1:
                        continue
                        
                    metadata = data[:metadata_end].decode()
                    img_data = data[metadata_end + 1:]
                    
                    # Parse metadata
                    width, height = map(int, metadata.split(','))
                    
                    # Decode image
                    img_array = np.frombuffer(img_data, dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        # Convert OpenCV image to ROS message
                        ros_image = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                        
                        self.image_publisher.publish(ros_image)
                        
            except Exception as e:
                self.get_logger().error(f'Error in subscriber loop: {e}')
                continue
    
    def destroy_node(self):
        self.running = False
        if self.subscriber_thread.is_alive():
            self.subscriber_thread.join(timeout=1.0)
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = VisionToAction()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()