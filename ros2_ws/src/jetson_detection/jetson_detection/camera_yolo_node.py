import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO


class CameraYoloNode(Node):
    def __init__(self):
        super().__init__('camera_yolo_node')

        # GStreamer pipeline for IMX477 CSI
        gst = (
            'nvarguscamerasrc sensor-id=0 ! '
    	    'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1 ! '
    	    'nvvidconv ! video/x-raw,width=640,height=480,format=BGRx ! '
    	    'videoconvert ! video/x-raw,format=BGR ! appsink drop=1'
        )
        self.cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            self.get_logger().error('CSI 카메라 열기 실패')
            return

        self.model = YOLO('yolo11s.engine')
        self.bridge = CvBridge()

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.image_pub = self.create_publisher(Image, '/image_raw', qos)
        self.det_pub = self.create_publisher(Detection2DArray, '/detections', qos)

        # 30fps 타이머
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)
        self.get_logger().info('Camera + YOLOv11s 노드 시작')

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # 이미지 publish
        img_msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = 'camera_frame'
        self.image_pub.publish(img_msg)

        # YOLOv11s 추론
        results = self.model(frame, verbose=False)[0]

        det_array = Detection2DArray()
        det_array.header = img_msg.header

        for box in results.boxes:
            det = Detection2D()
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            det.bbox.center.position.x = float((x1 + x2) / 2)
            det.bbox.center.position.y = float((y1 + y2) / 2)
            det.bbox.size_x = float(x2 - x1)
            det.bbox.size_y = float(y2 - y1)

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(int(box.cls[0]))
            hyp.hypothesis.score = float(box.conf[0])
            det.results.append(hyp)

            det_array.detections.append(det)

        self.det_pub.publish(det_array)

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraYoloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
