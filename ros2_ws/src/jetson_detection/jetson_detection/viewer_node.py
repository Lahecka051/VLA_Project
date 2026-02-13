import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import json
import threading


class ViewerNode(Node):
    def __init__(self):
        super().__init__('viewer_node')
        self.bridge = CvBridge()
        self.latest_fusion = {}
        self.display_frame = None
        self.lock = threading.Lock()
        self.running = True

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(Image, '/image_raw', self.image_cb, qos)
        self.create_subscription(String, '/fusion/objects', self.fusion_cb, 1)

        self.display_thread = threading.Thread(target=self.display_loop, daemon=True)
        self.display_thread.start()

    def fusion_cb(self, msg):
        try:
            self.latest_fusion = json.loads(msg.data)
        except:
            self.latest_fusion = {}

    def image_cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        h, w = frame.shape[:2]
        scale = 640.0 / w
        small = cv2.resize(frame, (640, int(h * scale)))

        fusion = self.latest_fusion
        objects = fusion.get('objects', [])
        heading = fusion.get('heading')
        compass = fusion.get('compass')
        bearing = fusion.get('true_bearing')
        roll = fusion.get('roll', 0)
        pitch = fusion.get('pitch', 0)
        yaw = fusion.get('yaw', 0)

        for obj in objects:
            x, y, bw, bh = obj['bbox']
            x = int(x * scale)
            y = int(y * scale)
            bw = int(bw * scale)
            bh = int(bh * scale)
            dist = obj['distance']
            cls = obj['class']

            color = (0, 255, 0) if dist > 0 else (0, 0, 255)
            cv2.rectangle(small, (x, y), (x + bw, y + bh), color, 2)

            if dist > 0:
                label = f'{cls} {dist:.1f}m'
            else:
                label = f'{cls} --'
            cv2.putText(small, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 좌상단에 heading 표시
        if compass is not None and heading is not None:
            hud = f'{compass} {heading:.0f}'
            imu_hud = f'R:{roll:.1f} P:{pitch:.1f} Y:{yaw:.1f}'
            cv2.putText(small, hud, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(small, imu_hud, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        with self.lock:
            self.display_frame = small

    def display_loop(self):
        while self.running:
            with self.lock:
                frame = self.display_frame
            if frame is not None:
                cv2.imshow('Fusion', frame)
            if cv2.waitKey(16) & 0xFF == ord('q'):
                self.running = False
                break

    def destroy_node(self):
        self.running = False
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ViewerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
