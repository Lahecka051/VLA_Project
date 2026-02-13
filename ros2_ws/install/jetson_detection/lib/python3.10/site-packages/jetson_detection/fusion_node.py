import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan, Imu
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String
import math
import json


class FusionNode(Node):
    def __init__(self):
        super().__init__('fusion_node')

        self.img_w = 640
        self.img_h = 480
        self.fov_h = math.radians(62.2)
        self.mag_declination = 0.0  # 부산 자기편차 (도)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(Detection2DArray, '/detections', self.det_cb, qos)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, qos)
        self.create_subscription(Imu, '/imu/data', self.imu_cb, 10)

        self.result_pub = self.create_publisher(String, '/fusion/objects', 10)

        self.latest_scan = None
        self.latest_imu = None
        self.latest_det = None

        self.timer = self.create_timer(0.05, self.fuse)

        self.coco_names = {
            0:'person',1:'bicycle',2:'car',3:'motorcycle',4:'airplane',5:'bus',
            6:'train',7:'truck',8:'boat',9:'traffic light',10:'fire hydrant',
            11:'stop sign',12:'parking meter',13:'bench',14:'bird',15:'cat',
            16:'dog',17:'horse',18:'sheep',19:'cow',20:'elephant',21:'bear',
            22:'zebra',23:'giraffe',24:'backpack',25:'umbrella',26:'handbag',
            27:'tie',28:'suitcase',29:'frisbee',30:'skis',31:'snowboard',
            32:'sports ball',33:'kite',34:'baseball bat',35:'baseball glove',
            36:'skateboard',37:'surfboard',38:'tennis racket',39:'bottle',
            40:'wine glass',41:'cup',42:'fork',43:'knife',44:'spoon',45:'bowl',
            46:'banana',47:'apple',48:'sandwich',49:'orange',50:'broccoli',
            51:'carrot',52:'hot dog',53:'pizza',54:'donut',55:'cake',56:'chair',
            57:'couch',58:'potted plant',59:'bed',60:'dining table',61:'toilet',
            62:'tv',63:'laptop',64:'mouse',65:'remote',66:'keyboard',67:'cell phone',
            68:'microwave',69:'oven',70:'toaster',71:'sink',72:'refrigerator',
            73:'book',74:'clock',75:'vase',76:'scissors',77:'teddy bear',
            78:'hair drier',79:'toothbrush'
        }
        self.get_logger().info('Fusion node started')

    def det_cb(self, msg):
        self.latest_det = msg

    def scan_cb(self, msg):
        self.latest_scan = msg

    def imu_cb(self, msg):
        self.latest_imu = msg


    def quat_to_euler(self, q):
        sinr = 2.0 * (q.w * q.x + q.y * q.z)
        cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        roll = math.degrees(math.atan2(sinr, cosr))
        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        sinp = max(-1.0, min(1.0, sinp))
        pitch = math.degrees(math.asin(sinp))
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.degrees(math.atan2(siny, cosy))
        return roll, pitch, yaw

    def quat_to_yaw(self, q):
        """쿼터니언에서 yaw(도) 추출"""
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.degrees(math.atan2(siny, cosy))

    def get_heading(self):
        """동쪽=0°, 반시계=양수 기준 heading (0~360)"""
        if self.latest_imu is None:
            return None, None
        yaw = self.quat_to_yaw(self.latest_imu.orientation)
        # yaw: 0=magnetic north, 양수=동쪽 방향 (EBIMU 기준)
        # 자기편차 보정 → true north 기준
        true_yaw = yaw + self.mag_declination
        # 북쪽=0° (나침반과 동일)
        heading = true_yaw % 360.0
        return heading, true_yaw

    def heading_to_compass(self, true_yaw):
        """true north 기준 yaw → 16방위"""
        bearing = true_yaw % 360.0
        dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        idx = int((bearing + 11.25) / 22.5) % 16
        return dirs[idx]

    def fuse(self):
        if self.latest_det is None or self.latest_scan is None:
            return

        scan = self.latest_scan
        det = self.latest_det

        fx = self.img_w / (2.0 * math.tan(self.fov_h / 2.0))
        cx = self.img_w / 2.0

        lidar_points = []
        angle = scan.angle_min
        for i, r in enumerate(scan.ranges):
            if r < scan.range_min or r > scan.range_max or math.isinf(r) or math.isnan(r):
                angle += scan.angle_increment
                continue
            lx = r * math.cos(angle)
            ly = r * math.sin(angle)
            if lx <= 0.1:
                angle += scan.angle_increment
                continue
            px = cx - (ly / lx) * fx
            if 0 <= px < self.img_w:
                lidar_points.append((px, r))
            angle += scan.angle_increment

        heading, true_yaw = self.get_heading()

        results = []
        for d in det.detections:
            bbox = d.bbox
            x_center = bbox.center.position.x
            y_center = bbox.center.position.y
            w = bbox.size_x
            h = bbox.size_y
            x_min = x_center - w / 2.0
            x_max = x_center + w / 2.0

            dists = [dist for px, dist in lidar_points if x_min <= px <= x_max]

            obj_id = ''
            obj_score = 0.0
            if d.results:
                obj_id = self.coco_names.get(int(d.results[0].hypothesis.class_id), d.results[0].hypothesis.class_id)
                obj_score = d.results[0].hypothesis.score

            obj = {
                'class': obj_id,
                'score': round(obj_score, 2),
                'bbox': [int(x_min), int(y_center - h / 2), int(w), int(h)],
                'distance': round(min(dists), 2) if dists else -1.0,
                'lidar_hits': len(dists)
            }
            results.append(obj)

        roll, pitch, yaw = (0, 0, 0)
        if self.latest_imu is not None:
            roll, pitch, yaw = self.quat_to_euler(self.latest_imu.orientation)

        output = {
            'objects': results,
            'heading': round(heading, 1) if heading is not None else None,
            'compass': self.heading_to_compass(true_yaw) if true_yaw is not None else None,
            'true_bearing': round(true_yaw % 360, 1) if true_yaw is not None else None,
            'roll': round(roll, 1),
            'pitch': round(pitch, 1),
            'yaw': round(yaw, 1)
        }

        msg = String()
        msg.data = json.dumps(output, ensure_ascii=False)
        self.result_pub.publish(msg)

    def destroy_node(self):
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
