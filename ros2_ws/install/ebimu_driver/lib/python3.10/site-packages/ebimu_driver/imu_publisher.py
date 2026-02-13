import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import serial
import math
from transforms3d.euler import euler2quat

class EbimuPublisher(Node):
    def __init__(self):
        super().__init__('ebimu_publisher')
        self.publisher_ = self.create_publisher(Imu, 'imu/data', 10)
    
        # 포트 설정 (/dev/ttyUSB1 확인)
        self.ser = serial.Serial('/dev/ttyUSB1', 115200, timeout=0.1)
    
        # [추가됨] 센서를 ASCII 모드로 강제 전환하는 명령어 전송
        import time
        time.sleep(0.5) # 포트 열고 잠시 대기
        self.ser.write(b'<sof2>') # Output Format: ASCII
        self.ser.flush()
        self.get_logger().info("Sent <sof2> command to sensor.")
    
        self.timer = self.create_timer(0.01, self.timer_callback) # 100Hz
  
    def timer_callback(self):
        if self.ser.in_waiting:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                # EBIMU 데이터 포맷 파싱 (예: *,-10.5,20.1,5.5...)
                if line.startswith('*'):
                    data = line.split(',')
                    if len(data) >= 4:
                        roll = math.radians(float(data[1]))
                        pitch = math.radians(float(data[2]))
                        yaw = math.radians(float(data[3]))

                        msg = Imu()
                        msg.header.stamp = self.get_clock().now().to_msg()
                        msg.header.frame_id = "imu_link"

                        # Euler -> Quaternion 변환
                        q = euler2quat(roll, pitch, yaw, 'sxyz')
                        msg.orientation.x = q[1]
                        msg.orientation.y = q[2]
                        msg.orientation.z = q[3]
                        msg.orientation.w = q[0]

                        # 필요 시 가속도/자이로 데이터 추가 파싱

                        self.publisher_.publish(msg)
            except Exception as e:
                self.get_logger().warn(f'Error reading IMU: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = EbimuPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
