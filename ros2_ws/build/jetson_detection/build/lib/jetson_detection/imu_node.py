import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import serial
import math

class EbimuNode(Node):
    def __init__(self):
        super().__init__('ebimu_node')
        self.pub = self.create_publisher(Imu, '/imu/data', 10)
        self.ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.01)
        self.buf = b''
        self.timer = self.create_timer(0.01, self.read_serial)
        self.get_logger().info('EBIMU node started (ASCII mode, /dev/ttyUSB0)')

    def read_serial(self):
        raw = self.ser.read(512)
        if not raw:
            return
        self.buf += raw

        while b'\r\n' in self.buf:
            line, self.buf = self.buf.split(b'\r\n', 1)
            line = line.decode('ascii', errors='ignore').strip()
            if not line.startswith('*'):
                continue
            try:
                parts = line[1:].split(',')
                roll = float(parts[0])
                pitch = float(parts[1])
                yaw = float(parts[2])
            except (ValueError, IndexError):
                continue

            r = math.radians(roll)
            p = math.radians(pitch)
            y = math.radians(yaw)
            cr, sr = math.cos(r/2), math.sin(r/2)
            cp, sp = math.cos(p/2), math.sin(p/2)
            cy, sy = math.cos(y/2), math.sin(y/2)

            msg = Imu()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'imu_link'
            msg.orientation.w = cr*cp*cy + sr*sp*sy
            msg.orientation.x = sr*cp*cy - cr*sp*sy
            msg.orientation.y = cr*sp*cy + sr*cp*sy
            msg.orientation.z = cr*cp*sy - sr*sp*cy
            msg.orientation_covariance[0] = 0.01
            msg.orientation_covariance[4] = 0.01
            msg.orientation_covariance[8] = 0.01
            msg.angular_velocity_covariance[0] = -1.0
            msg.linear_acceleration_covariance[0] = -1.0
            self.pub.publish(msg)

    def destroy_node(self):
        self.ser.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = EbimuNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
