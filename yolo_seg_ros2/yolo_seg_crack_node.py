import os
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory


class YoloSegCrackNode(Node):
    def __init__(self):
        super().__init__('yolo_seg_crack_node')

        self.declare_parameter('input_topic', '/rgb_relay')
        self.declare_parameter('depth_topic', '/depth_relay')
        self.declare_parameter('output_topic', '/yolo_result')
        self.declare_parameter('conf_threshold', 0.25)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('process_fps', 5.0)

        try:
            package_share_dir = get_package_share_directory('yolo_seg_ros2')
            default_model_path = os.path.join(package_share_dir, 'yolo26n-seg.pt')
        except Exception:
            default_model_path = 'yolo26n-seg.pt'

        self.declare_parameter('model_path', default_model_path)

        self.input_topic = self.get_parameter('input_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.conf_threshold = float(self.get_parameter('conf_threshold').value)
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.process_fps = float(self.get_parameter('process_fps').value)
        self.model_path = self.get_parameter('model_path').value

        self.bridge = CvBridge()
        self.model = YOLO(self.model_path)

        self.latest_rgb = None
        self.latest_rgb_msg = None
        self.latest_depth = None
        self.processing = False

        sub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(Image, self.input_topic, self.rgb_callback, sub_qos)
        self.create_subscription(Image, self.depth_topic, self.depth_callback, sub_qos)

        self.result_pub = self.create_publisher(Image, self.output_topic, pub_qos)

        timer_period = 1.0 / max(self.process_fps, 0.1)
        self.timer = self.create_timer(timer_period, self.process_frame)

        self.get_logger().info(f'Loaded model: {self.model_path}')
        self.get_logger().info(
            f'Subscribed RGB: {self.input_topic}\n'
            f'Subscribed Depth: {self.depth_topic}\n'
            f'Publishing Result: {self.output_topic}'
        )

    def rgb_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_rgb = frame
            self.latest_rgb_msg = msg
        except Exception as e:
            self.get_logger().error(f'RGB callback error: {e}')

    def depth_callback(self, msg: Image):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Depth callback error: {e}')

    def get_mask_distance_m(self, mask, depth_img):
        if depth_img is None:
            return None

        mask_pixels = depth_img[mask > 0]
        valid = mask_pixels[np.isfinite(mask_pixels)]

        if valid.size == 0:
            return None

        # 常见深度图有两种：mm 或 m
        # 这里做一个简单判断
        if valid.mean() > 20:
            valid = valid[(valid > 100) & (valid < 10000)]
            if valid.size == 0:
                return None
            return float(np.median(valid)) / 1000.0
        else:
            valid = valid[(valid > 0.1) & (valid < 10.0)]
            if valid.size == 0:
                return None
            return float(np.median(valid))

    def process_frame(self):
        if self.processing:
            return
        if self.latest_rgb is None:
            return

        self.processing = True

        try:
            frame = self.latest_rgb.copy()
            depth = None if self.latest_depth is None else self.latest_depth.copy()
            rgb_msg = self.latest_rgb_msg

            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                imgsz=self.imgsz,
                verbose=False
            )

            result = results[0]
            annotated = result.plot()

            h, w = frame.shape[:2]
            best_distance = None
            best_center = None

            if result.masks is not None and len(result.masks.data) > 0:
                masks = result.masks.data.cpu().numpy()

                max_area = 0
                best_mask = None

                for m in masks:
                    mask = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask_bin = (mask > 0.5).astype(np.uint8)

                    area = int(mask_bin.sum())
                    if area > max_area:
                        max_area = area
                        best_mask = mask_bin

                if best_mask is not None:
                    ys, xs = np.where(best_mask > 0)
                    if len(xs) > 0 and len(ys) > 0:
                        cx = int(np.mean(xs))
                        cy = int(np.mean(ys))
                        best_center = (cx, cy)

                        best_distance = self.get_mask_distance_m(best_mask, depth)

                        cv2.circle(annotated, (cx, cy), 6, (0, 0, 255), -1)
                        cv2.putText(
                            annotated,
                            f'Target center: ({cx}, {cy})',
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA
                        )

            if best_distance is not None:
                cv2.putText(
                    annotated,
                    f'Distance: {best_distance:.2f} m',
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )
            else:
                cv2.putText(
                    annotated,
                    'Distance: N/A',
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

            img_center_x = w // 2
            cv2.line(annotated, (img_center_x, 0), (img_center_x, h), (255, 255, 0), 2)

            if best_center is not None:
                error_x = best_center[0] - img_center_x
                cv2.putText(
                    annotated,
                    f'Error X: {error_x}',
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA
                )

            out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            if rgb_msg is not None:
                out_msg.header = rgb_msg.header
            self.result_pub.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f'Process frame error: {e}')
        finally:
            self.processing = False


def main(args=None):
    rclpy.init(args=args)
    node = YoloSegCrackNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()