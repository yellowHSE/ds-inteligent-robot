#!/usr/bin/env python3
import os, functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node

class WebServer(Node):
    def __init__(self):
        super().__init__('web_server')
        self.declare_parameter('port', 8080)
        port = int(self.get_parameter('port').value)

        web_dir = os.path.join(get_package_share_directory('turtle_parking'), 'web')
        os.chdir(web_dir)  # 루트 변경

        handler = functools.partial(SimpleHTTPRequestHandler, directory=web_dir)
        self.httpd = ThreadingHTTPServer(('0.0.0.0', port), handler)
        self.get_logger().info(f'Web server serving {web_dir} at http://0.0.0.0:{port}')

    def spin_forever(self):
        try:
            self.httpd.serve_forever()
        finally:
            self.httpd.server_close()

def main():
    rclpy.init()
    node = WebServer()
    try:
        node.spin_forever()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()