import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/natalia/Documents/STUDING/ROS_bit_pel/install/referee_console'
