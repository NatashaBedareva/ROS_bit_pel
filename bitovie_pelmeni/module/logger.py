from collections import deque

from module.config import INFO_LEVEL, LOGGING_POOL_SIZE

previous_message_id = -1
previous_messages = deque(maxlen=LOGGING_POOL_SIZE)

def log_info(node, message, msg_id=-1, allow_repeat=False):
    global previous_message_id, previous_messages

    if not INFO_LEVEL:
        return

    if msg_id == -1 and message in previous_messages:
        if previous_messages[-1] != message:
            previous_messages.append(message)
        return

    if previous_message_id == msg_id and msg_id != -1:
        return
    
    message_ = f"--- {message}"
    
    node.get_logger().info(message_)
    previous_message_id = msg_id
    previous_messages.append(message)