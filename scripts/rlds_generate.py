import rosbag
import cv2
import numpy as np
import rlds

# Path to your ROS bag file
bag_file = 'path_to_your_rosbag.bag'

# Open the ROS bag
bag = rosbag.Bag(bag_file)

# Specify the topics you want to extract
robot_state_topic = '/robot/state'
video_topic = '/camera/image_raw'

# Create lists to hold your data
robot_states = []
video_frames = []

# Read through the bag file and extract data
for topic, msg, t in bag.read_messages(topics=[robot_state_topic, video_topic]):
    if topic == robot_state_topic:
        # Extract robot state
        robot_state = {
            'position': msg.pose.position,
            'orientation': msg.pose.orientation,
            # Add more fields as necessary
        }
        robot_states.append(robot_state)

    elif topic == video_topic:
        # Extract video frame
        np_arr = np.frombuffer(msg.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        video_frames.append(image_np)

bag.close()

# Now you have the robot states and video frames extracted


# Initialize a list to hold the RLDS episodes
episodes = []

# Create a new episode (trajectory)
episode = {
    'observations': [],
    'actions': [],
    'rewards': [],
    'is_terminal': [],
}

# Populate the episode with data
for i, (state, frame) in enumerate(zip(robot_states, video_frames)):
    observation = {
        'robot_state': state,
        'camera_image': frame,
    }
    action = {
        'linear_velocity': 0.0,  # Replace with actual action data
        'angular_velocity': 0.0,  # Replace with actual action data
    }
    reward = 0.0  # Replace with actual reward calculation
    is_terminal = False  # Set to True if this is a terminal step

    episode['observations'].append(observation)
    episode['actions'].append(action)
    episode['rewards'].append(reward)
    episode['is_terminal'].append(is_terminal)

# Add the episode to the list of episodes
episodes.append(episode)

# Save the RLDS dataset
dataset = rlds.Dataset(episodes)
dataset.save('rlds_dataset.tfrecord')

# Load the dataset for verification
loaded_dataset = rlds.Dataset.load('rlds_dataset.tfrecord')

# Verify the contents
for episode in loaded_dataset.episodes:
    print(episode['observations'])
    print(episode['actions'])
    print(episode['rewards'])
    print(episode['is_terminal'])

