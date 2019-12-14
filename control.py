#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from gazebo_msgs.msg import ModelStates, ModelState
import copy
import numpy as np

from utils import Environment, Drone, assign_targets
import time

offset = np.array([5000.,1000])
env = Environment()
count = 0
period = 100

publisher = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=10)

def callback(data):
    global count
    
    if count == 0:
        env.prev_states = np.array([d.xy.ravel() - offset for d in env.drones])

        _, _, points = assign_targets(*(env.get_data() + [None, 800]))
        if points.size > 0:
            env.step(points)

        env.new_states = np.array([d.xy.ravel() - offset for d in env.drones])
    
    drone_xy = (1-float(count)/period)*env.prev_states + float(count)/period*env.new_states

    count = (count+1)%period

    # coord = data.pose[1:-1]
    # coord = [c.position for c in coord]
    # coord = [[c.x, c.y, c.z] for c in coord]

    for i,xy in zip(range(1,6), drone_xy):
        msg = ModelState(model_name=data.name[i], 
                         pose=data.pose[i], 
                         twist=data.twist[i])
        msg.pose.position.z = 3.0*i
        msg.pose.position.x = xy[0]
        msg.pose.position.y = xy[1]
        publisher.publish(msg)


    #print(coord)
    
def listener():
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("/gazebo/model_states", ModelStates, callback)

    rospy.spin()

if __name__ == '__main__':
    print('hello')
    listener()