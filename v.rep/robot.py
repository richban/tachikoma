import vrep
import time
import math
from datetime import datetime, timedelta
import numpy as np

PI = math.pi
NUM_SENSORS = 16


class Robot:

    def __init__(self, client_id, id, op_mode):
        self.id = id
        self.client_id = client_id
        self.op_mode = op_mode

        # Initialize Motors
        res, self.left_motor = vrep.simxGetObjectHandle(
            self.client_id, "Pioneer_p3dx_leftMotor%s" %
            self.suffix, self.op_mode)
        res, self.right_motor = vrep.simxGetObjectHandle(
            self.client_id, "Pioneer_p3dx_rightMotor%s" %
            self.suffix, self.op_mode)

        # Initialize Proximity Sensors
        self.prox_sensors = []
        self.prox_sensors_val = []
        for i in range(1, NUM_SENSORS + 1):
            res, sensor = vrep.simxGetObjectHandle(
                self.client_id, 'Pioneer_p3dx_ultrasonicSensor%d%s' %
                (i, self.suffix), self.op_mode)
            self.prox_sensors.append(sensor)
            errorCode, detectionState, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector = vrep.simxReadProximitySensor(
                self.client_id, sensor, vrep.simx_opmode_streaming)
            np.append(self.prox_sensors_val, np.linalg.norm(detectedPoint))

        # Orientation of all the sensors:
        self.sensors_loc = np.array([-PI / 2, -50 / 180.0 * PI, -30 / 180.0 * PI, -10 / 180.0 * PI, 10 / 180.0 * PI, 30 / 180.0 * PI, 50 / 180.0 *
                                     PI, PI / 2, PI / 2, 130 / 180.0 * PI, 150 / 180.0 * PI, 170 / 180.0 * PI, -170 / 180.0 * PI, -150 / 180.0 * PI, -130 / 180.0 * PI, -PI / 2])

    @property
    def suffix(self):
        if self.id is not None:
            return '#%d' % self.id
        return ''

    def move_forward(self, speed=5.0):
        self.set_motors(speed, speed)

    def move_backward(self, speed=5.0):
        self.set_motors(-speed, -speed)

    def set_motors(self, left: float, right: float):
        vrep.simxSetJointTargetVelocity(
            self.client_id,
            self.left_motor,
            left,
            vrep.simx_opmode_streaming)
        vrep.simxSetJointTargetVelocity(
            self.client_id,
            self.right_motor,
            right,
            vrep.simx_opmode_streaming)

    def set_left_motor(self, left: float):
        vrep.simxSetJointTargetVelocity(
            self.client_id,
            self.left_motor,
            left,
            vrep.simx_opmode_streaming)

    def set_right_motor(self, right: float):
        vrep.simxSetJointTargetVelocity(
            self.client_id,
            self.right_motor,
            right,
            vrep.simx_opmode_streaming)

    def get_sensor_state(self, sensor):
        errorCode, detectionState, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector = vrep.simxReadProximitySensor(
            self.client_id, sensor, vrep.simx_opmode_streaming)
        return detectionState

    def get_sensor_distance(self, sensor):
        errorCode, detectionState, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector = vrep.simxReadProximitySensor(
            self.client_id, sensor, vrep.simx_opmode_buffer)
        return np.linalg.norm(detectedPoint)


def avoid_obstacles(robot):
    start_time = datetime.now()

    while datetime.now() - start_time < timedelta(seconds=60):
        sensors_val = np.array([])
        for s in robot.prox_sensors:
            detectedPoint = robot.get_sensor_distance(s)
            sensors_val = np.append(sensors_val, detectedPoint)

        # controller specific - take front sensor values.
        sensor_sq = sensors_val[0:8] * sensors_val[0:8]
        # find sensor where the obstacle is closest
        min_ind = np.where(sensor_sq == np.min(sensor_sq))
        min_ind = min_ind[0][0]

        if sensor_sq[min_ind] < 0.2:
            # sensor which has the obstacle closest to it
            steer = -1 / robot.sensors_loc[min_ind]
        else:
            steer = 0

        v = 1  # forward velocity
        kp = 0.5  # steering gain
        vl = v + kp * steer
        vr = v - kp * steer
        print("V_l = "+str(vl))
        print("V_r = "+str(vr))
        robot.set_motors(vl, vr)
        time.sleep(0.2)  # loop executes once every 0.2 seconds (= 5 Hz)

    #Post ALlocation
    errorCode=vrep.simxSetJointTargetVelocity(robot.client_id,robot.left_motor,0, robot.op_mode)
    errorCode=vrep.simxSetJointTargetVelocity(robot.cleint_id,robot.right_motor,0, robot.op_mode)




if __name__ == '__main__':
    print('Program started')
    port_num = 20010
    vrep.simxFinish(-1)  # just in case, close all opened connections
    client_id = vrep.simxStart(
        '127.0.0.1',
        port_num,
        True,
        True,
        5000,
        5)  # Connect to V-REP
    if client_id != -1:
        print('Connected to remote API server')
        op_mode = vrep.simx_opmode_oneshot_wait
        thymio = Robot(client_id, None, op_mode)
        # vrep.simxStopSimulation(client_id, op_mode)
        # time.sleep(1)
        # vrep.simxStartSimulation(client_id, op_mode)
        avoid_obstacles(thymio)
        # vrep.simxStopSimulation(client_id, op_mode)
        # vrep.simxFinish(client_id)

        print('Program ended')
    else:
        print('Failed connecting to remote API server')
