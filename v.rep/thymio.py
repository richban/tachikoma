import vrep
import time


class Thymio:
    def __init__(self, client_id, id, op_mode):
        self.id = id
        self.client_id = client_id
        self.op_mode = op_mode

        # Initialize Motors
        res, self.left_motor = vrep.simxGetObjectHandle(self.client_id, "Pioneer_p3dx_leftMotor%s" %self.suffix, self.op_mode)
        print(res)
        res, self.right_motor = vrep.simxGetObjectHandle(self.client_id, "Pioneer_p3dx_rightMotor%s" %self.suffix, self.op_mode)

    @property
    def suffix(self):
        if self.id != None:
            return '#%d' %self.id
        return ''


    def move_forward(self, speed=11.0):
        self._set_two_motor(speed, speed)


    def _set_two_motor(self, left: float, right: float):
        vrep.simxSetJointTargetVelocity(self.client_id, self.left_motor, left, vrep.simx_opmode_streaming)
        vrep.simxSetJointTargetVelocity(self.client_id, self.right_motor, right, vrep.simx_opmode_streaming)




if __name__ == '__main__':
    print('Program started')
    port_num = 20010
    vrep.simxFinish(-1) # just in case, close all opened connections
    client_id=vrep.simxStart('127.0.0.1',port_num,True,True,5000,5) # Connect to V-REP
    if client_id != -1:
        print ('Connected to remote API server')
        op_mode = vrep.simx_opmode_oneshot_wait
        thymio = Thymio(client_id, None, op_mode)
        # vrep.simxStopSimulation(client_id, op_mode)
        # time.sleep(1)
        # vrep.simxStartSimulation(client_id, op_mode)
        startTime=time.time()
        thymio.move_forward()
        thymio0.move_forward()
        while time.time() - startTime < 10:
            print(startTime)
            time.sleep(0.005)

        # vrep.simxStopSimulation(client_id, op_mode)
        # vrep.simxFinish(client_id)

        print ('Program ended')
    else:
        print ('Failed connecting to remote API server')
