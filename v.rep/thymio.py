import vrep
import time


class Thymio:
    def __init__(self, client_id, id, op_mode):
        self.id = id
        self.client_id = client_id
        self.op_mode = op_mode
        res, self.body = vrep.simxGetObjectHandle(self.client_id, "ePuck_base%s" %self.suffix, self.op_mode)

        # Initialize Motors
        ret, self.left_motor = vrep.simxGetObjectHandle(self.client_id, "ePuck_leftWheel%s" %self.suffix, self.op_mode)
        ret, self.right_motor = vrep.simxGetObjectHandle(self.client_id, "ePuck_rightWheel%s" %self.suffix, self.op_mode)

    @property
    def suffix(self):
        if self.id != None:
            return '#%d' %self.id
        return ''


    def move_forward(self, speed=11.0):
        self._set_two_motor(speed, speed)


    def _set_two_motor(self, left: float, right: float):
        vrep.simxSetJointTargetVelocity(self.client_id, self.left_motor, left, self.op_mode)
        vrep.simxSetJointTargetVelocity(self.client_id, self.right_motor, right, self.op_mode)




if __name__ == '__main__':
    print('Program started')
    port_num = 20010
    vrep.simxFinish(-1) # just in case, close all opened connections
    client_id=vrep.simxStart('127.0.0.1',port_num,True,True,5000,5) # Connect to V-REP
    if client_id != -1:
        print ('Connected to remote API server')
        op_mode = vrep.simx_opmode_oneshot_wait
        thymio = Thymio(client_id, 0, op_mode)
        vrep.simxStopSimulation(client_id, op_mode)
        time.sleep(1)
        vrep.simxStartSimulation(client_id, op_mode)
        startTime=time.time()
        while time.time()-startTime < 10:
            thymio.move_forward()
            print ('ePock move_forward')
            time.sleep(0.005)

        vrep.simxStopSimulation(client_id, op_mode)
        vrep.simxFinish(client_id)

        print ('Program ended')
    else:
        print ('Failed connecting to remote API server')
