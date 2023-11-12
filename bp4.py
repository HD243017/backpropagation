import numpy as np
import math
import random

data_training = np.loadtxt("training.dat")
data_testing = np.loadtxt("testing.dat")

TRAINING = 1000

INPUT_NUM   = 4
HIDDEN_NUM  = 3
OUTPUT_NUM  = 3
CNUM        = 75

LEARNING_RATE = 0.2

W1 = [[0]*4 for i in range(3)]
W2 = [[0]*3 for i in range(3)]

## 타겟 초기화
TARGET = [[1,0,0],[0,1,0],[0,0,1]]

## 연결강도 초기화
for i in range(3):
    for j in range(4):
        W1[i][j] = random.uniform(-0.5, 0.5)

for i in range(3):
    for j in range(3):
        W2[i][j] = random.uniform(-1, 1)
#####################


### 시그모이드
def sigmoid(x):

    return 1/(1+math.exp(-x))

### 에러 토탈 (미분)
def diffEtotal_h(output_y,target,i):
    diffEh = 0
    for j in range(OUTPUT_NUM):
        diffEh += (target[j]-output_y[j])*output_y[j]*(1-output_y[j])*W2[j][i]

    return diffEh

### 에러 토탈
def error_toral(target,output_y):
    e0 = (pow(target[0]-output_y[0],2))/2
    e1 = (pow(target[1]-output_y[1],2))/2
    e2 = (pow(target[2]-output_y[2],2))/2
    
    return e0 + e1 + e2




def training(test_num):
    hidden_x = [0] * HIDDEN_NUM
    hidden_y = [0] * HIDDEN_NUM

    output_x = [0] * OUTPUT_NUM
    output_y = [0] * OUTPUT_NUM

    Etoral = 0

    for cass_num in range(CNUM):
        data = data_training[cass_num][...]
        target = TARGET[((cass_num)//25)]

        for hidden_num in range(HIDDEN_NUM):
            hidden_x[hidden_num] = 0
            hidden_y[hidden_num] = 0

            for input_num in range(INPUT_NUM):
                hidden_x[hidden_num] += data[input_num] * W1[hidden_num][input_num]

            hidden_y[hidden_num] = sigmoid(hidden_x[hidden_num])
        

        for output_num in range(OUTPUT_NUM):
            output_x[output_num] = 0
            output_y[output_num] = 0

            for input_num in range(HIDDEN_NUM):
                output_x[output_num] += hidden_y[input_num] * W2[output_num][input_num]
            
            output_y[output_num] = sigmoid(output_x[output_num])


            delta = (target[output_num]-output_y[output_num])*output_y[output_num]*(1-output_y[output_num])
            for i in range(HIDDEN_NUM):
                W2[output_num][i] += LEARNING_RATE*delta*hidden_y[i]

        Etoral = error_toral(target, output_y)

        for i in range(INPUT_NUM):
            for j in range(HIDDEN_NUM):
                hidden_delta = diffEtotal_h(output_y,target,j)*hidden_y[j]*(1-hidden_y[j])
                W1[j][i] += LEARNING_RATE*hidden_delta*data[i]
    
    if test_num%100 == 0:
        print(Etoral)
    

def testing():
    t_t = [0 for a in range(75)]
    hidden_x = [0] * HIDDEN_NUM
    hidden_y = [0] * HIDDEN_NUM

    output_x = [0] * OUTPUT_NUM
    output_y = [0] * OUTPUT_NUM
    for i in range(75):
        imax = 0
        data = data_testing[i][...]
        for hidden_num in range(HIDDEN_NUM):
            hidden_x[hidden_num] = 0
            hidden_y[hidden_num] = 0

            for input_num in range(INPUT_NUM):
                hidden_x[hidden_num] += data[input_num] * W1[hidden_num][input_num]

            hidden_y[hidden_num] = sigmoid(hidden_x[hidden_num])
        

        for output_num in range(OUTPUT_NUM):
            output_x[output_num] = 0
            output_y[output_num] = 0

            for input_num in range(HIDDEN_NUM):
                output_x[output_num] += hidden_y[input_num] * W2[output_num][input_num]
            
            output_y[output_num] = sigmoid(output_x[output_num])

        for j in range(OUTPUT_NUM):
            if imax < output_y[j]:
                imax = output_y[j]
                t_t[i] = j
        
        print("output : ", output_y," class : ", t_t[i])

for testnum in range(TRAINING):
    training(testnum)
testing()            
            


