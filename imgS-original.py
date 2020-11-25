import matplotlib.pyplot as plt
import gym
import numpy as np
import cv2

# Enter N 3 channel pictures array
# Output: an array shape (84 84 N)
# : 1. resize ==>(84 84 3)[uint 0-255]
#       2. gray   ==> (84 84 1) [uint 0-255]
#       3. norm   ==> (84 84 1) [float32 0.0-1.0]
#       4. concat ===>(84 84 N) [float32 0.0-1.0]
def imgbuffer_process(imgbuffer, out_shape = (84, 84)):
    img_list = []
    for img in imgbuffer:
        tmp = cv2.resize(src=img, dsize=out_shape)
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
        ## Need to convert data type to 32F
        tmp = cv2.normalize(tmp, tmp, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # Expand a dimension
        tmp = np.expand_dims(tmp, len(tmp.shape))
        img_list.append(tmp)
    ret =  np.concatenate(tuple(img_list), axis=2)
    #print('ret_shape = ' + str(ret.shape))
    return ret

def test():
    env = gym.make('Breakout-v4')
    env.seed(1)  # reproducible
    # env = env.unwrapped
    N_F = env.observation_space.shape[0]  #Status space dimension
    N_A = env.action_space.n  #

    img_buffer = []
    img_buffer_size = 4
    s = env.reset()
    max_loop = 100000

    for i in range(max_loop):
        a = np.random.randint(0, N_A - 1)
        s_, r, done, info = env.step(a)
        env.render()

        if len(img_buffer) < img_buffer_size:
            img_buffer.append(s_)
            continue
        else:
            img_buffer.pop(0)
            img_buffer.append(s_)

        img_input = imgbuffer_process(img_buffer)
        print('img_input_shape = ' + str(img_input.shape))
        plt.subplot(2, 2, 1)
        plt.imshow(np.uint8(img_input[:, :, 0] * 255), cmap='gray')
        plt.subplot(2, 2, 2)
        plt.imshow(np.uint8(img_input[:, :, 1] * 255), cmap='gray')
        plt.subplot(2, 2, 3)
        plt.imshow(np.uint8(img_input[:, :, 2] * 255), cmap='gray')
        plt.subplot(2, 2, 4)
        plt.imshow(np.uint8(img_input[:, :, 3] * 255), cmap='gray')
        plt.show()

if __name__ == '__main__':
    test()