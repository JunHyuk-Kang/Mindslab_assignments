import numpy as np
from PIL import Image
import os
import copy
import matplotlib.pyplot as plt
import sys

# IMAGE[위를 판단]
# 회전이 안됨 -> up [0, 1,2,3,4]
# 90도 회전 -> left
# 180도 회전 -> down
# 270도 회전 -> right
# 미러링 -> up_flip [4,3,2,1,0]
# 90도 회전 -> [....]

class Image_op:
    def __init__(self, file):
        self.img = np.array(Image.open(file))
        self.flip_img = self.img[:, ::-1, :]
        # list[8] => up, left, down, right, up_flip, left_flip, down_flip, right_flip
        # self.dirs_pixels = [self.img[0, : , :], self.img[:, 0, :], self.img[len(self.img) - 1, :, :], self.img[:,len(self.img[0])-1,:],
        #                     self.img_flip[0, : , :], self.img_flip[:, 0, :], self.img_flip[len(self.img_flip) - 1, :, :], self.img_flip[:,len(self.img_flip[0])-1,:]]
        self.dirs_pixels = [
            self.img[0, :, :],
            self.img[:, -1, :],
            self.img[-1, ::-1, :],
            self.img[::-1, 0, :],
            self.flip_img[0, :, :],
            self.flip_img[:, -1, :],
            self.flip_img[-1, ::-1, :],
            self.flip_img[::-1, 0, :],
        ]
        self.dir = 0 # -> [0, 0도], [1, 90도], [2, 180도], [3, 270도] /  [0, up], [1, left], [2=right], [3=down]
        self.isFlips = False # true -> flip이 됨, false -> flip이 안된거
        # for i in self.dirs_pixels:
        #     print(i.shape)
        # input()


def cost_check(current_use):
    global N
    cost = 0
    pix_list = []
    threshold = 5
    # 0 1
    # 2 3 => cost 비교
    # -> [0, up], [1, left], [2=right], [3=down]
    # 왼위 오른쪽 - 오위 왼쪽
    if (imgs[current_use[0]].dirs_pixels[((2 + imgs[current_use[0]].dir) % 4) + (4 * imgs[current_use[0]].isFlips)].shape) != \
            (imgs[current_use[1]].dirs_pixels[((1 + imgs[current_use[1]].dir) % 4) + (4 * imgs[current_use[1]].isFlips)].shape):
        cost = 987654321
        pass
    else:
        cost = cost + np.sum(np.sum(np.abs(imgs[current_use[0]].dirs_pixels[((2 + imgs[current_use[0]].dir) % 4) + (4 * imgs[current_use[0]].isFlips)] - \
                                  imgs[current_use[1]].dirs_pixels[((1 + imgs[current_use[1]].dir) % 4) + (4 * imgs[current_use[1]].isFlips)]), axis=1)>threshold)
        # cost = cost + sum(sum(abs(imgs[current_use[0]].dirs_pixels[((2 + imgs[current_use[0]].dir) % 4) + (4 * imgs[current_use[0]].isFlips)]-\
        #     imgs[current_use[1]].dirs_pixels[((1 + imgs[current_use[1]].dir) % 4) + (4 * imgs[current_use[1]].isFlips)])))

    # 왼위 아래쪽 - 왼아래 위쪽
    if (imgs[current_use[0]].dirs_pixels[((3 + imgs[current_use[0]].dir) % 4) + (4 * imgs[current_use[0]].isFlips)].shape) !=\
            (imgs[current_use[2]].dirs_pixels[((0 + imgs[current_use[2]].dir) % 4) + (4 * imgs[current_use[2]].isFlips)].shape):
        cost = 987654321
        pass
    else:
        cost = cost + np.sum(np.sum(np.abs(imgs[current_use[0]].dirs_pixels[((3 + imgs[current_use[0]].dir) % 4) + (4 * imgs[current_use[0]].isFlips)] -\
            imgs[current_use[2]].dirs_pixels[((0 + imgs[current_use[2]].dir) % 4) + (4 * imgs[current_use[2]].isFlips)]), axis=1)>threshold)
        # cost = cost + sum(sum(abs(
        #     imgs[current_use[0]].dirs_pixels[((3 + imgs[current_use[0]].dir) % 4) + (4 * imgs[current_use[0]].isFlips)] - \
        #     imgs[current_use[2]].dirs_pixels[((0 + imgs[current_use[2]].dir) % 4) + (4 * imgs[current_use[2]].isFlips)])))

    # 오위 아래쪽 - 오아래 위쪽
    if (imgs[current_use[1]].dirs_pixels[((3 + imgs[current_use[1]].dir) % 4) + (4 * imgs[current_use[1]].isFlips)].shape) !=\
            (imgs[current_use[3]].dirs_pixels[((0 + imgs[current_use[3]].dir) % 4) + (4 * imgs[current_use[3]].isFlips)].shape):
        cost = 987654321
        pass
    else:
        cost = cost + np.sum(np.sum(np.abs(imgs[current_use[1]].dirs_pixels[((3 + imgs[current_use[1]].dir) % 4) + (4 * imgs[current_use[1]].isFlips)] - \
            imgs[current_use[3]].dirs_pixels[((0 + imgs[current_use[3]].dir) % 4) + (4 * imgs[current_use[3]].isFlips)]), axis=1)>threshold)
        # cost = cost + sum(sum(abs(
        #     imgs[current_use[1]].dirs_pixels[((3 + imgs[current_use[1]].dir) % 4) + (4 * imgs[current_use[1]].isFlips)] - \
        #     imgs[current_use[3]].dirs_pixels[((0 + imgs[current_use[3]].dir) % 4) + (4 * imgs[current_use[3]].isFlips)])))

    # 왼아래 오른쪽 - 오아래 왼쪽
    if (imgs[current_use[2]].dirs_pixels[((2 + imgs[current_use[2]].dir) % 4) + (4 * imgs[current_use[2]].isFlips)].shape) !=\
            (imgs[current_use[3]].dirs_pixels[((1 + imgs[current_use[3]].dir) % 4) + (4 * imgs[current_use[3]].isFlips)].shape):
        cost = 987654321
        pass
    else:
        cost = cost + np.sum(np.sum(np.abs(imgs[current_use[2]].dirs_pixels[((2 + imgs[current_use[2]].dir) % 4) + (4 * imgs[current_use[2]].isFlips)] - \
                                  imgs[current_use[3]].dirs_pixels[((1 + imgs[current_use[3]].dir) % 4) + (4 * imgs[current_use[3]].isFlips)]), axis=1)>threshold)
        # cost = cost + sum(sum(abs(imgs[current_use[2]].dirs_pixels[((2 + imgs[current_use[2]].dir) % 4) + (4 * imgs[current_use[2]].isFlips)] - \
        #     imgs[current_use[3]].dirs_pixels[((1 + imgs[current_use[3]].dir) % 4) + (4 * imgs[current_use[3]].isFlips)])))

    return cost

def back_tracking(step, current_use=[]):
    global min_cost
    global imgs
    global result
    global min_ref
    global cnt

    # 4, [0, 1, 2, 3]
    if step == 4:
        cost = cost_check(current_use)
        # print(cost)
        # print(min_cost)
        if cost < min_cost:
            min_cost = cost
            min_ref = copy.deepcopy(current_use)
            result = copy.deepcopy(imgs)
            print("===================")
            print(min_cost)
            print(min_ref)
            print(current_use)
            print(imgs[0].isFlips)
            print(imgs[1].isFlips)
            print(imgs[2].isFlips)
            print(imgs[3].isFlips)
            print("===================")
            print(result[0].isFlips)
            print(result[1].isFlips)
            print(result[2].isFlips)
            print(result[3].isFlips)
            print("===================")
        return

    for i in range(N):
        # 0~3
        if i in current_use:
            continue

        current_use.append(i) #
        print(current_use)
        for j in range(2): # 0 flip X / 1 flip O
            for k in range(4): # 회전이 얼마나 됐는지
                # print(bool(j))
                imgs[current_use[-1]].isFlips = bool(j)
                imgs[current_use[-1]].dir = k
                back_tracking(step+1, current_use)

        current_use.pop()

N = 0
imgs = []
result = []
min_cost = 987654321
min_ref = -1
cnt = 0

def merge(prefix_name=None, col=2, row=2, output_img='output'):
    global N
    global imgs
    global result
    global min_ref

    # min_ref = [3,0,1,2]

    N = col * row
    img_dir = './Photo'
    out_path = os.path.join(img_dir, output_img)
    files = os.listdir(img_dir)
    files = [os.path.join(img_dir, i) for i in files if i.startswith(prefix_name)]
    plot = []

    for i in range(N):
        imgs.append(Image_op('{}'.format(files[i])))

    # result = imgs
    # cost = cost_check(min_ref)
    # print(cost)

    back_tracking(0, [])
    print(min_cost)
    print(min_ref)
    print(result)
    print(result[0].isFlips)
    print(result[1].isFlips)
    print(result[2].isFlips)
    print(result[3].isFlips)
    print(result[0].dir)
    print(result[1].dir)
    print(result[2].dir)
    print(result[3].dir)

    for i in min_ref:
        result[i].img = Image.fromarray(result[i].img)
        if result[i].isFlips == True :
            result[i].img = Image.fromarray(np.array(result[i].img)[:, ::-1, :])
            # result[i].img = result[i].img.transpose(Image.FLIP_LEFT_RIGHT)

        if result[i].dir > 0:
            for j in range(result[i].dir):
                result[i].img = result[i].img.transpose(Image.ROTATE_90)

        plot.append(result[i].img)
        result[i].img.show()


    a = np.hstack((plot[0], plot[1]))
    b = np.hstack((plot[2], plot[3]))
    c = np.vstack((a,b))

    c = Image.fromarray(c)
    c.save('%s.jpg' % (out_path.replace('.jpg', '')))
    # c.show()

def main(argv):
    merge(argv[1],int(argv[2]), int(argv[3]), argv[4])

if __name__ == '__main__':
    main(sys.argv)