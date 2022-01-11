import sys
import random
from PIL import Image
import os

def calSize(file):
    image = Image.open(file)
    # print(image)
    # image.show()
    size_x = image.size[0]
    size_y = image.size[1]

    return size_x, size_y, image

def divide(file_name, col=1, row=1, sub_img=None):
    img_dir = './Photo'
    path = os.path.join(img_dir, file_name)
    out_path = os.path.join(img_dir, sub_img)
    x_size, y_size, img = calSize(path)

    # 이미지 가로
    # 가로 길이가 홀수일 때
    if x_size % row != 0:
        x_size = x_size-(x_size % row)
    # 이미지 세로
    if y_size % col != 0:
        y_size = y_size - (y_size % col)

    x_chop = x_size//row
    y_chop = y_size//col

    num_file = [i for i in range(row*col)]
    random.shuffle(num_file)

    cnt = 0
    for x0 in range(0, x_size, x_chop):
        for y0 in range(0, y_size, y_chop):
            box = (x0, y0,
                   x0 + x_chop if x0 + x_chop < x_size else x_size,
                   y0 + y_chop if y0 + y_chop < y_size else y_size)

            tmp_img = img.crop(box)
            a = random.randrange(0, 2)
            b = random.randrange(0, 2)
            c = random.randrange(0, 2)

            # 미러링 적용
            if a == 1 :
                tmp_img = tmp_img.transpose(Image.FLIP_LEFT_RIGHT)
                # print('미러링 적용')

            # flipping 적용
            if b == 1:
                tmp_img = tmp_img.transpose(Image.FLIP_TOP_BOTTOM)
                # print('플리핑 적용')

            # Rotate 90 적용
            if c == 1:
                tmp_img = tmp_img.transpose(Image.ROTATE_90)
                # print('90도 전환 적용')

            print('%s %s' % (path, box))
            tmp_img.save('%s_crop%03d.jpg' % (out_path.replace('.jpg', ''), num_file[cnt]))
            cnt+=1

    return

def main(argv):
    divide(argv[1],int(argv[2]), int(argv[3]), argv[4])

if __name__ == '__main__':
    main(sys.argv)