import numpy as np
import cv2


class Dot():
    """t为时间戳，P为极性，x,y为坐标"""

    def __init__(self, d: list[int]):
        h = d[0:4]
        l = d[4:8]
        self.t = (l[0] << (8 * 3)) | (l[1] << (8 * 2)) | (l[2] << 8) | l[3]
        self.P = h[3] & 0x01
        self.x = (h[2] << 7) | (h[3] >> 1)
        self.y = (h[1] & 0x7F) | (h[0] << 8)


class Img():
    def __init__(self, dots: list[Dot]):
        self.img = np.full((800, 1280, 3), 255, dtype=np.uint8)
        self.t = dots[0].t
        for i in dots:
            if i.P == 0:
                cv2.circle(self.img, (i.x, i.y), 3, (255, 0, 0), 3)
            elif i.P == 1:
                cv2.circle(self.img, (i.x, i.y), 3, (0, 255, 0), 3)


class Imgs():
    def __init__(self):
        self.cnt = 0
        self.dots: list[Dot] = []

    def add_dot(self, dot_t: Dot):
        # 判断dot与上一个dot是否同时发生
        if len(self.dots):
            if self.dots[0].t == dot_t.t:
                self.dots.append(dot_t)
            else:
                ### 由于爆内存，存到文件里
                img = Img(self.dots)
                text = "current timestamp is : {}".format(img.t)
                cv2.putText(img.img, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                cv2.imshow("img", img.img)
                cv2.imwrite("./IMG/image_{}.jpg".format(self.cnt), img.img)
                self.cnt += 1
                cv2.waitKey(10)

                ###
                self.dots.clear()
                self.dots.append(dot_t)
        else:
            self.dots.append(dot_t)

    def cheak_empty(self):
        if len(self.dots):
            self.imgs.append(Img(self.dots))
            self.dots.clear()

    def img_show(self):
        self.cheak_empty()
        for img in self.imgs:
            text = "current timestamp is : {}".format(img.t)
            cv2.putText(img.img, text, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
            cv2.imshow("img", img.img)
            cv2.waitKey(0)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    path = "event_record1699173527482016086.aedat"
    f = open(path, "rb")

    # 找到文件结尾
    f.seek(0, 2)
    eof = f.tell()
    f.seek(0, 0)

    # 初始化imgs
    imgs = Imgs()

    while True:
        # 判断是否到结尾
        if f.tell() >= eof:
            break

        # 读取高32位与低32位
        data = f.read(8)
        d = [data[i] for i in range(8)]

        dot_t1 = Dot(d)

        imgs.add_dot(dot_t1)
