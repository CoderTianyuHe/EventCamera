import numpy as np
import cv2



# dot[4] : t, x, y, P
# img_dots : {t1:[dot1,dot2,...], t2:[dot1,dot2,...], ...}
# imgs[10] : [[dot1,dot2,...],[dot1,dot2,...],...]

def get_dot(d: list[int]) -> list[int]:
    h = d[0:4]
    l = d[4:8]
    ls = [0, 0, 0, 0]
    ls[0] = (l[0] << (8 * 3)) | (l[1] << (8 * 2)) | (l[2] << 8) | l[3]
    ls[0] = int(ls[0] / 1000)
    ls[1] = (h[2] << 7) | (h[3] >> 1)
    ls[2] = (h[1] & 0x7F) | (h[0] << 8)
    ls[3] = h[3] & 0x01
    return ls


def img_save(t: int, d: list[list[int]], counter: int) -> None:
    img = np.full((800, 1280, 3), 255, dtype=np.uint8)
    text = "current timestamp is : {}".format(t)
    cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    for i in d:
        if i[2] == 0:
            img[i[0], i[1]] = np.array([255, 0, 0])
        else:
            img[i[0], i[1]] = np.array([0, 255, 0])

    cv2.imshow("img", img)
    cv2.imwrite("./IMG/image_{}.jpg".format(counter), img)
    cv2.waitKey(5)


img_size = 200
batch_size = 20

if __name__ == "__main__":
    counter = 0
    path = "event_record1699173527482016086.aedat"
    f = open(path, "rb")

    # 找到文件结尾
    f.seek(0, 2)
    eof = f.tell()
    f.seek(0, 0)

    img_dots = {}

    while True:
        # 判断是否到结尾
        if f.tell() >= eof:
            break

        # 读取高32位与低32位
        data = f.read(8)
        d = [data[i] for i in range(8)]

        dot = get_dot(d)

        # 添加点，先查找缓冲区是否存在t
        if img_dots.get(dot[0]) is None:
            # 不存在t，看看缓冲区是不是溢出了
            if len(img_dots) >= img_size:
                # 溢出了，排序之后保存图像
                _img_dots = sorted(img_dots.items(), key=lambda f: f[0], reverse=False)
                for key, val in _img_dots[0:batch_size]:
                    img_save(key, val, counter)
                    counter += 1
                    img_dots.pop(key)
                # 添加刚刚没放进去的dot
                img_dots[dot[0]] = [dot[1:4]]
            else:
                # 没有溢出，直接添加
                img_dots[dot[0]] = [dot[1:4]]
        else:
            # 缓冲区存在t，直接添加
            img_dots[dot[0]].append(dot[1:4])
