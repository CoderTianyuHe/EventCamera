import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
import cv2

def get_dot(d) :
    h = d[0:4]
    l = d[4:8]
    ls = [0, 0, 0, 0]
    ls[0] = (l[0] << (8 * 3)) | (l[1] << (8 * 2)) | (l[2] << 8) | l[3]
    ls[0] = int(ls[0] / 1000)
    ls[1] = (h[2] << 7) | (h[3] >> 1)
    # ls[2] = (h[1])  | (h[0] << 8)
    ls[2] = (h[0] & 0x7F)<<8 | h[1]
    ls[3] = h[3] & 0x01
    return ls, ls[3]
t=0
def get_t(data):
    global t
    t=data
if __name__ == "__main__":
    counter = 0
    path = "event_record1699173527482016086.aedat"
    f = open(path, "rb")

    # 找到文件结尾
    f.seek(0, 2)
    eof = f.tell()
    f.seek(0, 0)

    dot_lst_p = []
    dot_lst_n = []
    while True:
        # 判断是否到结尾
        if f.tell() >= eof:
            break

        # 读取高32位与低32位
        data = f.read(8)
        d = [data[i] for i in range(8)]

        dot, p = get_dot(d)
        if p == 0:
            dot_lst_n.append(dot)
        else:
            dot_lst_p.append(dot)

    # 绘制散点图
    # plt.plot(dot_lst_n[1], dot_lst_n[2], dot_lst_n[0], c='g', label='N')
    n_matrix=np.array(dot_lst_n)
    p_matrix = np.array(dot_lst_p)
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")

    # Creating plot
    ax.scatter3D(dot_lst_p[1], dot_lst_p[2], dot_lst_p[0], color="green")
    plt.title("simple 3D scatter plot")
    print(n_matrix[:,0])
    # show plot
    plt.show()
    cv2.namedWindow('time', cv2.WINDOW_AUTOSIZE)
    fourcc=cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter('output.mp4v', cv2.VideoWriter_fourcc(*'MP4V'), 10.0, (1200, 800))

    print(n_matrix.dtype)
    for t in range(100):
        img = np.full((800, 1280, 3), 255, dtype="uint8")
        n_dot_show=n_matrix[(n_matrix[:,0]>=t*100) & (n_matrix[:,0]<(t+1)*100) ]
        p_dot_show = p_matrix[(p_matrix[:, 0] >= t*100) & (p_matrix[:, 0] < (t+1)*100)]

        img[n_dot_show[:,1],n_dot_show[:,2]]=np.array([255, 0, 0], dtype = "uint8")
        img[p_dot_show[:, 1], p_dot_show[:, 2]] = np.array([0, 255, 0], dtype="uint8")
        cv2.imshow('imshow', img)
        cv2.imwrite("./img/{}.jpg".format(t), img)
        cv2.waitKey(100)
        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            break
        out.release()
    # while 1:
    #     img = np.full((800, 1280, 3), 255, dtype="uint8")
    #     n_dot_show=n_matrix[(n_matrix[:,0]>=t*100) & (n_matrix[:,0]<(t+1)*100) ]
    #     p_dot_show = p_matrix[(p_matrix[:, 0] >= t*100) & (p_matrix[:, 0] < (t+1)*100)]
    #
    #     img[n_dot_show[:,1],n_dot_show[:,2]]=np.array([255, 0, 0], dtype = "uint8")
    #     img[p_dot_show[:, 1], p_dot_show[:, 2]] = np.array([0, 255, 0], dtype="uint8")
    #     cv2.imshow('imshow', img)
    #     out.write(img)
    #     cv2.waitKey(20)
    #     if cv2.waitKey(1) == ord("q"):
    #         cv2.destroyAllWindows()
    #         break
