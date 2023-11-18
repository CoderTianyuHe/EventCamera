import numpy as np
import matplotlib.pyplot as plt
f = open("/home/hatiy/event_camera/event_camera_1/event_record1699173527482016086.aedat", "rb")
np.set_printoptions(threshold=np.inf)

for i in range(1000):
    df = f.read(8)
    a = np.frombuffer(df, dtype=np.uint8, count=-1, offset=0)
    # array = hex(int(array))
    # a = data.tobytes('F')
    if(i==0):
        y = np.array([a[0]])
        x = np.array([a[1]])
    np.append(y,1)
    np.append(x,2)
    
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([1, 4, 9, 16, 7, 11, 23, 18])

plt.scatter(x, y)
plt.show()
    
f.close()
