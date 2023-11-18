import os
import pandas as pd
from dv import AedatFile
current_dir = os.getcwd()

print("当前工作目录为：", current_dir)
import aedat

decoder = aedat.Decoder("/home/hatiy/event_camera/event_camera_1/event_record1699173527482016086.aedat")
print(decoder.id_to_stream())

for packet in decoder:
    print(packet["stream_id"], end=": ")
    if "events" in packet:
        print("{} polarity events".format(len(packet["events"])))
    elif "frame" in packet:
        print("{} x {} frame".format(packet["frame"]["width"], packet["frame"]["height"]))
    elif "imus" in packet:
        print("{} IMU samples".format(len(packet["imus"])))
    elif "triggers" in packet:
        print("{} trigger events".format(len(packet["triggers"])))
# print(pd.read_csv("/home/hatiy/event_camera/event_camera_1/event_record1699173527482016086.aedat",))
