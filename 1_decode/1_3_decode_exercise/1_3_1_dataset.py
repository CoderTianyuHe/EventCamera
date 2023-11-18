import torch
from torch.utils.data import Dataset, DataLoader
from typing import Any
import os
import numpy as np
from matterhorn_pytorch.util.plotter import event_tensor_plot_tyx


def extract(data: np.ndarray, mask: int, shift: int) -> np.ndarray:
    """
    从事件数据中提取x,y,p值所用的函数。
    @params:
        data: np.ndarray 事件数据
        mask: int 对应的掩模
        shift: int 对应的偏移量
    @return:
        data: np.ndarray 处理后的数据（x,y,p）
    """
    return (data >> shift) & mask


def filename_2_data(filename: str, endian: str = ">", datatype: str = "u4") -> np.ndarray:
    """
    输入文件名，读取文件内容。
    @params:
        filename: str 文件名
    @return:
        data: np.ndarray 文件内容（数据）
    """
    data_str = ""
    with open(filename, 'rb') as f:
        data_str = f.read()
        lines = data_str.split(b'\n')
        for line in range(len(lines)):
            if not lines[line].startswith(b'#'):
                break
        lines = lines[line:]
        data_str = b'\n'.join(lines)
    data = np.frombuffer(data_str, dtype = endian + datatype)
    return data


def data_2_file(filename: str, data: np.ndarray, endian: str = ">", datatype: str = "u4") -> np.ndarray:
    """
    输入文件名，读取文件内容。
    @params:
        filename: str 文件名
    @return:
        data: np.ndarray 文件内容（数据）
    """
    data = data.astype(endian + datatype)
    with open(filename, "w+b") as file:
        file.write(data.tobytes())


def data_2_tpyx(data: np.ndarray, p_mask = 0x1, p_shift = 0, y_mask = 0x7F, y_shift = 8, x_mask = 0x7F, x_shift = 1) -> np.ndarray:
    """
    将数据分割为t,p,y,x数组。
    @params:
        data: np.ndarray 数据，形状为[2n]
    @return:
        data_tpyx: np.ndarray 分为t,p,y,x的数据，形状为[n, 4]
    """
    res = np.zeros((data.shape[0] // 2, 4), dtype = "uint32") # [n, 4]
    xyp = data[::2] # [n]
    t = data[1::2] # [n]
    res[:, 0] = t # [n]
    res[:, 1] = extract(xyp, p_mask, p_shift) # [n]
    res[:, 2] = extract(xyp, y_mask, y_shift) # [n]
    res[:, 3] = extract(xyp, x_mask, x_shift) # [n]
    return res


def tpyx_2_data(data: np.ndarray, p_shift = 0, y_shift = 16, x_shift = 1) -> np.ndarray:
    """
    将t,p,y,x数组合并回数据。
    @params:
        data_tpyx: np.ndarray 分为t,p,y,x的数据，形状为[n, 4]
    @return:
        data: np.ndarray 数据，形状为[2n]
    """
    res = np.zeros((2 * data.shape[0], 2), dtype = "uint32")
    res[1::2] = data[:, 0]
    res[::2] = (data[:, 1] << p_shift) + (data[:, 2] << y_shift) + (data[:, 3] << x_shift)
    return res


def data_to_tensor(event_data: np.ndarray, T: int, H: int = 128, W: int = 128):
    res = torch.zeros(T, 2, H, W, dtype = torch.float)
    event_data = event_data.astype("float32")
    t_min = np.min(event_data[:, 0])
    t_max = np.max(event_data[:, 0])
    event_data[:, 0] = (event_data[:, 0] - t_min) / (t_max - t_min) * (T - 1)
    event_data = event_data.astype("int32")
    res[event_data[:, 0], event_data[:, 1], event_data[:, 2], event_data[:, 3]] = 1
    return res


class MyData(Dataset):
    labels = ("airplane",)
    def __init__(self, root: str, T: int) -> None:
        self.root = root # 根目录
        self.t = T
        self.file_names = []
        self.data_labels = []
        for label in self.labels:
            sub_dir = os.path.join(self.root, label)
            file_list = os.listdir(sub_dir)
            self.file_names += [os.path.join(sub_dir, file_name) for file_name in file_list]
            self.data_labels += [self.labels.index(label) for file_name in file_list]
    

    def __len__(self) -> int:
        return len(self.data_labels)
    

    def __getitem__(self, index) -> Any:
        data_file_name = self.file_names[index]
        raw_data = filename_2_data(data_file_name) # [2n]
        event_data = data_2_tpyx(raw_data) # [n, 4]
        data = data_to_tensor(event_data, self.t)
        label = self.data_labels[index]
        return data, label


class RNNInput(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 2) # [T, B, L]
        return x


class RNNHidden(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, f: torch.nn.Module = torch.nn.ReLU()):
        super().__init__()
        U = torch.Tensor(out_features, in_features)
        torch.nn.init.normal_(U)
        self.U = torch.nn.Parameter(U)
        W = torch.Tensor(out_features, out_features)
        torch.nn.init.normal_(W)
        self.W = torch.nn.Parameter(W)
        self.f = f
    

    def forward(self, x: torch.Tensor):
        x = x.permute(1, 0, 2) # [B, T, I] -> [T, B, I]
        s_seq = []
        for t in range(x.shape[0]):
            x_t = x[t] # [B, I]
            s1 = torch.nn.functional.linear(x_t, self.U) # [B, O]
            if t == 0:
                s_t = torch.zeros_like(s1) # [B, O]
            s2 = torch.nn.functional.linear(s_t, self.W) # [B, O]
            s_t = self.f(s1 + s2) # [B, O]
            s_seq.append(s_t)
        s = torch.stack(s_seq) # [T, B, O]
        s = s.permute(1, 0, 2) # [T, B, O] -> [B, T, O]
        return s


class RNNOutput(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, g: torch.nn.Module = torch.nn.ReLU()) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(in_features, out_features, bias = False)
        self.g = g
    

    def forward(self, s: torch.Tensor):
        s = s.permute(1, 0, 2) # [B, T, I] -> [T, B, I]
        o_seq = []
        for t in range(s.shape[0]):
            s_t = s[t] # [B, I]
            o_t = self.g(self.fc(s_t)) # [B, O]
            o_seq.append(o_t)
        o = torch.stack(o_seq) # [T, B, O]
        o = o.permute(1, 0, 2) # [T, B, O] -> [B, T, O]
        return o


if __name__ == "__main__":
    data = MyData(
        root = "./data",
        T = 16
    )
    loader = DataLoader(
        dataset = data,
        batch_size = 16,
        shuffle = True
    )

    model = torch.nn.Sequential(
        RNNInput(),
        RNNHidden(2 * 128 * 128, 800),
        RNNOutput(800, 10)
    )

    for x, y in loader:
        o = model(x)[:, -1]
        print(o)