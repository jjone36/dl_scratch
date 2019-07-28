import numpy as np

fp = open('../data/yolov3.weights', 'rb')
header = np.fromfile(fp, dtype = np.int32, count = 5)
weights = np.fromfile(fp, dtype = np.float32)
print(header)
print('======')
print(weights)

# self.header = torch.from_numpy(header)
# self.seen = self.header[3]
