import pickle
import numpy as np

with open(r'C:\Users\evan8\OneDrive\桌面\SemiAgent_v3\SemiAgent\data\wafer\raw\LSWMD.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')

print(type(data))
print(data.shape if hasattr(data, 'shape') else len(data))
