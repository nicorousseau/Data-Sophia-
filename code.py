from netCDF4 import Dataset
import matplotlib.pyplot as plt 

ds = Dataset("tsi_v02r01-preliminary_daily_s20230101_e20230331_c20230411.nc")

#print(ds.variables.keys())

data1_var = ds.variables["TSI"]
time = ds.variables["time"]
all_data = data1_var[:]

print(time.variables.keys())
#plt.plot(all_data)
#plt.show()