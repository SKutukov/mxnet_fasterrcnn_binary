import subprocess
import numpy as np
import scipy.stats as st

filename = '/home/skutukov/work/CI/fasterrcnnVGG16/5/CI_{}/test.log'
mAPs = []
for i in range(10):
    test_filename = filename.format(i)
    cmd = ["grep \"Mean AP\"  {}".format(test_filename)]
    a = subprocess.check_output(cmd, shell=True)
    a = str(a).split('=')[1].strip().split('\\')[0]
    mAPs.append(float(a))

print(mAPs)
interval = st.t.interval(0.95, len(mAPs)-1, loc=np.mean(mAPs), scale=st.sem(mAPs))
print(round((interval[0] + interval[1])/2, 2), round((interval[1] - interval[0])/2, 3))

