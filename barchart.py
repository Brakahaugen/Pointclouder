with open("bartchart_data2.txt") as file_in:
    lines = []
    for line in file_in:
        line = line.replace(" ", "")
        if len(line) < 5:
            continue
        lines.append(line.split("|")[1:-2])
        # lines.append(.split("|"))
    print(lines)

entries = []
for l in lines[1:]:
    entries.append([float(i) for i in l])
print("entries", entries)

bboxes = []
append = False
for l in entries:
    if append:
        append=False
        continue
    append=True
    bboxes.append(l)
print(bboxes)

segm = []
append = True
for l in entries:
    if append:
        append=False
        continue
    append=True
    segm.append(l)
print(segm)

bboxes = segm

vals = [[],[],[]]
for i in range(3):
    for l in bboxes:
        vals[i].append(l[i])
print(vals)

import numpy as np
import matplotlib.pyplot as plt
  
N = 6
ind = np.arange(N) 
width = 0.25
  
xvals = vals[1]
print("xvals", xvals)
bar1 = plt.bar(ind, xvals, width, color = 'salmon')
  
yvals = vals[2]
bar2 = plt.bar(ind+width, yvals, width, color='gold')
  
zvals = vals[0]
bar3 = plt.bar(ind+width*2, zvals, width, color = 'dodgerblue')


plt.xlabel("Number of Unique Trees")
plt.ylabel('AP Scores')
plt.title("")
  
plt.xticks(ind+width,['5', '10', '20', '50', '100', '200'])
plt.legend( (bar1, bar2, bar3), ('AP50', 'AP75', 'mAP') )
plt.show()