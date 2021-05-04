import os
import matplotlib.pyplot as plt

def F1(prec, rec):
    return 2*(prec*rec)/(prec+rec)

your_path = './'
files = os.listdir(your_path)
lines = {}

for file in files:
    if file.endswith(".txt"):
        if "25" in file:
            continue
        id = (int(file.strip().replace("images.txt","")))
        with open(file) as f:
            lines[id] = [line.rstrip() for line in f]

# print(lines)

file_dics = {}
# print(lines)
plt_dics = {}
lines = dict(sorted(lines.items()))

for id,file in lines.items():
    plt_dics[id] = [[],[]]
    for i,line in enumerate(file):
        # print(line)
        if "iou_THRESH" in line:
            print(line.replace("iou_THRESH ", ""))
            thresh = float(line.replace("iou_THRESH ", ""))
            plt_dics[id][0].append(thresh)

            splited1 = file[i+2].split(": ")
            splited2 = file[i+3].split(": ")

            plt_dics[id][1].append(F1(float(splited1[1]), float(splited2[1])))


                # file_dics[id] = {
                #     splited1[0]: float(splited1[1]),
                #     splited2[0]: float(splited2[1]),
                #     "F1": F1(float(splited1[1]), float(splited2[1]))
                # }
print(plt_dics)
# print(file_dics)
# prints = ""
# file_dics = dict(sorted(file_dics.items()))
# for id, file in file_dics.items():
#     # print(id, file)
#     prints = "& " + str(round(file['F1'], 3)) +" "+prints
# print(prints)



from scipy.ndimage.filters import gaussian_filter1d

# plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
for i,plt_dic in plt_dics.items():
    print(i)
    ysmoothed = gaussian_filter1d(plt_dic[1], sigma=4)
    plt.plot(plt_dic[0], ysmoothed, label=str(i)+" imgs")



plt.xlabel("iou_threshold")
plt.ylabel("F1-Score")
plt.legend()
plt.show()

