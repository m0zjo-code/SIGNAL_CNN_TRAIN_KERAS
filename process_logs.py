import os, csv

in_dir = "2DCNN/"
ls = os.listdir(in_dir)
res = [k for k in ls if 'results' in k]

with open('ProcessedLogs.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(["Opt", "CNN_Layers", "Dense_Layers", "Arch", "Loss", "Acc"])
    for i in res:
        F = open(in_dir+i,'r')
        name = i.split("_")
        dat = F.readlines()[1]
        dat = dat.split("""%""")[0]
        dat = dat.split(",")
        dat[1] = dat[1].replace(" ", "")
        print(name)
        spamwriter.writerow([name[1], name[2], name[3], "%s_%s"%(name[2], name[3])]+dat)
