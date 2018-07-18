import os
path="/home/saikantareddy/Downloads/raccoon_dataset-master/annotations/"
with open("trainval.txt","w") as f:
	for i in os.listdir(path):
		f.write(path+i.split(".")[0])
		f.write("\n")
	f.close()
