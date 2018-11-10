import json
import os
import re


path = "E:\__Programming\__hackathon\Madhack_2018\dress"

files = os.listdir(path)

result = {}

for file in files:
    raw_data = open(path+"/"+file, "r").read()
    #print(raw_data)
    findObj = re.findall(r'ProductID\":\"(\d*)\"', raw_data, re.M|re.I)
    if not findObj:
        print(file + "NOOOOOOO")
        exit()
    else:
        for id in findObj:
            if id in result:
                if file[:-6] in result[id]:
                    continue
                result[id].append(file[:-6])
                #print(id)
                #print(result[id])
            else:
                result[id] = []
                result[id].append(file[:-6])

output = open("dress_label.json", "w");
output.write(json.dumps(result))
