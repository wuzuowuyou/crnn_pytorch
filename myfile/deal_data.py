import os

dir_data = "/data_1/everyday/xian/20210308_crnn/IIIT5K/"

data_txt = "/data_1/everyday/xian/20210308_crnn/IIIT5K/traindata.txt"

with open(data_txt,"r")as fr:
    context = fr.readlines()
    for line in context:
        line = line.strip()
        print(line)
        list_txt = line.split(",")
        path_part = list_txt[0]
        words = list_txt[1]
        path_img = dir_data + path_part
        with open(path_img.replace(".png",".txt"),"w")as fw:
            fw.write(words)
