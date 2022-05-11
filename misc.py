import os
import json
# with open("texts_bad_format.txt", "r") as f:
#     lst = f.readlines()
# string = ""
# for l in lst:
#     l = l[2:-3].split("\\t\\t")
#     string += "\t\t".join(l) + "\n"
# with open("texts.txt", "w") as f:
#     f.write(string)
lst = []
with open("./MSR2019/experiment/sub_enhanced_dataset_th_100.txt", "r") as f:
    lst = json.load(f)
with open("./prediction/php/php_enhanced_dataset_th_100.txt", "r") as f:
    lst += json.load(f)
with open("./prediction/hermes+php/enhanced_dataset_th_100.txt", "w+") as f:
    json.dump(lst, f)
