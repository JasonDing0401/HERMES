import os

with open("texts_bad_format.txt", "r") as f:
    lst = f.readlines()
string = ""
for l in lst:
    l = l[2:-3].split("\\t\\t")
    string += "\t\t".join(l) + "\n"
with open("texts.txt", "w") as f:
    f.write(string)
