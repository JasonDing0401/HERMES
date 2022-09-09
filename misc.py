import os
import json
import utils
# with open("sim_scores/redis_sim_scores.txt", "r") as f:
#     lst = f.readlines()
# string = ""
# for l in lst:
#     l = l[2:-3].split("\\t\\t")
#     string += "\t\t".join(l) + "\n"
# with open("sim_scores/redis_sim_scores.txt", "w") as f:
#     f.write(string)
# lst = []
# with open("./MSR2019/experiment/sub_enhanced_dataset_th_100.txt", "r") as f:
#     lst = json.load(f)
# with open("./prediction/php/php_enhanced_dataset_th_100.txt", "r") as f:
#     lst += json.load(f)
# with open("./prediction/hermes+php/enhanced_dataset_th_100.txt", "w+") as f:
#     json.dump(lst, f)
# with open("prediction/v8/full_dataset_with_all_features.json", "r") as f:
#     lst = json.load(f)
# with open("prediction/v8/full_dataset_with_all_features.txt", "w+") as f:
#     json.dump(lst, f)

# commits_pool = []
# commit_result_dict = {}
# for name in ["ImageMagick", "php-src", "redis", "v8"]:
#     with open("../dl-vulnerability-detection/data/" + name + "_hermes.json", "r") as f:
#         lst = json.load(f)
#     for d in lst:
#         commits_pool.append(d["commit_id"])
#         commit_result_dict[d["commit_id"]] = d["cls"]
# for name in ["php-src"]:
#     with open("../dl-vulnerability-detection/data/" + name + "_hermes_2.json", "r") as f:
#         lst = json.load(f)
#     for d in lst:
#         commits_pool.append(d["commit_id"])
#         commit_result_dict[d["commit_id"]] = d["cls"]
# with open("../dl-vulnerability-detection/data/100_c_cpp_projects.json", "r") as f:
#         lst = json.load(f)
# for d in lst:
#     if d["commit_id"] in commits_pool:
#         continue
#     commits_pool.append(d["commit_id"])
#     commit_result_dict[d["commit_id"]] = d["label"]

# referenced_lst = []
# cnt = 0
# # linked = 0
# # total = 0
# file_names = utils.read_lines('github_issue_corpus_names.txt')
# all_issues = []
# for file_name in file_names:
#     with open("issue_corpus_new/" + file_name) as file:
#         json_raw = file.read()
#         json_dict_list = json.loads(json_raw)
#         for json_dict in json_dict_list:
#             linked_commits_lst = json_dict["linked_commits"]
#             lst = []
#             result = ""
#             for commit_dict in linked_commits_lst:
#                 commit_id = commit_dict["commit_id"]
#                 if commit_id in commits_pool:
#                     label = commit_result_dict[commit_id]
#                     commit_dict["label"] = label
#                     lst.append(commit_dict)
#                     if result != "" and label != result:
#                         # raise Exception("Conflict labels with one issue!")
#                         result = "conflict"
#                         continue
#                     result = label
#                 else:
#                     commit_dict["label"] = "unk"
#                     lst.append(commit_dict)
#             json_dict["linked_commits"] = lst
#             if result != "":
#                 json_dict["label"] = result
#             else:
#                 json_dict["label"] = "unk"
#             all_issues.append(json_dict)

# print("total issues:", len(all_issues))
# pos, neg, conf, unk = 0, 0, 0, 0
# for issue in all_issues:
#     if issue["label"] == "pos":
#         pos += 1
#     elif issue["label"] == "neg":
#         neg += 1
#     elif issue["label"] == "conflict":
#         conf += 1
#     else:
#         unk += 1
# print(pos, neg, conf, unk)

# with open("four_issue_repos_with_labels.txt", "w+") as f:
#     f.write(json.dumps(all_issues))
# print(linked, total)
# print(cnt, len(referenced_lst))
# pos = 0
# neg = 0
# for i in referenced_lst:
#     if commit_result_dict[i] == "pos":
#         pos += 1
#     if commit_result_dict[i] == "neg":
#         neg += 1
# print(pos, neg)

cnt = 0
have = 0
file_names = utils.read_lines('github_issue_corpus_names.txt')
all_issues = []
for file_name in file_names:
    with open("issue_corpus_new/" + file_name) as file:
        json_raw = file.read()
        json_dict_list = json.loads(json_raw)
        for json_dict in json_dict_list:
            cnt += 1
            if json_dict["linked_commits"]:
                have += 1
print("Total number of issues:", str(cnt))
print(str(have))