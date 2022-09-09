import os
from entities import EntityEncoder, GithubIssue
from issue_linker import *
directory = os.path.dirname(os.path.abspath(__file__))

def load_github_issues():
    print("Start loading crawled github issues...")
    github_issues = []

    file_names = utils.read_lines(os.path.join(directory, 'github_issue_corpus_names.txt'))
    id_count = -1
    for file_name in file_names:
        with open("issue_corpus_new/" + file_name) as file:
            json_raw = file.read()
            json_dict_list = json.loads(json_raw)
            for json_dict in json_dict_list:
                if json_dict is not None and json_dict != 'null':
                    id_count += 1
                    ticket = GithubIssue(json_value=json.dumps(json_dict))
                    ticket.id = id_count
                    github_issues.append(ticket)
    print("Finished loading crawled github issues")
    return github_issues