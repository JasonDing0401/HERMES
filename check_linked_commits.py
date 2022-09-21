import json
import os
import pickle
import pydriller
from pydriller import Repository
from entities import *
from issue_linker import *

def commit_to_data(commit, i, repo_link):
    return {
        "id": str(i),
        "repo": repo_link,
        "commit_id": commit.hash,
        "commit_message": commit.msg,
        "label": -1,
        "jira_ticket_list": [],
        "github_issue_list": [],
        "commit": {
            "author_name": commit.committer.name,
            "created_date": commit.committer_date.strftime("%a, %d %b %Y %H:%M:%S %z").strip(),
            "files": [{
                "file_name": file.old_path,
                "patch": file.diff,
                "status": file.change_type.name.lower(),
                "additions": file.added_lines,
                "deletions": file.deleted_lines,
                "changes": file.added_lines + file.deleted_lines
            } for file in commit.modified_files]
        }
    }

def calculate_similarity_scores(records, issue_id, issue_code_id,\
    issue_code_matrix, issue_text_matrix, tfidf_vectorizer, using_code_terms_only=False):
    
    record_corpus = []
    for record in records:
        record_corpus.append(record.code_terms)
        if not using_code_terms_only:
            for text_terms in record.text_terms_parts:
                record_corpus.append(text_terms)
    record_matrix = tfidf_vectorizer.transform(record_corpus)

    for record in records:   
        score_matrix_code = cosine_similarity(record_matrix[0, :], issue_code_matrix[issue_id, :], dense_output=False)
        max_score_code = csr_matrix.max(score_matrix_code)

        if record_matrix.shape[0] >= 2:
            start = issue_code_id[issue_id-1] + 1
            end = issue_code_id[issue_id] + 1
            if start == end:
                max_score_text = 0.0
            else:    
                score_matrix_text = cosine_similarity(record_matrix[1:, :], issue_text_matrix[start:end, :], dense_output=False)
                max_score_text = csr_matrix.max(score_matrix_text)
        else:
            max_score_text = 0.0

    return max(max_score_code, max_score_text)

def main():
    repo_path = "/home/ubuntu/dlvp/dl-vulnerability-detection/scripts/snorkel/repos/redis"
    with open("github_issues/redis-github_issue_corpus_processed.txt", "rb") as f:
        github_issues = pickle.load(f)
    print("Start loading files from commit_linker_data/")
    with open("commit_linker_data/issue_code_ind.npy", "rb") as f:
        issue_code_id = np.load(f, allow_pickle=True)
    with open("commit_linker_data/issue_code_mat.npz", "rb") as f:
        issue_code_matrix = load_npz(f)
    with open("commit_linker_data/issue_text_mat.npz", "rb") as f:
        issue_text_matrix = load_npz(f)
    with open("commit_linker_data/tfidf_vectorizer.joblib", "rb") as f:
        tfidf_vectorizer = load(f)
    print("Finish loading files from commit_linker_data/")
    with open("tmp-redis-thresh-0.8.txt", "r") as f:
        for line in f:
            parts = line.split("\t\t")
            record_id = int(parts[0])
            issue_id = int(parts[2])
            score = float(parts[3])
            print(record_id, issue_id, score)
            issue = github_issues[issue_id]
            # commit_id_lst = []
            for c in issue.linked_commits:
                # commit_id_lst.append(c["commit_id"])
                try:
                    repo = list(Repository(repo_path, single=c["commit_id"]).traverse_commits())[0]
                except ValueError:
                    continue
                # repo to hermes format
                data = commit_to_data(repo, 0, "https://github.com/redis/redis")
                record = Record(json_value=json.dumps(data))
                record.code_terms = extract_commit_code_terms(record)
                record.text_terms_parts = extract_commit_text_terms_parts(record)
                score = calculate_similarity_scores([record], issue_id,\
                issue_code_id, issue_code_matrix, issue_text_matrix, tfidf_vectorizer)
                print(score)

if __name__ == '__main__':
    main()