import os
import json
import pickle
from tkinter import E
from issue_linker import *
from entities import EntityEncoder, GithubIssue

def extract_issue_code_terms(issue):
    terms = []

    for comment in issue.comments:
        terms.extend(retrieve_code_terms(comment.body))

    return " ".join(terms)

def extract_issue_text_terms_parts(issue, limit_feature):
    terms_parts = []

    if not limit_feature:
        for comment in issue.comments:
            text_term = extract_text(comment.body)
            if len(text_term) > 0:
                terms_parts.extend(text_term)

    return terms_parts

def load_github_issues(file_names):
    print("Start loading crawled github issues...")
    github_issues = []

    # file_names = utils.read_lines(os.path.join(directory, 'github_issue_corpus_names.txt'))
    id_count = -1
    file_names = ["php-php-src_issue_corpus.txt"]
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

def prepare_for_calculation(records, using_code_terms_only=False):
    print("Preparing files for calculating similarity.")
    record_codes = []
    record_texts = []
    record_index = -1
    record_code_id = [-1]
    for record in records:
        record_codes.append(record.code_terms)
        if not using_code_terms_only:
            for text_terms in record.text_terms_parts:
                record_index += 1
                record_texts.append(text_terms)
            record_code_id.append(record_index)
    record_code_id = np.array(record_code_id)

    corpus = record_codes + record_texts
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer = tfidf_vectorizer.fit(corpus)
    record_code_matrix = tfidf_vectorizer.transform(record_codes)
    record_text_matrix = tfidf_vectorizer.transform(record_texts)
    with open("commit_linker_data/record_code_ind.npy", "wb+") as f:
        np.save(f, np.array(record_code_id))
    with open("commit_linker_data/tfidf_vectorizer.joblib", "wb+") as f:
        dump(tfidf_vectorizer, f)
    with open("commit_linker_data/record_code_mat.npz", "wb+") as f:
        save_npz(f, record_code_matrix)
    with open("commit_linker_data/record_text_mat.npz", "wb+") as f:
        save_npz(f, record_text_matrix)
    print("Finished.")

def calculate_similarity_scores(records, github_issues, record_code_id,\
    record_code_matrix, record_text_matrix, tfidf_vectorizer, using_code_terms_only=False):
    
    issue_corpus = []
    for issue in github_issues:
        issue_corpus.append(issue.code_terms)
        if not using_code_terms_only:
            for text_terms in issue.text_terms_parts:
                issue_corpus.append(text_terms)
    issue_matrix = tfidf_vectorizer.transform(issue_corpus)

    score_lines = []
    issue_count = 0
    for issue in github_issues:
        issue_count += 1
        best_record = None
        
        score_matrix_code = cosine_similarity(issue_matrix[0, :], record_code_matrix, dense_output=False)
        max_score_code = csr_matrix.max(score_matrix_code)

        if issue_matrix.shape[0] >= 2:
            score_matrix_text = cosine_similarity(issue_matrix[1:, :], record_text_matrix, dense_output=False)
            max_score_text = csr_matrix.max(score_matrix_text)
        else:
            max_score_text = 0.0

        if max_score_code != 0.0 or max_score_text != 0.0:
            if max_score_code >= max_score_text:
                max_ind = csr_matrix.argmax(score_matrix_code) % record_code_matrix.shape[0]
                best_record = records[max_ind]
            else:
                max_ind = csr_matrix.argmax(score_matrix_text) % record_text_matrix.shape[0]
                record_ind = np.where(record_code_id < max_ind)[0][-1]
                best_record = records[record_ind]

        if best_record is not None:
            score_lines.append(str(issue.id) + '\t\t' + best_record.repo + '/commit/' + best_record.commit_id + '\t\t'
                               + str(best_record.id)
                               + '\t\t' + str(max(max_score_code, max_score_text)))
        else:
            score_lines.append(
                str(issue.id) + '\t\t' + 'None' + '/commit/' + 'None' + '\t\t' + 'None'
                + '\t\t' + '0')

        if issue_count % 50 == 0:
            print("Finished {} issues".format(issue_count))
    # TODO: the problem with issues is that it might take long time to transform
    return score_lines

# sim_scores_file = score_lines
def write_dataset_with_enhanced_issue(issues, records, score_lines, thresh=0.5, limit=-1):
    id_to_issue = {}
    for issue in issues:
        id_to_issue[int(issue.id)] = issue

    scores = []
    for line in score_lines:
        parts = line.split("\t\t")
        issue_id = int(parts[0])
        if parts[2] == 'None':
            continue
        record_id = int(parts[2])
        score = float(parts[3])
        # ticket_key = parts[4]
        if score >= thresh:
            scores.append((issue_id, record_id, score))

    # print("Sorting scores...")
    scores.sort(key=lambda x: x[2], reverse=True)
    # print("Finish sorting scores")

    count = 0
    for issue_id, record_id, score in scores:
        issue = id_to_issue[issue_id]
        try:
            record = records[record_id]
            dict = {
                "commit_id": record.commit_id, 
                "event": "link_recovery",
                "label": record.label
            }
            issue.linked_commits.append(dict)
        except IndexError:
            print(record_id)
        count += 1
        if limit != -1 and count == limit:
            break
        issue = id_to_issue[issue_id]

    return issues

def main():
    records = data_loader.load_records(os.path.join(directory,
    '/space2/ding/dl-vulnerability-detection/scripts/snorkel/repo_data/php-full_dataset_with_all_features-limited.txt'))
    print("length of the records is:", len(records))
    for record in records:
        record.code_terms = extract_commit_code_terms(record)
        record.text_terms_parts = extract_commit_text_terms_parts(record)
    # If first time running, please run the following function.
    # Otherwise, comment it out.
    prepare_for_calculation(records)
    
    issues = load_github_issues(["php-php-src_issue_corpus.txt"])
    print("length of the issues is:", len(issues))
    issue_linked = []
    cnt = 0
    
    print("Start loading files from commit_linker_data/")
    with open("commit_linker_data/record_code_ind.npy", "rb") as f:
        record_code_id = np.load(f, allow_pickle=True)
    with open("commit_linker_data/record_code_mat.npz", "rb") as f:
        record_code_matrix = load_npz(f)
    with open("commit_linker_data/record_text_mat.npz", "rb") as f:
        record_text_matrix = load_npz(f)
    with open("commit_linker_data/tfidf_vectorizer.joblib", "rb") as f:
        tfidf_vectorizer = load(f)
    print("Finish loading files from commit_linker_data/")
    in_the_list = 0
    for issue in issues:
        if len(issue.linked_commits) != 0:
            issue.code_terms = extract_issue_code_terms(issue)
            issue.text_terms_parts = extract_issue_text_terms_parts(issue, False)
            score_lines = calculate_similarity_scores(records, [issue], record_code_id,\
                record_code_matrix, record_text_matrix, tfidf_vectorizer)
            print(score_lines[0])
            issue = write_dataset_with_enhanced_issue([issue], records, score_lines)[0]
            issue.__dict__.pop("code_terms")
            issue.__dict__.pop("text_terms_parts")
            commit_id_lst = []
            linked_one = ""
            for c in issue.linked_commits:
                if c["event"] == "link_recovery":
                    linked_one = c["commit_id"]
                else:
                    commit_id_lst.append(c["commit_id"])
            if linked_one != "":
                cnt += 1
            if linked_one in commit_id_lst:
                in_the_list += 1
                print("found one")
        issue.__dict__.pop("id")
        issue_linked.append(issue)
    
    print("The number of issues with linked commits is:", str(cnt))
    print("The number of correct linkage is:", str(in_the_list))
    entity_encoder = EntityEncoder()
    issues_with_record = entity_encoder.encode(issue_linked)
    with open(os.path.join(directory, "enhanced_issue_corpus/php.txt"), "w+") as f:
        f.write(issues_with_record)

def store_records():
    combined_records = []
    for repo in ["php", "redis", "v8", "image"]:
        with open("prediction/"+repo+"/full_dataset_with_all_features.txt", "r") as f:
            combined_records += json.load(f)
    with open("prediction/four_repo_commits.txt", "w+") as f:
        f.write(json.dumps(combined_records))

if __name__ == '__main__':
    main()