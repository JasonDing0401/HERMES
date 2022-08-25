import os
import json
import pickle
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

def load_github_issues():
    print("Start loading crawled github issues...")
    github_issues = []

    file_names = utils.read_lines(os.path.join(directory, 'github_issue_corpus_names.txt'))
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

def calculate_similarity_scores(records, github_issues, tfidf_vectorizer, using_code_terms_only):
    # TODO: save and load record corpus instead as it will get very big
    record_codes = []
    record_texts = []
    record_index = -1
    record_code_id = []
    for record in records:
        record_codes.append(record.code_terms)
        if not using_code_terms_only:
            for text_terms in record.text_terms_parts:
                record_index += 1
                record_texts.append(text_terms)
            record_code_id.append(record_index)
            # add an index counter here to get record index
            # have another array to store record text terms
    record_code_id = np.array(record_code_id)

    # print("Calculating TF-IDF vectorizer...")
    # print("record matrix shape is:", record_matrix.shape)
    
    issue_corpus = []
    for issue in github_issues:
        issue_corpus.append(issue.code_terms)
        if not using_code_terms_only:
            for text_terms in issue.text_terms_parts:
                issue_corpus.append(text_terms)
    corpus = record_codes + record_texts + issue_corpus
    tfidf_vectorizer = tfidf_vectorizer.fit(corpus)
    record_code_matrix = tfidf_vectorizer.transform(record_codes)
    record_text_matrix = tfidf_vectorizer.transform(record_texts)
    issue_matrix = tfidf_vectorizer.transform(issue_corpus)
    # print("issue matrix shape is:", issue_matrix.shape)
    # print("Finish calculating TF-IDF vectorizer")

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
                record_ind = np.where(record_code_id <= max_ind)[0][-1]
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

# input: record -- single record with no issue linked
# records -- [record] used in for loop
def process_linking(issues, records, testing=False, min_df=1, using_code_terms_only=False, limit_feature=False, 
                    text_feature_min_length=0, output_file_name='texts.txt', chunk=-1, 
                    relevant_ticket=False, test_true_link=False, merge_link=False, max_df=1):
    global terms_min_length
    terms_min_length = text_feature_min_length

    global similarity_scores_file_path
    similarity_scores_file_path = os.path.join(directory, output_file_name)

    global chunk_size
    chunk_size = chunk

    global use_relevant_ticket
    use_relevant_ticket = relevant_ticket

    # print("length of the commits is:", len(records))
    # for record in records:
    #     record.code_terms = extract_commit_code_terms(record)
    #     if not using_code_terms_only:
    #         record.text_terms_parts = extract_commit_text_terms_parts(record)

    # github_issues = load_github_issues()
    # print("Issues length: {}".format(len(github_issues)))
    # print("Start extracting issue features...")
    for issue in issues:
        issue.code_terms = extract_issue_code_terms(issue)
        if not using_code_terms_only:
            issue.text_terms_parts = extract_issue_text_terms_parts(issue, limit_feature)
    # print("Finish extracting issue features")
    
    tfidf_vectorizer = TfidfVectorizer()
    if min_df != 1:
        tfidf_vectorizer.min_df = min_df
    if max_df != 1:
        tfidf_vectorizer.max_df = max_df

    score_lines = calculate_similarity_scores(records, issues, tfidf_vectorizer, using_code_terms_only)
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
    records = data_loader.load_records(os.path.join(directory, 'prediction/php/full_dataset_with_all_features.txt'))
    for record in records:
        record.code_terms = extract_commit_code_terms(record)
        record.text_terms_parts = extract_commit_text_terms_parts(record)
    issues = load_github_issues()
    print("length of the issues is:", len(issues))
    issue_linked = []
    cnt = 0
    for issue in issues:
        if len(issue.linked_commits) == 0:
            cnt += 1
            score_lines = process_linking([issue], records)
            print(score_lines[0])
            issue = write_dataset_with_enhanced_issue([issue], records, score_lines)[0]
            issue.__dict__.pop("code_terms")
            issue.__dict__.pop("text_terms_parts")
        issue.__dict__.pop("id")
        issue_linked.append(issue)
        if cnt == 30:
            break

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