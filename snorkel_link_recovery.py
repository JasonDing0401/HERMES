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

# input: record -- single record with no issue linked
# records -- [record] used in for loop
def process_linking(records, testing=False, min_df=1, using_code_terms_only=False, limit_feature=False, 
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

    for record in records:
        record.code_terms = extract_commit_code_terms(record)
        if not using_code_terms_only:
            record.text_terms_parts = extract_commit_text_terms_parts(record)

    # github_issues = load_github_issues()
    # print("Issues length: {}".format(len(github_issues)))
    # print("Start extracting issue features...")
    # for issue in github_issues:
    #     issue.code_terms = extract_issue_code_terms(issue)
    #     if not using_code_terms_only:
    #         issue.text_terms_parts = extract_issue_text_terms_parts(issue, limit_feature)
    # print("Finish extracting issue features")

    # with open("github_issue_corpus_processed.txt", "wb+") as f:
    #     pickle.dump(github_issues, f)

    with open("github_issue_corpus_processed.txt", "rb") as f:
        github_issues = pickle.load(f)
    
    tfidf_vectorizer = TfidfVectorizer(max_features=500)
    if min_df != 1:
        tfidf_vectorizer.min_df = min_df
    if max_df != 1:
        tfidf_vectorizer.max_df = max_df

    score_lines = calculate_similarity_scores(records, github_issues, tfidf_vectorizer, using_code_terms_only)
    return score_lines

# sim_scores_file = score_lines
def write_dataset_with_enhanced_issue(records, score_lines, thresh=0.5, limit=-1):
    with open("github_issue_corpus_processed.txt", "rb") as f:
        github_issues = pickle.load(f)

    id_to_record = {}
    for record in records:
        id_to_record[int(record.id)] = record

    scores = []
    for line in score_lines:
        parts = line.split("\t\t")
        record_id = int(parts[0])
        if parts[2] == 'None':
            continue
        ticket_id = int(parts[2])
        score = float(parts[3])
        # ticket_key = parts[4]
        if score >= thresh:
            scores.append((record_id, ticket_id, score))

    print("Sorting scores...")
    scores.sort(key=lambda x: x[2], reverse=True)
    print("Finish sorting scores")

    count = 0
    for record_id, ticket_id, score in scores:
        record = id_to_record[record_id]
        try:
            ticket = github_issues[ticket_id]
            record.github_issue_list.append(ticket)
        except IndexError:
            print(ticket_id)
        count += 1
        if limit != -1 and count == limit:
            break
        record = id_to_record[record_id]

    return records

def main():
    records = data_loader.load_records(os.path.join(directory, 'prediction/v8/full_dataset_with_all_features.txt'))
    records_linked = []
    for record in records:
        if len(record.jira_ticket_list) == 0 and len(record.github_issue_list) == 0:
            score_lines = process_linking([record])
            print(score_lines)
            record = write_dataset_with_enhanced_issue([record], score_lines)[0]
        records_linked.append(record)
    entity_encoder = EntityEncoder()
    records_with_issue = entity_encoder.encode(records_linked)
    with open(os.path.join(directory, "prediction/v8/enhanced_dataset_with_issues.txt"), "w+") as f:
        f.write(records_with_issue)

if __name__ == '__main__':
    main()