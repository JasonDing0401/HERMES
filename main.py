# import argparse
import os
import requests
import sys
import re
import pickle
import json
import time
import warnings
from github import Github
from pydriller import Git
from issue_linker import *
from predict import *
from entities import EntityEncoder

# parser = argparse.ArgumentParser()
# parser.add_argument('--name', type=str, required=True)
# args = parser.parse_args()

def collect_data(gh_pat, folder, commit_id):
    if folder.endswith("/"):
        folder = folder[:-1]

    if not os.path.isdir(folder):
        print(f"Error: could not find folder {folder}")
        sys.exit(1)

    if not os.path.isdir(f"{folder}/.git/"):
        print(f"Error: {folder} does not contain a git repository")
        sys.exit(1)

    # Don't push it to github
    gh = Github(gh_pat)

    git = Git(f"{folder}/.git/")
    repo_name = git.repo.remotes[0].url.replace("git@github.com:", "").replace(".git", "")
    if repo_name.endswith("/"):
        repo_name = repo_name[:-1]
    repo = gh.get_repo(repo_name)

    commit = git.get_commit(commit_id)
    data = {
        "id": 0,
        "repo": git.repo.remotes[0].url,
        "commit_id": commit_id,
        "commit_message": commit.msg,
        "label": None,
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

    if "#" in data["commit_message"]:
        pattern = r'\W(GH\-\d+|#\d+)'
        issues = re.findall(pattern, data["commit_message"])

        for potential_issue in issues:
            try:
                issue_num = int(potential_issue.replace("#", "").replace("GH-", ""))
                r = requests.get(f"{git.repo.remotes[0].url}/issues/{issue_num}")
                if r.status_code == 404:
                    continue

                issue = repo.get_issue(issue_num)

                issue_data = {
                    "title": issue.title,
                    "body": issue.body,
                    "author_name": issue.user.name,
                    "created_at": issue.created_at.strftime("%a, %d %b %Y %H:%M:%S %z").strip(),
                    "closed_at": issue.closed_at.strftime("%a, %d %b %Y %H:%M:%S %z").strip(),
                    "closed_by": issue.closed_by.name,
                    "last_modified": issue.last_modified,
                    "comments": []
                }

                for comment in issue.get_comments():
                    issue_data["comments"].append({
                        "body": comment.body,
                        "created_at": comment.created_at.strftime("%a, %d %b %Y %H:%M:%S %z").strip(),
                        "created_by": comment.user.name,
                        "last_modified": comment.last_modified
                    })

                data["github_issue_list"].append(issue_data)
            except Exception as e:
                print(f"note: {data['commit_id']} had a potential GitHub issue {potential_issue}, which failed to load.")
                print(e)
    return [data]

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

    with open("issue_corpus_processed.txt", "rb") as f:
        jira_tickets = pickle.load(f)

    tfidf_vectorizer = TfidfVectorizer(max_features=500)
    if min_df != 1:
        tfidf_vectorizer.min_df = min_df
    if max_df != 1:
        tfidf_vectorizer.max_df = max_df

    score_lines = calculate_similarity_scores(records, jira_tickets, tfidf_vectorizer, using_code_terms_only)
    return score_lines

# sim_scores_file = score_lines
def write_dataset_with_enhanced_issue(records, score_lines, limit=-1):
    with open("issue_corpus_processed.txt", "rb") as f:
        jira_tickets = pickle.load(f)

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
        if score > 0:
            scores.append((record_id, ticket_id, score))

    print("Sorting scores...")
    scores.sort(key=lambda x: x[2], reverse=True)
    print("Finish sorting scores")

    count = 0
    for record_id, ticket_id, score in scores:
        record = id_to_record[record_id]
        try:
            ticket = jira_tickets[ticket_id]
            record.jira_ticket_list.append(ticket)
        except IndexError:
            print(ticket_id)
        count += 1
        if limit != -1 and count == limit:
            break
        record = id_to_record[record_id]

    return records

def measure_joint_model_using_logistic_regression(ensemble_classifier, test_data, 
                                                  log_message_test_predict_prob, id_to_issue_test_predict_prob,
                                                  patch_test_predict_prob, options):
    # issue_train_mean_probability = None
    issue_test_mean_probability = None

    if options.use_issue_classifier:
        # issue_train_mean_probability = np.mean([prob for id, prob in id_to_issue_train_predict_prob.items()])
        issue_test_mean_probability = np.mean([prob for id, prob in id_to_issue_test_predict_prob.items()])

    y_test = retrieve_label(test_data)
    X_test = []
    for index in range(len(test_data)):
        if options.use_issue_classifier:
            if test_data[index].id in id_to_issue_test_predict_prob:
                X_test.append(
                    [log_message_test_predict_prob[index],
                     id_to_issue_test_predict_prob[test_data[index].id],
                     patch_test_predict_prob[index]])
            else:
                # TODO: This might be a bug in the original file
                X_test.append(
                    [log_message_test_predict_prob[index],
                     issue_test_mean_probability,
                     patch_test_predict_prob[index]])
        else:
            X_test.append([log_message_test_predict_prob[index], patch_test_predict_prob[index]])

    # ensemble_classifier.fit(X=X_train, y=y_train)
    y_pred = ensemble_classifier.predict(X=X_test)
    # prob that the label is 1=vulnerability-related
    y_prob = ensemble_classifier.predict_proba(X=X_test)[:, 1]

    return y_pred, y_prob

def predict_label(records, size=-1, ignore_number=True, github_issue=True, jira_ticket=True, use_comments=True, positive_weights=[0.5], 
                    n_gram=1, min_df=5, use_linked_commits_only=False, use_issue_classifier=True, fold_to_run=1, use_stacking_ensemble=True,
                    tf_idf_threshold=0.005, use_patch_context_lines=False):
    # python predict.py --min_df 5 --use_linked_commits_only False --use_issue_classifier True 
    # --use_stacking_ensemble True --use-patch-context-lines False --tf-idf-threshold 0.005
    options = feature_options.read_option_from_command_line(size, 0, ignore_number,
                                                            github_issue, jira_ticket, use_comments,
                                                            positive_weights,
                                                            n_gram, min_df, use_linked_commits_only,
                                                            use_issue_classifier,
                                                            fold_to_run,
                                                            use_stacking_ensemble,
                                                            tf_idf_threshold,
                                                            use_patch_context_lines)

    print("loading vectorizer from model/")
    # TODO: Haven't moved these pretrained stuff to model/php/ yet
    with open("./model/commit_message_vectorizer.joblib", "rb") as f:
        commit_message_vectorizer = load(f)
    with open("./model/issue_vectorizer.joblib", "rb") as f:
        issue_vectorizer = load(f)
    with open("./model/patch_vectorizer.joblib", "rb") as f:
        patch_vectorizer = load(f)
    print("done")

    records = preprocess_data(records, options)

    weight_to_log_classifier = {}
    weight_to_patch_classifier = {}
    weight_to_issue_classifier = {}
    weight_to_joint_classifier = {}

    for positive_weight in positive_weights:
        print("loading model from model/ with pos weight {}".format(positive_weight))
        with open('./model/log_classifier_weight-{}.joblib'.format(positive_weight), "rb") as f:
            weight_to_log_classifier[positive_weight] = load(f)
        if options.use_issue_classifier:
            with open('./model/issue_classifier_weight-{}.joblib'.format(positive_weight), "rb") as f:
                weight_to_issue_classifier[positive_weight] = load(f)
        with open('./model/patch_classifier_weight-{}.joblib'.format(positive_weight), "rb") as f:
            weight_to_patch_classifier[positive_weight] = load(f)
        if options.use_stacking_ensemble:
            with open('./model/joint_classifier_weight-{}.joblib'.format(positive_weight), "rb") as f:
                weight_to_joint_classifier[positive_weight] = load(f)
        print("done")

    # train_data, test_data = retrieve_data(records, train_data_indices, test_data_indices)
    test_data = records
    # log_x_train, log_y_train = calculate_log_message_feature_vector(train_data, commit_message_vectorizer)
    log_x_test, log_y_test = calculate_log_message_feature_vector(test_data, commit_message_vectorizer)

    # issue_x_train, issue_y_train, issue_x_test, issue_y_test = None, None, None, None
    issue_x_test, issue_y_test = None, None

    if options.use_issue_classifier:
        # issue_x_train, issue_y_train = calculate_issue_feature_vector(train_data, issue_vectorizer)
        issue_x_test, issue_y_test = calculate_issue_feature_vector(test_data, issue_vectorizer)

    # patch_x_train, patch_y_train = calculate_patch_feature_vector(train_data, patch_vectorizer)
    patch_x_test, patch_y_test = calculate_patch_feature_vector(test_data, patch_vectorizer)

    for positive_weight in positive_weights:
        print("Current processing weight set ({},{})".format(positive_weight, 1 - positive_weight))
        log_classifier = weight_to_log_classifier[positive_weight]

        issue_classifier = None
        # id_to_issue_train_predict_prob = None
        id_to_issue_test_predict_prob = None
        if options.use_issue_classifier:
            issue_classifier = weight_to_issue_classifier[positive_weight]

        patch_classifier = weight_to_patch_classifier[positive_weight]

        # calculate precision, recall for log message classification
        log_classifier, precision, recall, f1, log_message_prediction, log_message_test_predict_prob, false_positives, false_negatives\
            = log_message_classify(log_classifier, log_x_test, log_y_test)

        # calculate precision, recall for issue classification
        precision, recall, f1, issue_prediction = None, None, None, None
        if options.use_issue_classifier:
            issue_classifier, precision, recall, f1, issue_prediction, id_to_issue_test_predict_prob, false_positives, false_negatives\
                = issue_classify(issue_classifier, issue_x_test, issue_y_test, test_data)

        patch_classifier, precision, recall, f1, patch_prediction, patch_test_predict_prob, false_positives, false_negatives\
            = patch_classify(patch_classifier, patch_x_test, patch_y_test)

        if options.use_stacking_ensemble:
            ensemble_classifier = weight_to_joint_classifier[positive_weight]
            y_pred, y_prob \
                = measure_joint_model_using_logistic_regression(ensemble_classifier,
                                                                test_data=test_data,
                                                                log_message_test_predict_prob=log_message_test_predict_prob,
                                                                id_to_issue_test_predict_prob=id_to_issue_test_predict_prob,
                                                                patch_test_predict_prob=patch_test_predict_prob,
                                                                options=options)
        else:
            raise Exception("Should use joint classifier")

    # Write data and label to file
    print("probability that it is vulnerability-related is:", y_prob[0])

    return int(y_pred[0])

def main():
    # format: [{...}] single record
    
    with open(os.path.join(directory, "tmp.txt"), "w") as f:
        f.write(json.dumps(data))
    # format record, now just assume single record
    records = data_loader.load_records(os.path.join(directory, 'tmp.txt'))
    record = records[0]
    assert record.label == None
    # arbitrarily assign it to 0
    record.label = 0
    records = [record]
    # issue recover if not linked
    if len(record.jira_ticket_list) == 0 and len(record.github_issue_list) == 0:
        score_lines = process_linking(records)
        print(score_lines)
        records = write_dataset_with_enhanced_issue(records, score_lines)
    entity_encoder = EntityEncoder()
    records_with_issue = entity_encoder.encode(records)
    with open(os.path.join(directory, "tmp.txt"), "w") as f:
        f.write(records_with_issue)
    # predict label
    label = predict_label(records)
    print("predicted label is:", label)
    record = records[0]
    record.label = label
    records = [record]
    json_value = entity_encoder.encode(records)
    with open(os.path.join(directory, "prediction.txt"), "w") as f:
        f.write(json_value)
    return records

if __name__ == '__main__':
    # TODO: Improvements:
    # 1. store ticket_id_to_corpus_id and corpus; 
    # 2. set threshold of similarity score and assign a above 0.5 one instead of the highest
    warnings.simplefilter("ignore")
    start_time = time.time()
    records = main()
    end_time = time.time()
    print("time spent is:", end_time-start_time)
    record = records[0]
    print(record)
    print("label is:", record.label)