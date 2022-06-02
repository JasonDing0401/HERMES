# import argparse
import os
import pickle
from issue_linker import *
from predict import *

# parser = argparse.ArgumentParser()
# parser.add_argument('--name', type=str, required=True)
# args = parser.parse_args()

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

    tfidf_vectorizer = TfidfVectorizer()
    if min_df != 1:
        tfidf_vectorizer.min_df = min_df
    if max_df != 1:
        tfidf_vectorizer.max_df = max_df

    score_lines = calculate_similarity_scores(records, jira_tickets, tfidf_vectorizer, using_code_terms_only)
    return score_lines

# sim_scores_file = score_lines
def write_dataset_with_enhanced_issue(records, score_lines, limit=-1):
    jira_tickets = load_jira_tickets(testing=False)

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

        # calculate precision, recall for joint-model
        joint_precision, joint_recall, joint_f1 = None, None, None

        if options.use_stacking_ensemble:
            ensemble_classifier = weight_to_joint_classifier[positive_weight]
            y_pred, joint_precision, joint_recall, joint_f1, joint_auc_roc, joint_auc_pr, false_positive_joint_records, false_negative_joint_records, output_lines \
                = measure_joint_model_using_logistic_regression(ensemble_classifier,
                                                                test_data=test_data,
                                                                log_message_test_predict_prob=log_message_test_predict_prob,
                                                                id_to_issue_test_predict_prob=id_to_issue_test_predict_prob,
                                                                patch_test_predict_prob=patch_test_predict_prob,
                                                                options=options)
        else:
            y_pred, joint_precision, joint_recall, joint_f1, joint_auc_roc, joint_auc_pr \
                = measure_joint_model(log_message_prediction, issue_prediction,
                                        patch_prediction, log_message_test_predict_prob, patch_test_predict_prob, retrieve_label(test_data), options)

    # Write data and label to file
    pred_list = []
    assert len(test_data) == len(y_pred)
    with open(file_path, "r") as f:
        pred_list = json.load(f)
    assert len(test_data) == len(pred_list)
    for i in range(len(test_data)):
        data = pred_list[i]
        assert data['id'] == test_data[i].id
        data['label'] = int(y_pred[i])
        pred_list[i] = data

    return pred_list

def main():
    # format record
    records = data_loader.load_records(os.path.join(directory, 'prediction/php/full_dataset_with_all_features.txt'))
    for record in records:
        assert record.label == None
        # arbitrarily assign it to 0
        record.label = 0
    # issue recover if not linked
    # predict label
    pass