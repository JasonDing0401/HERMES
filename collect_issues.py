from github import Github
from github import RateLimitExceededException
import os
import json
# from ratelimit import limits
from datetime import datetime, timezone
import time

def rate_limit_exceed(gh, cnt):
    print("Number of issues collected before encountering error:", cnt)
    limits = gh.get_rate_limit()
    if limits.search.remaining == 0:
        limited = limits.search
    elif limits.graphql.remaining == 0:
        limited = limits.graphql
    else:
        limited = limits.core
    reset = limited.reset.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    seconds = (reset - now).total_seconds() + 30
    print("Rate limit exceeded.")
    print(f"Reset is in {seconds:.3g} seconds.")
    if seconds > 0.0:
        print(f"Waiting for {seconds:.3g} seconds...")
        time.sleep(seconds)
        print("Done waiting - resume!")

# @limits(calls=5000, period=3600)
def collect_issue(gh_pat, repo_name):
    # Closed issues only
    cnt = 0
    num = 0
    gh = Github(gh_pat)
    issue_list = []
    while True:
        try:
            repo = gh.get_repo(repo_name)
            break
        except RateLimitExceededException:
            rate_limit_exceed(gh, cnt)
    while True:
        try:
            total_issues = repo.get_issues(state="closed")
            total_count = total_issues.totalCount
            break
        except RateLimitExceededException:
            rate_limit_exceed(gh, cnt)
    print("length of issues:", total_count)
    repo_name = repo_name.replace("/", "-")

    for cnt in range(5000, total_count):
        while True:
            try:
                issue = total_issues[cnt]
                try:
                    issue_data = {
                        "title": issue.title,
                        "body": issue.body,
                        "author_name": issue.user.name,
                        "created_at": issue.created_at.strftime("%a, %d %b %Y %H:%M:%S %z").strip(),
                        "closed_at": issue.closed_at.strftime("%a, %d %b %Y %H:%M:%S %z").strip(),
                        "closed_by": issue.closed_by.name,
                        "last_modified": issue.last_modified,
                        "comments": [],
                        "linked_commits": [],
                        "labels": []
                    }
                    for comment in issue.get_comments():
                        issue_data["comments"].append({
                            "body": comment.body,
                            "created_at": comment.created_at.strftime("%a, %d %b %Y %H:%M:%S %z").strip(),
                            "created_by": comment.user.name,
                            "last_modified": comment.last_modified
                        })
                except AttributeError:
                    break

                for event in issue.get_events():
                    try:
                        if event.commit_id:
                            issue_data["linked_commits"].append({
                                "commit_id": event.commit_id,
                                "event": event.event
                            })
                    except AttributeError:
                        continue
                for label in issue.get_labels():
                    try:
                        issue_data["labels"].append({
                            "name": label.name,
                            "description": label.description
                        })
                    except AttributeError:
                        continue
                issue_list.append(issue_data)
                
                if (cnt+1) % 50 == 0:
                    print("Already collected:", (cnt+1))
                if (cnt+1) % 5000 == 0:
                    print("Number of issues exceeds " + str((cnt+1)) + ". Will store in separate files.")
                    num = (cnt+1) // 5000
                    with open("issue_corpus_new/" + repo_name + "_issue_corpus_" + str(num) + ".txt", "w+") as f:
                        f.write(json.dumps(issue_list))
                    with open("github_issue_corpus_names.txt", "a") as f:
                        f.write(repo_name + "_issue_corpus_" + str(num) + ".txt\n")
                    issue_list = []
                break
            except RateLimitExceededException:
                rate_limit_exceed(gh, (cnt+1))
    
    if num == 0:
        with open("issue_corpus_new/" + repo_name + "_issue_corpus.txt", "w+") as f:
            f.write(json.dumps(issue_list))
        with open("github_issue_corpus_names.txt", "a") as f:
            f.write(repo_name + "_issue_corpus.txt\n")
    else:
        if issue_list:
            num += 1
            with open("issue_corpus_new/" + repo_name + "_issue_corpus_" + str(num) + ".txt", "w+") as f:
                f.write(json.dumps(issue_list))
            with open("github_issue_corpus_names.txt", "a") as f:
                f.write(repo_name + "_issue_corpus_" + str(num) + ".txt\n")


if __name__ == '__main__':
    # ["php/php-src", "redis/redis", "v8/v8", "ImageMagick/ImageMagick", "openssl/openssl"]
    # ["tensorflow/tensorflow", "electron/electron", "bitcoin/bitcoin", "opencv/opencv", "apple/swift", "netdata/netdata", "pytorch/pytorch", "protocolbuffers/protobuf"]
    # ["godotengine/godot", "tesseract-ocr/tesseract", "git/git", "ocornut/imgui", "obsproject/obs-studio", "grpc/grpc", "FFmpeg/FFmpeg"]
    # ["topjohnwu/Magisk", "aria2/aria2", "curl/curl", "rethinkdb/rethinkdb", "tmux/tmux", "ClickHouse/ClickHouse", "dmlc/xgboost", "facebook/rocksdb"]
    # ["emscripten-core/emscripten", "facebook/folly", "mongodb/mongo", "ApolloAuto/apollo", "yuzu-emu/yuzu", "SerenityOS/serenity"]
    # ["apache/incubator-mxnet", "envoyproxy/envoy", "libuv/libuv", "taichi-dev/taichi", "telegramdesktop/tdesktop"]
    # ["mpv-player/mpv", "fish-shell/fish-shell", "osquery/osquery", "taosdata/TDengine"]
    for repo in ["mpv-player/mpv", "fish-shell/fish-shell", "osquery/osquery", "taosdata/TDengine"]:
        collect_issue("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", repo)