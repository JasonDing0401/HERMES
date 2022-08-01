from github import Github
from github import RateLimitExceededException
import os
import json
# from ratelimit import limits
from datetime import datetime, timezone
import time

# @limits(calls=5000, period=3600)
def collect_issue(gh_pat, repo_name):
    # Closed issues only
    gh = Github(gh_pat)
    repo = gh.get_repo(repo_name)
    total_issues = repo.get_issues(state="closed")
    issue_list = []
    repo_name = repo_name.replace("/", "-")

    print("length of issues:", total_issues.totalCount)
    cnt = 0
    for issue in total_issues:
        try:
            cnt += 1
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
            issue_list.append(issue_data)
            if cnt % 50 == 0:
                print("Already collected:", cnt)
            if cnt > 10000:
                print("Number of issues exceeds 10000. Will collect the next repo.")
                break
        except AttributeError:
            continue
        except RateLimitExceededException:
            print("Number of issues collected before encountering error:", cnt)
            limits = gh.get_rate_limit()
            reset = limits.core.reset.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            seconds = (reset - now).total_seconds() + 5
            print("Rate limit exceeded.")
            print(f"Reset is in {seconds:.3g} seconds.")
            if seconds > 0.0:
                print(f"Waiting for {seconds:.3g} seconds...")
                time.sleep(seconds)
                print("Done waiting - resume!")
    
    with open("issue_corpus_new/" + repo_name + "_issue_corpus.txt", "w+") as f:
        f.write(json.dumps(issue_list))
    
    with open("github_issue_corpus_names.txt", "a") as f:
        f.write(repo_name + "_issue_corpus.txt\n")

if __name__ == '__main__':
    # repo_list = ["php/php-src", "redis/redis", "v8/v8", "ImageMagick/ImageMagick", "openssl/openssl"]
    # repos = ["tensorflow/tensorflow", "electron/electron", "bitcoin/bitcoin", "opencv/opencv", "apple/swift", "netdata/netdata", "pytorch/pytorch", "protocolbuffers/protobuf"]
    # ["godotengine/godot", "tesseract-ocr/tesseract", "git/git", "ocornut/imgui", "obsproject/obs-studio", "grpc/grpc", "FFmpeg/FFmpeg"]
    # ["topjohnwu/Magisk", "aria2/aria2", "curl/curl", "rethinkdb/rethinkdb", "tmux/tmux", "ClickHouse/ClickHouse", "dmlc/xgboost", "facebook/rocksdb"]
    # ["emscripten-core/emscripten", "facebook/folly", "mongodb/mongo", "ApolloAuto/apollo", "yuzu-emu/yuzu", "SerenityOS/serenity"]
    collect_issue("123", "mongodb/mongo")