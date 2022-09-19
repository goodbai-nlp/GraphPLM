# coding:utf-8
import sys
import json

with open(sys.argv[1], 'r', encoding="utf-8") as fin:
    # for idx, line in enumerate(fin):
    #     json_data = json.loads(line.strip())
    #     print(idx, end=',')
    print(fin.readlines())
