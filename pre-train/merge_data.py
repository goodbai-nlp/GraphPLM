# coding:utf-8
import sys
import os

res = []
for itm in sys.argv[1:]:
    tmp = open(itm, 'r', encoding='utf-8').readlines()
    tmp[-1] = tmp[-1] + '\n'
    res.extend(tmp)

with open("merged.jsonl", 'w', encoding='utf-8') as fout:
    fout.write(''.join(res)+'\n')