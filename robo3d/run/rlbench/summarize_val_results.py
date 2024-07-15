import os
import numpy as np
import jsonlines
import collections
import tap


class Arguments(tap.Tap):
    result_file: str


def main(args):
    results = collections.defaultdict(list)
    taskvar_set = set()
    with jsonlines.open(args.result_file, 'r') as f:
        for item in f:
            if (item['checkpoint'], item['task'], item['variation']) in taskvar_set:
                continue
            results[item['checkpoint']].append((item['task'], item['variation'], item['sr']))
            taskvar_set.add((item['checkpoint'], item['task'], item['variation']))

    ckpts = list(results.keys())
    ckpts.sort(key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))

    # show task results
    tasks = set()
    for ckpt in ckpts:
        for x in results[ckpt]:
            tasks.add(x[0])
    tasks = list(tasks)
    tasks.sort()
    for task in tasks:
        res = []
        for ckpt in ckpts:
            ckpt_res = []
            for x in results[ckpt]:
                if x[0] == task:
                    ckpt_res.append(x[-1])
            res.append(np.mean(ckpt_res))
        print('\n', task, len(ckpt_res))
        print(', '.join(['%.2f' % (x*100) for x in res]))
    print()

    avg_results = []
    for k in ckpts:
        v = results[k]
        sr = collections.defaultdict(list)
        for x in v:
            sr[x[0]].append(x[-1])
        sr = [np.mean(x) for x in sr.values()]
        print(k, len(v), np.mean(sr)*100)
        avg_results.append((k, np.mean(sr)))

    print()
    print('Best checkpoint and SR')
    avg_results.sort(key=lambda x: -x[1])
    for x in avg_results:
        if x[-1] < avg_results[0][-1]:
            break
        print((x[0], x[1]*100))
    print('\n')


if __name__ == '__main__':
    args = Arguments().parse_args()
    main(args)
