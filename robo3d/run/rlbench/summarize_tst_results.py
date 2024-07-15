
import os
import numpy as np
import jsonlines
import collections
import tap


class Arguments(tap.Tap):
    result_file: str
    show_variation: bool = False


def main(args):
    results = collections.defaultdict(dict)
    taskvar_results = collections.defaultdict(list)
    with jsonlines.open(args.result_file, 'r') as f:
        for item in f:
            results[item['checkpoint']].setdefault(item['task'], [])
            results[item['checkpoint']][item['task']].append(item['sr'])
            taskvar_results[item['checkpoint']].append((item['task'], item['variation'], item['sr']))

    ckpt_task_results = collections.defaultdict(list)
    for ckpt, res in results.items():
        for task, v in res.items():
            ckpt_task_results[ckpt].append((task, np.mean(v)))

    print('\nnum_tasks', len(ckpt_task_results[ckpt]))
    for ckpt, res in ckpt_task_results.items():
        print()
        print(ckpt)
        res.sort(key=lambda x: x[0])

        print(','.join([x[0] for x in res]))

        print(','.join(['%.2f' % (x[1]*100) for x in res]))

        print('#tasks: %d, average: %.2f' % (len(res), np.mean([x[1] for x in res]) * 100))

        if args.show_variation:
            taskvar_results[ckpt].sort(key=lambda x: (x[0], x[1]))
            #print('\n'.join(['%s+%d %.2f'%(x[0], x[1], x[2]*100) for x in taskvar_results[ckpt]]))
            print('#taskvars', len(taskvar_results[ckpt]))
            print(','.join(['%s+%d'%(x[0], x[1]) for x in taskvar_results[ckpt]]))
            print(','.join(['%.2f'%(x[2]*100) for x in taskvar_results[ckpt]]))
            print()


if __name__ == '__main__':
    args = Arguments().parse_args()
    main(args)
