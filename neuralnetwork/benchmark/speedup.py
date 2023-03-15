import matplotlib.pyplot as plt
import subprocess
#import pickle

avgs = {}

tests = ['10', '25', '50']

# sequential

sum = 0
for test in tests:
    for _ in range(5):
        result = subprocess.run(['go', 'run', 'proj3/editor', test, 's'], stdout=subprocess.PIPE)
        seconds = float(result.stdout.decode('utf-8'))
        sum += seconds
    avgs[test + ' ' + 's'] = sum / 5

# Parallel

threadcounts = ['2','4','6','8','12']
parallelversions = ['ws', 'wb']

for threads in threadcounts:
    for test in tests:
        for parallelversion in parallelversions:
            sum = 0
            for _ in range(5): # run each test 5 times
                result = subprocess.run(['go', 'run', 'proj3/editor', test, parallelversion, threads], stdout=subprocess.PIPE)
                try:
                    seconds = float(result.stdout.decode('utf-8'))
                except:
                    pass
                sum += seconds
            avgs[test + ' ' + threads + ' ' + parallelversion] = sum / 5

# pickling the avgs so we don't have to run the tests every time

# with open('speedup.pickle', 'wb') as handle:
#     pickle.dump(avgs, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('speedup.pickle', 'rb') as handle:
#     avgs = pickle.load(handle)

# graphing and saving

plt.title('Editor Speedup Graph (Work Stealing)')
plt.plot([2, 4, 6, 8, 12], [avgs['10 s'] / x for x in [avgs['10 2 ws'], avgs['10 4 ws'], avgs['10 6 ws'], avgs['10 8 ws'], avgs['10 12 ws']]], 'b', label='10 epochs')
plt.plot([2, 4, 6, 8, 12], [avgs['20 s'] / x for x in [avgs['20 2 ws'], avgs['20 4 ws'], avgs['20 6 ws'], avgs['20 8 ws'], avgs['20 12 ws']]], 'g', label='20 epochs')
plt.plot([2, 4, 6, 8, 12], [avgs['30 s'] / x for x in [avgs['30 2 ws'], avgs['30 4 ws'], avgs['30 6 ws'], avgs['30 8 ws'], avgs['30 12 ws']]], 'r', label='30 epochs')
plt.legend(loc='lower right')
plt.ylabel('Speedup')
plt.xlabel('# Threads')
plt.savefig('speedup-ws')
#plt.show()
plt.clf()

plt.title('Editor Speedup Graph (Work Balancing)')
plt.plot([2, 4, 6, 8, 12], [avgs['10 s'] / x for x in [avgs['10 2 wb'], avgs['10 4 wb'], avgs['10 6 wb'], avgs['10 8 wb'], avgs['10 12 wb']]], 'b', label='10 epochs')
plt.plot([2, 4, 6, 8, 12], [avgs['20 s'] / x for x in [avgs['20 2 wb'], avgs['20 4 wb'], avgs['20 6 wb'], avgs['20 8 wb'], avgs['20 12 wb']]], 'g', label='20 epochs')
plt.plot([2, 4, 6, 8, 12], [avgs['30 s'] / x for x in [avgs['30 2 wb'], avgs['30 4 wb'], avgs['30 6 wb'], avgs['30 8 wb'], avgs['30 12 wb']]], 'r', label='30 epochs')
plt.legend(loc='lower right')
plt.ylabel('Speedup')
plt.xlabel('# Threads')
plt.savefig('speedup-wb')
#plt.show()