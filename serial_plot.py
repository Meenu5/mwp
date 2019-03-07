import matplotlib.pyplot as plt

f = open('plot_data/parallel_dynamic_sync.txt', 'r')

x = [10, 50, 100, 200, 500, 1000]

master = [0.0 for i in range(6)];
shuffle = [0.0 for i in range(6)];
random = [0.0 for i in range(6)];

line_num = 0
cur_list = -1
cur_ind = -1
for line in f.readlines():
    if(len(line.strip()) == 0):
        continue
    if(line_num % 2 == 0):
        line_split = line.split('/')
        cur_ind = x.index(int(line_split[-1].split('_')[0]))
        if(line_split[-2] == 'master'):
            cur_list = 1
        elif(line_split[-2] == 'shuffle'):
            cur_list = 2
        else:
            cur_list = 3

    else:
        time = float(line.split(', ')[1].split(' ')[3])
        if(cur_list == 1):
            master[cur_ind] += time
        elif(cur_list == 2):
            shuffle[cur_ind] += time
        else:
            random[cur_ind] += time

    line_num += 1

for i in range(6):
    master[i] /= 5.0
    shuffle[i] /= 5.0
    random[i] /= 5.0

f.close()
print(master)
print(shuffle)
print(random)

plt.plot(x, master, '-*')
plt.plot(x, shuffle, '-*')
plt.plot(x, random, '-*')
plt.legend(['master', 'shuffle', 'random'])
plt.xlabel('n')
plt.ylabel('execution time (us)')
plt.show()