import os

folders = ['master', 'shuffle', 'random']
sizes = ['10', '50', '100', '200', '500', '1000']

for folder in folders:
    for size in sizes:
        for i in range(1,6):
            f = 'data/complete/' + size + '_' + size + '/' + folder + '/'
            f += size + '_' + size + '_' +  str(i) + '.txt'
            print(f)
            os.system('./a.out < ' + f)