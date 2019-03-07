def formatData(path, outpath):
    choice = 0
    tot_residents = 0
    tot_hosp = 0
    residents = [[]]
    hospitals = [[]]
    f = open(path, "r")
    f_w = open(outpath, "w", buffering=0)

    for line in f.readlines():
        line = line.strip()
        if(len(line) == 0 or line == '@End'):
            continue
        if(line == '@PartitionA'):
            choice = 1
            continue
        if(line == '@PartitionB'):
            choice = 2
            continue
        if(line == '@PreferenceListsA'):
            choice = 3
            continue
        if(line == '@PreferenceListsB'):
            choice = 4
            continue

        if(choice == 1):
            line_trim = line.replace(' ', '')[:-1]
            line_split = line_trim.split(',')
            tot_residents = len(line_split)
            for i in range(tot_residents):
                residents.append([])

        if(choice == 2):
            line_trim = line.replace(' ', '')[:-1]
            line_split = line_trim.split(',')
            tot_hosp = len(line_split)
            for i in range(tot_hosp):
                hospitals.append([])

        if(choice == 3):
            line_trim = line.replace(' ', '')[:-1]
            temp_split = line_trim.split(':')
            res = temp_split[0]
            r_ind = int(res[1:])
            pref_list = temp_split[1].split(',')
            for h in pref_list:
                h_ind = h[1:]
                residents[r_ind].append(h_ind)

        if(choice == 4):
            line_trim = line.replace(' ', '')[:-1]
            temp_split = line_trim.split(':')
            hosp = temp_split[0]
            h_ind = int(hosp[1:])
            pref_list = temp_split[1].split(',')
            for r in pref_list:
                if(r != ''):
                    r_ind = r[1:]
                    hospitals[h_ind].append(r_ind)

    f_w.write(str(tot_residents) + '\n')
    for i in range(1,tot_residents+1):
        s = str(len(residents[i])) + ' '
        for h in residents[i]:
            s += h + ' '
        f_w.write(s + '\n')

    for i in range(1,tot_hosp+1):
        s = str(len(residents[i])) + ' '
        for r in hospitals[i]:
            s += r + ' '
        f_w.write(s + '\n')

    f.close()
    f_w.close()

folders = ['master', 'shuffle', 'random']
# sizes = ['10', '50', '100', '200', '500']
sizes = ['1000']
for folder in folders:
    for size in sizes:
        for i in range(1,6):
            fin = 'raw_data/complete/' + size + '_' + size + '/' + folder + '/'
            fout = 'data/complete/' + size + '_' + size + '/' + folder + '/'
            fin += size + '_' + size + '_' +  str(i) + '.txt'
            fout += size + '_' + size + '_' +  str(i) + '.txt'
            print(fin)
            formatData(fin, fout)