# Matching With Preference

This is the git repo for the matching with preference project for gpu programming.

---------------------------------------------------------------------------------

Compilation Instructions (with dynamic parallelism):

nvcc -arch=sm_35 -rdc=true  parallelCode.cu -lcudadevrt -o parallel


To run the program with say data/complete/10_10/master/10_10_1.txt input file:

./parallel < data/complete/10_10/master/10_10_5.txt


Custom input file format:

First line contains 'n', the number of men/women

Next n lines correspond to men's preference list:

	Each line starts with 'm', number of women in the man's preference list,
	followed by m numbers denoting women in decreasing order of his preference

Next n lines correspond to women's preference list in a similar way
