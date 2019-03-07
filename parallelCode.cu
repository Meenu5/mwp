#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <cuda.h>

void initialize(int *menacc, int *womenacc, int *menpre, int *womenlock, int n) {
    int i;
    for(i=0; i<=n; i++) {
        menacc[i] = -1;
        womenacc[i] = -1;
        menpre[i] = 1;
        womenlock[i] = 0;
    }
}

__global__ void stable_matching(int n, int *d_men, int *d_women,
        int *d_menacc, int *d_womenacc, int *d_menpre, int *d_matched, int *d_womenlock) {
    int j = threadIdx.x + 1, idx;
    idx = d_men[j*(n+1) + d_menpre[j]];
    if(j <= n && d_menacc[j] == -1) {
        *d_matched = 0;
        // locking mechanism
        bool isSet = false;
        do {
            if(isSet = atomicCAS(&d_womenlock[idx], 0, 1) == 0) {
                if(d_womenacc[idx] == -1) {
                    d_womenacc[idx] = j;
                    d_menacc[j] = idx;
                }
                else if(d_women[idx*(n+1) + d_womenacc[idx]] > d_women[idx*(n+1) + j]) {
                    d_menacc[d_womenacc[idx]] = -1;
                    d_menacc[j] = idx;
                    d_womenacc[idx] = j;
                }
            }
            if(isSet) {
                atomicCAS(&d_womenlock[idx], 1, 0);
            }
        } while(!isSet);
        d_menpre[j]++;
    }
}

// driver function to utilize CUDA dynamic parallelism
__global__ void driver_function(int n, int *d_men, int *d_women,
        int *d_menacc, int *d_womenacc, int *d_menpre, int *d_matched, int *d_womenlock) {
    *d_matched = 0;
    while(!(*d_matched)) {
        *d_matched = 1;
        stable_matching <<< 1, n >>>(n, d_men, d_women, d_menacc, d_womenacc, d_menpre, d_matched, d_womenlock);
        cudaDeviceSynchronize();
    }
}


int main()
{
    int n,i,j,k;
    int *d_matched;
    int *men, *women;
    int *menacc, *womenacc, *menpre, *womenlock;
    int *d_men, *d_women;
    int *d_menacc, *d_womenacc, *d_menpre, *d_womenlock;
    clock_t beg, end;
    double time_taken;

    scanf("%d",&n);
    men = (int *) malloc((n+1)*(n+1)*sizeof(int));
    women = (int *) malloc((n+1)*(n+1)*sizeof(int));
    menacc = (int *) malloc((n+1)*sizeof(int));
    womenacc = (int *) malloc((n+1)*sizeof(int));
    womenlock = (int *) malloc((n+1)*sizeof(int));
    menpre = (int *) malloc((n+1)*sizeof(int));

    cudaMalloc(&d_men, (n+1)*(n+1)*sizeof(int));
    cudaMalloc(&d_women, (n+1)*(n+1)*sizeof(int));
    cudaMalloc(&d_menacc, (n+1)*sizeof(int));
    cudaMalloc(&d_womenacc, (n+1)*sizeof(int));
    cudaMalloc(&d_womenlock, (n+1)*sizeof(int));
    cudaMalloc(&d_menpre, (n+1)*sizeof(int));
    cudaMalloc(&d_matched, sizeof(int));

    initialize(menacc, womenacc, menpre, womenlock, n);

    beg = clock();
    for(i=1; i<=n; i++) {
        for(j=0; j<=n; j++) {
            scanf("%d", &men[i*(n+1) + j]);
        }
    }

    for(i=1; i<=n; i++) {
        for(j=0; j<=n; j++) {
            scanf("%d", &k);
            women[i*(n+1) + k] = j;
        }
    }
    end = clock();
    time_taken = ((double)(end-beg) * 1000000)/CLOCKS_PER_SEC;
    printf("read time : %f us, ", time_taken);

    cudaMemcpy(d_men, men, (n+1)*(n+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_women, women, (n+1)*(n+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_menacc, menacc, (n+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_womenlock, womenlock, (n+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_womenacc, womenacc, (n+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_menpre, menpre, (n+1)*sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    cudaEventRecord(start,0);

    // uncomment this part to use kernel-2

    // int matched = 0;
    // while(!matched) {
    //     matched = 1;
    //     cudaMemcpy(d_matched, &matched, sizeof(int), cudaMemcpyHostToDevice);
    //     stable_matching <<< 1, n >>>(n, d_men, d_women, d_menacc, d_womenacc, d_menpre, d_matched, d_womenlock);
    //     cudaMemcpy(&matched, d_matched, sizeof(int), cudaMemcpyDeviceToHost);
    // }

    // kernel-3 implementation

    driver_function <<< 1, 1 >>>(n, d_men, d_women, d_menacc, d_womenacc, d_menpre, d_matched, d_womenlock);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(menacc, d_menacc, (n+1)*sizeof(int), cudaMemcpyDeviceToHost);
    printf("compute time : %f us\n", milliseconds*1000);

    for(j=1;j<=n;j++)
        printf("%d %d\n", j, menacc[j]);

    free(men); free(women);
    free(menacc); free(womenacc); free(menpre); free(womenlock);
    cudaFree(&d_men); cudaFree(&d_women); cudaFree(&d_matched);
    cudaFree(&d_menacc); cudaFree(&d_womenacc); cudaFree(&d_menpre); cudaFree(&d_womenlock);

    return 0;
}