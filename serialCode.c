#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

struct node {
  int val;
  struct node *next;
};

struct queue {
  struct node *front, *rear;
};

struct node* newNode(int n) {
  struct node* temp = (struct node*) malloc(sizeof(struct node));
  temp->val = n;
  temp->next = NULL;
  return temp;
}

struct queue* createQueue() {
  struct queue* q = (struct queue*) malloc(sizeof(struct queue));
  q->front = NULL;
  q->rear = NULL;
  return q;
}

void enQueue(struct queue *q, int n) {
  struct node* temp = newNode(n);
  if(q->rear == NULL) {
    q->front = q->rear = temp;
    return;
  }
  q->rear->next = temp;
  q->rear = temp;
}

struct node* deQueue(struct queue *q) {
  if(q->front == NULL) {
    return NULL;
  }
  struct node* temp = q->front;
  q->front = q->front->next;
  if(q->front == NULL) {
    q->rear = NULL;
  }
  return temp;
}

bool isEmpty(struct queue *q) {
  return (q->front == NULL);
}

void initialize(int *menacc, int *womenacc, int *menpre, int n)
{
  int i;
  for(i=0; i<=n; i++)
  {
    menacc[i]=-1;
    womenacc[i]=-1;
    menpre[i]=1;
  }
}

int main()
{
  int n,i,j,k,idx;
  int **men, **women;
  int *menacc, *womenacc, *menpre;
  clock_t start, end;
  double time_taken;

  start = clock();
  scanf("%d",&n);
  men = (int **) malloc((n+1)*sizeof(int*));
  women = (int **) malloc((n+1)*sizeof(int*));
  menacc = (int *) malloc((n+1)*sizeof(int));
  womenacc = (int *) malloc((n+1)*sizeof(int));
  menpre = (int *) malloc((n+1)*sizeof(int));

  for(i=0; i<=n; i++) {
    men[i] = (int *) malloc((n+1)*sizeof(int));
    women[i] = (int *) malloc((n+1)*sizeof(int));
  }
  initialize(menacc, womenacc, menpre, n);

  for(i=1;i<=n;i++) {
    for(j=0;j<=n;j++) {
      scanf("%d",&men[i][j]);
    }
  }
  for(i=1;i<=n;i++) {
    for(j=0;j<=n;j++) {
      scanf("%d",&k);
      women[i][k]=j;
    }
  }

  end = clock();
  time_taken = ((double)(end-start) * 1000000)/CLOCKS_PER_SEC;
  printf("read time : %f us, ", time_taken);

  struct queue* q = createQueue();
  for(i=1; i<=n; i++) {
    enQueue(q, i);
  }

  start = clock();
  int ct=0;
  while(!isEmpty(q)) {
    ct++;
    struct node* cur = deQueue(q);
    j = cur->val;
    free(cur);

    // j is the index of man, menpre contains the next preference to be checked 
    // idx is the index of woman he proposes
    idx = men[j][menpre[j]];
    if(womenacc[idx] == -1) {
      womenacc[idx] = j;
      menacc[j] = idx;
    }
    else if(women[idx][womenacc[idx]] > women[idx][j]) {
      menacc[womenacc[idx]] = -1;
      menacc[j] = idx;
      if(menpre[womenacc[idx]] <= n)
        enQueue(q, womenacc[idx]);
      womenacc[idx] = j;
    }
    else if(menpre[j] <= n) {
      enQueue(q, j);
    }
    menpre[j]++;
  }

  end = clock();
  time_taken = ((double)(end-start) * 1000000)/CLOCKS_PER_SEC;
  printf("compute time : %f us\n", time_taken);
  printf("count : %d\n", ct);

  for(j=1;j<=n;j++)
    printf("%d %d\n", j, menacc[j]);

  free(q);
  for(i=0; i<=n; i++) {
    free(men[i]); free(women[i]);
  }
  free(men); free(women);
  free(menacc); free(womenacc); free(menpre);

  return 0;
}