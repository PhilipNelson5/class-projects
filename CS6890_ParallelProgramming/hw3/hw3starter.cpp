#include <iostream>
#include <mpi.h>
//#include "/usr/local/include/mpi.h"
#define MCW MPI_COMM_WORLD

using namespace std;

int ring(const char *m, int icf, int data){
    // TODO: write your code here
    /* ignore m */
}

void printRunning(const char* m, int running[64]){
    int size;
    MPI_Comm_size(MCW, &size); 
    cout<<m<<": ";
    for(int i=0;i<size;++i)cout<<running[i]<<" ";
    cout<<endl;
}
using namespace std;

int main(int argc, char **argv){

    int rank, size;
    int data;
    int running[64];
    const char *m="11111111";
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MCW, &rank); 
    MPI_Comm_size(MCW, &size); 

    data = ring(m,1,rank);
    MPI_Gather(&data,1,MPI_INT,running,1,MPI_INT,0,MCW);
    if(rank==0)printRunning(m,running);

    data = ring(m,-1,rank);
    MPI_Gather(&data,1,MPI_INT,running,1,MPI_INT,0,MCW);
    if(rank==0)printRunning(m,running);

    data = ring(m,-4,rank);
    MPI_Gather(&data,1,MPI_INT,running,1,MPI_INT,0,MCW);
    if(rank==0)printRunning(m,running);

    MPI_Finalize();

    return 0;
}


