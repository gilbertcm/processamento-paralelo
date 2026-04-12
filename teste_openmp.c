#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main()
{
    int num_threads = 4;              // Número de threads a serem usadas
    omp_set_num_threads(num_threads); // Configura o número de threads

#pragma omp parallel
    {
        int thread_id = omp_get_thread_num(); // Obtém o ID da thread
        printf("Hello from thread %d\n", thread_id);
    }

    return 0;
}