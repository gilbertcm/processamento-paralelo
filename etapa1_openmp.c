/* 
 * Projeto 1 - Filtro Gaussiano com OpenMP
 * Disciplina: DEC107 - Processamento Paralelo
 * Alunos: Gilbert Carmo e Henio Pedro

 * gcc -O2 -fopenmp etapa1_openmp.c -o etapa1
 * ./etapa1
*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// tamanhos de imagem a testar
#define TAM1 512
#define TAM2 1024
#define TAM3 2048

// numero de vezes que o filtro e aplicado
#define ITERACOES 10

// kernel gaussiano 3x3 (soma = 16, entao divide por 16)
float kernel3[3][3] = {
    {1, 2, 1},
    {2, 4, 2},
    {1, 2, 1}
};

// kernel gaussiano 5x5 (soma = 256, entao divide por 256)
float kernel5[5][5] = {
    {1,  4,  6,  4,  1},
    {4, 16, 24, 16,  4},
    {6, 24, 36, 24,  6},
    {4, 16, 24, 16,  4},
    {1,  4,  6,  4,  1}
};

// aloca uma matriz de floats
float** alocar(int linhas, int colunas) {
    float** mat = (float**)malloc(linhas * sizeof(float*));
    for (int i = 0; i < linhas; i++) {
        mat[i] = (float*)malloc(colunas * sizeof(float));
    }
    return mat;
}

// preenche a matriz com valores aleatorios (simula imagem em tons de cinza)
void preencher(float** mat, int linhas, int colunas) {
    for (int i = 0; i < linhas; i++) {
        for (int j = 0; j < colunas; j++) {
            mat[i][j] = rand() % 256;
        }
    }
}

// libera a memoria da matriz
void liberar(float** mat, int linhas) {
    for (int i = 0; i < linhas; i++) {
        free(mat[i]);
    }
    free(mat);
}

// aplica o kernel 3x3 em um pixel (ignora borda de 1 pixel)
float aplicar3x3(float** entrada, int linha, int coluna) {
    float soma = 0.0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            soma += entrada[linha + i][coluna + j] * kernel3[i + 1][j + 1];
        }
    }
    return soma / 16.0;
}

// aplica o kernel 5x5 em um pixel (ignora borda de 2 pixels)
float aplicar5x5(float** entrada, int linha, int coluna) {
    float soma = 0.0;
    for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= 2; j++) {
            soma += entrada[linha + i][coluna + j] * kernel5[i + 2][j + 2];
        }
    }
    return soma / 256.0;
}

// versao serial com kernel 3x3
void filtro_serial_3x3(float** entrada, float** saida, int tam) {
    for (int it = 0; it < ITERACOES; it++) {
        for (int i = 1; i < tam - 1; i++) {
            for (int j = 1; j < tam - 1; j++) {
                saida[i][j] = aplicar3x3(entrada, i, j);
            }
        }
        // troca entrada e saida para proxima iteracao
        float** temp = entrada;
        entrada = saida;
        saida = temp;
    }
}

// versao serial com kernel 5x5
void filtro_serial_5x5(float** entrada, float** saida, int tam) {
    for (int it = 0; it < ITERACOES; it++) {
        for (int i = 2; i < tam - 2; i++) {
            for (int j = 2; j < tam - 2; j++) {
                saida[i][j] = aplicar5x5(entrada, i, j);
            }
        }
        float** temp = entrada;
        entrada = saida;
        saida = temp;
    }
}

// versao paralela com kernel 3x3
void filtro_paralelo_3x3(float** entrada, float** saida, int tam, int num_threads) {
    omp_set_num_threads(num_threads);
    for (int it = 0; it < ITERACOES; it++) {
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < tam - 1; i++) {
            for (int j = 1; j < tam - 1; j++) {
                saida[i][j] = aplicar3x3(entrada, i, j);
            }
        }
        float** temp = entrada;
        entrada = saida;
        saida = temp;
    }
}

// versao paralela com kernel 5x5
void filtro_paralelo_5x5(float** entrada, float** saida, int tam, int num_threads) {
    omp_set_num_threads(num_threads);
    for (int it = 0; it < ITERACOES; it++) {
        #pragma omp parallel for collapse(2)
        for (int i = 2; i < tam - 2; i++) {
            for (int j = 2; j < tam - 2; j++) {
                saida[i][j] = aplicar5x5(entrada, i, j);
            }
        }
        float** temp = entrada;
        entrada = saida;
        saida = temp;
    }
}

// roda os testes para um tamanho de imagem e um kernel
void rodar_teste(int tam, int kernel) {
    float** entrada = alocar(tam, tam);
    float** saida   = alocar(tam, tam);

    printf("\n--- Imagem %dx%d | Kernel %dx%d ---\n", tam, tam, kernel, kernel);

    // tempo serial
    preencher(entrada, tam, tam);
    double inicio = omp_get_wtime();
    if (kernel == 3)
        filtro_serial_3x3(entrada, saida, tam);
    else
        filtro_serial_5x5(entrada, saida, tam);
    double tempo_serial = omp_get_wtime() - inicio;
    printf("Serial:         %.4f s\n", tempo_serial);

    // teste com 2, 4 e 8 threads
    int configs[] = {2, 4, 8};
    for (int c = 0; c < 3; c++) {
        int threads = configs[c];
        preencher(entrada, tam, tam);
        inicio = omp_get_wtime();
        if (kernel == 3)
            filtro_paralelo_3x3(entrada, saida, tam, threads);
        else
            filtro_paralelo_5x5(entrada, saida, tam, threads);
        double tempo_paralelo = omp_get_wtime() - inicio;
        double speedup = tempo_serial / tempo_paralelo;
        double eficiencia = (speedup / threads) * 100.0;
        printf("%d threads:     %.4f s | Speedup: %.2f | Eficiencia: %.1f%%\n",
               threads, tempo_paralelo, speedup, eficiencia);
    }

    liberar(entrada, tam);
    liberar(saida, tam);
}

int main() {
    printf("=== Benchmark Filtro Gaussiano Iterativo ===\n");
    printf("Iteracoes por rodada: %d\n", ITERACOES);

    int tamanhos[] = {TAM1, TAM2, TAM3};
    int kernels[]  = {3, 5};

    for (int t = 0; t < 3; t++) {
        for (int k = 0; k < 2; k++) {
            rodar_teste(tamanhos[t], kernels[k]);
        }
    }

    printf("\n=== Fim ===\n");
    return 0;
}
