/* Construção: Versão Serial, Versão paralela openmp, Medição de tempo, Base pronta pra relatorio*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h> //funções do OpenMP
#include <time.h>

#define N 1000  // Tamanho da matriz
#define M 1000  // Tamanho da matriz
#define ITER 10 // Número de iterações para a medição de tempo
// N x M - dimensão da imagem e número de vezes que o filtro será aplicado

// alocação da matriz
float **alocar_matriz(int linhas, int colunas)
{
    float **matriz = (float **)malloc(linhas * sizeof(float *));
    for (int i = 0; i < linhas; i++)
    {
        matriz[i] = (float *)malloc(colunas * sizeof(float));
    }
    return matriz;
}

// inicialização da matriz
void inicializar_matriz(float **matriz, int linhas, int colunas)
{
    for (int i = 0; i < linhas; i++)
    {
        for (int j = 0; j < colunas; j++)
        {
            matriz[i][j] = rand() % 256; // Valores entre 0 e 255 (simulando uma imagem em tons de cinza)
        }
    }
}

// Kernel Gaussiano
float kernel[3][3] = {
    {1, 2, 1},
    {2, 4, 2},
    {1, 2, 1}};

// função de convolução
float aplicar_kernel(float **input, int x, int y)
{
    float sum = 0.0;
    for (int i = -1; i <= 1; i++)
    {
        for (int j = -1; j <= 1; j++)
        {
            sum += input[x + i][y + j] * kernel[i + 1][j + 1];
        }
    }
    return sum / 16.0; // Normalização do kernel
}
/* Percorre vizinhos
Aplica pesos do kernel
Divide por 16 (normalização) */

// versão serial da aplicação do kernel
void gauss_serial(float **input, float **output)
{
    for (int iter = 0; iter < ITER; iter++)
    {
        for (int i = 1; i < N - 1; i++)
        {
            for (int j = 1; j < M - 1; j++)
            {
                output[i][j] = aplicar_kernel(input, i, j);
            }
        }
        // troca de matrizes para a próxima iteração
        float **temp = input;
        input = output;
        output = temp;
    }
}

//! PONTO CRÍTICO: usamos input e output para evitar sobrescrever os dados durante a aplicação do kernel, garantindo que cada iteração use os resultados da iteração anterior.

// versão paralela da aplicação do kernel usando OpenMP
void gauss_paralelo(float **input, float **output)
{
    for (int iter = 0; iter < ITER; iter++)
    {
#pragma omp parallel for collapse(2) // Paraleliza os loops aninhados
        for (int i = 1; i < N - 1; i++)
        {
            for (int j = 1; j < M - 1; j++)
            {
                output[i][j] = aplicar_kernel(input, i, j);
            }
        }
        // troca de matrizes para a próxima iteração
        float **temp = input;
        input = output;
        output = temp;
    }
}

// MAIN + Medição de tempo
//? A Fazer