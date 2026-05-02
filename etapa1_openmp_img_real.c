/* 
 * Projeto 1 - Filtro Gaussiano com OpenMP
 * Disciplina: DEC107 - Processamento Paralelo
 * Alunos: Gilbert Carmo e Henio Pedro
 *
 * Compilar: gcc -O2 -fopenmp etapa1_openmp.c -o etapa1
 * Usar:     ./etapa1 imagem.pgm saida.pgm
 *
 * Para converter JPG/PNG para PGM:
 *   convert imagem.jpg imagem.pgm        (ImageMagick)
 *   convert saida.pgm saida.jpg          (ImageMagick)
*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

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

// ─── leitura e escrita de imagem PGM ──────────────────────────────────────

// le imagem PGM e devolve matriz de floats. preenche largura e altura.
float** ler_pgm(const char* arquivo, int* largura, int* altura) {
    FILE* f = fopen(arquivo, "rb");
    if (!f) { printf("Erro: nao abriu %s\n", arquivo); exit(1); }

    char magic[3];
    fscanf(f, "%2s", magic);
    if (magic[0] != 'P' || magic[1] != '5') {
        printf("Erro: arquivo nao e PGM binario (P5)\n"); exit(1);
    }

    // pula comentarios
    int c = fgetc(f);
    while (c == '#') { while (fgetc(f) != '\n'); c = fgetc(f); }
    ungetc(c, f);

    int maxval;
    fscanf(f, "%d %d %d", largura, altura, &maxval);
    fgetc(f); // consome o '\n' final do cabecalho

    int W = *largura, H = *altura;

    // aloca matriz
    float** mat = (float**)malloc(H * sizeof(float*));
    for (int i = 0; i < H; i++) {
        mat[i] = (float*)malloc(W * sizeof(float));
    }

    // le pixels (1 byte por pixel em PGM P5)
    unsigned char* linha = (unsigned char*)malloc(W);
    for (int i = 0; i < H; i++) {
        fread(linha, 1, W, f);
        for (int j = 0; j < W; j++) {
            mat[i][j] = (float)linha[j];
        }
    }
    free(linha);
    fclose(f);
    return mat;
}

// salva matriz de floats como PGM
void salvar_pgm(const char* arquivo, float** mat, int largura, int altura) {
    FILE* f = fopen(arquivo, "wb");
    if (!f) { printf("Erro: nao criou %s\n", arquivo); exit(1); }

    fprintf(f, "P5\n%d %d\n255\n", largura, altura);

    unsigned char* linha = (unsigned char*)malloc(largura);
    for (int i = 0; i < altura; i++) {
        for (int j = 0; j < largura; j++) {
            float v = mat[i][j];
            if (v < 0) v = 0;
            if (v > 255) v = 255;
            linha[j] = (unsigned char)v;
        }
        fwrite(linha, 1, largura, f);
    }
    free(linha);
    fclose(f);
}

// ─── alocacao e liberacao de matrizes ─────────────────────────────────────

float** alocar(int linhas, int colunas) {
    float** mat = (float**)malloc(linhas * sizeof(float*));
    for (int i = 0; i < linhas; i++) {
        mat[i] = (float*)malloc(colunas * sizeof(float));
    }
    return mat;
}

void liberar(float** mat, int linhas) {
    for (int i = 0; i < linhas; i++) free(mat[i]);
    free(mat);
}

// ─── aplicacao dos kernels ─────────────────────────────────────────────────

float aplicar3x3(float** entrada, int linha, int coluna) {
    float soma = 0.0;
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++)
            soma += entrada[linha + i][coluna + j] * kernel3[i + 1][j + 1];
    return soma / 16.0;
}

float aplicar5x5(float** entrada, int linha, int coluna) {
    float soma = 0.0;
    for (int i = -2; i <= 2; i++)
        for (int j = -2; j <= 2; j++)
            soma += entrada[linha + i][coluna + j] * kernel5[i + 2][j + 2];
    return soma / 256.0;
}

// ─── versoes serial e paralela ────────────────────────────────────────────

void filtro_serial(float** entrada, float** saida, int H, int W, int ksize) {
    int borda = ksize / 2;
    for (int it = 0; it < ITERACOES; it++) {
        for (int i = borda; i < H - borda; i++) {
            for (int j = borda; j < W - borda; j++) {
                saida[i][j] = (ksize == 3) ? aplicar3x3(entrada, i, j)
                                           : aplicar5x5(entrada, i, j);
            }
        }
        float** temp = entrada; entrada = saida; saida = temp;
    }
}

void filtro_paralelo(float** entrada, float** saida, int H, int W, int ksize, int num_threads) {
    int borda = ksize / 2;
    omp_set_num_threads(num_threads);
    for (int it = 0; it < ITERACOES; it++) {
        #pragma omp parallel for collapse(2)
        for (int i = borda; i < H - borda; i++) {
            for (int j = borda; j < W - borda; j++) {
                saida[i][j] = (ksize == 3) ? aplicar3x3(entrada, i, j)
                                           : aplicar5x5(entrada, i, j);
            }
        }
        float** temp = entrada; entrada = saida; saida = temp;
    }
}

// ─── benchmark por tamanho/kernel ─────────────────────────────────────────

void rodar_benchmark(int tam, int ksize) {
    float** entrada = alocar(tam, tam);
    float** saida   = alocar(tam, tam);

    // preenche com valores aleatorios para benchmark de desempenho
    for (int i = 0; i < tam; i++)
        for (int j = 0; j < tam; j++)
            entrada[i][j] = rand() % 256;

    printf("\n--- Benchmark %dx%d | Kernel %dx%d ---\n", tam, tam, ksize, ksize);

    double inicio = omp_get_wtime();
    filtro_serial(entrada, saida, tam, tam, ksize);
    double tempo_serial = omp_get_wtime() - inicio;
    printf("Serial:      %.4f s\n", tempo_serial);

    int configs[] = {2, 4, 8};
    for (int c = 0; c < 3; c++) {
        int threads = configs[c];
        for (int i = 0; i < tam; i++)
            for (int j = 0; j < tam; j++)
                entrada[i][j] = rand() % 256;
        inicio = omp_get_wtime();
        filtro_paralelo(entrada, saida, tam, tam, ksize, threads);
        double tp = omp_get_wtime() - inicio;
        printf("%d threads:   %.4f s | Speedup: %.2f | Eficiencia: %.1f%%\n",
               threads, tp, tempo_serial / tp, (tempo_serial / tp / threads) * 100.0);
    }

    liberar(entrada, tam);
    liberar(saida, tam);
}

// ─── main ─────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {

    // MODO 1: aplicar filtro em imagem real
    // uso: ./etapa1 entrada.pgm saida.pgm
    if (argc == 3) {
        int W, H;
        printf("Lendo imagem: %s\n", argv[1]);
        float** entrada = ler_pgm(argv[1], &W, &H);
        float** saida   = alocar(H, W);

        printf("Tamanho: %d x %d pixels\n", W, H);
        printf("Aplicando filtro Gaussiano 3x3 com %d iteracoes...\n", ITERACOES);

        double t0 = omp_get_wtime();
        filtro_paralelo(entrada, saida, H, W, 3, 4);
        double dt = omp_get_wtime() - t0;

        printf("Tempo: %.4f s\n", dt);
        printf("Salvando imagem suavizada: %s\n", argv[2]);
        salvar_pgm(argv[2], saida, W, H);
        printf("Pronto!\n");

        liberar(entrada, H);
        liberar(saida, H);
        return 0;
    }

    // MODO 2: benchmark completo (sem argumentos)
    printf("=== Benchmark Filtro Gaussiano Iterativo ===\n");
    printf("Iteracoes por rodada: %d\n\n", ITERACOES);

    int tamanhos[] = {512, 1024, 2048};
    int kernels[]  = {3, 5};

    for (int t = 0; t < 3; t++)
        for (int k = 0; k < 2; k++)
            rodar_benchmark(tamanhos[t], kernels[k]);

    printf("\n=== Fim ===\n");
    return 0;
}