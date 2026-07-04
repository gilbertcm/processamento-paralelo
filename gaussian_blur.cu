// gaussian_blur.cu — Suavização gaussiana CUDA com leitura/escrita PGM
// Compilar: nvcc -O2 -arch=native -o gaussian_blur gaussian_blur.cu
// Uso:      ./gaussian_blur input.pgm output.pgm [raio] [sigma] [iteracoes] [repeticoes]
//           input.pgm pode ser "gen:N" para gerar imagem sintetica NxN
//           (gradiente radial + ruido, semente 42 — mesma das Etapas 1 e 2)
// Ex.:      ./gaussian_blur gen:1024 out.pgm 1 1.5 10 5

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// ─────────────────────────────────────────────
// Configuração
// ─────────────────────────────────────────────
#define BLOCK_DIM  16
#define MAX_RADIUS 16
#define MAX_MASK   ((2*MAX_RADIUS+1)*(2*MAX_RADIUS+1))
#define TOLERANCIA 1e-5f   // tolerância do teste de corretude (float, ~7 dígitos)

__constant__ float d_mask[MAX_MASK];

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error em %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ─────────────────────────────────────────────
// Leitura de PGM
// ─────────────────────────────────────────────
typedef struct { int W, H, maxval; unsigned char* data; } PGM;

static void skipWS(FILE* f) {
    int c;
    while ((c = fgetc(f)) != EOF) {
        if (c == '#') { while ((c = fgetc(f)) != EOF && c != '\n'); }
        else if (c != ' ' && c != '\t' && c != '\r' && c != '\n') { ungetc(c, f); break; }
    }
}

PGM* readPGM(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) { fprintf(stderr, "Erro: não foi possível abrir '%s'\n", filename); return NULL; }
    char magic[3];
    if (!fgets(magic, sizeof(magic), f) || magic[0] != 'P' || (magic[1] != '5' && magic[1] != '2')) {
        fprintf(stderr, "Erro: '%s' não é PGM válido\n", filename);
        fclose(f); return NULL;
    }
    int binary = (magic[1] == '5');
    PGM* img = (PGM*)malloc(sizeof(PGM));
    skipWS(f); fscanf(f, "%d", &img->W);
    skipWS(f); fscanf(f, "%d", &img->H);
    skipWS(f); fscanf(f, "%d", &img->maxval);
    fgetc(f);
    int N = img->W * img->H;
    img->data = (unsigned char*)malloc(N);
    if (binary) fread(img->data, 1, N, f);
    else { for (int i = 0; i < N; i++) { int v; fscanf(f, "%d", &v); img->data[i] = (unsigned char)v; } }
    fclose(f);
    printf("Lido: %s (%dx%d, maxval=%d, %s)\n",
           filename, img->W, img->H, img->maxval, binary ? "binário P5" : "ASCII P2");
    return img;
}

// ─────────────────────────────────────────────
// Imagem sintética: gradiente radial + ruído (semente 42)
// Mesmo padrão usado no benchmark das Etapas 1 e 2.
// ─────────────────────────────────────────────
PGM* genSynthetic(int size) {
    PGM* img = (PGM*)malloc(sizeof(PGM));
    img->W = size; img->H = size; img->maxval = 255;
    img->data = (unsigned char*)malloc((size_t)size * size);
    srand(42);
    float cx = size / 2.f, cy = size / 2.f;
    float maxDist = sqrtf(cx*cx + cy*cy);
    for (int y = 0; y < size; y++)
        for (int x = 0; x < size; x++) {
            float d = sqrtf((x-cx)*(x-cx) + (y-cy)*(y-cy)) / maxDist;  // 0..1
            int noise = rand() % 51 - 25;                              // -25..+25
            int v = (int)((1.f - d) * 255.f) + noise;
            if (v < 0) v = 0; if (v > 255) v = 255;
            img->data[y*size + x] = (unsigned char)v;
        }
    printf("Gerada: imagem sintética %dx%d (gradiente radial + ruído, semente 42)\n", size, size);
    return img;
}

// ─────────────────────────────────────────────
// Escrita de PGM
// ─────────────────────────────────────────────
void writePGM(const char* filename, unsigned char* data, int W, int H, int maxval) {
    FILE* f = fopen(filename, "wb");
    if (!f) { fprintf(stderr, "Erro: não foi possível escrever '%s'\n", filename); return; }
    fprintf(f, "P5\n%d %d\n%d\n", W, H, maxval);
    fwrite(data, 1, W * H, f);
    fclose(f);
    printf("Salvo : %s (%dx%d)\n", filename, W, H);
}

// ─────────────────────────────────────────────
// Gerar máscara gaussiana
// ─────────────────────────────────────────────
void buildGaussianMask(float* mask, int radius, float sigma) {
    int size = 2 * radius + 1;
    float sum = 0.f;
    for (int dy = -radius; dy <= radius; dy++)
        for (int dx = -radius; dx <= radius; dx++) {
            float v = expf(-(dx*dx + dy*dy) / (2.f * sigma * sigma));
            mask[(dy+radius)*size + (dx+radius)] = v;
            sum += v;
        }
    for (int i = 0; i < size*size; i++) mask[i] /= sum;
}

// ─────────────────────────────────────────────
// Versão SERIAL de referência (CPU) — usada no teste de corretude.
// Mesma lógica do kernel: convolução com zero-padding nas bordas.
// ─────────────────────────────────────────────
void cpuGaussianBlur(const float* in, float* out, int W, int H,
                     const float* mask, int radius) {
    int sz = 2 * radius + 1;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            float sum = 0.f;
            for (int dy = -radius; dy <= radius; dy++)
                for (int dx = -radius; dx <= radius; dx++) {
                    int px = x + dx, py = y + dy;
                    float v = (px >= 0 && px < W && py >= 0 && py < H)
                              ? in[py*W + px] : 0.f;   // zero-padding (igual à GPU)
                    sum += v * mask[(dy+radius)*sz + (dx+radius)];
                }
            out[y*W + x] = sum;
        }
}

// Executa 'iters' iterações da versão serial (ping-pong em buffers do host)
// e devolve o ponteiro para o buffer com o resultado final.
float* cpuIterate(float* bufA, float* bufB, int W, int H,
                  const float* mask, int radius, int iters) {
    float *src = bufA, *dst = bufB;
    for (int it = 0; it < iters; it++) {
        cpuGaussianBlur(src, dst, W, H, mask, radius);
        float* tmp = src; src = dst; dst = tmp;   // troca os papéis
    }
    return src;  // último buffer escrito
}

// ─────────────────────────────────────────────
// KERNEL: blur 2D com shared memory + halo
// ─────────────────────────────────────────────
__global__ void gaussianBlur2D(
    const float* __restrict__ in,
    float*       __restrict__ out,
    int W, int H, int radius)
{
    const int TILE = BLOCK_DIM + 2 * radius;
    extern __shared__ float smem[];

    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    int lx = threadIdx.x + radius;
    int ly = threadIdx.y + radius;

    // Pixel central
    smem[ly * TILE + lx] = (gx < W && gy < H) ? in[gy * W + gx] : 0.f;

    // Halo horizontal
    if (threadIdx.x < radius) {
        int px = gx - radius;
        smem[ly * TILE + (lx - radius)]    = (px >= 0 && gy < H) ? in[gy * W + px] : 0.f;
        px = gx + BLOCK_DIM;
        smem[ly * TILE + (lx + BLOCK_DIM)] = (px < W && gy < H)  ? in[gy * W + px] : 0.f;
    }

    // Halo vertical
    if (threadIdx.y < radius) {
        int py = gy - radius;
        smem[(ly - radius)  * TILE + lx] = (py >= 0 && gx < W) ? in[py * W + gx] : 0.f;
        py = gy + BLOCK_DIM;
        smem[(ly + BLOCK_DIM) * TILE + lx] = (py < H && gx < W) ? in[py * W + gx] : 0.f;
    }

    // Cantos do halo
    if (threadIdx.x < radius && threadIdx.y < radius) {
        int px, py;
        px = gx - radius;  py = gy - radius;
        smem[(ly-radius)*TILE + (lx-radius)]    = (px>=0 && py>=0) ? in[py*W+px] : 0.f;
        px = gx + BLOCK_DIM;
        smem[(ly-radius)*TILE + (lx+BLOCK_DIM)] = (px<W  && py>=0) ? in[py*W+px] : 0.f;
        px = gx - radius;  py = gy + BLOCK_DIM;
        smem[(ly+BLOCK_DIM)*TILE + (lx-radius)] = (px>=0 && py<H)  ? in[py*W+px] : 0.f;
        px = gx + BLOCK_DIM;
        smem[(ly+BLOCK_DIM)*TILE + (lx+BLOCK_DIM)] = (px<W && py<H) ? in[py*W+px] : 0.f;
    }

    __syncthreads();

    if (gx >= W || gy >= H) return;

    float sum = 0.f;
    int sz = 2 * radius + 1;
    for (int dy = 0; dy < sz; dy++)
        for (int dx = 0; dx < sz; dx++)
            sum += smem[(ly-radius+dy)*TILE + (lx-radius+dx)] * d_mask[dy*sz+dx];

    out[gy * W + gx] = sum;
}

// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────
int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Uso: %s input.pgm output.pgm [raio] [sigma] [iteracoes] [repeticoes]\n", argv[0]);
        printf("  input.pgm  : arquivo PGM ou gen:N (imagem sintetica NxN, semente 42)\n");
        printf("  raio       : raio da mascara gaussiana  (default: 3)\n");
        printf("  sigma      : desvio padrao              (default: 1.5)\n");
        printf("  iteracoes  : numero de passes do filtro (default: 1)\n");
        printf("  repeticoes : execucoes cronometradas    (default: 5, apos 1 warm-up)\n");
        return 1;
    }

    const char* inFile  = argv[1];
    const char* outFile = argv[2];
    int   radius = (argc > 3) ? atoi(argv[3]) : 3;
    float sigma  = (argc > 4) ? atof(argv[4]) : 1.5f;
    int   iters  = (argc > 5) ? atoi(argv[5]) : 1;
    int   reps   = (argc > 6) ? atoi(argv[6]) : 5;

    if (radius > MAX_RADIUS) { fprintf(stderr, "Raio maximo: %d\n", MAX_RADIUS); return 1; }
    if (iters  < 1)          { fprintf(stderr, "Iteracoes minimas: 1\n"); return 1; }
    if (reps   < 1)          { fprintf(stderr, "Repeticoes minimas: 1\n"); return 1; }

    printf("═══════════════════════════════════════════\n");
    printf("  Gaussian Blur CUDA — PGM\n");
    printf("  Raio: %d  |  Sigma: %.2f  |  Iteracoes: %d  |  Repeticoes: %d\n",
           radius, sigma, iters, reps);
    printf("═══════════════════════════════════════════\n\n");

    // ── Propriedades da GPU (dados para a secao Ambiente do relatorio) ──
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU               : %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("SMs               : %d\n", prop.multiProcessorCount);
    printf("Memoria global    : %.1f GB\n", prop.totalGlobalMem / (1024.f*1024.f*1024.f));
    printf("Shared mem/bloco  : %.0f KB\n\n", prop.sharedMemPerBlock / 1024.f);

    // ── Lê ou gera a imagem ─────────────────────────────────────────────
    PGM* img = NULL;
    if (strncmp(inFile, "gen:", 4) == 0) img = genSynthetic(atoi(inFile + 4));
    else                                 img = readPGM(inFile);
    if (!img) return 1;
    int W = img->W, H = img->H, N = W * H;

    // ── Converte uchar → float [0,1] ───────────────────────────────────
    float* h_in  = (float*)malloc(N * sizeof(float));
    float* h_out = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++)
        h_in[i] = img->data[i] / (float)img->maxval;

    // ── Máscara gaussiana na constant memory ───────────────────────────
    int   maskSz = 2 * radius + 1;
    float h_mask[MAX_MASK];
    buildGaussianMask(h_mask, radius, sigma);
    CUDA_CHECK(cudaMemcpyToSymbol(d_mask, h_mask, maskSz*maskSz*sizeof(float)));
    printf("Mascara gaussiana %dx%d carregada na constant memory.\n\n", maskSz, maskSz);

    // ── Aloca dois buffers na GPU (ping-pong) ───────────────────────────
    float *d_buf[2];
    size_t imgBytes = N * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_buf[0], imgBytes));
    CUDA_CHECK(cudaMalloc(&d_buf[1], imgBytes));

    // ── Configuração do kernel ──────────────────────────────────────────
    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((W + BLOCK_DIM-1)/BLOCK_DIM, (H + BLOCK_DIM-1)/BLOCK_DIM);
    int    tileW       = BLOCK_DIM + 2 * radius;
    size_t sharedBytes = (size_t)tileW * tileW * sizeof(float);

    printf("Grid : %u x %u blocos  |  Block: %u x %u threads\n", grid.x, grid.y, block.x, block.y);
    printf("Shared memory por bloco: %.1f KB\n\n", sharedBytes / 1024.f);

    // ── Medicao: 1 warm-up + 'reps' execucoes cronometradas ────────────
    // O warm-up descarta custos de inicializacao do contexto CUDA.
    // Cada repeticao recomeca da imagem original (re-upload) e cronometra
    // apenas as 'iters' iteracoes do kernel com eventos CUDA.
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    int   resultBuf = iters % 2;   // com ping-pong, resultado fica neste buffer
    float somaMs = 0.f, melhorMs = 1e30f;

    for (int rep = 0; rep <= reps; rep++) {          // rep 0 = warm-up
        CUDA_CHECK(cudaMemcpy(d_buf[0], h_in, imgBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(t0));
        for (int it = 0; it < iters; it++) {
            int ping = it % 2;
            int pong = 1 - ping;
            gaussianBlur2D<<<grid, block, sharedBytes>>>(d_buf[ping], d_buf[pong], W, H, radius);
        }
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        if (rep == 0) continue;                      // descarta warm-up
        somaMs += ms;
        if (ms < melhorMs) melhorMs = ms;
    }
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    float mediaMs = somaMs / reps;
    printf("Iteracoes            : %d\n", iters);
    printf("Tempo medio (%d reps): %.2f ms   |  melhor: %.2f ms\n", reps, mediaMs, melhorMs);
    printf("Tempo por iteracao   : %.2f ms (media)\n", mediaMs / iters);
    printf("Throughput medio     : %.1f MP/s\n\n", (W*H/1e6f) / ((mediaMs/iters)/1000.f));

    // ── Copia resultado de volta ────────────────────────────────────────
    CUDA_CHECK(cudaMemcpy(h_out, d_buf[resultBuf], imgBytes, cudaMemcpyDeviceToHost));

    // ── TESTE DE CORRETUDE: compara GPU com a versao serial (CPU) ──────
    printf("Teste de corretude: executando versao serial de referencia na CPU...\n");
    float* cpuA = (float*)malloc(imgBytes);
    float* cpuB = (float*)malloc(imgBytes);
    memcpy(cpuA, h_in, imgBytes);
    float* cpuRes = cpuIterate(cpuA, cpuB, W, H, h_mask, radius, iters);

    float maxAbs = 0.f;
    double somaRel = 0.0;
    for (int i = 0; i < N; i++) {
        float dif = fabsf(h_out[i] - cpuRes[i]);
        if (dif > maxAbs) maxAbs = dif;
        somaRel += dif / (fabsf(cpuRes[i]) + 1e-8f);
    }
    float mediaRel = (float)(somaRel / N);
    printf("  Erro maximo absoluto : %.3e\n", maxAbs);
    printf("  Erro relativo medio  : %.3e\n", mediaRel);
    printf("  Tolerancia           : %.1e\n", TOLERANCIA);
    printf("  Resultado            : %s\n\n", (maxAbs <= TOLERANCIA) ? "OK" : "FALHA");
    free(cpuA); free(cpuB);

    // ── Converte float [0,1] → uchar e salva ───────────────────────────
    unsigned char* h_outU = (unsigned char*)malloc(N);
    for (int i = 0; i < N; i++) {
        float v = h_out[i] * img->maxval;
        if (v < 0.f) v = 0.f;
        if (v > (float)img->maxval) v = (float)img->maxval;
        h_outU[i] = (unsigned char)(v + 0.5f);
    }
    writePGM(outFile, h_outU, W, H, img->maxval);

    // ── Libera ──────────────────────────────────────────────────────────
    CUDA_CHECK(cudaFree(d_buf[0]));
    CUDA_CHECK(cudaFree(d_buf[1]));
    free(h_in); free(h_out); free(h_outU);
    free(img->data); free(img);

    printf("\nConcluido.\n");
    return 0;
}
