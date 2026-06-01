/* 
 * Projeto 1 - Filtro Gaussiano com MPI
 * Disciplina: DEC107 - Processamento Paralelo
 * Alunos: Gilbert e Henio
 *
 * Compilar:
 *   mpicc -O2 etapa2_mpi.c -o etapa2
 *
 * Executar:
 *   mpirun -np 2 ./etapa2
 *   mpirun -np 4 ./etapa2
 *   mpirun -np 8 --oversubscribe ./etapa2
 *
 * Imagem real:
 *   mpirun -np 4 ./etapa2 imagem.pgm saida.pgm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#define ITERACOES  10
#define TOLERANCIA 0.0001f

float kernel3[3][3] = { 
    {1,2,1},
    {2,4,2},
    {1,2,1} 
};
float kernel5[5][5] = {
    {1, 4, 6, 4,1},
    {4,16,24,16,4},
    {6,24,36,24,6},
    {4,16,24,16,4},
    {1, 4, 6, 4,1}
};

/* ========= PGM ==================================================== */
float** ler_pgm(const char* f, int* W, int* H){
    FILE* fp=fopen(f,"rb"); if(!fp){printf("Erro: %s\n",f);exit(1);}
    char m[3]; fscanf(fp,"%2s",m);
    if(m[0]!='P'||m[1]!='5'){printf("Nao e PGM P5\n");exit(1);}
    int c=fgetc(fp);
    while(c=='#'){while(fgetc(fp)!='\n');c=fgetc(fp);}
    ungetc(c,fp);
    int mv; fscanf(fp,"%d %d %d",W,H,&mv); fgetc(fp);
    float** mat=(float**)malloc(*H*sizeof(float*));
    for(int i=0;i<*H;i++) mat[i]=(float*)malloc(*W*sizeof(float));
    unsigned char* ln=(unsigned char*)malloc(*W);
    for(int i=0;i<*H;i++){
        fread(ln,1,*W,fp);
        for(int j=0;j<*W;j++) mat[i][j]=(float)ln[j];
    }
    free(ln); fclose(fp); return mat;
}
void salvar_pgm(const char* f, float** mat, int W, int H){
    FILE* fp=fopen(f,"wb"); if(!fp){printf("Erro: %s\n",f);exit(1);}
    fprintf(fp,"P5\n%d %d\n255\n",W,H);
    unsigned char* ln=(unsigned char*)malloc(W);
    for(int i=0;i<H;i++){
        for(int j=0;j<W;j++){float v=mat[i][j];if(v<0)v=0;if(v>255)v=255;ln[j]=(unsigned char)v;}
        fwrite(ln,1,W,fp);
    }
    free(ln); fclose(fp);
}

/* ===== Alocação ========================================================== */

/*
 * IMPORTANTE: usa calloc (não malloc).
 * O serial tem duas matrizes: A (valores originais) e B (zeros).
 * O MPI replica esse comportamento com cur (A) e prx (B).
 * prx DEVE comecar zerada e NUNCA deve receber uma copia de cur -
 * caso contrario as linhas de borda teriam valores originais em vez
 * de zeros, quebrando a corretude a partir da iteracao 2.
 */
float** alocar(int linhas, int colunas){
    float** mat=(float**)malloc(linhas*sizeof(float*));
    for(int i=0;i<linhas;i++)
        mat[i]=(float*)calloc(colunas,sizeof(float)); /* calloc = zeros */
    return mat;
}
void liberar(float** mat, int linhas){
    for(int i=0;i<linhas;i++) free(mat[i]); free(mat);
}

/* ===== Kernels ========================================================== */
float aplicar3x3(float** m, int i, int j){
    float s=0;
    for(int di=-1;di<=1;di++) for(int dj=-1;dj<=1;dj++)
        s+=m[i+di][j+dj]*kernel3[di+1][dj+1];
    return s/16.0f;
}
float aplicar5x5(float** m, int i, int j){
    float s=0;
    for(int di=-2;di<=2;di++) for(int dj=-2;dj<=2;dj++)
        s+=m[i+di][j+dj]*kernel5[di+2][dj+2];
    return s/256.0f;
}

/* ===== Serial ========================================================== */
void filtro_serial(float** entrada, float** saida, int H, int W, int ksize){
    int borda=ksize/2;
    for(int it=0;it<ITERACOES;it++){
        for(int i=borda;i<H-borda;i++)
            for(int j=borda;j<W-borda;j++)
                saida[i][j]=(ksize==3)?aplicar3x3(entrada,i,j):aplicar5x5(entrada,i,j);
        float** t=entrada; entrada=saida; saida=t;
    }
}

/* ==== Conversão ======================================================= */
void mat2vec(float** mat, float* vec, int L, int C){
    for(int i=0;i<L;i++) for(int j=0;j<C;j++) vec[i*C+j]=mat[i][j];
}
void vec2mat(float* vec, float** mat, int L, int C){
    for(int i=0;i<L;i++) for(int j=0;j<C;j++) mat[i][j]=vec[i*C+j];
}

/* ==== Corretude ============================================================== */
int verificar(float* serial, float* mpi, int H, int W, int ksize){
    int borda=ksize/2, erros=0; float dmax=0;
    for(int i=borda;i<H-borda;i++)
        for(int j=borda;j<W-borda;j++){
            float d=fabsf(serial[i*W+j]-mpi[i*W+j]);
            if(d>dmax) dmax=d;
            if(d>TOLERANCIA) erros++;
        }
    if(erros==0) printf("  [CORRETUDE: OK] Dif. maxima: %.6f\n",dmax);
    else printf("  [CORRETUDE: FALHOU] %d pixels | Dif. max: %.6f\n",erros,dmax);
    return (erros==0);
}

/* =========================================================================
 * FILTRO MPI
 *
 * Cada processo recebe uma fatia horizontal da imagem via MPI_Scatter.
 * Antes de cada iteração, troca as linhas de borda com os vizinhos
 * (troca de halos) para ter os dados necessarios ao kernel.
 *
 * Duas matrizes alternam o papel de "atual" e "próximo":
 *   cur (A): comeca com os dados originais
 *   prx (B): comeca zerada (calloc) — igual ao serial
 *
 * NUNCA copiar cur para prx antes do loop: o serial não faz isso,
 * e copiar causaria corretude errada nas iterações pares.
 * ============================================================================ */
void filtro_mpi(int H, int W, int ksize, int rank, int nprocs,
                float* vec_entrada, float* vec_saida)
{
    int borda = ksize/2;
    int lpp   = H/nprocs;          /* linhas por processo */
    int elem  = lpp*W;
    int lh    = lpp + 2*borda;     /* linhas com halos */

    float** cur = alocar(lh, W);   /* dados originais (calloc, enchido abaixo) */
    float** prx = alocar(lh, W);   /* zerada — NAO copiar cur aqui */

    float* buf = (float*)malloc(elem*sizeof(float));

    /* distribui a imagem: cada processo recebe lpp linhas */
    MPI_Scatter(vec_entrada, elem, MPI_FLOAT,
                buf,         elem, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    /* carrega na regiao real de cur (halos ficam zero do calloc) */
    vec2mat(buf, cur+borda, lpp, W);

    int acima  = (rank>0)        ? rank-1 : MPI_PROC_NULL;
    int abaixo = (rank<nprocs-1) ? rank+1 : MPI_PROC_NULL;
    int g0     = rank*lpp; /* linha global do inicio desta fatia */

    /*
     * Buffers contiguos para a troca de halos.
     * Como float** aloca cada linha separadamente (não contiguo),
     * precisamos empacotar as linhas num buffer 1D antes de enviar.
     */
    float* env_cima  = (float*)malloc(borda*W*sizeof(float));
    float* env_baixo = (float*)malloc(borda*W*sizeof(float));
    float* rcb_cima  = (float*)malloc(borda*W*sizeof(float));
    float* rcb_baixo = (float*)malloc(borda*W*sizeof(float));

    for(int it=0; it<ITERACOES; it++){

        /* ==== TROCA DE HALOS ======================================
         *
         * Empacota as primeiras e ultimas 'borda' linhas reais,
         * envia para os vizinhos e recebe as deles.
         *
         * Para 3x3 (borda=1): 1 linha por direcao.
         * Para 5x5 (borda=2): 2 linhas por direcao.
         *
         * Indexacao das linhas reais em cur:
         *   primeira linha real : cur[borda]
         *   ultima linha real   : cur[borda + lpp - 1]
         *
         * Indexacao dos halos em cur:
         *   halos do topo    : cur[0] .. cur[borda-1]
         *   halos do fundo   : cur[lh-borda] .. cur[lh-1]
         */

        /* empacota primeiras 'borda' linhas reais → envia para cima */
        for(int b=0; b<borda; b++)
            memcpy(env_cima + b*W, cur[borda+b], W*sizeof(float));

        /* empacota ultimas 'borda' linhas reais → envia para baixo
         *
         * ultima linha real  : cur[borda + lpp - 1] = cur[lpp+borda-1]
         * 2a-a-ultima        : cur[lpp+borda-2]
         * ...
         * (borda)-esima-a-ultima: cur[lpp]   (= cur[lpp+borda-1 - (borda-1)])
         *
         * Empacotamos em ordem crescente de b para que o vizinho
         * abaixo receba em rcb_cima[0..borda-1] os valores que
         * correspondem a suas posicoes de halo[0..borda-1]:
         *   rcb_cima[0] → halo[0] = linha global g0_abaixo - borda
         *   rcb_cima[1] → halo[1] = linha global g0_abaixo - borda + 1
         *   ...
         *   rcb_cima[borda-1] → halo[borda-1] = linha global g0_abaixo - 1
         *
         * Portanto env_baixo[b] deve conter a linha global
         *   g0 + lpp - borda + b
         * que e o indice local cur[borda + (lpp-borda+b)] = cur[lpp+b].
         */
        for(int b=0; b<borda; b++)
            memcpy(env_baixo + b*W, cur[lpp+b], W*sizeof(float));

        /* Passo 1: envia topo para cima, recebe fundo de baixo */
        MPI_Sendrecv(env_cima,  borda*W, MPI_FLOAT, acima,  10,
                     rcb_baixo, borda*W, MPI_FLOAT, abaixo, 10,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Passo 2: envia fundo para baixo, recebe topo de cima */
        MPI_Sendrecv(env_baixo, borda*W, MPI_FLOAT, abaixo, 20,
                     rcb_cima,  borda*W, MPI_FLOAT, acima,  20,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* desempacota halos recebidos nas posicoes de halo de cur */
        for(int b=0; b<borda; b++){
            memcpy(cur[b],          rcb_cima  + b*W, W*sizeof(float));
            memcpy(cur[lh-borda+b], rcb_baixo + b*W, W*sizeof(float));
        }

        /* ==== APLICA FILTRO ====================================================
         * Processa apenas as linhas reais que não são borda global.
         * O serial tambem pula as primeiras e ultimas 'borda' linhas
         * da imagem completa, replicamos esse comportamento aqui.
         */
        for(int i=borda; i<lpp+borda; i++){
            int g = g0 + (i-borda);
            if(g<borda || g>=H-borda) continue;
            for(int j=borda; j<W-borda; j++)
                prx[i][j] = (ksize==3) ? aplicar3x3(cur,i,j)
                                       : aplicar5x5(cur,i,j);
        }

        /* alterna cur e prx para a proxima iteracao */
        float** t=cur; cur=prx; prx=t;
    }

    /* salva resultado da ultima iteração (cur apos 10 swaps) */
    mat2vec(cur+borda, buf, lpp, W);

    /* junta as fatias no processo 0 */
    MPI_Gather(buf,       elem, MPI_FLOAT,
               vec_saida, elem, MPI_FLOAT,
               0, MPI_COMM_WORLD);

    liberar(cur,lh); liberar(prx,lh); free(buf);
    free(env_cima); free(env_baixo);
    free(rcb_cima); free(rcb_baixo);
}

/* ==== Benchmark ==================================================== */
void rodar_benchmark(int tam, int ksize, int rank, int nprocs){
    float *ve=NULL, *vs=NULL, *vser=NULL;
    double tser=0;

    if(rank==0){
        printf("\n--- Benchmark %dx%d | Kernel %dx%d ---\n",tam,tam,ksize,ksize);
        float** A=alocar(tam,tam);
        float** B=alocar(tam,tam);

        /* mesma semente para serial e MPI */
        srand(42);
        for(int i=0;i<tam;i++) for(int j=0;j<tam;j++) A[i][j]=rand()%256;

        double t0=MPI_Wtime();
        filtro_serial(A,B,tam,tam,ksize);
        tser=MPI_Wtime()-t0;
        printf("Serial:      %.4f s\n",tser);

        /*
         * Salva o resultado serial.
         * Apos 10 iteracoes (par), a funcao filtro_serial deixa o
         * resultado final em 'A' (o ponteiro original de 'entrada').
         * Salvar de 'B' seria o resultado da 9a iteracao — errado.
         */
        vser=(float*)malloc(tam*tam*sizeof(float));
        mat2vec(A, vser, tam, tam);

        /* reinicia A com a mesma semente para o teste MPI */
        srand(42);
        for(int i=0;i<tam;i++) for(int j=0;j<tam;j++) A[i][j]=rand()%256;

        ve=(float*)malloc(tam*tam*sizeof(float));
        vs=(float*)malloc(tam*tam*sizeof(float));
        mat2vec(A, ve, tam, tam);

        liberar(A,tam); liberar(B,tam);
    }

    MPI_Bcast(&tser, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double t1=MPI_Wtime();
    filtro_mpi(tam,tam,ksize,rank,nprocs,ve,vs);
    double tmpi=MPI_Wtime()-t1;

    if(rank==0){
        double sp=tser/tmpi;
        printf("%d processos: %.4f s | Speedup: %.2f | Eficiencia: %.1f%%\n",
               nprocs,tmpi,sp,(sp/nprocs)*100.0);
        verificar(vser,vs,tam,tam,ksize);
        printf("  [BALANCEAMENTO] %d linhas/processo (uniforme)\n", tam/nprocs);
        free(ve); free(vs); free(vser);
    }
}

/* ==== Main ======================================================================== */
int main(int argc, char* argv[]){
    int rank, nprocs;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);

    if(argc==3){
        int W=0,H=0; float** img=NULL; float *ve=NULL,*vs=NULL;
        if(rank==0){
            printf("Lendo: %s\n",argv[1]);
            img=ler_pgm(argv[1],&W,&H);
            printf("Tamanho: %dx%d | Processos: %d\n",W,H,nprocs);
            ve=(float*)malloc(H*W*sizeof(float));
            vs=(float*)malloc(H*W*sizeof(float));
            mat2vec(img,ve,H,W); liberar(img,H);
        }
        MPI_Bcast(&H,1,MPI_INT,0,MPI_COMM_WORLD);
        MPI_Bcast(&W,1,MPI_INT,0,MPI_COMM_WORLD);
        double t0=MPI_Wtime();
        filtro_mpi(H,W,3,rank,nprocs,ve,vs);
        if(rank==0){
            printf("Tempo: %.4f s\n",MPI_Wtime()-t0);
            float** saida=alocar(H,W); vec2mat(vs,saida,H,W);
            salvar_pgm(argv[2],saida,W,H); printf("Salvo: %s\n",argv[2]);
            liberar(saida,H); free(ve); free(vs);
        }
        MPI_Finalize(); return 0;
    }

    if(rank==0){
        printf("=== Benchmark Filtro Gaussiano Iterativo (MPI) ===\n");
        printf("Processos: %d | Iteracoes: %d\n",nprocs,ITERACOES);
    }

    int tams[]={512,1024,2048};
    int ks[]={3,5};
    for(int t=0;t<3;t++) for(int k=0;k<2;k++)
        rodar_benchmark(tams[t],ks[k],rank,nprocs);

    if(rank==0) printf("\n=== Fim ===\n");
    MPI_Finalize(); return 0;
}

/*
 * ALTERNATIVA MPI_Scatterv (para H nao divisivel por nprocs):
 *
 *   int cnt[nprocs], dsp[nprocs], off=0;
 *   for(int p=0;p<nprocs;p++){
 *       cnt[p]=(H/nprocs+(p<H%nprocs?1:0))*W;
 *       dsp[p]=off; off+=cnt[p];
 *   }
 *   MPI_Scatterv(ve,cnt,dsp,MPI_FLOAT, buf,cnt[rank],MPI_FLOAT, 0,MPI_COMM_WORLD);
 *   MPI_Gatherv (buf,cnt[rank],MPI_FLOAT, vs,cnt,dsp,MPI_FLOAT, 0,MPI_COMM_WORLD);
 */
