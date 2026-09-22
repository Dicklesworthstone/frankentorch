/* Native instruction-level review of the production proposal in FrankenTorch #1.
 * This is a C translation for differential tests, NOT a build of the Rust crate.
 * Scalar reference does not call the candidate quantizer or dot helper.
 *
 * Compile from the repository root (GCC on x86-64):
 *   gcc -O2 -fno-tree-vectorize -ffp-contract=off \
 *     scripts/issue1_vnni_native.c -lm -o /tmp/issue1-vnni-native
 *   /tmp/issue1-vnni-native
 * Sanitizers:
 *   gcc -O1 -g -fno-tree-vectorize -ffp-contract=off \
 *     -fsanitize=address,undefined -fno-omit-frame-pointer \
 *     scripts/issue1_vnni_native.c -lm -o /tmp/issue1-vnni-sanitized
 *   /tmp/issue1-vnni-sanitized
 *
 * Exit 77 means the required host features are absent, NOT a passing test.
 * Do not use -ffast-math. This does not establish Rust dispatch or performance.
 * Source proposal: https://github.com/Dicklesworthstone/frankentorch/issues/1
 */
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <fenv.h>

#define VNNI __attribute__((target("avx512f,avx512bw,avx512vnni"),noinline))
#define AVX512 __attribute__((target("avx512f"),noinline))
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); exit(1); } } while(0)
static size_t quant_cases, dot_cases, matrix_cases;
static void *alloc(size_t n) { void *p=malloc(n ? n : 1); CHECK(p); return p; }
static uint32_t bits(float x) { uint32_t u; memcpy(&u,&x,4); return u; }
static float from_bits(uint32_t u) { float x; memcpy(&x,&u,4); return x; }

/* Independent explicit nearest-even reference; NaN preserved until Rust-like cast. */
static float nearest_even(float x) {
    if (!isfinite(x) || fabsf(x)>=8388608.0f) return x;
    float lo=floorf(x), frac=x-lo;
    if (frac<0.5f) return lo;
    if (frac>0.5f) return lo+1.0f;
    return fmodf(lo,2.0f)==0.0f ? lo : lo+1.0f;
}
static int8_t clamp_cast(float x) {
    if (isnan(x)) return 0; /* Rust float -> integer cast */
    if (x>127.0f) return 127;
    if (x< -127.0f) return -127;
    return (int8_t)x;
}
static void quant_ref(int8_t *dst,const float *src,size_t k,float scale) {
    for(size_t j=0;j<k;j++) dst[j]=clamp_cast(nearest_even(src[j]/scale));
}
AVX512 static void quant_candidate(int8_t *dst,const float *src,size_t k,float scale) {
    __m512 sv=_mm512_set1_ps(scale);
    float rounded[16]; size_t j=0;
    for(;j+16<=k;j+=16) {
        __m512 x=_mm512_loadu_ps(src+j);
        __m512 q=_mm512_div_ps(x,sv);
        __m512 r=_mm512_roundscale_ps(q,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
        _mm512_storeu_ps(rounded,r);
        for(size_t lane=0;lane<16;lane++) dst[j+lane]=clamp_cast(rounded[lane]);
    }
    quant_ref(dst+j,src+j,k-j,scale); /* proposal's scalar tail */
}
static float row_scale(const float *x,size_t k) {
    float amax=0.0f;
    for(size_t i=0;i<k;i++) amax=fmaxf(amax,fabsf(x[i]));
    return amax>0.0f ? amax/127.0f : 1.0f;
}
static void check_quant(const float *x,size_t k,float scale) {
    /* Deliberately offset destination buffers. ASan also checks their tails. */
    int8_t *aa=alloc(k+1),*bb=alloc(k+1),*a=aa+1,*b=bb+1;
    quant_ref(a,x,k,scale); quant_candidate(b,x,k,scale);
    if(memcmp(a,b,k)!=0) {
        for(size_t i=0;i<k;i++) if(a[i]!=b[i])
            fprintf(stderr,"quant k=%zu i=%zu x=%g scale=%g ref=%d got=%d\n",k,i,x[i],scale,a[i],b[i]);
        exit(1);
    }
    free(aa);free(bb);quant_cases++;
}
static int64_t dot_ref(const int8_t *a,const int8_t *b,size_t k) {
    int64_t s=0; for(size_t j=0;j<k;j++) s+=(int32_t)a[j]*(int32_t)b[j]; return s;
}
static int32_t weight_sum(const int8_t *b,size_t k) {
    int64_t s=0;for(size_t j=0;j<k;j++)s+=b[j];
    CHECK(s>=INT_MIN && s<=INT_MAX);return (int32_t)s;
}
/* Mirrors ordinary Rust checked scalar arithmetic around the SIMD reduction.
 * Returning false models an overflow panic rather than invoking C signed UB. */
VNNI static int dot_checked(const int8_t *a,const int8_t *b,size_t k,int32_t sum,int32_t *out) {
    __m512i acc=_mm512_setzero_si512(),sign=_mm512_set1_epi8((char)0x80);
    size_t j=0;
    for(;j+64<=k;j+=64) {
        __m512i av=_mm512_xor_si512(_mm512_loadu_si512(a+j),sign);
        acc=_mm512_dpbusd_epi32(acc,av,_mm512_loadu_si512(b+j));
    }
    /* GCC's reduction header uses signed C vector arithmetic, which UBSan
     * diagnoses for the deliberate wide-K probe. Model Rust intrinsic-style
     * modulo reduction explicitly there, without introducing C signed UB. */
    int32_t reduced;
    if (k <= (size_t)INT_MAX/(255u*128u)) {
        reduced=_mm512_reduce_add_epi32(acc);
    } else {
        uint32_t lanes[16], u=0;
        _mm512_storeu_si512(lanes,acc);
        for(size_t lane=0;lane<16;lane++)u+=lanes[lane];
        memcpy(&reduced,&u,sizeof(reduced));
    }
    int64_t total=reduced;
    for(;j<k;j++) {
        total+=((int32_t)a[j]+128)*(int32_t)b[j];
        if(total<INT_MIN || total>INT_MAX)return 0;
    }
    int64_t correction=128LL*sum;
    if(correction<INT_MIN || correction>INT_MAX)return 0;
    total-=correction;
    if(total<INT_MIN || total>INT_MAX)return 0;
    *out=(int32_t)total;return 1;
}
static float dequant(int32_t sum,float as,float ws,const float *bias,size_t o) {
    float y=(float)sum*as; y=y*ws; if(bias)y=y+bias[o];return y;
}
VNNI static void tile_candidate(const int8_t *a,const float *as,size_t start,size_t rows,
    size_t k,const int8_t *w,const float *ws,const int32_t *sums,size_t n,
    const float *bias,float *out) {
    __m512i sign=_mm512_set1_epi8((char)0x80),zero=_mm512_setzero_si512();
    size_t end=k/64*64,o=0;
    for(;o+4<=n;o+=4) {
        __m512i acc[4][4];for(size_t r=0;r<4;r++)for(size_t c=0;c<4;c++)acc[r][c]=zero;
        for(size_t j=0;j<end;j+=64) {
            __m512i ww[4];for(size_t c=0;c<4;c++)ww[c]=_mm512_loadu_si512(w+(o+c)*k+j);
            for(size_t r=0;r<rows;r++) {
                __m512i av=_mm512_xor_si512(_mm512_loadu_si512(a+(start+r)*k+j),sign);
                for(size_t c=0;c<4;c++)acc[r][c]=_mm512_dpbusd_epi32(acc[r][c],av,ww[c]);
            }
        }
        for(size_t r=0;r<rows;r++)for(size_t c=0;c<4;c++) {
            int32_t total=_mm512_reduce_add_epi32(acc[r][c]);
            for(size_t j=end;j<k;j++)total+=((int32_t)a[(start+r)*k+j]+128)*(int32_t)w[(o+c)*k+j];
            total-=128*sums[o+c];
            out[r*n+o+c]=dequant(total,as[start+r],ws[o+c],bias,o+c);
        }
    }
    for(;o<n;o++)for(size_t r=0;r<rows;r++) {
        int32_t total=0;CHECK(dot_checked(a+(start+r)*k,w+o*k,k,sums[o],&total));
        out[r*n+o]=dequant(total,as[start+r],ws[o],bias,o);
    }
}
static void check_matrix(size_t m,size_t k,size_t n) {
    int8_t *a=alloc(m*k),*w=alloc(n*k);float *as=alloc(m*sizeof(float)),*ws=alloc(n*sizeof(float));
    int32_t *sums=alloc(n*sizeof(int32_t));float *bias=alloc(n*sizeof(float)),*out=alloc(m*n*sizeof(float));
    for(size_t i=0;i<m*k;i++)a[i]=(int8_t)((int)((i*73+19)%255)-127);
    for(size_t i=0;i<n*k;i++)w[i]=(int8_t)((int)((i*29+7)%256)-128);
    for(size_t r=0;r<m;r++)as[r]=(float)(r+1)*0.00371f;
    for(size_t c=0;c<n;c++){ws[c]=(float)(c+1)*0.00037f;bias[c]=(float)((int)c-4)*0.0019375f;sums[c]=weight_sum(w+c*k,k);}
    for(int b=0;b<2;b++) {
        const float *bp=b?bias:NULL;
        for(size_t r=0;r<m;r+=4)tile_candidate(a,as,r,m-r<4?m-r:4,k,w,ws,sums,n,bp,out+r*n);
        for(size_t r=0;r<m;r++)for(size_t c=0;c<n;c++) {
            int64_t d=dot_ref(a+r*k,w+c*k,k);CHECK(d>=INT_MIN&&d<=INT_MAX);
            float expected=dequant((int32_t)d,as[r],ws[c],bp,c);
            CHECK(bits(expected)==bits(out[r*n+c]));
        }
        matrix_cases++;
    }
    free(a);free(w);free(as);free(ws);free(sums);free(bias);free(out);
}
int main(void) {
    __builtin_cpu_init();
    if(!__builtin_cpu_supports("avx512f") || !__builtin_cpu_supports("avx512bw") || !__builtin_cpu_supports("avx512vnni")) {
        fprintf(stderr,"SKIP: AVX512F/BW/VNNI host required\n");return 77;
    }
    CHECK(fesetround(FE_TONEAREST)==0);
    _mm_setcsr(_mm_getcsr() & ~(0x8000u|0x0040u)); /* no FTZ or DAZ */
    printf("Native AVX512F/BW/VNNI enabled; MXCSR=0x%x\n",_mm_getcsr());
    float boundaries[800];size_t nb=0;
    for(int i=-127;i<127;i++){float h=(float)i+0.5f;boundaries[nb++]=nextafterf(h,-INFINITY);boundaries[nb++]=h;boundaries[nb++]=nextafterf(h,INFINITY);}
    float specials[]={0.0f,-0.0f,FLT_MIN,FLT_MAX,INFINITY,-INFINITY,NAN};
    for(size_t i=0;i<sizeof(specials)/sizeof(*specials);i++)boundaries[nb++]=specials[i];
    boundaries[nb++]=from_bits(1);boundaries[nb++]=-from_bits(1);
    float scales[]={0.0f,from_bits(1),FLT_MIN,0.00390625f,0.1f,1.0f,3.7f,65536.0f,INFINITY};
    size_t widths[]={0,1,7,15,16,17,31,32,63,64,65,127,128,129,383,384,385,1536,1537};
    for(size_t q=0;q<sizeof(widths)/sizeof(*widths);q++) {
        size_t k=widths[q];float *storage=alloc((k+1)*sizeof(float)),*x=storage+1;
        for(size_t offset=0;offset<nb;offset+=13) {
            for(size_t i=0;i<k;i++)x[i]=boundaries[(i+offset)%nb];
            for(size_t s=0;s<sizeof(scales)/sizeof(*scales);s++)check_quant(x,k,scales[s]);
            check_quant(x,k,row_scale(x,k));
        }
        free(storage);
    }
    uint32_t rng=0x71a50001u;float random_row[257];
    for(size_t trial=0;trial<1000;trial++) {
        for(size_t i=0;i<257;i++){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;random_row[i]=from_bits(rng);}
        check_quant(random_row,257,row_scale(random_row,257));
    }
    for(size_t k=0;k<=2049;k++) {
        int8_t *a=alloc(k),*b=alloc(k);
        for(int mode=0;mode<5;mode++) {
            for(size_t i=0;i<k;i++) {
                a[i]= mode==1?127:mode==2?-128:(int8_t)((int)((i*17+11)%256)-128);
                b[i]= mode==3?127:mode==4?-128:(int8_t)((int)((i*29+7)%256)-128);
            }
            int32_t got=0;CHECK(dot_checked(a,b,k,weight_sum(b,k),&got));CHECK(got==dot_ref(a,b,k));dot_cases++;
        }
        free(a);free(b);
    }
    size_t ks[]={1,63,64,65,127,128,129,383,384,385,1536,1537};
    for(size_t m=1;m<=9;m++)for(size_t n=1;n<=9;n++)for(size_t i=0;i<sizeof(ks)/sizeof(*ks);i++)check_matrix(m,ks[i],n);
    check_matrix(8,384,384);check_matrix(5,1536,384);check_matrix(4,384,1536);
    printf("PASS: %zu quantizer cases (including nonfinites, subnormals, ties and random bits)\n",quant_cases);
    printf("PASS: %zu signed-dot cases (K=0..2049; full i8 range)\n",dot_cases);
    printf("PASS: %zu tiled matrix/bias cases; exact f32 output bits\n",matrix_cases);
    /* Mathematical dot fits i32, but the candidate's checked correction does not. */
    size_t k=70000;int8_t *a=alloc(k),*b=alloc(k);memset(a,127,k);memset(b,127,k);
    int32_t got=0;int success=dot_checked(a,b,k,weight_sum(b,k),&got);
    int64_t expected=dot_ref(a,b,k);CHECK(expected<=INT_MAX);CHECK(!success);
    printf("CONFIRMED overflow boundary: K=%zu, all a=b=127; exact signed dot=%lld fits i32, but proposed checked intermediate/correction overflows\n",k,(long long)expected);
    printf("Conservative no-overflow bound for u8 x full-i8 correction: K <= %d\n",INT_MAX/(255*128));
    free(a);free(b);return 0;
}
