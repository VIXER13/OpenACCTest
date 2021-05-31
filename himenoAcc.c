#include <stdio.h>
#include <sys/time.h>

#define SMALL

#ifdef SMALL
#define MIMAX            129
#define MJMAX            65
#define MKMAX            65
#endif

#ifdef MIDDLE
#define MIMAX            257
#define MJMAX            129
#define MKMAX            129
#endif

#ifdef LARGE
#define MIMAX            513
#define MJMAX            257
#define MKMAX            257
#endif

#ifdef EXTLARGE
#define MIMAX            1025
#define MJMAX            513
#define MKMAX            513
#endif

#define NN               200



static float  p[MIMAX][MJMAX][MKMAX];
static float  a[MIMAX][MJMAX][MKMAX][4]; 
static float b[MIMAX][MJMAX][MKMAX][3]; 
static float c[MIMAX][MJMAX][MKMAX][3];
static float  bnd[MIMAX][MJMAX][MKMAX];
static float  wrk1[MIMAX][MJMAX][MKMAX];
static float wrk2[MIMAX][MJMAX][MKMAX];
double gosa = 0.0;


static int imax = MIMAX-1, jmax = MJMAX-1, kmax = MKMAX-1;
static float omega = 0.8;

void initmt()
{
	
	#pragma acc parallel loop present(a,b,c,p,wrk1,bnd)	
		for(size_t i = 0; i < imax; ++i)
			for(size_t j = 0; j < jmax; ++j)
				for(size_t k = 0; k < kmax; ++k)
				{
					a[i][j][k][0] = 0.0;
					a[i][j][k][1] = 0.0;
					a[i][j][k][2] = 0.0;
					a[i][j][k][3] = 0.0;
					b[i][j][k][0] = 0.0;
					b[i][j][k][1] = 0.0;
					b[i][j][k][2] = 0.0;
					c[i][j][k][0] = 0.0;
					c[i][j][k][1] = 0.0;
					c[i][j][k][2] = 0.0;
					p[i][j][k]    = 0.0;
					wrk1[i][j][k] = 0.0;
					bnd[i][j][k]  = 0.0;
				}
	#pragma acc parallel loop present(a,b,c,p,wrk1,bnd)			
		for(size_t i = 0; i < imax; ++i)
			for(size_t j = 0; j < jmax; ++j)
				for(size_t k = 0; k < kmax; ++k)
				{
					a[i][j][k][0] = 1.0;
					a[i][j][k][1] = 1.0;
					a[i][j][k][2] = 1.0;
					a[i][j][k][3] = 1.0 / 6.0;
					b[i][j][k][0] = 0.0;
					b[i][j][k][1] = 0.0;
					b[i][j][k][2] = 0.0;
					c[i][j][k][0] = 1.0;
					c[i][j][k][1] = 1.0;
					c[i][j][k][2] = 1.0;
					p[i][j][k]    = (float)(k*k)/(float)((kmax-1)*(kmax-1));
					wrk1[i][j][k] = 0.0;
					bnd[i][j][k]  = 1.0;
				}
}

float jacobi(const size_t nn)
{
    float gosa = 0.;//, s0 = 0., ss = 0.;
	#pragma acc update self(gosa)	
    for(size_t n = 0; n < nn; ++n)
    {
        gosa = 0.0;
		#pragma acc parallel loop present(p,a,b,c,bnd,wrk1,wrk2) reduction(+: gosa)
			for(size_t i = 1; i < imax-1; ++i)
				for(size_t j = 1; j < jmax-1; ++j)
					for(size_t k = 1; k < kmax-1; ++k)
					{
						float s0 = a[i][j][k][0] * p[i+1][j  ][k  ]
						   + a[i][j][k][1] * p[i  ][j+1][k  ]
						   + a[i][j][k][2] * p[i  ][j  ][k+1]

						   + b[i][j][k][0] * (p[i+1][j+1][k  ] - p[i+1][j-1][k  ]
											- p[i-1][j+1][k  ] + p[i-1][j-1][k  ])
						   + b[i][j][k][1] * (p[i  ][j+1][k+1] - p[i  ][j-1][k+1]
											- p[i  ][j+1][k-1] + p[i  ][j-1][k-1])
						   + b[i][j][k][2] * (p[i+1][j  ][k+1] - p[i-1][j  ][k+1]
											- p[i+1][j  ][k-1] + p[i-1][j  ][k-1])

						   + c[i][j][k][0] * p[i-1][j  ][k  ]
						   + c[i][j][k][1] * p[i  ][j-1][k  ]
						   + c[i][j][k][2] * p[i  ][j  ][k-1]
						   + wrk1[i][j][k];

						float ss = (s0 * a[i][j][k][3] - p[i][j][k]) * bnd[i][j][k];

						gosa += ss*ss;

						wrk2[i][j][k] = p[i][j][k] + omega * ss;
					}

		#pragma acc parallel loop present(p,wrk2)
			for(size_t i = 1; i < imax-1; ++i)
				for(size_t j = 1; j < jmax-1; ++j)
					for(size_t k = 1; k < kmax-1; ++k)
						p[i][j][k] = wrk2[i][j][k];
		
		#pragma acc update self(gosa)	
    } /* end n loop */

    return gosa;
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

int main()
{
	#pragma acc enter data create(p[MIMAX][MJMAX][MKMAX],a[MIMAX][MJMAX][MKMAX][4],b[MIMAX][MJMAX][MKMAX][3],c[MIMAX][MJMAX][MKMAX][3])
	#pragma acc enter data create(bnd[MIMAX][MJMAX][MKMAX],wrk1[MIMAX][MJMAX][MKMAX], wrk2[MIMAX][MJMAX][MKMAX], gosa)
	
	double time = get_time();
    initmt();

    printf("mimax = %d mjmax = %d mkmax = %d\n", MIMAX, MJMAX, MKMAX);
    printf("imax = %d jmax = %d kmax =%d\n", imax, jmax, kmax);

    
    gosa = jacobi(NN);
    time = get_time() - time;

    double nflop = (kmax-2)*(jmax-2)*(imax-2)*34,
           xmflops2 = nflop / time * 1.0e-6 * NN,
           score = xmflops2 / 32.27;
    
    printf("cpu : %f sec.\n", time);
    printf("Loop executed for %d times\n",NN);
    printf("Gosa : %e \n",gosa);
    printf("MFLOPS measured : %f\n",xmflops2);
    printf("Score based on MMX Pentium 200MHz : %f\n",score);
    
	#pragma acc exit data delete(p[MIMAX][MJMAX][MKMAX],a[MIMAX][MJMAX][MKMAX][4], b[MIMAX][MJMAX][MKMAX][3],c[MIMAX][MJMAX][MKMAX][3])
	#pragma acc exit data delete(bnd[MIMAX][MJMAX][MKMAX],wrk1[MIMAX][MJMAX][MKMAX], wrk2[MIMAX][MJMAX][MKMAX], gosa)
    
	return 0;
}