#include <Python.h>
#include <numpy/arrayobject.h>

#include <math.h>
#include <time.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// CUDA
#include <cuda_runtime_api.h>

class CRBM_Data;

void crbm_inference2D(CRBM_Data/*<float>*/ *p);
void crbm_reconstruct2D(CRBM_Data/*<float>*/ *p);

//template<class T>
class CRBM_Data
{
public:
    /*T*/float       *data_input;                // input data
    /*T*/float       *data_kernel;               // kernel
    /*T*/float       *h_bias;                    // bias of hidden layer
    /*T*/float       *v_bias;                    // bias of visible layer
    /*T*/float       *h_sample;                  // hidden values of h_sample
    /*T*/float       *h_sample_init;             // initialization of hidden layer
    /*T*/float       *h_state;                   // the active matrix
    /*T*/float       *v_sample;                  // the visible layer values
    /*T*/float        gauss;                     // gaussian parameter

    int     H, W;                       // input image H & W
    int     N;                          // image number
    int     Wfilter, Hfilter;           // kernel W & H
    int     Wres, Hres;                 // output data W & H
    int     Hstride, Wstride;           // stride of H & W
    int     Hpool, Wpool;               // pool size of H & W
    int     n_map_v, n_map_h;           // map number of v & h
    char    type_input;                 // type of inputdata

    int     run_on_gpu;                 // run on GPU (1, default) or CPU (0)

public:
    CRBM_Data(void) {
        run_on_gpu = 1;
        init();
    }

    ~CRBM_Data(void) {
        if( run_on_gpu )
            release();
        else
            release_no_free();
    }

    int init(void) {
        data_input    = NULL;
        data_kernel   = NULL;
        h_bias        = NULL;
        v_bias        = NULL;
        h_sample      = NULL;
        h_sample_init = NULL;
        h_state       = NULL;
        v_sample      = NULL;

        return 0;
    }

    void release(void) {
        if( data_input    != NULL ) delete [] data_input;
        if( data_kernel   != NULL ) delete [] data_kernel;
        if( h_bias        != NULL ) delete [] h_bias;
        if( v_bias        != NULL ) delete [] v_bias;
        if( h_sample      != NULL ) delete [] h_sample;
        if( h_sample_init != NULL ) delete [] h_sample_init;
        if( h_state       != NULL ) delete [] h_state;
        if( v_sample      != NULL ) delete [] v_sample;

        data_input    = NULL;
        data_kernel   = NULL;
        h_bias        = NULL;
        v_bias        = NULL;
        h_sample      = NULL;
        h_sample_init = NULL;
        h_state       = NULL;
        v_sample      = NULL;

    }

    void release_no_free(void) {
        data_input    = NULL;
        data_kernel   = NULL;
        h_bias        = NULL;
        v_bias        = NULL;
        h_sample      = NULL;
        h_sample_init = NULL;
        h_state       = NULL;
        v_sample      = NULL;
    }

    void setDevice(int d) {
        cudaSetDevice(d);
    }

    void gibbs_sample(void) {
        int i, j;

	setDevice(0);

        crbm_inference2D(this);

        j = Hres * Wres * n_map_h * N;
        for(i = 0; i < j; i++)
		h_sample_init[i] = h_sample[i];

        crbm_reconstruct2D(this);

        j = H * W * n_map_v * N;
        for(i = 0; i < j; i++)
		data_input[i] = v_sample[i];

        crbm_inference2D(this);
    }

    double *calculate_dW(npy_intp *dim_w) {
        int i, j, ii, jj, ni, nv, nh, id;

        int total_size = dim_w[0] * dim_w[1] * dim_w[2] * dim_w[3];
        double *dW = (double*)malloc(sizeof(double) * total_size + 100);

        for(ni = 0; ni < N; ni++) {
            for(nh = 0; nh < n_map_h; nh++) {
                for (j = 0; j < Wfilter; j++) {
                    for (i = 0; i < Hfilter; i++) {
                        for(nv = 0; nv < n_map_v; nv++) {
                            for(jj = 0; jj < Wres; jj++) {
                                for (ii = 0; ii < Hres; ii++) {
                                    id = i + Hfilter*j +
                                        Hfilter*Wfilter*nv +
                                        Hfilter*Wfilter*n_map_v*nh;
                                    dW[id] += (data_input[(ii*Hstride+i) +
                                                        H*(jj*Wstride+j) +
                                                        H*W*nv + H*W*n_map_v*ni] *
                                                this->h_sample_init[(ii+Hres*jj) +
                                                        Hres*Wres*nh +
                                                        Hres*Wres*n_map_h*ni] -

                                                this->v_sample[(ii*Hstride+i) +
                                                        H*(jj*Wstride+j) +
                                                        H*W*nv + H*W*n_map_v*ni] *
                                                this->h_sample[(ii+Hres*jj) +
                                                        Hres*Wres*nh +
                                                        Hres*Wres*n_map_h*ni]);
                                }
                            }
                        }
                    }
                }
            }
        }

        return dW;
    }

};

