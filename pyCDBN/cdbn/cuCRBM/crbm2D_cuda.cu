#include "crbm2D_cuda.h"

/*** ------------------CUDA CONVOLUTION INFERENCE------------------------- ***/

__global__ void conv_cuda_infer(float *da, float *db, float *dc, int H, int W,
                                int Hres, int Wres, int Hfilter,int Wfilter,
                                int Hstride, int Wstride, int Hpool, int Wpool,
                                int n_map_v, int n_map_h,
                                int ni)
{
    int vmap_idx = blockIdx.x, hmap_idx = blockIdx.y;
    int conv_xi  = threadIdx.x, conv_yi;
    int ii, jj;

    float   *da_, *db_, *dc_;
    float   sum;

    // debug
    //arr_calc[vmap_idx*n_map_h + hmap_idx] = 1;

    // get array pointers
    da_ = da + ni*H*W*n_map_v + H*W*vmap_idx;       // input data
    db_ = db + Hfilter*Wfilter*n_map_v*hmap_idx +   // conv kernel
               Hfilter*Wfilter*vmap_idx;
    dc_ = dc + ni*Hres*Wres*n_map_h*n_map_v +       // output data
               Hres*Wres*n_map_v*hmap_idx +
               Hres*Wres*vmap_idx;

    // begin calculation
    for(conv_yi=0; conv_yi<Hres; conv_yi++) {
        sum = 0;

        for(jj =0; jj < Wfilter; jj++) {
            for(ii = 0; ii<Hfilter; ii++) {
                sum += da_[conv_yi*Hstride+ii + H*(conv_xi*Wstride+jj)]
                        * db_[ii + jj*Hfilter];
            }
        }

        dc_[conv_yi+Hres*conv_xi] = sum;
    }
}



/*** ------------------CUDA CONVOLUTION RECONSTRUCTION--------------------- ***/

__global__ void conv_cuda_recon(float *da, float *db, float *dc, int H_off, int W_off,
                                int H, int W, int Hfilter,int Wfilter,
                                int Hstride, int Wstride, int Hpool, int Wpool,
                                int n_map_v, int n_map_h,
                                int ni)
{
    int hmap_idx = blockIdx.x, vmap_idx = blockIdx.y;
    int conv_xi  = threadIdx.x, conv_yi;
    int ii, jj;

    float   *da_, *db_, *dc_;
    float   sum;


    // get array pointers
    da_ = da + ni*H_off*W_off*n_map_h + H_off*W_off*hmap_idx;       // input data
    db_ = db + Hfilter*Wfilter*n_map_v*hmap_idx +   // conv kernel
               Hfilter*Wfilter*vmap_idx;
    dc_ = dc + ni*H*W*n_map_v*n_map_h +       // output data
               H*W*n_map_h*vmap_idx +
               H*W*hmap_idx;

    // begin calculation
    for(conv_yi=0; conv_yi<H; conv_yi++) {
        sum = 0;

        for(jj =0; jj < Wfilter; jj++) {
            for(ii = 0; ii<Hfilter; ii++) {
                sum += da_[conv_yi*Hstride+ii + H_off*(conv_xi*Wstride+jj)]
                        * db_[Hfilter*Wfilter-1-(ii + jj*Hfilter)];
            }
        }

        dc_[conv_yi+H*conv_xi] = sum;
    }
}


/*** -------------------------CUDA MERGE INFERENCE------------------------- ***/

__global__ void conv_merge_infer(float *dc, float *dh, float *dd, int H, int W,
                                 int Hres, int Wres,int Hfilter,int Wfilter,
                                 int Hstride, int Wstride, int Hpool, int Wpool,
                                 int n_map_v, int n_map_h,
                                 int ni, char type_input, float gauss)
{
    int hmap_idx = blockIdx.x, vmap_idx;
    int jj,ii;

    float   *dc_, *dd_;


    dd_ = dd + ni*Hres*Wres*n_map_h + Hres*Wres*hmap_idx;

    // merge maps to single feature map
    for(vmap_idx = 0; vmap_idx < n_map_v; vmap_idx++) {
        dc_ = dc + ni*Hres*Wres*n_map_h*n_map_v + Hres*Wres*n_map_v*hmap_idx + Hres*Wres*vmap_idx;

        for(jj = 0; jj < Wres; jj++) {
            for(ii = 0; ii < Hres; ii++) {
                dd_[ii+jj*Hres] += dc_[ii+jj*Hres];
            }
        }
    }

    // apply bias
    for(jj = 0; jj < Wres; jj++) {
        for(ii = 0; ii < Hres; ii++) {

            if (type_input == 'B')
                dd_[ii+jj*Hres] = exp(dd_[ii+jj*Hres] + dh[hmap_idx]);
            if (type_input == 'G')
                dd_[ii+jj*Hres] = exp(1.0/(gauss*gauss)*(dd_[ii+jj*Hres] + dh[hmap_idx]));
        }
    }
}



/*** -------------------------CUDA MERGE RECONSTRUCTION---------------------- ***/

__global__ void conv_merge_recon(float *dc, float *dv, float *dd, int H_off, int W_off,
                                 int H, int W,int Hfilter,int Wfilter,
                                 int Hstride, int Wstride, int Hpool, int Wpool,
                                 int n_map_v, int n_map_h,
                                 int ni, char type_input)
{
    int vmap_idx = blockIdx.x, hmap_idx;
    int jj,ii;

    float   *dc_, *dd_;


    dd_ = dd + ni*H*W*n_map_v + H*W*vmap_idx;

    // merge maps to single feature map
    for(hmap_idx = 0; hmap_idx < n_map_h; hmap_idx++) {
        dc_ = dc + ni*H*W*n_map_v*n_map_h + H*W*n_map_h*vmap_idx + H*W*hmap_idx;

        for(jj = 0; jj < W; jj++) {
            for(ii = 0; ii < H; ii++) {
                dd_[ii+jj*H] += dc_[ii+jj*H];
            }
        }
    }

    // apply bias
    for(jj = 0; jj < W; jj++) {
        for(ii = 0; ii < H; ii++) {

            if (type_input == 'B')
                dd_[ii+jj*H] = 1.0/(1.0+exp(-(dd_[ii+jj*H] + dv[vmap_idx])));
            if (type_input == 'G')
                dd_[ii+jj*H] = dd_[ii+jj*H] + dv[vmap_idx];
        }
    }
}


/*** ------------------------- NON-KERNEL FUNCTIONS ---------------------- ***/

//BOTTOM-UP: POSITIVE UPDATE
void crbm_inference2D(CRBM_Data/*<float>*/ *p)
{
    int         ni, i, j, ii, jj, nh,
                H, W, n_map_v, n_map_h, N,
                Hfilter, Wfilter, Hstride, Wstride,
                Hpool, Wpool, Hres, Wres;

    int         *_id;

    float       sum, rnd, pro_sum, gauss;
    float       *block;
    bool        done;
    char        type_input;

    H       = p->H;
    W       = p->W;
    N       = p->N;
    Hres    = p->Hres;
    Wres    = p->Wres;
    Hpool   = p->Hpool;
    Wpool   = p->Wpool;
    n_map_v = p->n_map_v;
    n_map_h = p->n_map_h;
    Hfilter = p->Hfilter;
    Wfilter = p->Wfilter;
    Hstride = p->Hstride;
    Wstride = p->Wstride;
    gauss   = p->gauss;
    type_input = p->type_input;


    // Initialize matrixs
    j       = Hres*Wres*n_map_h*N;
    block   = new float[j];
    for(i= 0; i< j; i++) block[i] = 0;

    _id     = new int[Hpool*Wpool];
    for(i= 0; i< Hpool*Wpool; i++) _id[i] = 0;

    /***---------------------------CUDA CODE------------------------------***/

    int SIZE_IMAGE, SIZE_FILTER, SIZE_OUTPUT;
    float *da, *db, *dc, *dd, *dh, *fc;

    j = Hres*Wres*n_map_v*n_map_h*N;
    fc = new float[j];
    for(i=0; i< j; i++) fc[i] = 0;
    //cudaMallocHost(&fc, sizeof(float)*Hres*Wres*n_map_v*n_map_h*N);
    //memset(fc, 0, sizeof(float)*Hres*Wres*n_map_v*n_map_h*N);

    SIZE_IMAGE  = H * W * n_map_v * N;
    SIZE_FILTER = Hfilter * Wfilter * n_map_v * n_map_h;
    SIZE_OUTPUT = Hres * Wres * n_map_h * N;

    cudaMalloc(&da, sizeof(float) * SIZE_IMAGE);
    cudaMalloc(&db, sizeof(float) * SIZE_FILTER);
    cudaMalloc(&dc, sizeof(float) * SIZE_OUTPUT*n_map_v);
    cudaMalloc(&dd, sizeof(float) * SIZE_OUTPUT);
    cudaMalloc(&dh, sizeof(float) * n_map_h);

    cudaMemcpy(da,p->data_input,    sizeof(float)*SIZE_IMAGE,         cudaMemcpyHostToDevice);
    cudaMemcpy(db,p->data_kernel,   sizeof(float)*SIZE_FILTER,        cudaMemcpyHostToDevice);
    cudaMemcpy(dc,fc,               sizeof(float)*SIZE_OUTPUT*n_map_v,cudaMemcpyHostToDevice);
    cudaMemcpy(dd,block,            sizeof(float)*SIZE_OUTPUT        ,cudaMemcpyHostToDevice);
    cudaMemcpy(dh,p->h_bias,        sizeof(float)*n_map_h,            cudaMemcpyHostToDevice);

    dim3    blocks(n_map_v, n_map_h);
    dim3    threads(Wres, 1);

    dim3    blocks2(n_map_h, 1);
    dim3    threads2(1, 1);

    for(ni=0; ni< N; ni++){

        conv_cuda_infer<<<blocks, threads>>>(da, db, dc,
                                             H, W, Hres, Wres, Hfilter, Wfilter,
                                             Hstride, Wstride, Hpool, Wpool,
                                             n_map_v,n_map_h, ni);

        conv_merge_infer<<<blocks2, threads2>>>(dc,dh, dd,
                                             H, W, Hres, Wres, Hfilter, Wfilter,
                                             Hstride, Wstride, Hpool, Wpool,
                                             n_map_v, n_map_h, ni, type_input, gauss);
    }

    cudaMemcpy(block, dd, sizeof(float) * SIZE_OUTPUT, cudaMemcpyDeviceToHost);


    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    cudaFree(dd);
    cudaFree(dh);
    //cudaFreeHost(fc);
    delete [] fc;


    /***---------------------------CUDA END------------------------------***/

    /*** CONVOLUTION & GET HIDDEN ACTIVATION STATE ***/
    for(ni=0; ni< N; ni++){
        for(nh=0; nh< n_map_h; nh++){

            //GET HIDDEN ACTIVATION STATE

            for(j=0; j< floor(Wres/Wpool); j++){
                for(i=0; i< floor(Hres/Hpool); i++){

                    sum = 0;

                    for(jj=0; jj< Wpool; jj++){

                        _id[jj*Hpool] = i*Hpool + (j*Wpool+jj)*Hres + Hres*Wres*nh + Hres*Wres*n_map_h*ni;
                        sum          += block[_id[jj*Hpool]];

                        for(ii=1; ii< Hpool; ii++){
                            _id[jj*Hpool+ii] = _id[jj*Hpool+ii-1] + 1;
                            sum             += block[_id[jj*Hpool+ii]];
                        }
                    }

                    done     = false;
                    rnd      = rand() % 10000 / 10000.0;
                    pro_sum  = 0.0;

                    for(jj=0; jj< Hpool*Wpool; jj++){
                        p->h_sample[_id[jj]] = block[_id[jj]]/(1.0+sum);
                        pro_sum += p->h_sample[_id[jj]];

                        //Randomly generate the hidden state: at most one unit is activated
                        if(done == false){
                            if(pro_sum >= rnd){
                                p->h_state[_id[jj]] = 1;
                                done = true;
                            }
                        }
                    }

                }
            }

        }
    }

    delete [] _id;
    delete [] block;
    return;
}


// UP-DOWN: NEGATIVE UPDATE
void crbm_reconstruct2D(CRBM_Data/*<float>*/ *p)
{
    int         ni, i, j, ii, jj, nh, nv, id,
                H, W, n_map_v, n_map_h, N,
                Hfilter, Wfilter, Hstride, Wstride,
                Hpool, Wpool, Hres, Wres,
                offset_h, offset_w, H_off, W_off;

    float       *h_state_off, *v;
    char        type_input;

    H       = p->H;
    W       = p->W;
    N       = p->N;
    Hres    = p->Hres;
    Wres    = p->Wres;
    Hpool   = p->Hpool;
    Wpool   = p->Wpool;
    n_map_v = p->n_map_v;
    n_map_h = p->n_map_h;
    Hfilter = p->Hfilter;
    Wfilter = p->Wfilter;
    Hstride = p->Hstride;
    Wstride = p->Wstride;
    type_input = p->type_input;

    j = H*W*n_map_v*N;
    v = new float[j];
    for(i=0; i< j; i++) v[i] = 0;

    //extend the matrix of h_state
    offset_h  = (H-1)*Hstride*Hstride+(Hfilter-1)*Hstride+Hfilter-H;
    offset_w  = (W-1)*Wstride*Wstride+(Wfilter-1)*Wstride+Wfilter-W;
    H_off     = Hres + offset_h;
    W_off     = Wres + offset_w;

    j           = H_off*W_off*n_map_h*N;
    h_state_off = new float[j];
    for(i=0; i< j; i++) h_state_off[i] = 0;

    for(ni=0; ni< N; ni++){
        for(nh=0; nh< n_map_h; nh++){
            for(j=0; j< Wres; j++){
                for(i=0; i< Hres; i++){

                    h_state_off[i + offset_h/2 + H_off*(j+offset_w/2) + H_off*W_off*nh + H_off*W_off*n_map_h*ni]
                   = p->h_state[i + Hres*j + Hres*Wres*nh + Hres*Wres*n_map_h*ni];

                }
            }
        }
    }

    /***--------------------------CUDA CODE----------------------------***/
    if (0) {
    int SIZE_IMAGE, SIZE_FILTER, SIZE_OUTPUT;
    float *da, *db, *dc, *dd, *dv, *fc;

    j = H*W*n_map_h*n_map_v*N;
    fc = new float[j];
    for(i=0; i< j; i++) fc[i] = 0;

    SIZE_IMAGE  = H_off * W_off * n_map_h * N;
    SIZE_FILTER = Hfilter * Wfilter * n_map_v * n_map_h;
    SIZE_OUTPUT = H * W * n_map_v * N;

    cudaMalloc(&da, sizeof(float) * SIZE_IMAGE);
    cudaMalloc(&db, sizeof(float) * SIZE_FILTER);
    cudaMalloc(&dc, sizeof(float) * SIZE_OUTPUT*n_map_h);
    cudaMalloc(&dd, sizeof(float) * SIZE_OUTPUT);
    cudaMalloc(&dv, sizeof(float) * n_map_h);

    cudaMemcpy(da,h_state_off,    sizeof(float)*SIZE_IMAGE,         cudaMemcpyHostToDevice);
    cudaMemcpy(db,p->data_kernel, sizeof(float)*SIZE_FILTER,        cudaMemcpyHostToDevice);
    cudaMemcpy(dc,fc,             sizeof(float)*SIZE_OUTPUT*n_map_v,cudaMemcpyHostToDevice);
    cudaMemcpy(dd,v,              sizeof(float)*SIZE_OUTPUT        ,cudaMemcpyHostToDevice);
    cudaMemcpy(dv,p->v_bias,      sizeof(float)*n_map_h,            cudaMemcpyHostToDevice);

    dim3    blocks(n_map_h, n_map_v);
    dim3    threads(W, 1);

    dim3    blocks2(n_map_v, 1);
    dim3    threads2(1, 1);

    for(ni=0; ni< N; ni++){

        conv_cuda_recon<<<blocks, threads>>>(da, db, dc,
                                             H_off, W_off, H, W, Hfilter, Wfilter,
                                             Hstride, Wstride, Hpool, Wpool,
                                             n_map_v,n_map_h, ni);

        conv_merge_recon<<<blocks2, threads2>>>(dc,dv, dd,
                                             H_off, W_off, H, W, Hfilter, Wfilter,
                                             Hstride,Wstride,Hpool,Wpool,
                                             n_map_v,n_map_h, ni,type_input);
    }

    cudaMemcpy(v, dd, sizeof(float) * SIZE_OUTPUT, cudaMemcpyDeviceToHost);

    for(i=0; i< H*W*n_map_v*N; i++) p->v_sample[i] = v[i];
    }

    /***---------------------------CUDA END---------------------------***/


    //do the convolution
    for(ni=0; ni< N; ni++){
        for(nv=0; nv< n_map_v; nv++){
            for(j=0; j< W; j++){
                for(i=0; i< H; i++){

                    id    = i + H*j + H*W*nv + H*W*n_map_v*ni;
                    v[id] = 0;

                    for (nh = 0; nh< n_map_h; nh++){
                        for (jj = 0; jj< Wfilter; jj++){
                            for (ii = 0; ii < Hfilter; ii++){

                                v[id] += h_state_off[(i*Hstride+ii) + H_off*(j*Wstride+jj) + H_off*W_off*nh + H_off*W_off*n_map_h*ni]
                                        * p->data_kernel[Hfilter*Wfilter-1-(ii+Hfilter*jj) + Hfilter*Wfilter*nv + Hfilter*Wfilter*n_map_v*nh];

                            }

                        }
                    }

                    v[id]          += p->v_bias[nv];

                    if (type_input == 'B')
                        p->v_sample[id] = 1.0/(1.0+exp(-v[id]));
                    if (type_input == 'G')
                        p->v_sample[id] = v[id];

                }
            }
        }
    }


    delete [] h_state_off;
    delete [] v;
    //delete [] fc;
    return;
}


