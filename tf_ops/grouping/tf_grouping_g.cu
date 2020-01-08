// input: radius (1), nsample (1), xyz1 (b,n,c), xyz2 (b,m,c)
// output: idx (b,m,nsample), pts_cnt (b,m)
__global__ void knn_gpu(int b, int n, int c, int m, int nsample, const float *xyz1, const float *xyz2, int *idx, float *dist) {
    int batch_index = blockIdx.x;
    xyz1 += n*c*batch_index;
    xyz2 += m*c*batch_index;
    idx += m*nsample*batch_index;
    dist += m*nsample*batch_index;
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int j=index;j<m;j+=stride) {
        for(int nI=0;nI<nsample;nI++)
        {
            dist[j*nsample+nI]=-1;
            idx[j*nsample+nI]=-1;
        }
        for (int k=0;k<n;++k) {
            float d=0;
            for(int ci=0;ci<c;ci++)
                d+=(xyz2[j*c+ci]-xyz1[k*c+ci])*(xyz2[j*c+ci]-xyz1[k*c+ci]);
    	    d=sqrtf(d);
            if(d<dist[j*nsample+nsample-1]||dist[j*nsample+nsample-1]==-1)
            {
                float d_=d;
                int k_=k;
                bool placed=false;
                for(int nI=0;nI<nsample;nI++)
                {
                    if(placed || d_<=dist[j*nsample+nI]||dist[j*nsample+nI]==-1)
                    {
                        float d_tmp=dist[j*nsample+nI];
                        dist[j*nsample+nI]=d_;
                        d_=d_tmp;
                        int k_tmp=idx[j*nsample+nI];
                        idx[j*nsample+nI]=k_;
                        k_=k_tmp;
                        placed=true;
                    }
                }
            }
        }
    }
}

// input: radius (1), nsample (1), xyz1 (b,n,c), xyz2 (b,m,c)
// output: idx (b,m,nsample), pts_cnt (b,m)
__global__ void query_ball_point_gpu(int b, int n, int c, int m, float radius, int nsample, const float *xyz1, const float *xyz2, int *idx, int *pts_cnt) {
    int batch_index = blockIdx.x;
    xyz1 += n*c*batch_index;
    xyz2 += m*c*batch_index;
    idx += m*nsample*batch_index;
    pts_cnt += m*batch_index; // counting how many unique points selected in local region

    int index = threadIdx.x;
    int stride = blockDim.x;
    
    for (int j=index;j<m;j+=stride) {
        int cnt = 0;
        for (int k=0;k<n;++k) {
            if (cnt == nsample)
                break; // only pick the FIRST nsample points in the ball
            float d=0;
            for(int ci=0;ci<c;ci++)
                d+=(xyz2[j*c+ci]-xyz1[k*c+ci])*(xyz2[j*c+ci]-xyz1[k*c+ci]);
    	    d=max(sqrtf(d),1e-20f);
            if (d<radius) {
                if (cnt==0) { // set ALL indices to k, s.t. if there are less points in ball than nsample, we still have valid (repeating) indices
                    for (int l=0;l<nsample;++l)
                        idx[j*nsample+l] = k;
                }
                idx[j*nsample+cnt] = k;
                cnt+=1;
            }
        }
        pts_cnt[j] = cnt;
    }
}

// input: points (b,n,c), idx (b,m,nsample)
// output: out (b,m,nsample,c)
__global__ void group_point_gpu(int b, int n, int c, int m, int nsample, const float *points, const int *idx, float *out) {
    int batch_index = blockIdx.x;
    points += n*c*batch_index;
    idx += m*nsample*batch_index;
    out += m*nsample*c*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;
    
    for (int j=index;j<m;j+=stride) {
        for (int k=0;k<nsample;++k) {
            int ii = idx[j*nsample+k];
            for (int l=0;l<c;++l) {
                out[j*nsample*c+k*c+l] = points[ii*c+l];
            }
        }
    }
}

// input: grad_out (b,m,nsample,c), idx (b,m,nsample), 
// output: grad_points (b,n,c)
__global__ void group_point_grad_gpu(int b, int n, int c, int m, int nsample, const float *grad_out, const int *idx, float *grad_points) {
    int batch_index = blockIdx.x;
    idx += m*nsample*batch_index;
    grad_out += m*nsample*c*batch_index;
    grad_points += n*c*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int j=index;j<m;j+=stride) {
        for (int k=0;k<nsample;++k) {
            int ii = idx[j*nsample+k];
            for (int l=0;l<c;++l) {
                 atomicAdd(&grad_points[ii*c+l], grad_out[j*nsample*c+k*c+l]);
            }
        }
    }
}

// input: k (1), distance matrix dist (b,m,n)
// output: idx (b,m,n), dist_out (b,m,n)
// only the top k results within n are useful
__global__ void selection_sort_gpu(int b, int n, int m, int k, const float *dist, int *outi, float *out) {
    int batch_index = blockIdx.x;
    dist+=m*n*batch_index;
    outi+=m*n*batch_index;
    out+=m*n*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;

    // copy from dist to dist_out
    for (int j=index;j<m;j+=stride) {
        for (int s=0;s<n;++s) {
            out[j*n+s] = dist[j*n+s];
            outi[j*n+s] = s;
        }
    }

    float *p_dist;
    for (int j=index;j<m;j+=stride) {
        p_dist = out+j*n;
        // selection sort for the first k elements
        for (int s=0;s<k;++s) {
            int min=s; 
            // find the min
            for (int t=s+1;t<n;++t) {
                if (p_dist[t]<p_dist[min]) {
                    min = t;
                }
            }
            // swap min-th and i-th element
            if (min!=s) {
                float tmp = p_dist[min];
                p_dist[min] = p_dist[s];
                p_dist[s] = tmp;
                int tmpi = outi[j*n+min];
                outi[j*n+min] = outi[j*n+s];
                outi[j*n+s] = tmpi;
            }
        }
    }
}

void knnLauncher(int b, int n, int c, int m, int nsample, const float *xyz1, const float *xyz2, int *idx, float *dist) 
{
    knn_gpu<<<b,1024>>>(b,n,c,m,nsample,xyz1,xyz2,idx,dist);
    //cudaDeviceSynchronize();
}
void queryBallPointLauncher(int b, int n, int c, int m, float radius, int nsample, const float *xyz1, const float *xyz2, int *idx, int *pts_cnt) {
    query_ball_point_gpu<<<b,1024>>>(b,n,c,m,radius,nsample,xyz1,xyz2,idx,pts_cnt);
    //cudaDeviceSynchronize();
}
void selectionSortLauncher(int b, int n, int m, int k, const float *dist, int *outi, float *out) {
    selection_sort_gpu<<<b,1024>>>(b,n,m,k,dist,outi,out); 
    //cudaDeviceSynchronize();
}
void groupPointLauncher(int b, int n, int c, int m, int nsample, const float *points, const int *idx, float *out){
    group_point_gpu<<<b,1024>>>(b,n,c,m,nsample,points,idx,out);
    //cudaDeviceSynchronize();
}
void groupPointGradLauncher(int b, int n, int c, int m, int nsample, const float *grad_out, const int *idx, float *grad_points){
    group_point_grad_gpu<<<b,1024>>>(b,n,c,m,nsample,grad_out,idx,grad_points);
    //group_point_grad_gpu<<<1,1>>>(b,n,c,m,nsample,grad_out,idx,grad_points);
    //cudaDeviceSynchronize();
}
