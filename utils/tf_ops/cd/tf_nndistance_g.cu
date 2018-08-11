#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

__global__ void NmDistanceKernel(int b,int n,const float * xyz,int m,const float * xyz2,float * result,int * result_i){
	const int batch=512;
	__shared__ float buf[batch*2];
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		for (int k2=0;k2<m;k2+=batch){
			int end_k=min(m,k2+batch)-k2;
			for (int j=threadIdx.x;j<end_k*2;j+=blockDim.x){
				buf[j]=xyz2[(i*m+k2)*2+j];
			}
			__syncthreads();
			for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
				float x1=xyz[(i*n+j)*2+0];
				float y1=xyz[(i*n+j)*2+1];
				int best_i=0;
				float best=0;
				int end_ka=end_k-(end_k&3);//
				if (end_ka==batch){
					for (int k=0;k<batch;k+=4){//
						{
							float x2=buf[k*2+0]-x1;
							float y2=buf[k*2+1]-y1;
							float d=x2*x2+y2*y2;
							if (k==0 || d<best){
								best=d;
								best_i=k+k2;
							}
						}
						{
							float x2=buf[k*2+2]-x1;
							float y2=buf[k*2+3]-y1;
							float d=x2*x2+y2*y2;
							if (d<best){
								best=d;
								best_i=k+k2+1;
							}
						}
						{
							float x2=buf[k*2+4]-x1;
							float y2=buf[k*2+5]-y1;
							float d=x2*x2+y2*y2;
							if (d<best){
								best=d;
								best_i=k+k2+2;
							}
						}
						{
							float x2=buf[k*2+6]-x1;
							float y2=buf[k*2+7]-y1;
							float d=x2*x2+y2*y2;
							if (d<best){
								best=d;
								best_i=k+k2+3;
							}
						}
					}
				}else{
					for (int k=0;k<end_ka;k+=4){//
						{
							float x2=buf[k*2+0]-x1;
							float y2=buf[k*2+1]-y1;
							float d=x2*x2+y2*y2;
							if (k==0 || d<best){
								best=d;
								best_i=k+k2;
							}
						}
						{
							float x2=buf[k*2+2]-x1;
							float y2=buf[k*2+3]-y1;
							float d=x2*x2+y2*y2;
							if (d<best){
								best=d;
								best_i=k+k2+1;
							}
						}
						{
							float x2=buf[k*2+4]-x1;
							float y2=buf[k*2+5]-y1;
							float d=x2*x2+y2*y2;
							if (d<best){
								best=d;
								best_i=k+k2+2;
							}
						}
						{
							float x2=buf[k*2+6]-x1;
							float y2=buf[k*2+7]-y1;
							float d=x2*x2+y2*y2;
							if (d<best){
								best=d;
								best_i=k+k2+3;
							}
						}
					}
				}
				for (int k=end_ka;k<end_k;k++){
					float x2=buf[k*2+0]-x1;
					float y2=buf[k*2+1]-y1;
					float d=x2*x2+y2*y2;
					if (k==0 || d<best){
						best=d;
						best_i=k+k2;
					}
				}
				if (k2==0 || result[(i*n+j)]>best){
					result[(i*n+j)]=best;
					result_i[(i*n+j)]=best_i;
				}
			}
			__syncthreads();
		}
	}
}
void NmDistanceKernelLauncher(int b,int n,const float * xyz,int m,const float * xyz2,float * result,int * result_i,float * result2,int * result2_i){
	NmDistanceKernel<<<dim3(32,16,1),512>>>(b,n,xyz,m,xyz2,result,result_i);//
	NmDistanceKernel<<<dim3(32,16,1),512>>>(b,m,xyz2,n,xyz,result2,result2_i);//
}
__global__ void NmDistanceGradKernel(int b,int n,const float * xyz1,int m,const float * xyz2,const float * grad_dist1,const int * idx1,float * grad_xyz1,float * grad_xyz2){
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
			float x1=xyz1[(i*n+j)*2+0];
			float y1=xyz1[(i*n+j)*2+1];
			int j2=idx1[i*n+j];
			float x2=xyz2[(i*m+j2)*2+0];
			float y2=xyz2[(i*m+j2)*2+1];
			float g=grad_dist1[i*n+j]*2;
			atomicAdd(&(grad_xyz1[(i*n+j)*2+0]),g*(x1-x2));
			atomicAdd(&(grad_xyz1[(i*n+j)*2+1]),g*(y1-y2));
			atomicAdd(&(grad_xyz2[(i*m+j2)*2+0]),-(g*(x1-x2)));
			atomicAdd(&(grad_xyz2[(i*m+j2)*2+1]),-(g*(y1-y2)));
		}
	}
}
void NmDistanceGradKernelLauncher(int b,int n,const float * xyz1,int m,const float * xyz2,const float * grad_dist1,const int * idx1,const float * grad_dist2,const int * idx2,float * grad_xyz1,float * grad_xyz2){
	cudaMemset(grad_xyz1,0,b*n*2*4);
	cudaMemset(grad_xyz2,0,b*m*2*4);
	NmDistanceGradKernel<<<dim3(1,16,1),256>>>(b,n,xyz1,m,xyz2,grad_dist1,idx1,grad_xyz1,grad_xyz2);//
	NmDistanceGradKernel<<<dim3(1,16,1),256>>>(b,m,xyz2,n,xyz1,grad_dist2,idx2,grad_xyz2,grad_xyz1);//
}

#endif