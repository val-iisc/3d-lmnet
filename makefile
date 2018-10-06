nvcc = /usr/local/cuda-8.0/bin/nvcc
cudalib = /usr/local/cuda-8.0/lib64/
tensorflow = /usr/local/lib/python2.7/dist-packages/tensorflow/include

all: utils/tf_ops/cd/tf_nndistance_so.so utils/tf_ops/cd/tf_auctionmatch_so.so utils/show_3d/render_balls_so.so
.PHONY : all

utils/tf_ops/cd/tf_nndistance_so.so: utils/tf_ops/cd/tf_nndistance_g.cu.o utils/tf_ops/cd/tf_nndistance.cpp
	g++ -std=c++11 utils/tf_ops/cd/tf_nndistance.cpp utils/tf_ops/cd/tf_nndistance_g.cu.o -o utils/tf_ops/cd/tf_nndistance_so.so -shared -fPIC -I $(tensorflow) -lcudart -L $(cudalib) -O2 -D_GLIBCXX_USE_CXX11_ABI=0

utils/tf_ops/cd/tf_nndistance_g.cu.o: utils/tf_ops/cd/tf_nndistance_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o utils/tf_ops/cd/tf_nndistance_g.cu.o utils/tf_ops/cd/tf_nndistance_g.cu -I $(tensorflow) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2

utils/tf_ops/emd/tf_auctionmatch_so.so: utils/tf_ops/emd/tf_auctionmatch_g.cu.o utils/tf_ops/emd/tf_auctionmatch.cpp
	g++ -std=c++11 utils/tf_ops/emd/tf_auctionmatch.cpp utils/tf_ops/emd/tf_auctionmatch_g.cu.o -o utils/tf_ops/emd/tf_auctionmatch_so.so -shared -fPIC -I $(tensorflow) -lcudart -L $(cudalib) -O2 -D_GLIBCXX_USE_CXX11_ABI=0

utils/tf_ops/emd/tf_auctionmatch_g.cu.o: utils/tf_ops/emd/tf_auctionmatch_g.cu
	$(nvcc) -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o utils/tf_ops/emd/tf_auctionmatch_g.cu.o utils/tf_ops/emd/tf_auctionmatch_g.cu -I $(tensorflow) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 -arch=sm_30

utils/show_3d/render_balls_so.so: utils/show_3d/render_balls_so.cpp
	g++ -std=c++11 utils/show_3d/render_balls_so.cpp -o utils/show_3d/render_balls_so.so -shared -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=0

