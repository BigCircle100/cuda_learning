#include <iostream>

class Vertex;
class Edge;

// 起始点
__global__ void
initialize_vertices(Vertex *vertices, int starting_vertex, int num_vertices){
  int v = blockDim.x * blockIdx.x + threadIdx.x;
  if (v == starting_vertex) 
    vertices[v] = 0; 
  else 
    vertices[v] = -1;
}

// 按照边去遍历相连的两个点，其中一个为当前深度，另一个没访问过，那就把没访问过的深度改为当前深度+1
// v是节点序号，d是深度
// 注：如果是相同深度的不同节点，同时访问相同的未访问过的节点，同时对该未访问节点写入的深度是相同的，因此不影响结果
__global__ void 
bfs(const Edge * edges, Vertex *vertices, int current_depth, bool &done){
  int e = blockDim.x*blockIdx.x + threadIdx.x;
  int vfirst = edges[e].first;
  int dfirst = vertices[vfirst];
  int vsecond = edges[e].second;
  int dsecond = vertices[vsecond];
  if ((dfirst == current_depth) && (dsecond == -1)){
    vertices[vsecond] = dfirst + 1;
    done = false;
  }
  if ((dsecond == current_depth) && (dfirst == -1)){
    vertices[vfirst] = dsecond + 1;
    done = false;
  }
}

int main(){
  // ...
  bool h_done, d_done, h_true;
  Edge *edges;
  Vertex *vertices;
  int current_depth = 0;

  while (!h_done){
    cudaMemcpy(&d_done, &h_true, sizeof(bool), cudaMemcpyHostToDevice);
    bfs(edges, vertices, current_depth, d_done);
    cudaMemcpy(&h_done, &d_done, sizeof(bool), cudaMemcpyDeviceToHost);
    current_depth ++;
  }
}