#include <stdio.h> 

#include <stdlib.h> 

#include <cuda_runtime.h> 

#define N 8 

 

// Función principal ejecutada en el host 

int main(int argc, char** argv) 

{ 

 // Declaración de variables 

 float *m_host; 

 float *m_device; 

 

// Reservar memoria en el host 

 m_host = (float *) malloc ( N * N * sizeof(float) ); 

 

// Reservar memoria en el device 

 cudaMalloc( (void**) &m_device, N * N * sizeof(float) ); 

 

// Inicializar la matriz 

 

 for (int i = 0; i < N * N; i++) 

 { 

     m_host[i] = (float) ( rand() % 10 ); 

 } 

 

// Copiar información al device 

 cudaMemcpy(m_device, m_host, N * N * sizeof(float), cudaMemcpyHostToDevice); 

 

// Liberar memoria  

 cudaFree( m_device ); 

 

 printf("\npulsa INTRO para finalizar..."); 

 fflush(stdin); 

 char tecla = getchar(); 

 

 return 0; 

} 