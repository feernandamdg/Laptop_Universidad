#include <cuda_runtime.h>
#include <iostream>
#include <string>

// Funciones de dispositivo para reemplazar isalpha, isupper, tolower
__device__ bool d_isalpha(char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

__device__ bool d_isupper(char c) {
    return (c >= 'A' && c <= 'Z');
}

__device__ char d_tolower(char c) {
    if (c >= 'A' && c <= 'Z') {
        return c - 'A' + 'a';
    }
    return c;
}

__global__ void vigenereDecryptKernel(char* input, const char* key, int inputLen, int keyLen) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < inputLen) {
        // Solo procesar caracteres alfabéticos
        if (d_isalpha(input[idx])) {
            char base = d_isupper(input[idx]) ? 'A' : 'a';
            int pi = input[idx] - base; // Posición de la letra en el alfabeto
            int ki = d_tolower(key[idx % keyLen]) - 'a'; // Posición de la letra de la clave
            
            // Descifrar: restar el valor de la clave
            input[idx] = base + (pi - ki + 26) % 26;
        }
    }
}

std::string vigenereDecryptCUDA(const std::string& input, const std::string& key) {
    // Preparar memoria para input
    char* h_input = new char[input.length() + 1];
    strcpy(h_input, input.c_str());
    h_input[input.length()] = '\0';
    
    // Preparar memoria para key
    char* h_key = new char[key.length() + 1];
    strcpy(h_key, key.c_str());
    h_key[key.length()] = '\0';
    
    // Alocar memoria en dispositivo
    char* d_input;
    char* d_key;
    
    cudaMalloc(&d_input, (input.length() + 1) * sizeof(char));
    cudaMalloc(&d_key, (key.length() + 1) * sizeof(char));
    
    // Copiar datos al dispositivo
    cudaMemcpy(d_input, h_input, (input.length() + 1) * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_key, h_key, (key.length() + 1) * sizeof(char), cudaMemcpyHostToDevice);
    
    // Configurar grid y block
    int blockSize = 256;
    int gridSize = (input.length() + blockSize - 1) / blockSize;
    
    // Lanzar kernel
    vigenereDecryptKernel<<<gridSize, blockSize>>>(d_input, d_key, input.length(), key.length());
    
    // Copiar resultado de vuelta al host
    cudaMemcpy(h_input, d_input, (input.length() + 1) * sizeof(char), cudaMemcpyDeviceToHost);
    
    // Liberar memoria
    cudaFree(d_input);
    cudaFree(d_key);
    
    std::string result(h_input);
    
    delete[] h_input;
    delete[] h_key;
    
    return result;
}

int main() {
    std::string mensaje;
    std::string key;
    
    // Pedir mensaje al usuario
    std::cout << "Ingrese el mensaje cifrado: ";
    std::getline(std::cin, mensaje);
    
    // Pedir clave al usuario
    std::cout << "Ingrese la clave de descifrado: ";
    std::getline(std::cin, key);
    
    std::string decrypted = vigenereDecryptCUDA(mensaje, key);
    std::cout << "\nMensaje descifrado:\n" << decrypted << std::endl;
    
    return 0;
}