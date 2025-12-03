#include <iostream>
#include <cuda_runtime.h>
#include <cstring>

// Función para verificar si es letra (versión en CPU)
bool esLetraCPU(char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

// Función para convertir a minúscula (versión en CPU)
char aMinusculaCPU(char c) {
    return (c >= 'A' && c <= 'Z') ? (c + 32) : c;
}

// Función en GPU para verificar si es letra
__device__ bool esLetraGPU(char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

// Función en GPU para verificar si es mayúscula
__device__ bool esMayusculaGPU(char c) {
    return (c >= 'A' && c <= 'Z');
}

// Kernel CUDA para descifrar con Vigenère
__global__ void vigenereDecrypt(char *input, int *keyMap, int inputLen) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = 0;  // Índice de la clave

    while (i < inputLen) {
        if (esLetraGPU(input[i])) {
            char base = esMayusculaGPU(input[i]) ? 'A' : 'a';
            int pi = input[i] - base;    // Posición en el alfabeto (0-25)
            int ki = keyMap[j];          // Usamos el mapa precalculado para la clave

            input[i] = base + (pi - ki + 26) % 26; // Descifrado Vigenère
            j++; // Solo avanzamos en la clave si desciframos una letra
        }
        i += blockDim.x * gridDim.x;  // Asegurar procesamiento paralelo
    }
}

int main() {
    std::string input, key;
    
    // Pedir el texto cifrado
    std::cout << "Ingrese el texto cifrado: ";
    std::getline(std::cin, input);

    // Pedir la clave de descifrado
    std::cout << "Ingrese la clave: ";
    std::getline(std::cin, key);

    int inputLen = input.length();
    int keyLen = key.length();
    
    // Crear un mapa de la clave en la CPU
    int *keyMap = new int[inputLen];
    int j = 0; // Índice para la clave

    for (int i = 0; i < inputLen; i++) {
        if (esLetraCPU(input[i])) {
            keyMap[j] = aMinusculaCPU(key[j % keyLen]) - 'a';
            j++;
        }
    }

    // Reservar memoria en la GPU
    char *d_input;
    int *d_keyMap;
    cudaMalloc((void **)&d_input, inputLen * sizeof(char));
    cudaMalloc((void **)&d_keyMap, j * sizeof(int));

    // Copiar datos a la GPU
    cudaMemcpy(d_input, input.c_str(), inputLen * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_keyMap, keyMap, j * sizeof(int), cudaMemcpyHostToDevice);

    // Configuración de hilos y bloques
    int blockSize = 256;
    int numBlocks = (inputLen + blockSize - 1) / blockSize;

    // Ejecutar el kernel en la GPU
    vigenereDecrypt<<<numBlocks, blockSize>>>(d_input, d_keyMap, inputLen);
    cudaDeviceSynchronize();

    // Copiar resultado de vuelta al host
    char *output = new char[inputLen + 1];
    cudaMemcpy(output, d_input, inputLen * sizeof(char), cudaMemcpyDeviceToHost);
    output[inputLen] = '\0';

    // Liberar memoria en la GPU
    cudaFree(d_input);
    cudaFree(d_keyMap);

    // Mostrar el resultado
    std::cout << "Texto descifrado: " << output << std::endl;

    // Liberar memoria en el host
    delete[] output;
    delete[] keyMap;

    return 0;
}



