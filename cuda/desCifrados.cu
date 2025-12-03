#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

// Funciones de archivos
char* leer_archivo(const char* rutaArchivo);
void guardarContenidoEnArchivo(const char* rutaArchivo, const char* contenido);

// Kernel CUDA
__global__ void cifrado_cesar_cuda(char* c, char* cesar, int n, int cif) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int ascii_letra = (int)c[i];
        int r = ascii_letra;

        if (ascii_letra >= 65 && ascii_letra <= 90) {
            r = ascii_letra + cif;
            while (r > 90) r -= 26;
        } else if (ascii_letra >= 97 && ascii_letra <= 122) {
            r = ascii_letra + cif;
            while (r > 122) r -= 26;
        }
        cesar[i] = (char)r;
    }
}

__global__ void descifrado_cesar_cuda(char* c, char* cesar, int n, int cif) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int ascii_letra = (int)c[i];
        int r = ascii_letra;

        if (ascii_letra >= 65 && ascii_letra <= 90) {
            r = ascii_letra - cif;
            while (r < 65) r += 26;
        } else if (ascii_letra >= 97 && ascii_letra <= 122) {
            r = ascii_letra - cif;
            while (r < 97) r += 26;
        }
        cesar[i] = (char)r;
    }
}

int main() {
    int opcion;
    char rutaArchivo[256];
    int clave;

    printf("Seleccione una opción:\n");
    printf("1. Cifrar\n");
    printf("2. Descifrar\n");
    printf("Opción: ");
    scanf("%d", &opcion);
    getchar(); // Limpiar buffer

    printf("Ingrese la ruta del archivo: ");
    fgets(rutaArchivo, sizeof(rutaArchivo), stdin);
    rutaArchivo[strcspn(rutaArchivo, "\n")] = 0;

    char* contenido = leer_archivo(rutaArchivo);
    if (contenido == NULL) {
        printf("Error al leer el archivo.\n");
        return 1;
    }

    int n = strlen(contenido);
    char* resultado = (char*)malloc(n + 1);
    if (resultado == NULL) {
        printf("Error al asignar memoria.\n");
        free(contenido);
        return 1;
    }

    printf("Ingrese la clave de cifrado: ");
    scanf("%d", &clave);

    // Reservar memoria en GPU
    char *d_c, *d_cesar;
    cudaMalloc((void**)&d_c, (n + 1) * sizeof(char));
    cudaMalloc((void**)&d_cesar, (n + 1) * sizeof(char));

    cudaMemcpy(d_c, contenido, (n + 1) * sizeof(char), cudaMemcpyHostToDevice);

    // Configurar malla de hilos
    int blockSize = 1024;
    int gridSize = (n + blockSize - 1) / blockSize;

    if (opcion == 1) {
        cifrado_cesar_cuda<<<gridSize, blockSize>>>(d_c, d_cesar, n, clave);
        cudaMemcpy(resultado, d_cesar, (n + 1) * sizeof(char), cudaMemcpyDeviceToHost);
        guardarContenidoEnArchivo("archivo_cifrado.txt", resultado);
        printf("Archivo cifrado guardado como 'archivo_cifrado.txt'\n");
    } else if (opcion == 2) {
        descifrado_cesar_cuda<<<gridSize, blockSize>>>(d_c, d_cesar, n, clave);
        cudaMemcpy(resultado, d_cesar, (n + 1) * sizeof(char), cudaMemcpyDeviceToHost);
        guardarContenidoEnArchivo("archivo_descifrado.txt", resultado);
        printf("Archivo descifrado guardado como 'archivo_descifrado.txt'\n");
    } else {
        printf("Opción no válida.\n");
    }

    cudaFree(d_c);
    cudaFree(d_cesar);
    free(contenido);
    free(resultado);

    return 0;
}


char* leer_archivo(const char* rutaArchivo) { // ruta del archivo como entrada
    FILE* archivo = fopen(rutaArchivo, "r");
    if (archivo == NULL) {
        perror("Error al abril el archivo");
        return NULL; 
    }

    // Ir al final del archivo
    fseek(archivo, 0, SEEK_END);

    //Obtener el tamaño del archivo
    long tamanoArchivo = ftell(archivo);

    //Volver al inicio del archivo para leerlo
    rewind(archivo);

    //Asignar memoria para contener el archivo completo
    // +1 para el caracter nulo

    char* buffer = (char*)malloc(tamanoArchivo + 1);
    if (buffer == NULL) {
        perror("Error al asignar memoria ");
        fclose(archivo);
        return NULL;
    }

    //Leer el erchivo en el buffer 
    size_t resultadoLectura = fread(buffer, 1, tamanoArchivo, archivo);
    if (resultadoLectura != tamanoArchivo){
        perror("Error al leer el archivo");
        free(buffer);
        fclose(archivo);
        return NULL;
    }

    //Añadir el caracter nulo al final del buffer
    //para indicar el final de la cadena
    buffer[tamanoArchivo] = '\0';

    fclose(archivo);
    return buffer;
}

// para guardar el contenido de un apuntador char en un archivo
void guardarContenidoEnArchivo(const char * rutaArchivo, const char* contenido)
{
    //intentar abrir el archivo para escritura
    FILE* archivo = fopen(rutaArchivo, "w");
    if (archivo == NULL) {
        perror("Error al abrir el archivo");
        return;
    }

    //Escribir el contenido en el archivo
    if(fputs(contenido, archivo) == EOF) {
        perror("Error al escribir el archivo");
    }

    //Cerrar el archivo
    fclose(archivo);
}