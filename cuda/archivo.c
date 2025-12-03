#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char* leer_archivo(const char* rutaArchivo);
void guardarContenidoEnArchivo(const char * rutaArchivo, const char* contenido);
void cifrado_cesar(char *c,char *cesar, int n, int cif);
void descifrado_cesar(char *c,char *cesar, int n, int cif);

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
    rutaArchivo[strcspn(rutaArchivo, "\n")] = 0; // Eliminar el salto de línea
    
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
    
    if (opcion == 1) {
        cifrado_cesar(contenido, resultado, n, clave);
        guardarContenidoEnArchivo("archivo_cifrado.txt", resultado);
        printf("Archivo cifrado guardado como 'archivo_cifrado.txt'\n");
    } else if (opcion == 2) {
        descifrado_cesar(contenido, resultado, n, clave);
        guardarContenidoEnArchivo("archivo_descifrado.txt", resultado);
        printf("Archivo descifrado guardado como 'archivo_descifrado.txt'\n");
    } else {
        printf("Opción no válida.\n");
    }
    
    free(contenido);
    free(resultado);
    return 0;
}
char* leer_archivo(const char* rutaArchivo) 
{ // ruta del archivo como entrada
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

/*
c es el nombre del arreglo de la cadena
cesar es el apuntador al arreglo final
n es la longitud de la cadena
cif es el numero cifrador */

void cifrado_cesar(char *c,char *cesar, int n, int cif)
{
    int r;

    for(int i = 0; i < n; i++){
        int ascii_letra= (int)*(c+i);

        if(ascii_letra >= 65 && ascii_letra <= 90){
            r = (int) *(c+i) + cif;
            while(1){
                if(r > 90 && ascii_letra <= 90){
                    r = r - 26;
                }else{
                    break;
                }
            }
            *(cesar+i) = (char) r;
        }else if(ascii_letra >= 97 && ascii_letra <= 122){
            r = (int) *(c+i) + cif;
            while(1){
                if(r > 122 && (ascii_letra <= 122)){
                    r = r - 26;
                }else{
                    break;
                }
            }
            *(cesar+i) = (char) r;
        }else{
            *(cesar + i) = *(c+i);
        }
    }
}

void descifrado_cesar(char *c,char *cesar, int n, int cif){
    int r;

    for(int i = 0; i < n; i++){
        int ascii_letra= (int)*(c+i);

        if(ascii_letra >= 65 && ascii_letra <= 90){
            r = (int) *(c+i) - cif;
            while(1){
                if(r < 65 && ascii_letra <= 90){
                    r = r + 26;
                }else{
                    break;
                }
            }
            *(cesar+i) = (char) r;
        }else if(ascii_letra >= 97 && ascii_letra <= 122){
            r = (int) *(c+i) - cif;
            while(1){
                if(r < 97 && (ascii_letra <= 97)){
                    r = r + 26;
                }else{
                    break;
                }
            }
            *(cesar-i) = (char) r;
        }else{
            *(cesar - i) = *(c - i);
        }
    }
}
