#include "Arduino.h"

int file_size(char* path) {
    // printf("Opening file %s...", path);

    // Open the file
    FILE* file = NULL;
    file = fopen(path, "r");
    if (file == NULL)
    {
        printf("\nFailed to open file: %s\n", path);
        return 0;
    }
    // printf("Opened...");

    // Determine the size of the file
    fseek(file, 0, SEEK_END);
    int size = ftell(file);
    fseek(file, 0, SEEK_SET);
    // printf("Size: %i bytes...", size);

    // Close the file
    fclose(file);
    // printf("Closed!\n");

    return size;
}

void read_file_data(char* path, int size, uint8_t* data)
{
    // printf("Opening file %s...", path);

    // Open the file
    FILE* file = NULL;
    file = fopen(path, "r");
    if (file == NULL)
    {
        printf("\nFailed to open file: %s\n", path);
        return;
    }
    // printf("Opened...");

    // Read the file data from the file
    if (fread(data, 1, size, file) != size)
    {
        printf("\nFailed to read file data\n");
        fclose(file);
        free(data);
        data = NULL;
        return;
    }
    // printf("Read...");

    // Close the file
    fclose(file);
    // printf("Closed!\n");
}