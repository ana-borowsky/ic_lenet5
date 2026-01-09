#include <Arduino.h>
#include <SPIFFS.h>
#include <esp_timer.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/core/c/c_api_types.h"

#include <dirent.h>

#include "io.h"
#include "ml.h"

namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroMutableOpResolver<10>* resolver = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  int tensor_arena_size = 256 * 1024;
  uint8_t* tensor_arena = nullptr;
}

void updateProgressBar(int, int);

void setup() {
  delay(3000);
  Serial.begin(115200);
  SPIFFS.begin();
  print_available_memory();

  char model_file[31] = "/spiffs/lenet.tflite";
  interpreter = initializeInterpreter(model_file, model, resolver, tensor_arena_size, tensor_arena);

  print_available_memory();
}

void loop() {
  char test_dir[30] = "/spiffs/test";
  printf("Opening test dir %s\n", test_dir);

  // Open the directory
  DIR* dir = NULL;
  dir = opendir(test_dir);
  if (dir == NULL) {
    printf("Failed to open test dir\n");
    return;
  }

  // Iterate through each entry in the directory
  struct dirent* entry;
  int number_of_files = 2115;
  int file_number = 0;
  char image_file[30];
  int lenet_image_size = 28 * 28 * 1;
  int squeeze_image_size = 32 * 32 * 1;
  uint8_t* image_data = (uint8_t*)malloc(lenet_image_size);
  uint8_t* image_32_data = (uint8_t*)malloc(squeeze_image_size);
  float* float_image_data = (float*)malloc(lenet_image_size * 4);
  float* float_image_32_data = (float*)malloc(squeeze_image_size * 4);
  int prediction;
  int predictions[100] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  int64_t start = esp_timer_get_time();

  while ((entry = readdir(dir)) != NULL) {
    snprintf(image_file, 30, "%s/%s", test_dir, entry->d_name);
    // printf("\nFile: %s\n", image_file);
    read_file_data(image_file, lenet_image_size, image_data);

    // Squeeze needs 32 x 32 images
    // resizeImage(image_data, image_32_data);
    // preprocessImageData(image_32_data, squeeze_image_size, float_image_32_data);
    // prediction = predict(interpreter, float_image_32_data, squeeze_image_size);

    // Other models use 28 x 28 images
    preprocessImageData(image_data, lenet_image_size, float_image_data);
    prediction = predict(interpreter, float_image_data, lenet_image_size);

    // printf("Prediction: %i\n", entry->d_name, prediction);
    include_prediction_in_confusion_matrix(predictions, entry->d_name, prediction);
    updateProgressBar(file_number++, number_of_files);
  }

  int64_t stop = esp_timer_get_time();
  printf("\n\nElapsed time: %lld microseconds\n", stop - start);

  print_confusion_matrix(predictions);

  print_available_memory();

  free(image_data);
  free(image_32_data);
  free(float_image_data);
  free(float_image_32_data);

  // Close the directory
  closedir(dir);
  printf("\nTest dir closed\n\n");

  delay(60000);
}

void updateProgressBar(int progress, int total) {
    const int barWidth = 80;
    float percentage = (float)progress / total;
    int pos = (int)(barWidth * percentage);

    printf("[");
    for (int i = 0; i < barWidth; i++) {
        if (i < pos) {
            printf("=");
        } else {
            printf(" ");
        }
    }
    printf("] %.2f%%\r", percentage * 100);
    fflush(stdout);
}