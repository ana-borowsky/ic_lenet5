#include <Arduino.h>
#include <SPIFFS.h>
#include <esp_timer.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/core/c/c_api_types.h"

#include <dirent.h>

#include "io.h"
#include "ml.h"

namespace
{
  const tflite::Model *model = nullptr;
  tflite::MicroMutableOpResolver<10> *resolver = nullptr;
  tflite::MicroInterpreter *interpreter = nullptr;
  TfLiteTensor *input = nullptr;
  TfLiteTensor *output = nullptr;

  int tensor_arena_size = 256 * 1024;
  uint8_t *tensor_arena = nullptr;
}

void updateProgressBar(int, int);

void setup()
{
  delay(3000);
  Serial.begin(115200);
  SPIFFS.begin();
  print_available_memory();

  char model_file[31] = "/spiffs/mobilenet.tflite";
  interpreter = initializeInterpreter(
      model_file, model, resolver, tensor_arena_size, tensor_arena);

  print_available_memory();
}

void loop()
{
  char test_dir[30] = "/spiffs/test";
  printf("Opening test dir %s\n", test_dir);

  DIR *dir = opendir(test_dir);
  if (dir == NULL)
  {
    printf("Failed to open test dir\n");
    return;
  }

  struct dirent *entry;
  int number_of_files = 2115;
  int file_number = 0;
  char image_file[30];

  int mobilenet_image_size = 32 * 32 * 1;

  uint8_t *image_data = (uint8_t *)malloc(32 * 32);
  float *float_image_data = (float *)malloc(mobilenet_image_size * sizeof(float));

  int prediction = -1;

  int predictions[100] = {0};

  int64_t start = esp_timer_get_time();

  while ((entry = readdir(dir)) != NULL)
  {

    snprintf(image_file, 30, "%s/%s", test_dir, entry->d_name);

    read_file_data(image_file, 32 * 32, image_data);

    for (int i = 0; i < 32 * 32; i++)
    {
      float pixel = image_data[i] / 255.0f;
      float_image_data[i * 3 + 0] = pixel;
      float_image_data[i * 3 + 1] = pixel;
      float_image_data[i * 3 + 2] = pixel;
    }

    prediction = predict(interpreter, float_image_data, mobilenet_image_size);

    include_prediction_in_confusion_matrix(
        predictions, entry->d_name, prediction);

    updateProgressBar(file_number++, number_of_files);
  }

  int64_t stop = esp_timer_get_time();
  printf("\n\nElapsed time: %lld microseconds\n", stop - start);

  print_confusion_matrix(predictions);
  print_available_memory();

  free(image_data);
  free(float_image_data);

  closedir(dir);
  printf("\nTest dir closed\n\n");

  delay(60000);
}

void updateProgressBar(int progress, int total)
{
  const int barWidth = 80;
  float percentage = (float)progress / total;
  int pos = (int)(barWidth * percentage);

  printf("[");
  for (int i = 0; i < barWidth; i++)
  {
    if (i < pos)
      printf("=");
    else
      printf(" ");
  }
  printf("] %.2f%%\r", percentage * 100);
  fflush(stdout);
}
