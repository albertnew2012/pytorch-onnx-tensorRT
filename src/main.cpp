#include <fstream>
#include <iostream>
#include <vector>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <opencv2/opencv.hpp>
//#include <torch/torch.h>

// Define a macro to check CUDA errors
#define CHECK_CUDA(call) \
    do { \
        cudaError_t status = call; \
        if (status != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(status) << std::endl; \
            return false; \
        } \
    } while(0)


int indexOfMax(float *array, int size) {
    if (size <= 0) return -1; // Handle empty array

    int maxIndex = 0; // Start with the first index as the max
    float maxValue = array[0]; // And the first element as the max value

    for (int i = 1; i < size; ++i) {
        if (array[i] > maxValue) {
            maxValue = array[i]; // Update max value
            maxIndex = i; // Update index of max value
        }
    }

    return maxIndex; // Return the index of the max value
}

void preprocessImage(const std::string &imagePath, float *inputBuffer) {
    // Load the image as grayscale with OpenCV:
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);

    // Check if the image is loaded successfully
    if (img.empty()) {
        throw std::runtime_error("Error: Image at " + imagePath + " could not be loaded.");
    }

    // Ensure the image is 28x28:
    if (img.rows != 28 || img.cols != 28) {
        cv::resize(img, img, cv::Size(28, 28));
    }

    // Convert to float32:
    img.convertTo(img, CV_32FC1);

    // Normalize the image if required by your model:
    img = (img - 127.5f) / 127.5f;

    // Copy the image to the input buffer:
    std::memcpy(inputBuffer, img.data, 28 * 28 * sizeof(float));
}

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char *msg) noexcept override {
        // Print log messages only if they are warnings (or more severe)
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
} gLogger;

std::shared_ptr<nvinfer1::ICudaEngine> loadEngine(const std::string &enginePath, nvinfer1::IRuntime *&runtime) {
    std::ifstream engineFile(enginePath, std::ios::binary);
    if (!engineFile) {
        std::cerr << "Error opening engine file: " << enginePath << std::endl;
        return nullptr;
    }

    engineFile.seekg(0, engineFile.end);
    long int fsize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);
    if (!engineFile) {
        std::cerr << "Error loading engine file: " << enginePath << std::endl;
        return nullptr;
    }

    runtime = nvinfer1::createInferRuntime(gLogger);
    return std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr), [](nvinfer1::ICudaEngine* engine) { engine->destroy(); });
//    return std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr), [](nvinfer1::ICudaEngine* engine) { engine->destroy(); });
}

bool allocateBuffers(nvinfer1::ICudaEngine *engine, std::vector<void *> &buffers, int &inputIndex, int &outputIndex) {
    // Assuming the engine has one input and one output
    inputIndex = engine->getBindingIndex("input");  // Use your actual input name
    outputIndex = engine->getBindingIndex("output"); // Use your actual output name

    if (inputIndex == -1 || outputIndex == -1) {
        std::cerr << "Invalid input or output name!" << std::endl;
        return false;
    }

    // Allocate input buffer
    nvinfer1::Dims inputDims = engine->getBindingDimensions(inputIndex);
    size_t inputSize = 1;
    for (int i = 0; i < inputDims.nbDims; ++i) {
        inputSize *= inputDims.d[i];
    }
    CHECK_CUDA(cudaMalloc(&buffers[inputIndex], inputSize * sizeof(float)));

    // Allocate output buffer
    nvinfer1::Dims outputDims = engine->getBindingDimensions(outputIndex);
    size_t outputSize = 1;
    for (int i = 0; i < outputDims.nbDims; ++i) {
        outputSize *= outputDims.d[i];
    }
    CHECK_CUDA(cudaMalloc(&buffers[outputIndex], outputSize * sizeof(float)));

    return true;
}

void displayImageWithCaption(const std::string &imagePath, const std::string &prediction) {
    // Read the image
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        std::cerr << "Error: Image cannot be loaded..!!" << std::endl;
        return;
    }

    // Set parameters for the caption
    int fontFace = cv::FONT_HERSHEY_COMPLEX_SMALL;
    double fontScale = 1;
    int thickness = 2;
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(prediction, fontFace, fontScale, thickness, &baseline);
    baseline += thickness;

    // Calculate the height needed for the text and some padding
    int textHeight = textSize.height + 20;

    // Create a larger image to accommodate the original image and the caption
    cv::Mat canvas = cv::Mat::zeros(img.rows + textHeight, img.cols, img.type());

    // Copy the original image to the top part of the canvas
    img.copyTo(canvas(cv::Rect(0, 0, img.cols, img.rows)));

    // Set the position for the caption (somewhere in the middle of the allocated text area)
    cv::Point textOrg((canvas.cols - textSize.width) / 2, img.rows + textHeight - 5);

    // Draw the caption on the canvas
    cv::putText(canvas, prediction, textOrg, fontFace, fontScale, cv::Scalar(255, 255, 255), thickness);

    // Create a window for display.
    cv::namedWindow("Prediction", cv::WINDOW_AUTOSIZE);

    // Show our image inside it.
    cv::imshow("Prediction", canvas);

    // Wait for a keystroke in the window
    cv::waitKey(0);
}

int main() {
    const std::string enginePath = "simple_cnn.engine"; // Adjust as necessary

    nvinfer1::IRuntime *runtime;
    auto engine = loadEngine(enginePath, runtime);
    if (!engine) {
        std::cerr << "Failed to load engine." << std::endl;
        return -1;
    }

    auto context = engine->createExecutionContext();

    std::vector<void *> buffers(engine->getNbBindings());
    int inputIndex, outputIndex;
    if (!allocateBuffers(engine.get(), buffers, inputIndex, outputIndex)) {
        std::cerr << "Failed to allocate buffers." << std::endl;
        return -1;
    }

    const std::string imagePath = "image.png";  // Adjust as necessary
    // Allocate the input buffer for a 28x28 grayscale image:
    int inputSize = 28 * 28 * 1;
    float *inputBuffer = new float[inputSize];
    try {
        // Load and preprocess the image:
        preprocessImage(imagePath, inputBuffer);
    } catch (const std::exception &e) {
        std::cerr << "Exception occurred: " << e.what() << std::endl;
        delete[] inputBuffer;
        return -1;
    }

    // Copy image data to input buffer
    CHECK_CUDA(cudaMemcpy(buffers[inputIndex], inputBuffer, inputSize * sizeof(float), cudaMemcpyHostToDevice));

    // Run inference using executeV2
    context->executeV2(buffers.data());

    // Retrieve results
    int outputSize = 10;  // Adjust size as necessary for your model's output
    float *output = new float[outputSize];
    CHECK_CUDA(cudaMemcpy(output, buffers[outputIndex], outputSize * sizeof(float), cudaMemcpyDeviceToHost));

    // Process and print the results
    int prediction = indexOfMax(output, 10); // Find the index of the max value
    std::cout << "Predicted classes:\n" << prediction << std::endl;

//    torch::Tensor output_tensor = torch::from_blob(output, {1, 10});
//    torch::Tensor values, predicted;
//    std::tie(values, predicted) = torch::max(output_tensor, 1);
//
//    std::cout << "Predicted classes:\n" << predicted << std::endl;

    // visualize the results
    displayImageWithCaption(imagePath, std::to_string(prediction));


    // Cleanup
    delete[] inputBuffer;
    delete[] output;
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);
    context->destroy();

    return 0;
}
