#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <memory>
// opencv
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace c10;

void displayHeatmap(cv::Mat heatmap, const std::string& windowName) {
    double minVal, maxVal;
    cv::minMaxLoc(heatmap, &minVal, &maxVal);

    // Scale the floating point values to 8-bit range before converting
    heatmap = 255.0 * (heatmap - minVal) / (maxVal - minVal);

    // Convert the floating-point image to an 8-bit image
    heatmap.convertTo(heatmap, CV_8U);

    // Apply a colormap
    cv::applyColorMap(heatmap, heatmap, cv::COLORMAP_JET);

    // Show the image
    cv::imshow(windowName, heatmap);
}
// Add this function to your code
void printStats(const cv::Mat& mat, const std::string& name) {
    double minVal, maxVal;
    cv::minMaxLoc(mat, &minVal, &maxVal);
    cv::Scalar meanVal = cv::mean(mat);

    std::cout << name << " Min: " << minVal << ", Max: " << maxVal << ", Mean: " << meanVal.val[0] << std::endl;
}

int main() {
    const char* model_path = "C:\\Users\\joeli\\OneDrive\\Documents\\GitHub\\AE_Pytorch\\ts_316_trace\\encoder.pt";

    torch::jit::script::Module module;
    bool model_loaded = false;
    // Load the model
    {
        try {
            module = torch::jit::load(model_path);
            model_loaded = true;

        }
        catch (const c10::Error& e) {
            std::cerr << "Error loading the model: " << e.what() << std::endl;
            return -1;
        }
    }

    if (model_loaded) {
        // Load and process the image

        cv::Mat img = cv::imread("C:\\Users\\joeli\\Dropbox\\AE_InputModels\\m46.png", cv::IMREAD_COLOR);
        cv::Mat img_float;
        img.convertTo(img_float, CV_32FC3, 1.0f / 255.0f);

        // Reshape the image to a batch of pixels
        int width = img.cols;
        int height = img.rows;
        torch::Tensor input_tensor = torch::from_blob(img_float.ptr<float>(), { 3, height, width }, torch::kFloat32);
        input_tensor = input_tensor.view({ 3 , height * width });

        std::cout << "Image Shape: " << height << " " << width << std::endl;
        std::cout << "Reshaped Input Tensor Shape: ";
        for (auto s : input_tensor.sizes()) {
            std::cout << s << " ";
        }
        std::cout << std::endl;

        input_tensor = input_tensor.view({ -1,3 }); // Flatten the tensor

        std::cout << "Reshaped Input Tensor Shape: ";
        for (auto s : input_tensor.sizes()) {
            std::cout << s << " ";
        }
        std::cout << std::endl;

        // Perform inference
        torch::Tensor output;
        {
            try {
                output = module.forward({ input_tensor }).toTensor();
            }
            catch (const c10::Error& e) {
                std::cerr << "Error during model inference: " << e.what() << std::endl;
                return -1;
            }
            catch (const std::exception& e) {
                std::cerr << "Error during model inference: " << e.what() << std::endl;
                return -1;
            }
            catch (...) {
                std::cerr << "Unknown error during model inference." << std::endl;
                return -1;
            }
        }

        std::cout << "Output Tensor Shape: ";
        for (auto s : output.sizes()) {
            std::cout << s << " ";
        }
        std::cout << std::endl;


        // Convert the output tensor to a CV_32FC1 OpenCV matrix
        cv::Mat param_map(output.size(0), output.size(1), CV_32FC1, output.data_ptr<float>());

        // Display the original image
        cv::imshow("Original Image", img);

        // Assuming output is of shape [1, 4194304, 5]
        cv::Mat Cm = cv::Mat(height, width, CV_32FC1, output.select(1, 0).view({ height, width }).data_ptr<float>());
        cv::Mat Ch = cv::Mat(height, width, CV_32FC1, output.select(1, 1).view({ height, width }).data_ptr<float>());
        cv::Mat Bm = cv::Mat(height, width, CV_32FC1, output.select(1, 2).view({ height, width }).data_ptr<float>());
        cv::Mat Bh = cv::Mat(height, width, CV_32FC1, output.select(1, 3).view({ height, width }).data_ptr<float>());
        cv::Mat T = cv::Mat(height, width, CV_32FC1, output.select(1, 4).view({ height, width }).data_ptr<float>());
        //show Cm
        
        //normalize Cm, Ch, Bm, Bh, T
        printStats(Cm, "Cm");
        printStats(Ch, "Ch");
        printStats(Bm, "Bm");
        printStats(Bh, "Bh");
        printStats(T, "T");


        // Print the stats after normalization
        printStats(Cm, "Cm (Normalized)");
        printStats(Ch, "Ch (Normalized)");
        printStats(Bm, "Bm (Normalized)");
        printStats(Bh, "Bh (Normalized)");
        printStats(T, "T (Normalized)");

        displayHeatmap(Cm, "Cm Heatmap");
        displayHeatmap(Ch, "Ch Heatmap");
        displayHeatmap(Bm, "Bm Heatmap");
        displayHeatmap(Bh, "Bh Heatmap");
        displayHeatmap(T, "T Heatmap");

        cv::waitKey(0);
    }

    std::cout << "Press any key to exit...";
    std::cin.get();
}
