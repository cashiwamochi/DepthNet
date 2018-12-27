#include <torch/torch.h>
#include <torch/script.h> // One-stop header.

#include <iostream>
#include <vector>
#include <memory>
#include <cmath>

#include <opencv2/opencv.hpp>

cv::Mat forwardProcess(const std::shared_ptr<torch::jit::script::Module>& module,
                       const std::pair<cv::Mat,cv::Mat>& pm_images);

int main(int argc, char* argv[]) {

  if (argc != 4){
    std::cout << "usage : this.out [/path/to/model.pt] [/path/to/image0] [/path/to/image1]";
    std::cout << std::endl;
    return -1;
  }


  // Loading your model
  const std::string s_model_name = argv[1];
  std::cout << " >>> Loading " << s_model_name;
  std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(s_model_name);

  assert(module != nullptr);
  std::cout << " ... Done! " << std::endl;;

  // Loading your images
  const std::string s_image_name0 = argv[2];
  const std::string s_image_name1 = argv[3];

  std::cout << " >>> Loading " << s_image_name0 << " and " << s_image_name1;
  auto pm_images = std::make_pair(cv::imread(s_image_name0,1),cv::imread(s_image_name1, 1));
  std::cout << " ... Done! " << std::endl;

  cv::Mat m_color_map = forwardProcess(module, pm_images);

  cv::imshow("depth-color-map(press q to quit)", m_color_map);
  for(;;) {
    if(cv::waitKey(50)=='q')
      break;
  }

  cv::imwrite("depth_color_map.png", m_color_map);
  std::cout << " >>> Demo has finised !" << std::endl;

  return 0;
}


cv::Mat forwardProcess(const std::shared_ptr<torch::jit::script::Module>& module,
                       const std::pair<cv::Mat,cv::Mat>& pm_images)
{
  // define tensor dimention you will generate from cv::Mat
  const int channel = pm_images.first.channels();
  const int height = pm_images.first.rows;
  const int width = pm_images.first.cols;
  std::vector<int64_t> dims{static_cast<int64_t>(1), // 1
                            static_cast<int64_t>(channel * 2), // 6
                            static_cast<int64_t>(height), // h
                            static_cast<int64_t>(width)}; // w

  std::cout << " >>> [" << channel 
            << " x " << height 
            << " x " << width 
            << "]" << std::endl;

  cv::Mat mf_input = cv::Mat::zeros(height*channel*2, width, CV_32FC1);

  // cv::Mat m_bgr0[channel],m_bgr1[channel], mf_rgb_rgb[channel*2];
  cv::Mat m_bgr[channel], mf_rgb_rgb[channel*2];
  cv::split(pm_images.first, m_bgr);
  for(int i = 0; i < 3; i++) {
    m_bgr[i].convertTo(mf_rgb_rgb[2-i], CV_32FC1, 1.0/255.0, -0.5); 
    m_bgr[i].release();
  }

  cv::split(pm_images.second, m_bgr);
  for(int i = 0; i < 3; i++) {
    m_bgr[i].convertTo(mf_rgb_rgb[5-i], CV_32FC1, 1.0/255.0, -0.5); 
    m_bgr[i].release();
  }

  for (int i = 0; i < 6; i++) {
    mf_rgb_rgb[i].copyTo(mf_input.rowRange(i*height, (i+1)*height));
  }

  mf_input /= 0.2;

  at::TensorOptions options(at::kFloat);
  at::Tensor input_tensor = torch::from_blob(mf_input.data, at::IntList(dims), options);

  at::Tensor output_tensor = module->forward({input_tensor}).toTensor();
  std::cout << " >>> [" << output_tensor.size(0)
            << " x " << output_tensor.size(1)
            << " x " << output_tensor.size(2)
            << "]" << std::endl;
  cv::Mat m_output(cv::Size(output_tensor.size(1), output_tensor.size(2)), CV_32FC1, output_tensor.data<float>());
  cv::Mat m_depth = m_output.clone();
  cv::Mat m_upscaled_depth;
  cv::resize(m_depth, m_upscaled_depth, cv::Size(width, height), 0, 0);
  const float max_value = 100.0; // this is determined by the original python
  cv::Mat m_arranged_depth = 255.0*m_upscaled_depth/max_value; 
  for (int i = 0; i < m_arranged_depth.rows; i++) {
    for (int j = 0; j < m_arranged_depth.cols; j++) {
      if(m_arranged_depth.at<float>(i,j) > 255.0) {
        m_arranged_depth.at<float>(i,j) = 255.0;
      } 
      else if (m_arranged_depth.at<float>(i,j) < 0.0) {
        m_arranged_depth.at<float>(i,j) = 0.0;
      }
    }
  }
  m_arranged_depth.convertTo(m_arranged_depth, CV_8U); 
  cv::Mat m_color_map;
  cv::applyColorMap(m_arranged_depth, m_color_map, cv::COLORMAP_RAINBOW);
  
  return m_color_map;
}

