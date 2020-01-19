
//
// Created by chenyuan on 12/10/19.
//
#include <sys/types.h>
#include <dirent.h>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>

#include "gflags/gflags.h"
#include "mace/public/mace.h"
#include "mace/port/env.h"
#include "mace/port/file_system.h"
#include "mace/utils/logging.h"
#include "mace/utils/memory.h"
#include "mace/utils/string_util.h"
#include "mace/utils/statistics.h"

#include <opencv2/opencv.hpp>

#ifdef MODEL_GRAPH_FORMAT_CODE
#include "mace/codegen/engine/mace_engine_factory.h"
#endif

#define IMAGE 127.5
#define IMAGE_STD 127.5
#define IMAGE_SRC "/home/chenyuan/CLionProjects/mace/examples/host/004545.jpg"

union {
    float a;
    unsigned char bytes[4];
} ftob;

namespace mace {

    struct BBox {
        float xmin;
        float ymin;
        float xmax;
        float ymax;
        int label;
        float confidence;
    };

    namespace {
        inline float overlap(const BBox &a, const BBox &b) {
            if (a.xmin > b.xmax || a.xmax < b.xmin ||
                a.ymin > b.ymax || a.ymax < b.ymin) {
                return 0.f;
            }
            float overlap_w = std::min(a.xmax, b.xmax) - std::max(a.xmin, b.xmin);
            float overlap_h = std::min(a.ymax, b.ymax) - std::max(a.ymin, b.ymin);
            return overlap_w * overlap_h;
        }

        void NmsSortedBboxes(const std::vector<BBox> &bboxes,
                             const float nms_threshold,
                             const int top_k,
                             std::vector<BBox> *sorted_boxes) {
            const int n = std::min(top_k, static_cast<int>(bboxes.size()));
            std::vector<int> picked;

            std::vector<float> areas(n);
#pragma omp parallel for schedule(runtime)
            for (int i = 0; i < n; ++i) {
                const BBox &r = bboxes[i];
                float width = std::max(0.f, r.xmax - r.xmin);
                float height = std::max(0.f, r.ymax - r.ymin);
                areas[i] = width * height;
            }

            for (int i = 0; i < n; ++i) {
                const BBox &a = bboxes[i];
                int keep = 1;
                for (size_t j = 0; j < picked.size(); ++j) {
                    const BBox &b = bboxes[picked[j]];

                    float inter_area = overlap(a, b);
                    float union_area = areas[i] + areas[picked[j]] - inter_area;
                    MACE_CHECK(union_area > 0, "union_area should be greater than 0");
                    if (inter_area / union_area > nms_threshold) {
                        keep = 0;
                        break;
                    }
                }

                if (keep) {
                    picked.push_back(i);
                    sorted_boxes->push_back(bboxes[i]);
                }
            }
        }

        inline bool cmp(const BBox &a, const BBox &b) {
            return a.confidence > b.confidence;
        }
    }  // namespace

    int DetectionOutput(const float *loc_ptr,
                        const float *conf_ptr,
                        const float *pbox_ptr,
                        const int num_prior,
                        const int num_classes,
                        const float nms_threshold,
                        const int top_k,
                        const int keep_top_k,
                        const float confidence_threshold,
                        std::vector<BBox> *bbox_rects) {
        MACE_CHECK(keep_top_k > 0, "keep_top_k should be greater than 0");
        std::vector<float> bboxes(4 * num_prior);
#pragma omp parallel for schedule(runtime)
        for (int i = 0; i < num_prior; ++i) {
            int index = i * 4;
            const float *lc = loc_ptr + index;
            const float *pb = pbox_ptr + index;
            const float *var = pb + num_prior * 4;

            float pb_w = pb[2] - pb[0];
            float pb_h = pb[3] - pb[1];
            float pb_cx = (pb[0] + pb[2]) * 0.5f;
            float pb_cy = (pb[1] + pb[3]) * 0.5f;

            float bbox_cx = var[0] * lc[0] * pb_w + pb_cx;
            float bbox_cy = var[1] * lc[1] * pb_h + pb_cy;
            float bbox_w = std::exp(var[2] * lc[2]) * pb_w;
            float bbox_h = std::exp(var[3] * lc[3]) * pb_h;

            bboxes[0 + index] = bbox_cx - bbox_w * 0.5f;
            bboxes[1 + index] = bbox_cy - bbox_h * 0.5f;
            bboxes[2 + index] = bbox_cx + bbox_w * 0.5f;
            bboxes[3 + index] = bbox_cy + bbox_h * 0.5f;
        }
        // start from 1 to ignore background class

        for (int i = 1; i < num_classes; ++i) {
            // filter by confidence threshold
            std::vector<BBox> class_bbox_rects;
            for (int j = 0; j < num_prior; ++j) {
                float confidence = conf_ptr[j * num_classes + i];
                if (confidence > confidence_threshold) {
                    BBox c = {bboxes[0 + j * 4], bboxes[1 + j * 4], bboxes[2 + j * 4],
                              bboxes[3 + j * 4], i, confidence};
                    class_bbox_rects.push_back(c);
                }
            }
            std::sort(class_bbox_rects.begin(), class_bbox_rects.end(), cmp);

            // apply nms
            std::vector<BBox> sorted_boxes;
            NmsSortedBboxes(class_bbox_rects,
                            nms_threshold,
                            std::min(top_k,
                                     static_cast<int>(class_bbox_rects.size())),
                            &sorted_boxes);
            // gather
            bbox_rects->insert(bbox_rects->end(), sorted_boxes.begin(),
                               sorted_boxes.end());
        }

        std::sort(bbox_rects->begin(), bbox_rects->end(), cmp);

        // output
        int num_detected = keep_top_k < static_cast<int>(bbox_rects->size()) ?
                           keep_top_k : static_cast<int>(bbox_rects->size());
        bbox_rects->resize(num_detected);

        return num_detected;
    }
}  // namespace mace

namespace mace {
    namespace examples {

        using namespace cv;
        void imagePreProcess(const char *src, char * buff){
            cv::Mat bgr = cv::imread(src);//bgr
            cv::resize(bgr,bgr,cv::Size(300,300));
            for (int i = 0; i < bgr.rows; i++) {
                for (int j = 0; j < bgr.cols; j++) {
                    ftob.a = static_cast<float>(((bgr.at<Vec3b>(i, j)[2] & 0xFF) - IMAGE) / IMAGE_STD);
                    *(buff++) = ftob.bytes[0];
                    *(buff++) = ftob.bytes[1];
                    *(buff++) = ftob.bytes[2];
                    *(buff++) = ftob.bytes[3];
                    ftob.a = static_cast<float>(((bgr.at<Vec3b>(i, j)[1] & 0xFF) - IMAGE) / IMAGE_STD);
                    *(buff++) = ftob.bytes[0];
                    *(buff++) = ftob.bytes[1];
                    *(buff++) = ftob.bytes[2];
                    *(buff++) = ftob.bytes[3];
                    ftob.a = static_cast<float>(((bgr.at<Vec3b>(i, j)[0] & 0xFF) - IMAGE) / IMAGE_STD);
                    *(buff++) = ftob.bytes[0];
                    *(buff++) = ftob.bytes[1];
                    *(buff++) = ftob.bytes[2];
                    *(buff++) = ftob.bytes[3];
                }
            }
//            cv::imshow("",img);
//            cv::waitKey(0);
        }

        void  imgPostProcess(const char* src, BBox &box){
            cv::Mat img = cv::imread(src);
            auto x1 = box.xmin * img.cols;
            auto y1 = box.ymin * img.rows;
            auto x2 = box.xmax * img.cols;
            auto y2 = box.ymax * img.rows;
            cv::rectangle(img,Point(x1,y1),Point(x2,y2),(255,0,0),2,1);
            cv::imshow("",img);
            cv::waitKey(0);
        }

        void ParseShape(const std::string &str, std::vector<int64_t> *shape) {
            std::string tmp = str;
            while (!tmp.empty()) {
                int dim = atoi(tmp.data());
                shape->push_back(dim);
                size_t next_offset = tmp.find(",");
                if (next_offset == std::string::npos) {
                    break;
                } else {
                    tmp = tmp.substr(next_offset + 1);
                }
            }
        }

        std::string FormatName(const std::string input) {
            std::string res = input;
            for (size_t i = 0; i < input.size(); ++i) {
                if (!isalnum(res[i])) res[i] = '_';
            }
            return res;
        }

        DeviceType ParseDeviceType(const std::string &device_str) {
            if (device_str.compare("CPU") == 0) {
                return DeviceType::CPU;
            } else if (device_str.compare("GPU") == 0) {
                return DeviceType::GPU;
            } else if (device_str.compare("HEXAGON") == 0) {
                return DeviceType::HEXAGON;
            } else if (device_str.compare("HTA") == 0) {
                return DeviceType::HTA;
            } else if (device_str.compare("APU") == 0) {
                return DeviceType::APU;
            } else {
                return DeviceType::CPU;
            }
        }

        DataFormat ParseDataFormat(const std::string &data_format_str) {
            if (data_format_str == "NHWC") {
                return DataFormat::NHWC;
            } else if (data_format_str == "NCHW") {
                return DataFormat::NCHW;
            } else if (data_format_str == "OIHW") {
                return DataFormat::OIHW;
            } else {
                return DataFormat::NONE;
            }
        }

        DEFINE_string(model_name,
                      "",
                      "model name in yaml");
        DEFINE_string(input_node,
                      "data",
                      "input nodes, separated by comma");
        DEFINE_string(input_shape,
                      "1,300,300,3",
                      "input shapes, separated by colon and comma");
        DEFINE_string(output_node,
                      "mbox_loc,mbox_conf_flatten,mbox_priorbox",
                      "output nodes, separated by comma");
        DEFINE_string(output_shape,
                      "1,7668:1,3834:1,2,7668",
                      "output shapes, separated by colon and comma");
        DEFINE_string(input_data_format,
                      "NHWC",
                      "input data formats, NONE|NHWC|NCHW");
        DEFINE_string(output_data_format,
                      "NHWC,NHWC,NHWC",
                      "output data formats, NONE|NHWC|NCHW");
        DEFINE_string(input_file,
                      "",
                      "input file name | input file prefix for multiple inputs.");
        DEFINE_string(output_file,
                      "",
                      "output file name | output file prefix for multiple outputs");
        DEFINE_string(input_dir,
                      "",
                      "input directory name");
        DEFINE_string(output_dir,
                      "output",
                      "output directory name");
        DEFINE_string(opencl_binary_file,
                      "",
                      "compiled opencl binary file path");
        DEFINE_string(opencl_parameter_file,
                      "",
                      "tuned OpenCL parameter file path");
        DEFINE_string(model_data_file,
                      "/home/chenyuan/mace/build/ssd_mobilenet_v1/model/ssd_mobilenet_v1.data",
                      "model data file name, used when EMBED_MODEL_DATA set to 0 or 2");
        DEFINE_string(model_file,
                      "/home/chenyuan/mace/build/ssd_mobilenet_v1/model/ssd_mobilenet_v1.pb",
                      "model file name, used when load mace model in pb");
        DEFINE_string(device, "CPU", "CPU/GPU/HEXAGON/APU");
        DEFINE_int32(round, 1, "round");
        DEFINE_int32(restart_round, 1, "restart round");
        DEFINE_int32(malloc_check_cycle, -1, "malloc debug check cycle, -1 to disable");
        DEFINE_int32(gpu_perf_hint, 3, "0:DEFAULT/1:LOW/2:NORMAL/3:HIGH");
        DEFINE_int32(gpu_priority_hint, 3, "0:DEFAULT/1:LOW/2:NORMAL/3:HIGH");
        DEFINE_int32(omp_num_threads, -1, "num of openmp threads");
        DEFINE_int32(cpu_affinity_policy, 1,
                     "0:AFFINITY_NONE/1:AFFINITY_BIG_ONLY/2:AFFINITY_LITTLE_ONLY");
        DEFINE_bool(benchmark, false, "enable benchmark op");


        bool RunModel(const std::string &model_name,
                      const std::vector<std::string> &input_names,
                      const std::vector<std::vector<int64_t>> &input_shapes,
                      const std::vector<DataFormat> &input_data_formats,
                      const std::vector<std::string> &output_names,
                      const std::vector<std::vector<int64_t>> &output_shapes,
                      const std::vector<DataFormat> &output_data_formats,
                      float cpu_capability) {
            DeviceType device_type = ParseDeviceType(FLAGS_device);

            int64_t t0 = NowMicros();
            // config runtime
            MaceStatus status;
            MaceEngineConfig config(device_type);
            status = config.SetCPUThreadPolicy(
                    FLAGS_omp_num_threads,
                    static_cast<CPUAffinityPolicy >(FLAGS_cpu_affinity_policy));
            if (status != MaceStatus::MACE_SUCCESS) {
                LOG(WARNING) << "Set openmp or cpu affinity failed.";
            }


            std::unique_ptr<mace::port::ReadOnlyMemoryRegion> model_graph_data =
                    make_unique<mace::port::ReadOnlyBufferMemoryRegion>();
            if (FLAGS_model_file != "") {
                auto fs = GetFileSystem();
                status = fs->NewReadOnlyMemoryRegionFromFile(FLAGS_model_file.c_str(),
                                                             &model_graph_data);
                if (status != MaceStatus::MACE_SUCCESS) {
                    LOG(FATAL) << "Failed to read file: " << FLAGS_model_file;
                }
            }

            std::unique_ptr<mace::port::ReadOnlyMemoryRegion> model_weights_data =
                    make_unique<mace::port::ReadOnlyBufferMemoryRegion>();
            if (FLAGS_model_data_file != "") {
                auto fs = GetFileSystem();
                status = fs->NewReadOnlyMemoryRegionFromFile(FLAGS_model_data_file.c_str(),
                                                             &model_weights_data);
                if (status != MaceStatus::MACE_SUCCESS) {
                    LOG(FATAL) << "Failed to read file: " << FLAGS_model_data_file;
                }
            }

            std::shared_ptr<mace::MaceEngine> engine;
            MaceStatus create_engine_status;

            while (true) {
                // Create Engine
                int64_t t0 = NowMicros();
#ifdef MODEL_GRAPH_FORMAT_CODE
                if (model_name.empty()) {
      LOG(INFO) << "Please specify model name you want to run";
      return false;
    }
    create_engine_status =
          CreateMaceEngineFromCode(model_name,
                                   reinterpret_cast<const unsigned char *>(
                                     model_weights_data->data()),
                                   model_weights_data->length(),
                                   input_names,
                                   output_names,
                                   config,
                                   &engine);
#else
                (void)(model_name);
                if (model_graph_data == nullptr || model_weights_data == nullptr) {
                    LOG(INFO) << "Please specify model graph file and model data file";
                    return false;
                }
                create_engine_status =
                        CreateMaceEngineFromProto(reinterpret_cast<const unsigned char *>(
                                                          model_graph_data->data()),
                                                  model_graph_data->length(),
                                                  reinterpret_cast<const unsigned char *>(
                                                          model_weights_data->data()),
                                                  model_weights_data->length(),
                                                  input_names,
                                                  output_names,
                                                  config,
                                                  &engine);
#endif
                int64_t t1 = NowMicros();

                if (create_engine_status != MaceStatus::MACE_SUCCESS) {
                    LOG(ERROR) << "Create engine runtime error, retry ... errcode: "
                               << create_engine_status.information();
                } else {
                    double create_engine_millis = (t1 - t0) / 1000.0;
                    LOG(INFO) << "Create Mace Engine latency: " << create_engine_millis
                              << " ms";
                    break;
                }
            }

            int64_t t1 = NowMicros();
            double init_millis = (t1 - t0) / 1000.0;
            LOG(INFO) << "Total init latency: " << init_millis << " ms";

            const size_t input_count = input_names.size();
            const size_t output_count = output_names.size();
            std::map<std::string, mace::MaceTensor> inputs;
            std::map<std::string, mace::MaceTensor> outputs;
            std::map<std::string, int64_t> inputs_size;

            for (size_t i = 0; i < input_count; ++i) {
                // Allocate input and output
                // only support float and int32, use char for generalization
                // sizeof(int) == 4, sizeof(float) == 4
                int64_t input_size =
                        std::accumulate(input_shapes[i].begin(), input_shapes[i].end(), 4,
                                        std::multiplies<int64_t>());
                inputs_size[input_names[i]] = input_size;
                auto buffer_in = std::shared_ptr<char>(new char[input_size],
                                                       std::default_delete<char[]>());
                // load input
                mace::examples::imagePreProcess(IMAGE_SRC, buffer_in.get());
//                std::ifstream in_file(FLAGS_input_file + "_" + FormatName(input_names[i]),
//                                      std::ios::in | std::ios::binary);
//                if (in_file.is_open()) {
//                    in_file.read(buffer_in.get(), input_size);
//                    in_file.close();
//                } else {
//                    LOG(INFO) << "Open input file failed";
//                    return -1;
//                }
                inputs[input_names[i]] = mace::MaceTensor(input_shapes[i], buffer_in);
            }

            //4.output buffer
            for (size_t i = 0; i < output_count; ++i) {
                // only support float and int32, use char for generalization
                int64_t output_size =
                        std::accumulate(output_shapes[i].begin(), output_shapes[i].end(), 4,
                                        std::multiplies<int64_t>());
                auto buffer_out = std::shared_ptr<char>(new char[output_size],
                                                        std::default_delete<char[]>());
                outputs[output_names[i]] = mace::MaceTensor(output_shapes[i], buffer_out);
            }

           //engine->Run(inputs, &outputs);

            for (int i = 0; i < 100; ++i) {
                engine->Run(inputs, &outputs);

                const int num_prior = 1917;
                const int num_classes = 2;
                const float nms_threshold = 0.45;
                const int top_k = 100;
                const int keep_top_k = 100;
                const float confidence_threshold = 0.25;
                std::vector<mace::BBox> bbox_rects;
                //get detection result
                int detectionNum = 0;
                detectionNum = mace::DetectionOutput(outputs["mbox_loc"].data().get(),
                                                     outputs["mbox_conf_flatten"].data().get(),
                                                     outputs["mbox_priorbox"].data().get(),
                                                     num_prior,
                                                     num_classes,
                                                     nms_threshold,
                                                     top_k,
                                                     keep_top_k,
                                                     confidence_threshold,
                                                     &bbox_rects);
                if (detectionNum <= 0) {
                    printf("------------------>detect nothing!\n");
                }else{
                    imgPostProcess(IMAGE_SRC,bbox_rects[0]);
                    printf("------------------>detect %d!\n",detectionNum);

                }
            }
            return true;
        }


        int Main(int argc, char **argv) {

            std::vector<std::string> input_names = mace::Split(FLAGS_input_node, ',');
            std::vector<std::string> output_names = mace::Split(FLAGS_output_node, ',');
            if (input_names.empty() || output_names.empty()) {
                LOG(INFO) << "input names or output names empty";
                return 0;
            }

            std::vector<std::string> input_shapes = mace::Split(FLAGS_input_shape, ':');
            std::vector<std::string> output_shapes = mace::Split(FLAGS_output_shape, ':');

            const size_t input_count = input_shapes.size();
            const size_t output_count = output_shapes.size();
            std::vector<std::vector<int64_t>> input_shape_vec(input_count);
            std::vector<std::vector<int64_t>> output_shape_vec(output_count);
            for (size_t i = 0; i < input_count; ++i) {
                ParseShape(input_shapes[i], &input_shape_vec[i]);
            }
            for (size_t i = 0; i < output_count; ++i) {
                ParseShape(output_shapes[i], &output_shape_vec[i]);
            }
            if (input_names.size() != input_shape_vec.size()
                || output_names.size() != output_shape_vec.size()) {
                LOG(INFO) << "inputs' names do not match inputs' shapes "
                            "or outputs' names do not match outputs' shapes";
                return 0;
            }

            std::vector<std::string> raw_input_data_formats =
                    Split(FLAGS_input_data_format, ',');
            std::vector<std::string> raw_output_data_formats =
                    Split(FLAGS_output_data_format, ',');

            std::vector<DataFormat> input_data_formats(input_count);
            std::vector<DataFormat> output_data_formats(output_count);
            for (size_t i = 0; i < input_count; ++i) {
                input_data_formats[i] = ParseDataFormat(raw_input_data_formats[i]);
            }
            for (size_t i = 0; i < output_count; ++i) {
                output_data_formats[i] = ParseDataFormat(raw_output_data_formats[i]);
            }
            float cpu_float32_performance = 0.0f;
//            if (FLAGS_input_dir.empty()) {
//                // get cpu capability
//                Capability cpu_capability = GetCapability(DeviceType::CPU);
//                cpu_float32_performance = cpu_capability.float32_performance.exec_time;
//            }

            bool ret = false;
            for (int i = 0; i < FLAGS_restart_round; ++i) {
                VLOG(0) << "restart round " << i;
                ret = RunModel(FLAGS_model_name,
                               input_names, input_shape_vec, input_data_formats,
                               output_names, output_shape_vec, output_data_formats,
                               cpu_float32_performance);
            }
            if (ret) {
                return 0;
            }
            return -1;
        }
    }  // namespace tools
}  // namespace mace

int main(int argc, char **argv) {
    mace::examples::Main(argc, argv);
}

