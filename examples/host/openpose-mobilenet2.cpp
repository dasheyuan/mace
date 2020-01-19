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

#include "pafprocess.h"
#include "Human.h"

#ifdef MODEL_GRAPH_FORMAT_CODE
#include "mace/codegen/engine/mace_engine_factory.h"
#endif

//#define IMAGE 127.5
//#define IMAGE_STD 127.5
#define IMAGE 0
#define IMAGE_STD 1
#define IMAGE_SRC "/home/chenyuan/3rdparty/tf-pose-estimation/images/p11.jpg"

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


    cv::Mat max_pooling(const cv::Mat &img) {
        int height = img.rows;
        int width = img.cols;
        int channel = img.channels();

        cv::Mat result = img.clone();

        // prepare output
        cv::Mat out = cv::Mat::zeros(height + 2, width + 2, CV_32FC1);
        cv::Mat out_roi = out(cv::Rect(1, 1, img.rows, img.cols));
        img.copyTo(out_roi);
        double v = 0;

        for (int y = 1; y < height + 1; y++) {
            for (int x = 1; x < width + 1; x++) {
                v = 0;
                for (int dy = -1; dy < 2; dy++) {
                    for (int dx = -1; dx < 2; dx++) {
                        v = fmax(out.at<float>(y + dy, x + dx), v);
                    }
                }
                result.at<float>(y - 1, x - 1) = fabs(v - img.at<float>(y - 1, x - 1)) < FLT_EPSILON ? v : 0;
            }
        }
        return result;
    }

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

        void imagePreProcess(const char *src, char *buff) {
            cv::Mat bgr = cv::imread(src);//bgr
            cv::resize(bgr, bgr, cv::Size(288, 288));
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

        void draw_humans(const cv::Mat &image, const std::vector<Human> &humans) {
            cv::Mat bgr = cv::imread(IMAGE_SRC);//bgr
            cv::Mat image2 = bgr.clone();

            auto numberColors = COCO_COLORS_RENDER.size();

            for (const Human human:humans) {
                std::map<int ,cv::Point2f> bodyPart_idx;

                //draw point
                for (const BodyPart bodyPart:human.bodyPart) {
                    cv::Point2f center = cv::Point2f(bodyPart.getX() * image2.cols, bodyPart.getY() * image2.rows);
                    bodyPart_idx.emplace(std::make_pair(bodyPart.getPartIdx(),center));

                    cv::circle(image2, center, 3, Scalar(255, 0, 0), -1);
                }
                //draw line


                for (auto it = std::begin(COCO_PAIRS_RENDER); it!=std::end(COCO_PAIRS_RENDER); ++it){
                    for(const auto &id:bodyPart_idx){
                        if(id.first==(*it)){
                            for(const auto &id2:bodyPart_idx){
                                if(id2.first==*(it+1)){
                                    //draw
                                    const cv::Scalar color{
                                            COCO_COLORS_RENDER[((*it)+2) % numberColors],
                                            COCO_COLORS_RENDER[((*it)+1) % numberColors],
                                            COCO_COLORS_RENDER[(*it) % numberColors]
                                    };
                                    cv::line(image2,id.second,id2.second,color,3);
                                    break;
                                }
                            }
                        }
                    }
                    ++it;
                }
            }
            cv::imshow("image2", image2);
            cv::waitKey(0);
        }


        void imgPostProcess(const char *src, BBox &box) {
            cv::Mat img = cv::imread(src);
            auto x1 = box.xmin * img.cols;
            auto y1 = box.ymin * img.rows;
            auto x2 = box.xmax * img.cols;
            auto y2 = box.ymax * img.rows;
            cv::rectangle(img, Point(x1, y1), Point(x2, y2), (255, 0, 0), 2, 1);
            cv::imshow("", img);
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
                      "image",
                      "input nodes, separated by comma");
        DEFINE_string(input_shape,
                      "1,288,288,3",
                      "input shapes, separated by colon and comma");
        DEFINE_string(output_node,
                      "Openpose/concat_stage7",
                      "output nodes, separated by comma");
        DEFINE_string(output_shape,
                      "1,36,36,57",
                      "output shapes, separated by colon and comma");
        DEFINE_string(input_data_format,
                      "NHWC",
                      "input data formats, NONE|NHWC|NCHW");
        DEFINE_string(output_data_format,
                      "NHWC",
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
                      "/home/chenyuan/mace/build/mobilenet-thin/model/mobilenet-thin.data",
                      "model data file name, used when EMBED_MODEL_DATA set to 0 or 2");
        DEFINE_string(model_file,
                      "/home/chenyuan/mace/build/mobilenet-thin/model/mobilenet-thin.pb",
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
                (void) (model_name);
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


            engine->Run(inputs, &outputs);

            float *out = outputs["Openpose/concat_stage7"].data().get();
            Mat M = cv::Mat(36, 36, CV_32FC(57), out);
            Mat M_up_sample;
            //1.
            cv::resize(M, M_up_sample, Size(144, 144));
            std::vector<Mat> out_channels;
            cv::split(M_up_sample, out_channels);
            std::vector<Mat> heatmap_channels, paf_channels;
            heatmap_channels.assign(out_channels.begin(), out_channels.begin() + 19);

            paf_channels.assign(out_channels.begin() + 19, out_channels.begin() + 57);


            //显示heatmap
            Mat heatmaps = cv::Mat(144, 144, CV_32FC1, Scalar(0));
            for (Mat &heatmap : heatmap_channels) {
                for (int i = 0; i < heatmaps.rows; i++) {
                    for (int j = 0; j < heatmaps.cols; j++) {
                        if (heatmaps.at<float>(i, j) < heatmap.at<float>(i, j)) {
                            heatmaps.at<float>(i, j) = heatmap.at<float>(i, j);
                        }
                    }
                }
            }
            cv::imshow("heatmaps", heatmaps);
            //显示vectormap-x vectormap-y
            Mat vectormap_x = cv::Mat(144, 144, CV_32FC1, Scalar(0));
            Mat vectormap_y = cv::Mat(144, 144, CV_32FC1, Scalar(0));
            int paf_c = 0;
            for (Mat &paf : paf_channels) {
                std::cout << paf << std::endl;
                double minVal, maxVal;
                int minIdx[2] = {}, maxIdx[2] = {};    // minnimum Index, maximum Ind
                cv::minMaxIdx(paf, &minVal, &maxVal, minIdx, maxIdx);
                if (paf_c % 2 == 0) {
                    for (int i = 0; i < paf.rows; i++) {
                        for (int j = 0; j < paf.cols; j++) {
                            if (vectormap_x.at<float>(i, j) < paf.at<float>(i, j)) {
                                vectormap_x.at<float>(i, j) = paf.at<float>(i, j);
                            }
                        }
                    }
                } else {
                    for (int i = 0; i < paf.rows; i++) {
                        for (int j = 0; j < paf.cols; j++) {
                            if (vectormap_y.at<float>(i, j) < paf.at<float>(i, j)) {
                                vectormap_y.at<float>(i, j) = paf.at<float>(i, j);
                            }
                        }
                    }
                }
                paf_c++;
            }
            cv::imshow("vectormap-x", vectormap_x);
            cv::imshow("vectormap-y", vectormap_y);
            waitKey(0);


            int64_t peaks_size = 1 * 144 * 144 * 19;
            auto buffer_peaks = std::shared_ptr<float>(new float[peaks_size]);
            int64_t heatmap_size = 1 * 144 * 144 * 19;
            auto buffer_heatmap = std::shared_ptr<float>(new float[heatmap_size]);
            int64_t pafmap_size = 1 * 144 * 144 * 38;
            auto buffer_pafmap = std::shared_ptr<float>(new float[pafmap_size]);

            int c = 0;
            for (Mat &heatmap : heatmap_channels) {
                Mat heatmap_gauss, heatmap_peaks;
                cv::GaussianBlur(heatmap, heatmap_gauss, Size(25, 25), 5, 5);
                //求peaks
                heatmap_peaks = mace::max_pooling(heatmap_gauss);
//                std::cout<<"channel:"<<c<<std::endl;
//                std::cout << heatmap_peaks << std::endl;

                for (int y = 0; y < 144; y++) {
                    for (int x = 0; x < 144; x++) {
                        buffer_heatmap.get()[19 * (144 * y + x) + c] = heatmap.at<float>(y, x);
                        buffer_peaks.get()[19 * (144 * y + x) + c] = heatmap_peaks.at<float>(y, x);
                    }
                }
                c++;
            }
            c = 0;
            for (Mat &paf : paf_channels) {
                //std::cout << paf << std::endl;
                for (int y = 0; y < 144; y++) {
                    for (int x = 0; x < 144; x++) {
                        buffer_pafmap.get()[38 * (144 * y + x) + c] = paf.at<float>(y, x);
                    }
                }
                c++;
            }

            process_paf(144, 144, 19, buffer_peaks.get(), 144, 144, 19, buffer_heatmap.get(), 144, 144, 38,
                        buffer_pafmap.get());
            std::cout << "humans num:" << get_num_humans() << std::endl;
            int num_humans = get_num_humans();
            int human_id = 0;
            std::vector<Human> Humans;
            while (num_humans > 0) {
                Human human;
                bool is_added = false;
                for (int part_idx = 0; part_idx < 18; part_idx++) {
                    int c_idx = (int) get_part_cid(human_id, part_idx);
                    if (c_idx < 0) continue;
                    is_added = true;

                    BodyPart bodyPart = BodyPart(part_idx, (float)(get_part_x(c_idx)/144.0), (float)(get_part_y(c_idx)/144.0), get_part_score(c_idx));
                    human.bodyPart.emplace_back(bodyPart);
                }
                if (is_added) {
                    float score = get_score(human_id);
                    human.score = score;
                    Humans.emplace_back(human);
                }
                human_id++;
                num_humans--;
            }
            draw_humans(cv::Mat(), Humans);
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

