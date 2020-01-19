//
// Created by chenyuan on 1/6/20.
//

#ifndef MACE_HUMAN_H
#define MACE_HUMAN_H


#include <vector>

#define POSE_COCO_PAIRS_RENDER \
        1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   1,8,   8,9,   9,10,  1,11,  11,12, 12,13,  1,0,   0,14, 14,16,  0,15, 15,17

const std::vector<int> COCO_PAIRS_RENDER = std::vector<int>{POSE_COCO_PAIRS_RENDER};

#define POSE_COCO_COLORS_RENDER_GPU \
        255.f,     0.f,    85.f, \
        255.f,     0.f,     0.f, \
        255.f,    85.f,     0.f, \
        255.f,   170.f,     0.f, \
        255.f,   255.f,     0.f, \
        170.f,   255.f,     0.f, \
         85.f,   255.f,     0.f, \
          0.f,   255.f,     0.f, \
          0.f,   255.f,    85.f, \
          0.f,   255.f,   170.f, \
          0.f,   255.f,   255.f, \
          0.f,   170.f,   255.f, \
          0.f,    85.f,   255.f, \
          0.f,     0.f,   255.f, \
        255.f,     0.f,   170.f, \
        170.f,     0.f,   255.f, \
        255.f,     0.f,   255.f, \
         85.f,     0.f,   255.f
const std::vector<float> COCO_COLORS_RENDER = std::vector<float>{POSE_COCO_COLORS_RENDER_GPU};


enum CocoPart {
    Nose = 0,
    Neck = 1,
    RShoulder = 2,
    RElbow = 3,
    RWrist = 4,
    LShoulder = 5,
    LElbow = 6,
    LWrist = 7,
    RHip = 8,
    RKnee = 9,
    RAnkle = 10,
    LHip = 11,
    LKnee = 12,
    LAnkle = 13,
    REye = 14,
    LEye = 15,
    REar = 16,
    LEar = 17,
    Background = 18
};

/*
"""
part_idx : part index(eg. 0 for nose)
x, y: coordinate of body part
score : confidence score
"""
 */
class BodyPart {
    int part_idx_;
    float x_;
    float y_;
    float score_;

public:
    BodyPart(int part_idx,float x, float y, float score):part_idx_(part_idx),x_(x),y_(y),score_(score){

    }

public:
    int getPartIdx() const {
        return part_idx_;
    }

    void setPartIdx(int partIdx) {
        part_idx_ = partIdx;
    }

    float getX() const {
        return x_;
    }

    void setX(float x) {
        x_ = x;
    }

    float getY() const {
        return y_;
    }

    void setY(float y) {
        y_ = y;
    }

    float getScore() const {
        return score_;
    }

    void setScore(float score) {
        score_ = score;
    }

};

/*
"""
body_parts: list of BodyPart
"""
 */
class Human {
public:
    float score;
    std::vector<BodyPart> bodyPart;
};


#endif //MACE_HUMAN_H
