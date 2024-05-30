//
// Created by linh on 2023-01-05.
//

// Reference: https://github.com/sinecode/MeanShift

#ifndef MEANSHIFT_MEANSHIFTADAPTIVEBIRTH_H
#define MEANSHIFT_MEANSHIFTADAPTIVEBIRTH_H

#include <vector>
#include <utility>  // std::pair
#include <cmath>  // std::sqrt
#include <initializer_list>
#include <vector>
#include <algorithm>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/LU>

#define MAX_ITERATIONS 100

using namespace Eigen;
using namespace std;

class Point {

public:

    Point() {}

    Point(std::vector<float> values) {
        this->values = std::move(values);
    }

    Point(std::initializer_list<float> values) {
        this->values.assign(values);
    }

    Point(unsigned long dimensions) {
        this->values = std::vector<float>(dimensions, 0);
    }

    bool operator==(const Point &p) const {
        return this->values == p.values;
    }

    bool operator!=(const Point &p) const {
        return this->values != p.values;
    }

    Point operator+(const Point &p) const {
        Point point(this->values);
        return point += p;
    }

    Point &operator+=(const Point &p) {
        for (long i = 0; i < p.dimensions(); ++i)
            this->values[i] += p[i];
        return *this;
    }

    Point operator-(const Point &p) const {
        Point point(this->values);
        return point -= p;
    }

    Point &operator-=(const Point &p) {
        for (long i = 0; i < p.dimensions(); ++i)
            this->values[i] -= p[i];
        return *this;
    }

    Point operator*(const float d) const {
        Point point(this->values);
        return point *= d;
    }

    Point &operator*=(const float d) {
        for (long i = 0; i < dimensions(); ++i)
            this->values[i] *= d;
        return *this;
    }

    Point operator/(const float d) const {
        Point point(this->values);
        return point /= d;
    }

    Point &operator/=(const float d) {
        for (long i = 0; i < dimensions(); ++i)
            this->values[i] /= d;
        return *this;
    }

    float &operator[](const long index) {
        return values[index];
    }

    const float &operator[](const long index) const {
        return values[index];
    }

    unsigned long dimensions() const {
        return values.size();
    }

    std::vector<float>::const_iterator begin() const {
        return values.begin();
    }

    std::vector<float>::const_iterator end() const {
        return values.end();
    }

    float euclideanDistance(const Point &p) const {
        float sum = 0.0;
        for (std::pair<std::vector<float>::const_iterator, std::vector<float>::const_iterator> i(this->begin(),
                                                                                                 p.begin());
             i.first != this->end(); ++i.first, ++i.second) {
            float diff = *i.first - *i.second;
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    std::vector<float> getValues() {
        return values;
    }

private:
    std::vector<float> values;
};

class Cluster {
public:
    Cluster(Point centroid, int numSensors) {
        this->centroid = std::move(centroid);
        this->measureIndices = std::vector<int>(numSensors, 0);
    }

    Point getCentroid() const {
        return centroid;
    }

    void addPoint(Point point) {
        points.emplace_back(point);
    }

    long getSize() const {
        return points.size();
    }

    std::vector<Point>::iterator begin() {
        return points.begin();
    }

    std::vector<Point>::iterator end() {
        return points.end();
    }

    float getSse() const {
        float sum = 0.0;
        for (const Point &p : points)
            sum += std::pow(p.euclideanDistance(centroid), 2);
        return sum;
    }

    std::vector<int> getMeasurementIndices() const {
        return measureIndices;
    }

    void setMeasurementIndices(int sensor, int measurement) {
        measureIndices[sensor] = measurement;
    }

private:
    std::vector<Point> points;
    std::vector<int> measureIndices;
    Point centroid;
};

class ClustersBuilder {
public:
    ClustersBuilder(const std::vector<Point> &originalPoints, float clusterEps) {
        this->originalPoints = originalPoints;
        shiftedPoints = originalPoints;
        // vector of booleans such that the element in position i is false if the i-th point
        // has stopped to shift
        shifting = std::vector<bool>(originalPoints.size(), true);
        this->clusterEps = clusterEps;
        this->shiftingEps = clusterEps / 10;
    }

    Point &getShiftedPoint(long index) {
        return shiftedPoints[index];
    }

    void shiftPoint(const long index, const Point &newPosition) {
        if (newPosition.euclideanDistance(shiftedPoints[index]) <= shiftingEps)
            shifting[index] = false;
        else
            shiftedPoints[index] = newPosition;
    }

    bool hasStoppedShifting(long index) {
        return !shifting[index];
    }

    bool allPointsHaveStoppedShifting() {
        return std::none_of(shifting.begin(), shifting.end(), [](bool v) { return v; });
    }

    std::vector<Point>::iterator begin() {
        return shiftedPoints.begin();
    }

    std::vector<Point>::iterator end() {
        return shiftedPoints.end();
    }

    std::vector<Cluster> buildClusters(std::vector<std::vector<Point>> sensorsPoints) {
        std::vector<Cluster> clusters;
        // shifted points with distance minor or equal than clusterEps will go in the same cluster
        int shiftedIndex = 0;
        for (int i = 0; i < sensorsPoints.size(); ++i) {
            for (int j = 0; j < sensorsPoints[i].size(); j++) {
                Point orgPoint = sensorsPoints[i][j];
                int clusterIndx = -1;
                float smallestDistance = std::numeric_limits<float>::max();
                for (int k = 0; k < clusters.size(); k++) {
                    float kDistance = clusters[k].getCentroid().euclideanDistance(orgPoint);
                    if (kDistance <= clusterEps && clusters[k].getMeasurementIndices()[i] == 0) {
                        if (kDistance < smallestDistance) {
                            smallestDistance = kDistance;
                            clusterIndx = k;
                        }
                    }
                }
                if (clusterIndx == -1) {
                    // create a new cluster
                    Cluster cluster(shiftedPoints[shiftedIndex], sensorsPoints.size());
                    cluster.setMeasurementIndices(i, j + 1);
                    clusters.emplace_back(cluster);
                } else {
                    clusters[clusterIndx].setMeasurementIndices(i, j + 1);
                }
                ++shiftedIndex;
            }
        }
        return clusters;
    }

private:
    std::vector<Point> originalPoints;
    std::vector<Point> shiftedPoints;
    // vector of booleans such that the element in position i is false if the i-th point
    // has stopped to shift
    std::vector<bool> shifting;
    float clusterEps;
    float shiftingEps;
};

std::vector<float> z2xHomTrans(const Vector2d &z, const MatrixXd &T) {
    // https://www.petercorke.com/RTB/r9/html/homtrans.html
    Vector3d e2h = Vector3d::Ones();
    e2h(seq(0, 1)) = z; // E2H Euclidean to homogeneous
    Vector3d temp = T * e2h; // H2E Homogeneous to Euclidean
    Vector2d pt = temp(seq(0, 1)) / temp(2);
    std::vector<float> v2(2);
    v2[0] = pt(0);
    v2[1] = pt(1);
    return v2;
}


/*
 * Inputs:
 *     (1) Camera projection matrices (from 3D world coordinate to image plane)
 *     (2) Detection bounding boxes [Top, Left, log(Width), log(Height)]
 *     (2) Bandwidth (in meter) to run MeanShift algorithm
 * Output:
 *     Constraint: In a cluster, at most one measurement from a sensor. Measurements from the same sensor cannot be
 *     in a cluster. That is, a cluster has at most "Number of Sensors" elements and at lest one element.
 *
 *     List of cluster containing measurement indices
 *     Example: [[0, 1, 3, 4], [1, 5, 6, 9]], 0 for missed detection, 1:|Z| measurement index
 *     => Two clusters with detection from 4 sensors that generates two births
 *         (1) First birth get measurement #0, #2, #3 from sensor #1, #2, #3, sensor # missed
 *         (2) Second birth get measurement #0, #4, #5, #8 from sensor #0, #1, #2, #3
 */
tuple<vector<vector<int>>, vector<vector<float>>> meanShift(vector<MatrixXd> cMat, vector<MatrixXd> zMeasurements, float bandwidth = 0.5) {
    int nSensors = cMat.size();
    std::vector<std::vector<Point>> sensorsPoints(nSensors);
    std::vector<Point> points;
    Vector3i c_idxs;
    c_idxs << 0, 1, 3;
    for (int i = 0; i < nSensors; i++) {
        MatrixXd T = cMat[i](all, c_idxs).inverse();
        std::vector<Point> pointsTmp(zMeasurements[i].cols());
        MatrixXd zTmp = zMeasurements[i](seq(0, 1), all);
        // get center bottom of bbox
        zTmp.row(0) = zTmp.row(0).array() + zMeasurements[i].row(2).array().exp() / 2;
        zTmp.row(1) = zTmp.row(1).array() + zMeasurements[i].row(3).array().exp();
        for (int j = 0; j < zMeasurements[i].cols(); j++) {
            std::vector<float> values = z2xHomTrans(zTmp.col(j), T);
            pointsTmp[j] = Point(values);
            points.emplace_back(pointsTmp[j]);
        }
        sensorsPoints[i] = pointsTmp;
    }
    ClustersBuilder builder = ClustersBuilder(points, bandwidth);
    long iterations = 0;
    unsigned long dimensions = points[0].dimensions();
    float radius = bandwidth;
    float doubledSquaredBandwidth = 2 * bandwidth * bandwidth;
    while (!builder.allPointsHaveStoppedShifting() && iterations < MAX_ITERATIONS) {

#pragma omp parallel for default(none) \
shared(points, sensorsPoints, dimensions, builder, bandwidth, radius, doubledSquaredBandwidth) schedule(dynamic)

        for (long i = 0; i < points.size(); ++i) {
            if (builder.hasStoppedShifting(i))
                continue;

            Point newPosition(dimensions);
            Point pointToShift = builder.getShiftedPoint(i);
            float totalWeight = 0.0;
            for (int j = 0; j < sensorsPoints.size(); j++) {
                int updateKdx = -1;
                float kDistance = std::numeric_limits<float>::max();
                for (int k = 0; k < sensorsPoints[j].size(); k++) {
                    float distance = pointToShift.euclideanDistance(sensorsPoints[j][k]);
                    if (distance <= radius && kDistance > distance) {
                        updateKdx = k;
                        kDistance = distance;
                    }
                }
                if (updateKdx != -1) {
                    float gaussian = std::exp(-(kDistance * kDistance) / doubledSquaredBandwidth);
                    newPosition += sensorsPoints[j][updateKdx] * gaussian;
                    totalWeight += gaussian;
                }
            }

            // the new position of the point is the weighted average of its neighbors
            newPosition /= totalWeight;
            builder.shiftPoint(i, newPosition);
        }
        ++iterations;
    }
    if (iterations == MAX_ITERATIONS){
        std::cout << "WARNING: reached the maximum number of iterations" << std::endl;
    }

    std::vector<Cluster> clusters = builder.buildClusters(sensorsPoints);
    std::vector<std::vector<int>> birthSols(clusters.size());
    std::vector<std::vector<float>> centroids(clusters.size());
    for (int i = 0; i < clusters.size(); i++) {
        birthSols[i] = clusters[i].getMeasurementIndices();
        centroids[i] = clusters[i].getCentroid().getValues();
    }
    return {birthSols, centroids};
}

#endif //MEANSHIFT_MEANSHIFTADAPTIVEBIRTH_H
