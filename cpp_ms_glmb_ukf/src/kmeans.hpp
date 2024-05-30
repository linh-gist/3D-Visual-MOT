//
// Created by linh on 2023-02-02.
// Reference: https://github.com/aditya1601/kmeans-clustering-cpp
//

#ifndef MSADAPTIVEBIRTH_VISION_KMEAN_H
#define MSADAPTIVEBIRTH_VISION_KMEAN_H

#include <omp.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <Eigen/LU>
#include <random>        /*uniform distribution*/

using namespace std;
using namespace Eigen;

class PointK {
private:
    int pointId, clusterId;
    int dimensions;
    vector<double> values;

public:
    PointK(int id, double x, double y) {
        pointId = id;
        values.emplace_back(x);
        values.emplace_back(y);
        dimensions = values.size();
        clusterId = 0; // Initially not assigned to any cluster
    }

    int getDimensions() { return dimensions; }

    int getCluster() { return clusterId; }

    int getID() { return pointId; }

    void setCluster(int val) { clusterId = val; }

    double getVal(int pos) { return values[pos]; }
};

class ClusterK {
private:
    int clusterId;
    vector<double> centroid;
    vector<PointK> points;

public:
    ClusterK(int clusterId, PointK centroid) {
        this->clusterId = clusterId;
        for (int i = 0; i < centroid.getDimensions(); i++) {
            this->centroid.push_back(centroid.getVal(i));
        }
        this->addPoint(centroid);
    }

    void addPoint(PointK p) {
        p.setCluster(this->clusterId);
        points.push_back(p);
    }

    bool removePoint(int pointId) {
        int size = points.size();
        for (int i = 0; i < size; i++) {
            if (points[i].getID() == pointId) {
                points.erase(points.begin() + i);
                return true;
            }
        }
        return false;
    }

    void removeAllPoints() { points.clear(); }

    int getId() { return clusterId; }

    PointK getPoint(int pos) { return points[pos]; }

    int getSize() { return points.size(); }

    double getCentroidByPos(int pos) { return centroid[pos]; }

    void setCentroidByPos(int pos, double val) { this->centroid[pos] = val; }
};

class KMeans {
private:
    int K, iters, dimensions, total_points;
    vector<ClusterK> clusters;
    std::random_device rand_dev;
    std::mt19937 generator;

    void clearClusters() {
        for (int i = 0; i < K; i++) {
            clusters[i].removeAllPoints();
        }
    }

    int getNearestClusterId(PointK point) {
        double sum = 0.0, min_dist;
        int NearestClusterId;
        if (dimensions == 1) {
            min_dist = abs(clusters[0].getCentroidByPos(0) - point.getVal(0));
        } else {
            for (int i = 0; i < dimensions; i++) {
                sum += pow(clusters[0].getCentroidByPos(i) - point.getVal(i), 2.0);
                // sum += abs(clusters[0].getCentroidByPos(i) - point.getVal(i));
            }
            min_dist = sqrt(sum);
        }
        NearestClusterId = clusters[0].getId();
        for (int i = 1; i < K; i++) {
            double dist;
            sum = 0.0;
            if (dimensions == 1) {
                dist = abs(clusters[i].getCentroidByPos(0) - point.getVal(0));
            } else {
                for (int j = 0; j < dimensions; j++) {
                    sum += pow(clusters[i].getCentroidByPos(j) - point.getVal(j), 2.0);
                    // sum += abs(clusters[i].getCentroidByPos(j) - point.getVal(j));
                }
                dist = sqrt(sum);
                // dist = sum;
            }
            if (dist < min_dist) {
                min_dist = dist;
                NearestClusterId = clusters[i].getId();
            }
        }

        return NearestClusterId;
    }

    Vector2d z2xHomTrans(const Vector2d &z, const MatrixXd &T) {
        // https://www.petercorke.com/RTB/r9/html/homtrans.html
        Vector3d e2h = Vector3d::Ones();
        e2h(seq(0, 1)) = z; // E2H Euclidean to homogeneous
        Vector3d temp = T * e2h; // H2E Homogeneous to Euclidean
        Vector2d pt = temp(seq(0, 1)) / temp(2);
        return pt;
    }

public:
    KMeans(int iterations) {
        this->K = 0;
        this->iters = iterations;
        this->generator = std::mt19937(rand_dev());
    }

    tuple<vector<VectorXi>, vector<VectorXd>> run(vector<MatrixXd> cMat, vector<MatrixXd> zMeasurements) {
        int nSensors = cMat.size();
        std::vector<PointK> all_points;
        Vector3i c_idxs;
        c_idxs << 0, 1, 3;
        total_points = 0;
        vector<VectorXi> point_cluster_idxs;
        for (int i = 0; i < nSensors; i++) {
            MatrixXd T = cMat[i](all, c_idxs).inverse();
            MatrixXd zTmp = zMeasurements[i](seq(0, 1), all);
            // get center bottom of bbox
            zTmp.row(0) = zTmp.row(0).array() + zMeasurements[i].row(2).array().exp() / 2;
            zTmp.row(1) = zTmp.row(1).array() + zMeasurements[i].row(3).array().exp();
            for (int j = 0; j < zMeasurements[i].cols(); j++) {
                Vector2d value = z2xHomTrans(zTmp.col(j), T);
                all_points.emplace_back(PointK(total_points, value(0), value(1)));
                Vector3i point_i_j;
                point_i_j << i, j, total_points;
                point_cluster_idxs.emplace_back(point_i_j);
                total_points++;
            }
            if (this->K < zMeasurements[i].cols()) {
                this->K = zMeasurements[i].cols();
            }
        }
        uniform_int_distribution<int> distribution(0, total_points - 1); // Uniformly distributed numbers
        dimensions = all_points[0].getDimensions();
        // Initializing Clusters
        vector<int> used_pointIds;
        for (int i = 1; i <= K; i++) {
            while (true) {
                int index = distribution(this->generator); //rand() % total_points;
                if (find(used_pointIds.begin(), used_pointIds.end(), index) == used_pointIds.end()) {
                    used_pointIds.push_back(index);
                    all_points[index].setCluster(i);
                    ClusterK cluster(i, all_points[index]);
                    clusters.push_back(cluster);
                    break;
                }
            }
        }
        int iter = 1;
        while (true) {
            bool done = true;
            // Add all points to their nearest cluster
#pragma omp parallel for reduction(&&: done) num_threads(16)
            for (int i = 0; i < total_points; i++) {
                int currentClusterId = all_points[i].getCluster();
                int nearestClusterId = getNearestClusterId(all_points[i]);

                if (currentClusterId != nearestClusterId) {
                    all_points[i].setCluster(nearestClusterId);
                    done = false;
                }
            }
            // clear all existing clusters
            clearClusters();
            // reassign points to their new clusters
            for (int i = 0; i < total_points; i++) {
                // cluster index is ID-1
                clusters[all_points[i].getCluster() - 1].addPoint(all_points[i]);
            }

            // Recalculating the center of each cluster
            for (int i = 0; i < K; i++) {
                int ClusterSize = clusters[i].getSize();
                for (int j = 0; j < dimensions; j++) {
                    double sum = 0.0;
                    if (ClusterSize > 0) {
#pragma omp parallel for reduction(+: sum) num_threads(16)
                        for (int p = 0; p < ClusterSize; p++) {
                            sum += clusters[i].getPoint(p).getVal(j);
                        }
                        clusters[i].setCentroidByPos(j, sum / ClusterSize);
                    }
                }
            }

            if (done || iter >= iters) {
                break;
            }
            iter++;
        }
        vector<int> pointIDs(total_points);
        for (int i = 0; i < total_points; i++) {
            // sensor index, measurement index, cluster index
            point_cluster_idxs[i](2) = all_points[i].getCluster() - 1;
        }
        vector<VectorXd> meanXY(clusters.size());
        // Index of cluster clusters[i].getId() equals (i-1)
        for (int i = 0; i < clusters.size(); i++) {
            VectorXd centerXY(2);
            centerXY(0) = clusters[i].getCentroidByPos(0);
            centerXY(1) = clusters[i].getCentroidByPos(1);
            meanXY[i] = centerXY;
        }
        return {point_cluster_idxs, meanXY};
    }
};

#endif //MSADAPTIVEBIRTH_VISION_KMEAN_H
