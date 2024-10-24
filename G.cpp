#include <iostream>
#include <cmath>
#include <iomanip>
#include <vector>
using namespace std;

struct Point {
    double x, y, z;
};

Point read_point() {
    Point p;
    cin >> p.x >> p.y >> p.z;
    return p;
}

Point vector_subtract(const Point& p1, const Point& p2) {
    return {p1.x - p2.x, p1.y - p2.y, p1.z - p2.z};
}

Point vector_cross(const Point& v1, const Point& v2) {
    return {
        v1.y * v2.z - v1.z * v2.y,
        v1.z * v2.x - v1.x * v2.z,
        v1.x * v2.y - v1.y * v2.x
    };
}

double vector_dot(const Point& v1, const Point& v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

double vector_magnitude(const Point& v) {
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

Point vector_normalize(const Point& v) {
    double mag = vector_magnitude(v);
    return {v.x / mag, v.y / mag, v.z / mag};
}

void project_to_2d(const Point& A, const Point& B, const Point& C, Point& A_2d, Point& B_2d, Point& C_2d) {
    Point AB = vector_subtract(B, A);
    Point AC = vector_subtract(C, A);
    
    Point normal = vector_cross(AB, AC);
    
    Point u = vector_normalize(AB);
    
    Point v = vector_cross(normal, u);
    v = vector_normalize(v);
    
    auto project_point = [&](const Point& P) {
        Point AP = vector_subtract(P, A);
        double x = vector_dot(AP, u);
        double y = vector_dot(AP, v);
        return Point{x, y, 0};
    };
    
    A_2d = project_point(A);
    B_2d = project_point(B);
    C_2d = project_point(C);
}

double distance(const Point& p1, const Point& p2) {
    return hypot(p1.x - p2.x, p1.y - p2.y);
}

double distance_point_to_line(const Point& A, const Point& B, const Point& C) {
    double cross = abs((B.x - A.x) * (A.y - C.y) - (B.y - A.y) * (A.x - C.x));
    double dist = cross / distance(A, B);
    return dist;
}

vector<Point> tangent_points(const Point& P, const Point& C, double R) {
    double dx = P.x - C.x;
    double dy = P.y - C.y;
    double dist = hypot(dx, dy);
    vector<Point> tangents;
    if (dist < R) {
        return tangents;  // No tangent
    } else if (dist == R) {
        tangents.push_back(P);  // One tangent point (the point itself)
    } else {
        double angle_PC = atan2(dy, dx);
        double alpha = acos(R / dist);
        double t1 = angle_PC + alpha;
        double t2 = angle_PC - alpha;
        Point tp1 = {C.x + R * cos(t1), C.y + R * sin(t1), 0};
        Point tp2 = {C.x + R * cos(t2), C.y + R * sin(t2), 0};
        tangents.push_back(tp1);
        tangents.push_back(tp2);
    }
    return tangents;
}

double angle_between(const Point& C, const Point& P1, const Point& P2) {
    double v1x = P1.x - C.x;
    double v1y = P1.y - C.y;
    double v2x = P2.x - C.x;
    double v2y = P2.y - C.y;
    double dot = v1x * v2x + v1y * v2y;
    double mag1 = hypot(v1x, v1y);
    double mag2 = hypot(v2x, v2y);
    if (mag1 == 0 || mag2 == 0) {
        return 0;
    }
    double cos_theta = dot / (mag1 * mag2);
    cos_theta = max(min(cos_theta, 1.0), -1.0);
    double theta = acos(cos_theta);
    return theta;
}

double compute_path(const Point& A, const Point& B, const Point& C, double R) {
    double dist_AB = distance(A, B);
    double dist_to_line = distance_point_to_line(A, B, C);
    if (dist_to_line >= R) {
        return dist_AB;
    }
    vector<Point> tangents_A = tangent_points(A, C, R);
    vector<Point> tangents_B = tangent_points(B, C, R);
    if (tangents_A.empty() || tangents_B.empty()) {
        return -1;  // No possible path
    }
    double min_path = numeric_limits<double>::infinity();
    for (const auto& ta : tangents_A) {
        for (const auto& tb : tangents_B) {
            double angle = angle_between(C, ta, tb);
            double arc = min(angle, 2 * M_PI - angle) * R;
            double path = distance(A, ta) + arc + distance(B, tb);
            if (path < min_path) {
                min_path = path;
            }
        }
    }
    return min_path;
}

int main() {
    Point A = read_point();
    Point B = read_point();
    Point C = read_point();
    double R;
    cin >> R;
    
    Point A_2d, B_2d, C_2d;
    project_to_2d(A, B, C, A_2d, B_2d, C_2d);
    
    double path_length = compute_path(A_2d, B_2d, C_2d, R);
    if (path_length == -1) {
        cout << "No valid path" << endl;
    } else {
        cout << fixed << setprecision(2) << path_length << endl;
    }

    return 0;
}