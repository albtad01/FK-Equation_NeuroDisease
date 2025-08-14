#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>

struct Parameters
{
    std::string mesh_file_name;
    unsigned int degree;
    double T;
    double deltat;
    double theta;
    int matter_type;
    int protein_type;
    int axonal_field;
    double d_axn;
    double d_ext;
    double alpha;
};

Parameters read_params_from_csv(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Could not open parameter file: " + filename);

    std::string line;
    // Skip header
    if (!std::getline(file, line))
        throw std::runtime_error("Empty parameter file: " + filename);

    // Read first data row
    if (!std::getline(file, line))
        throw std::runtime_error("No data row in parameter file: " + filename);

    std::stringstream ss(line);
    std::string token;
    Parameters params;

    std::getline(ss, token, ','); params.mesh_file_name = token;
    std::getline(ss, token, ','); params.degree = static_cast<unsigned int>(std::stoul(token));
    std::getline(ss, token, ','); params.T = std::stod(token);
    std::getline(ss, token, ','); params.deltat = std::stod(token);
    std::getline(ss, token, ','); params.theta = std::stod(token);
    std::getline(ss, token, ','); params.matter_type = std::stoi(token);
    std::getline(ss, token, ','); params.protein_type = std::stoi(token);
    std::getline(ss, token, ','); params.axonal_field = std::stoi(token);
    std::getline(ss, token, ','); params.d_axn = std::stod(token);
    std::getline(ss, token, ','); params.d_ext = std::stod(token);
    std::getline(ss, token, ','); params.alpha = std::stod(token);

    return params;
}
