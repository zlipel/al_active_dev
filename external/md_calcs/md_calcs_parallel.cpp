#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>
#include <cmath>
#include <vector>
#include <iostream>

namespace py = pybind11;

py::array_t<double> msd_calc(py::array_t<double> pos, const std::string& dimension) {
    auto buf_pos = pos.request();

    size_t N_time = buf_pos.shape[0];
    size_t N_atom = buf_pos.shape[1];
    size_t N_dim = dimension == "2d" ? 2 : 3;

    if (dimension != "2d" && dimension != "3d") {
        throw std::runtime_error("Num of dimensions should be '2d' or '3d'.");
    }

    auto msd = py::array(py::buffer_info(
        nullptr,
        sizeof(double),
        py::format_descriptor<double>::value,
        1,
        { N_time },
        { sizeof(double) }
    ));

    auto buf_msd = msd.request();

    double *ptr_pos = static_cast<double *>(buf_pos.ptr),
           *ptr_msd = static_cast<double *>(buf_msd.ptr);
    std::fill_n(ptr_msd, N_time, 0.0);

    {
        py::gil_scoped_release release;  // Release GIL for the computational part

        #pragma omp parallel for
        for (size_t shift = 0; shift < N_time; ++shift) {

            // #pragma omp critical
            //     {
            //         std::cout << "Number of threads: " << omp_get_num_threads() << std::endl;
            //     }
            double sum_sq = 0.0;
            size_t count = 0;

            #pragma omp parallel for reduction(+:sum_sq, count)
            for (size_t start = 0; start < N_time - shift; ++start) {
                size_t end = start + shift;
                for (size_t j = 0; j < N_atom; ++j) {
                    double displacement_sq = 0.0;
                    for (size_t k = 0; k < N_dim; ++k) {
                        double start_pos = ptr_pos[start * N_atom * N_dim + j * N_dim + k];
                        double end_pos = ptr_pos[end * N_atom * N_dim + j * N_dim + k];
                        //double delta = end_pos - start_pos;

                        displacement_sq += start_pos*start_pos + end_pos*end_pos - 2.0*start_pos*end_pos;
                    }
                    sum_sq += displacement_sq;
                }
                ++count;
            }
            ptr_msd[shift] = count > 0 ? sum_sq / (N_atom * count) : 0.0;
        }
    }

    return msd;
}

py::array_t<double> msd_calc_decade(py::array_t<double> pos, const std::string& dimension) {
    auto buf_pos = pos.request();

    size_t N_time = buf_pos.shape[0];
    size_t N_atom = buf_pos.shape[1];
    size_t N_dim = dimension == "2d" ? 2 : 3;

    if (dimension != "2d" && dimension != "3d") {
        throw std::runtime_error("Num of dimensions should be '2d' or '3d'.");
    }

    auto msd = py::array(py::buffer_info(
        nullptr,
        sizeof(double),
        py::format_descriptor<double>::value,
        1,
        { N_time },
        { sizeof(double) }
    ));

    auto counts = py::array(py::buffer_info(
        nullptr,
        sizeof(size_t),
        py::format_descriptor<size_t>::value,
        1,
        { N_time },
        { sizeof(size_t) }
    ));

    auto buf_msd = msd.request();
    auto buf_counts = counts.request();

    double *ptr_pos = static_cast<double *>(buf_pos.ptr),
           *ptr_msd = static_cast<double *>(buf_msd.ptr);

    size_t *ptr_counts = static_cast<size_t *>(buf_counts.ptr);
    std::fill_n(ptr_msd, N_time, 0.0);
    std::fill_n(ptr_counts, N_time, 0);

    {
        py::gil_scoped_release release;  // Release GIL for the computational part

        #pragma omp parallel for
        for (size_t shift = 0; shift < N_time; ++shift) {

            // #pragma omp critical
            //     {
            //         std::cout << "Number of threads: " << omp_get_num_threads() << std::endl;
            //     }
            double sum_sq = 0.0;
            size_t count = 0;

            #pragma omp parallel for reduction(+:sum_sq, count)
            for (size_t start = 0; start < N_time - shift; ++start) {
                size_t end = start + shift;
                for (size_t j = 0; j < N_atom; ++j) {
                    double displacement_sq = 0.0;
                    for (size_t k = 0; k < N_dim; ++k) {
                        double start_pos = ptr_pos[start * N_atom * N_dim + j * N_dim + k];
                        double end_pos = ptr_pos[end * N_atom * N_dim + j * N_dim + k];
                        //double delta = end_pos - start_pos;

                        displacement_sq += start_pos*start_pos + end_pos*end_pos - 2.0*start_pos*end_pos;
                    }
                    sum_sq += displacement_sq;
                }
                ++count;
                ++ptr_counts[shift];
            }
            // ptr_counts[shift] = count;
            ptr_msd[shift] = count > 0 ? sum_sq / (N_atom * count) : 0.0;
        }
    }

    return py::make_tuple(msd, counts);
}

py::array_t<double> msd_xdir_calc(py::array_t<double> pos, const std::string& dimension) {
    auto buf_pos = pos.request();

    size_t N_time = buf_pos.shape[0];
    size_t N_atom = buf_pos.shape[1];
    size_t N_dim = dimension == "2d" ? 2 : 3;

    if (dimension != "2d" && dimension != "3d") {
        throw std::runtime_error("Num of dimensions should be '2d' or '3d'.");
    }

    auto msd = py::array(py::buffer_info(
        nullptr,
        sizeof(double),
        py::format_descriptor<double>::value,
        1,
        { N_time },
        { sizeof(double) }
    ));

    auto buf_msd = msd.request();

    double *ptr_pos = static_cast<double *>(buf_pos.ptr),
           *ptr_msd = static_cast<double *>(buf_msd.ptr);
    std::fill_n(ptr_msd, N_time, 0.0);

    {
        py::gil_scoped_release release;  // Release GIL for the computational part

        #pragma omp parallel for
        for (size_t shift = 0; shift < N_time; ++shift) {

            // #pragma omp critical
            //     {
            //         std::cout << "Number of threads: " << omp_get_num_threads() << std::endl;
            //     }
            double sum_sq = 0.0;
            size_t count = 0;

            #pragma omp parallel for reduction(+:sum_sq, count)
            for (size_t start = 0; start < N_time - shift; ++start) {
                size_t end = start + shift;
                for (size_t j = 0; j < N_atom; ++j) {
                        
                    double start_pos = ptr_pos[start * N_atom * N_dim + j * N_dim];
                    double end_pos = ptr_pos[end * N_atom * N_dim + j * N_dim];
                    
                    sum_sq += start_pos*start_pos + end_pos*end_pos - 2.0*start_pos*end_pos;
                }
                ++count;
            }
            ptr_msd[shift] = count > 0 ? sum_sq / (N_atom * count) : 0.0;
        }
    }

    return msd;
}

py::array_t<double> msd_ydir_calc(py::array_t<double> pos, const std::string& dimension) {
    auto buf_pos = pos.request();

    size_t N_time = buf_pos.shape[0];
    size_t N_atom = buf_pos.shape[1];
    size_t N_dim = dimension == "2d" ? 2 : 3;

    if (dimension != "2d" && dimension != "3d") {
        throw std::runtime_error("Num of dimensions should be '2d' or '3d'.");
    }

    auto msd = py::array(py::buffer_info(
        nullptr,
        sizeof(double),
        py::format_descriptor<double>::value,
        1,
        { N_time },
        { sizeof(double) }
    ));

    auto buf_msd = msd.request();

    double *ptr_pos = static_cast<double *>(buf_pos.ptr),
           *ptr_msd = static_cast<double *>(buf_msd.ptr);
    std::fill_n(ptr_msd, N_time, 0.0);

    {
        py::gil_scoped_release release;  // Release GIL for the computational part

        #pragma omp parallel for
        for (size_t shift = 0; shift < N_time; ++shift) {

            // #pragma omp critical
            //     {
            //         std::cout << "Number of threads: " << omp_get_num_threads() << std::endl;
            //     }
            double sum_sq = 0.0;
            size_t count = 0;

            #pragma omp parallel for reduction(+:sum_sq, count)
            for (size_t start = 0; start < N_time - shift; ++start) {
                size_t end = start + shift;
                for (size_t j = 0; j < N_atom; ++j) {
                        
                    double start_pos = ptr_pos[start * N_atom * N_dim + j * N_dim+1];
                    double end_pos = ptr_pos[end * N_atom * N_dim + j * N_dim+1];
                    
                    sum_sq += start_pos*start_pos + end_pos*end_pos - 2.0*start_pos*end_pos;
                }
                ++count;
            }
            ptr_msd[shift] = count > 0 ? sum_sq / (N_atom * count) : 0.0;
        }
    }

    return msd;
}

py::array_t<double> msd_zdir_calc(py::array_t<double> pos, const std::string& dimension) {
    auto buf_pos = pos.request();

    size_t N_time = buf_pos.shape[0];
    size_t N_atom = buf_pos.shape[1];
    size_t N_dim = dimension == "2d" ? 2 : 3;

    if (dimension != "2d" && dimension != "3d") {
        throw std::runtime_error("Num of dimensions should be '2d' or '3d'.");
    }

    auto msd = py::array(py::buffer_info(
        nullptr,
        sizeof(double),
        py::format_descriptor<double>::value,
        1,
        { N_time },
        { sizeof(double) }
    ));

    auto buf_msd = msd.request();

    double *ptr_pos = static_cast<double *>(buf_pos.ptr),
           *ptr_msd = static_cast<double *>(buf_msd.ptr);
    std::fill_n(ptr_msd, N_time, 0.0);

    {
        py::gil_scoped_release release;  // Release GIL for the computational part

        #pragma omp parallel for
        for (size_t shift = 0; shift < N_time; ++shift) {

            // #pragma omp critical
            //     {
            //         std::cout << "Number of threads: " << omp_get_num_threads() << std::endl;
            //     }
            double sum_sq = 0.0;
            size_t count = 0;

            #pragma omp parallel for reduction(+:sum_sq, count)
            for (size_t start = 0; start < N_time - shift; ++start) {
                size_t end = start + shift;
                for (size_t j = 0; j < N_atom; ++j) {
                        
                    double start_pos = ptr_pos[start * N_atom * N_dim + j * N_dim+2];
                    double end_pos = ptr_pos[end * N_atom * N_dim + j * N_dim+2];
                    
                    sum_sq += start_pos*start_pos + end_pos*end_pos - 2.0*start_pos*end_pos;
                }
                ++count;
            }
            ptr_msd[shift] = count > 0 ? sum_sq / (N_atom * count) : 0.0;
        }
    }

    return msd;
}

py::array_t<double> vacf_calc(py::array_t<double> vel, const std::string& dimension) {
    auto buf_vel = vel.request();

    size_t N_time = buf_vel.shape[0];
    size_t N_atom = buf_vel.shape[1];
    size_t N_dim = dimension == "2d" ? 2 : 3;

    if (dimension != "2d" && dimension != "3d") {
        throw std::runtime_error("Num of dimensions should be '2d' or '3d'.");
    }

    auto vacf = py::array(py::buffer_info(
        nullptr,
        sizeof(double),
        py::format_descriptor<double>::value,
        1,
        { N_time },
        { sizeof(double) }
    ));

    auto buf_vacf = vacf.request();

    double *ptr_vel = static_cast<double *>(buf_vel.ptr),
           *ptr_vacf = static_cast<double *>(buf_vacf.ptr);
    std::fill_n(ptr_vacf, N_time, 0.0);

    {
        py::gil_scoped_release release;  // Release GIL for the computational part

        #pragma omp parallel for
        for (size_t shift = 0; shift < N_time; ++shift) {
            double sum_vdot = 0.0;
            size_t count = 0;

            #pragma omp parallel for reduction(+:sum_vdot, count)
            for (size_t start = 0; start < N_time - shift; ++start) {
                size_t end = start + shift;
                for (size_t j = 0; j < N_atom; ++j) {
                    double vtdotv0 = 0.0;
                    for (size_t k = 0; k < N_dim; ++k) {
                        double start_vel = ptr_vel[start * N_atom * N_dim + j * N_dim + k];
                        double end_vel = ptr_vel[end * N_atom * N_dim + j * N_dim + k];
                        double vitdotvi0 = end_vel * start_vel;

                        vtdotv0 += vitdotvi0;
                    }
                    sum_vdot += vtdotv0;
                }
                ++count;
            }
            ptr_vacf[shift] = count > 0 ? sum_vdot / (N_atom * count) : 0.0;
        }
    }

    return vacf;
}

// py::array_t<double> vacf_calc_decade(py::array_t<double> vel, const std::string& dimension) {
//     auto buf_vel = vel.request();

//     size_t N_time = buf_vel.shape[0];
//     size_t N_atom = buf_vel.shape[1];
//     size_t N_dim = dimension == "2d" ? 2 : 3;

//     if (dimension != "2d" && dimension != "3d") {
//         throw std::runtime_error("Num of dimensions should be '2d' or '3d'.");
//     }

//     auto vacf = py::array(py::buffer_info(
//         nullptr,
//         sizeof(double),
//         py::format_descriptor<double>::value,
//         1,
//         { N_time },
//         { sizeof(double) }
//     ));

//     auto counts = py::array(py::buffer_info(
//         nullptr,
//         sizeof(size_t),
//         py::format_descriptor<size_t>::value,
//         1,
//         { N_time },
//         { sizeof(size_t) }
//     ));

//     auto buf_vacf = vacf.request();
//     auto buf_counts = counts.request();

//     double *ptr_vel = static_cast<double *>(buf_vel.ptr),
//            *ptr_vacf = static_cast<double *>(buf_vacf.ptr);

//     size_t  *ptr_counts = static_cast<size_t *>(buf_counts.ptr);
//     std::fill_n(ptr_vacf, N_time, 0.0);
//     std::fill_n(ptr_counts, N_time, 0);

//     {
//         py::gil_scoped_release release;  // Release GIL for the computational part

//         #pragma omp parallel for
//         for (size_t shift = 0; shift < N_time; ++shift) {
//             double sum_vdot = 0.0;
//             size_t count = 0;
//             #pragma omp parallel for reduction(+:sum_vdot, count)
//             for (size_t start = 0; start < N_time - shift; ++start) {
//                 size_t end = start + shift;
//                 for (size_t j = 0; j < N_atom; ++j) {
//                     double vtdotv0 = 0.0;
//                     for (size_t k = 0; k < N_dim; ++k) {
//                         double start_vel = ptr_vel[start * N_atom * N_dim + j * N_dim + k];
//                         double end_vel = ptr_vel[end * N_atom * N_dim + j * N_dim + k];
//                         double vitdotvi0 = end_vel * start_vel;

//                         vtdotv0 += vitdotvi0;
//                     }
//                     sum_vdot += vtdotv0;
//                 }
//                 ++count;
//             }
//             ptr_counts[shift] = count;
//             ptr_vacf[shift] = count > 0 ? sum_vdot / (N_atom * count) : 0.0;
//         }
//     }

//     return py::make_tuple(vacf, counts);
// }

py::array_t<double> vacf_calc_decade(py::array_t<double> vel, const std::string& dimension, double output_freq) {
    auto buf_vel = vel.request();

    size_t N_time = buf_vel.shape[0];
    size_t N_atom = buf_vel.shape[1];
    size_t N_dim = dimension == "2d" ? 2 : 3;

    if (dimension != "2d" && dimension != "3d") {
        throw std::runtime_error("Num of dimensions should be '2d' or '3d'.");
    }

    auto vacf = py::array(py::buffer_info(
        nullptr,
        sizeof(double),
        py::format_descriptor<double>::value,
        1,
        { N_time },
        { sizeof(double) }
    ));

    auto counts = py::array(py::buffer_info(
        nullptr,
        sizeof(size_t),
        py::format_descriptor<size_t>::value,
        1,
        { N_time },
        { sizeof(size_t) }
    ));

    auto buf_vacf = vacf.request();
    auto buf_counts = counts.request();

    double *ptr_vel = static_cast<double *>(buf_vel.ptr),
           *ptr_vacf = static_cast<double *>(buf_vacf.ptr);

    size_t  *ptr_counts = static_cast<size_t *>(buf_counts.ptr);
    std::fill_n(ptr_vacf, N_time, 0.0);
    std::fill_n(ptr_counts, N_time, 0);

    // uncomment this block to calculate the initial correlation
    // double initial_correlation = 0.0;
    // // Compute initial correlation (t=0)
    // for (size_t j = 0; j < N_atom; ++j) {
    //     for (size_t k = 0; k < N_dim; ++k) {
    //         double start_vel = ptr_vel[k + j * N_dim];  // Take the velocity at t=0 for normalization
    //         initial_correlation += start_vel * start_vel;
    //     }
    // }
    double initial_correlation = 1.0; // if not using above block

    //initial_correlation /= (N_atom * N_dim);  // Normalize by the number of atoms and dimensions

    {
        py::gil_scoped_release release;  // Release GIL for the computational part

        #pragma omp parallel for
        for (size_t shift = 0; shift < N_time; ++shift) {
            double sum_vdot = 0.0;
            size_t count = 0;
            #pragma omp parallel for reduction(+:sum_vdot, count)
            for (size_t start = 0; start < N_time - shift; ++start) {
                size_t end = start + shift;
                for (size_t j = 0; j < N_atom; ++j) {
                    double vtdotv0 = 0.0;
                    for (size_t k = 0; k < N_dim; ++k) {
                        double start_vel = ptr_vel[start * N_atom * N_dim + j * N_dim + k];
                        double end_vel = ptr_vel[end * N_atom * N_dim + j * N_dim + k];
                        vtdotv0 += start_vel * end_vel; // Calculate the dot product
                    }
                    sum_vdot += vtdotv0;
                }
                ++count;
            }
            ptr_counts[shift] = count;
            // Normalize by the initial correlation value at t=0
            ptr_vacf[shift] = count > 0 ? sum_vdot / (N_atom * count * initial_correlation * output_freq) : 0.0;
        }
    }

    return py::make_tuple(vacf, counts);
}


py::array_t<double> vacf_calc_notime(py::array_t<double> vel, const std::string& dimension) {
    auto buf_vel = vel.request();

    size_t N_time = buf_vel.shape[0];
    size_t N_atom = buf_vel.shape[1];
    size_t N_dim = dimension == "2d" ? 2 : 3;

    if (dimension != "2d" && dimension != "3d") {
        throw std::runtime_error("Num of dimensions should be '2d' or '3d'.");
    }

    auto vacf = py::array(py::buffer_info(
        nullptr,
        sizeof(double),
        py::format_descriptor<double>::value,
        1,
        { N_time },
        { sizeof(double) }
    ));

    auto buf_vacf = vacf.request();

    double *ptr_vel = static_cast<double *>(buf_vel.ptr),
           *ptr_vacf = static_cast<double *>(buf_vacf.ptr);
    std::fill_n(ptr_vacf, N_time, 0.0);

    // Calculate VACF
    {
        py::gil_scoped_release release;  // Release GIL for the computational part

        #pragma omp parallel for
        for (size_t shift = 0; shift < N_time; ++shift) {
            double sum_vdot = 0.0;

            // Calculate the dot product for each atom at the given shift
            for (size_t j = 0; j < N_atom; ++j) {
                double v0 = ptr_vel[0 * N_atom * N_dim + j * N_dim]; // vx of atom j at t=0
                for (size_t k = 0; k < N_dim; ++k) {
                    double vt = ptr_vel[shift * N_atom * N_dim + j * N_dim + k]; // vx of atom j at t=shift
                    sum_vdot += v0 * vt;  // Calculate the dot product
                }
            }

            // Average over the number of atoms
            ptr_vacf[shift] = sum_vdot / N_atom;
        }
    }

    return vacf;
}


py::array_t<double> shear_stress_acf(py::array_t<double> stress) {
    // Request buffer information
    auto buf_stress = stress.request();

    size_t N_time = buf_stress.shape[0];
    size_t N_components = buf_stress.shape[1];  // Should be 6 (p_xx, p_yy, p_zz, p_xy, p_xz, p_yz)
    
    if (N_components != 6) {
        throw std::runtime_error("Stress array should have 6 components per time step.");
    }

    // Prepare output array for shear stress autocorrelation
    auto acf = py::array(py::buffer_info(
        nullptr,
        sizeof(double),
        py::format_descriptor<double>::value,
        1,
        { N_time },
        { sizeof(double) }
    ));

    auto buf_acf = acf.request();

    double *ptr_stress = static_cast<double *>(buf_stress.ptr),
           *ptr_acf = static_cast<double *>(buf_acf.ptr);

    std::fill_n(ptr_acf, N_time, 0.0);

    {
        py::gil_scoped_release release;  // Release GIL for the computational part

        #pragma omp parallel for
        for (size_t shift = 0; shift < N_time; ++shift) {
            double sum_shear = 0.0;
            size_t count = 0;

            #pragma omp parallel for reduction(+:sum_shear, count)
            for (size_t start = 0; start < N_time - shift; ++start) {
                size_t end = start + shift;

                // Access stress components at time start and time end
                double p_xx_0 = ptr_stress[start * 6 + 0], p_xx_t = ptr_stress[end * 6 + 0];
                double p_yy_0 = ptr_stress[start * 6 + 1], p_yy_t = ptr_stress[end * 6 + 1];
                double p_zz_0 = ptr_stress[start * 6 + 2], p_zz_t = ptr_stress[end * 6 + 2];
                double p_xy_0 = ptr_stress[start * 6 + 3], p_xy_t = ptr_stress[end * 6 + 3];
                double p_xz_0 = ptr_stress[start * 6 + 4], p_xz_t = ptr_stress[end * 6 + 4];
                double p_yz_0 = ptr_stress[start * 6 + 5], p_yz_t = ptr_stress[end * 6 + 5];

                // Compute shear stress terms: <p_xy(0) * p_xy(t)> + <p_xz(0) * p_xz(t)> + <p_yz(0) * p_yz(t)>
                double shear_stress = p_xy_0 * p_xy_t + p_xz_0 * p_xz_t + p_yz_0 * p_yz_t;

                // Compute normal stress difference terms: N_ab = p_aa - p_bb
                double N_xy_0 = p_xx_0 - p_yy_0, N_xy_t = p_xx_t - p_yy_t;
                double N_xz_0 = p_xx_0 - p_zz_0, N_xz_t = p_xx_t - p_zz_t;
                double N_yz_0 = p_yy_0 - p_zz_0, N_yz_t = p_yy_t - p_zz_t;

                // Compute normal stress terms: <N_xy(0) * N_xy(t)> + <N_xz(0) * N_xz(t)> + <N_yz(0) * N_yz(t)>
                double normal_stress = N_xy_0 * N_xy_t + N_xz_0 * N_xz_t + N_yz_0 * N_yz_t;

                // Sum of shear stress and normal stress contributions
                sum_shear += (shear_stress / 5.0) + (normal_stress / 30.0);
                ++count;
            }

            ptr_acf[shift] = count > 0 ? sum_shear / count : 0.0;
        }
    }

    return acf;
}

py::tuple sacf_total(py::array_t<double> stress) {
    // Request buffer information
    auto buf_stress = stress.request();

    size_t N_time = buf_stress.shape[0];
    size_t N_components = buf_stress.shape[1];  // Should be 6 (p_xx, p_yy, p_zz, p_xy, p_xz, p_yz)
    
    if (N_components != 6) {
        throw std::runtime_error("Stress array should have 6 components per time step.");
    }

    // Prepare output array for shear stress autocorrelation
    auto pxx = py::array(py::buffer_info(
        nullptr,
        sizeof(double),
        py::format_descriptor<double>::value,
        1,
        { N_time },
        { sizeof(double) }
    ));

    auto pyy = py::array(py::buffer_info(
        nullptr,
        sizeof(double),
        py::format_descriptor<double>::value,
        1,
        { N_time },
        { sizeof(double) }
    ));

    auto pzz = py::array(py::buffer_info(
        nullptr,
        sizeof(double),
        py::format_descriptor<double>::value,
        1,
        { N_time },
        { sizeof(double) }
    ));

    auto pxy = py::array(py::buffer_info(
        nullptr,
        sizeof(double),
        py::format_descriptor<double>::value,
        1,
        { N_time },
        { sizeof(double) }
    ));

    auto pxz = py::array(py::buffer_info(
        nullptr,
        sizeof(double),
        py::format_descriptor<double>::value,
        1,
        { N_time },
        { sizeof(double) }
    ));

    auto pyz = py::array(py::buffer_info(
        nullptr,
        sizeof(double),
        py::format_descriptor<double>::value,
        1,
        { N_time },
        { sizeof(double) }
    ));

    auto counts = py::array(py::buffer_info(
        nullptr,
        sizeof(size_t),
        py::format_descriptor<size_t>::value,
        1,
        { N_time },
        { sizeof(size_t) }
    ));

    auto buf_pxx = pxx.request();
    auto buf_pyy = pyy.request();
    auto buf_pzz = pzz.request();
    auto buf_pxy = pxy.request();
    auto buf_pxz = pxz.request();
    auto buf_pyz = pyz.request();

    auto buf_counts = counts.request();

    double *ptr_stress = static_cast<double *>(buf_stress.ptr),
           *ptr_pxx = static_cast<double *>(buf_pxx.ptr),
           *ptr_pyy = static_cast<double *>(buf_pyy.ptr),
           *ptr_pzz = static_cast<double *>(buf_pzz.ptr),
           *ptr_pxy = static_cast<double *>(buf_pxy.ptr),
           *ptr_pxz = static_cast<double *>(buf_pxz.ptr),
           *ptr_pyz = static_cast<double *>(buf_pyz.ptr);

    size_t  *ptr_counts = static_cast<size_t *>(buf_counts.ptr);

    std::fill_n(ptr_pxx, N_time, 0.0);
    std::fill_n(ptr_pyy, N_time, 0.0);
    std::fill_n(ptr_pzz, N_time, 0.0);
    std::fill_n(ptr_pxy, N_time, 0.0);
    std::fill_n(ptr_pxz, N_time, 0.0);
    std::fill_n(ptr_pyz, N_time, 0.0);

    std::fill_n(ptr_counts, N_time, 0);

    {
        py::gil_scoped_release release;  // Release GIL for the computational part

        #pragma omp parallel for
        for (size_t shift = 0; shift < N_time; ++shift) {
            double sum_pxx = 0.0;
            double sum_pyy = 0.0;
            double sum_pzz = 0.0;
            double sum_pxy = 0.0;
            double sum_pxz = 0.0;
            double sum_pyz = 0.0;
            size_t count = 0;

            #pragma omp parallel for reduction(+:sum_pxx, sum_pyy, sum_pzz, sum_pxy, sum_pxz, sum_pyz, count)
            for (size_t start = 0; start < N_time - shift; ++start) {
                size_t end = start + shift;

                // Access stress components at time start and time end
                double p_xx_0 = ptr_stress[start * 6 + 0], p_xx_t = ptr_stress[end * 6 + 0];
                double p_yy_0 = ptr_stress[start * 6 + 1], p_yy_t = ptr_stress[end * 6 + 1];
                double p_zz_0 = ptr_stress[start * 6 + 2], p_zz_t = ptr_stress[end * 6 + 2];
                double p_xy_0 = ptr_stress[start * 6 + 3], p_xy_t = ptr_stress[end * 6 + 3];
                double p_xz_0 = ptr_stress[start * 6 + 4], p_xz_t = ptr_stress[end * 6 + 4];
                double p_yz_0 = ptr_stress[start * 6 + 5], p_yz_t = ptr_stress[end * 6 + 5];

                // Compute shear stress terms: <p_xy(0) * p_xy(t)> + <p_xz(0) * p_xz(t)> + <p_yz(0) * p_yz(t)>
                double pxx_val = p_xx_0 * p_xx_t;
                double pyy_val = p_yy_0 * p_yy_t;
                double pzz_val = p_zz_0 * p_zz_t;
                double pxy_val = p_xy_0 * p_xy_t;
                double pxz_val = p_xz_0 * p_xz_t;
                double pyz_val = p_yz_0 * p_yz_t;

                // Sum of stress correlations
                // sum_shear += (shear_stress / 5.0) + (normal_stress / 30.0);
                sum_pxx += pxx_val;
                sum_pyy += pyy_val;
                sum_pzz += pzz_val;
                sum_pxy += pxy_val;
                sum_pxz += pxz_val;
                sum_pyz += pyz_val;

                ++count;
            }
            if (shift == 0) {
                std::cout << "t=0 values (pxx, pyy, pzz, pxy, pxz, pyz): "
                    << sum_pxx / count << ", " << sum_pyy / count << ", "
                    << sum_pzz / count << ", " << sum_pxy / count << ", "
                    << sum_pxz / count << ", " << sum_pyz / count << std::endl;
            }
            ptr_counts[shift] = count;
            ptr_pxx[shift] = count > 0 ? sum_pxx / count : 0.0;
            ptr_pyy[shift] = count > 0 ? sum_pyy / count : 0.0;
            ptr_pzz[shift] = count > 0 ? sum_pzz / count : 0.0;
            ptr_pxy[shift] = count > 0 ? sum_pxy / count : 0.0;
            ptr_pxz[shift] = count > 0 ? sum_pxz / count : 0.0;
            ptr_pyz[shift] = count > 0 ? sum_pyz / count : 0.0;
            // ptr_acf[shift] = count > 0 ? sum_shear / count : 0.0;
        }
    }

    return py::make_tuple(pxx, pyy, pzz, pxy, pxz, pyz, counts);
}

py::tuple compute_moduli(py::array_t<double> g_t, py::array_t<double> t_data, py::array_t<double> omega_data) {
    // Request buffer information from input arrays
    auto buf_g_t = g_t.request();
    auto buf_t_data = t_data.request();
    auto buf_omega_data = omega_data.request();

    size_t N_time = buf_g_t.shape[0];
    size_t N_omega = buf_omega_data.shape[0];

    if (buf_t_data.shape[0] != N_time) {
        throw std::runtime_error("G(t) and time data must have the same length.");
    }

    // Extract pointers to the data
    double *ptr_g_t = static_cast<double*>(buf_g_t.ptr);
    double *ptr_t_data = static_cast<double*>(buf_t_data.ptr);
    double *ptr_omega_data = static_cast<double*>(buf_omega_data.ptr);

    // Create arrays to store G' and G'' results
    auto g_prime = py::array_t<double>(N_omega);
    auto g_double_prime = py::array_t<double>(N_omega);

    auto buf_g_prime = g_prime.request();
    auto buf_g_double_prime = g_double_prime.request();

    double *ptr_g_prime = static_cast<double*>(buf_g_prime.ptr);
    double *ptr_g_double_prime = static_cast<double*>(buf_g_double_prime.ptr);

    {
        py::gil_scoped_release release;  // Release the GIL for the computational part

        // Loop over each omega value to calculate G'(omega) and G''(omega)
        for (size_t i = 0; i < N_omega; ++i) {
            double omega = ptr_omega_data[i];
            double sum_sin = 0.0;
            double sum_cos = 0.0;

            // Perform numerical integration using the trapezoidal rule
            for (size_t j = 0; j < N_time - 1; ++j) {
                double t1 = ptr_t_data[j];
                double t2 = ptr_t_data[j + 1];
                double g1 = ptr_g_t[j];
                double g2 = ptr_g_t[j + 1];
                double dt = t2 - t1;

                // Compute average values for trapezoidal integration
                double avg_g = (g1 + g2) / 2.0;
                double avg_sin = (sin(omega * t1) + sin(omega * t2)) / 2.0;
                double avg_cos = (cos(omega * t1) + cos(omega * t2)) / 2.0;

                // Integrate G'(omega) and G''(omega)
                sum_sin += avg_g * avg_sin * dt;
                sum_cos += avg_g * avg_cos * dt;
            }

            // Calculate G' and G'' using the trapezoidal sums
            ptr_g_prime[i] = omega * sum_sin;
            ptr_g_double_prime[i] = omega * sum_cos;
        }
    }

    // Return the results as a tuple (G'(omega), G''(omega))
    return py::make_tuple(g_prime, g_double_prime);
}


PYBIND11_MODULE(md_calcs_par, m) {
    m.def("msd_calc", &msd_calc, "Calculate MSD for a trajectory.");
    m.def("msd_xdir_calc", &msd_xdir_calc, "Calculate x-dir MSD for a trajectory.");
    m.def("msd_ydir_calc", &msd_ydir_calc, "Calculate y-dir MSD for a trajectory.");
    m.def("vacf_calc", &vacf_calc, "Calculate ACF for a trajectory.");
    m.def("shear_stress_calc", &shear_stress_acf, "Calculate SACF for a trajectory.");
    m.def("compute_moduli", &compute_moduli, "Calculate SACF for a trajectory.");
    m.def("vacf_calc_notime", &vacf_calc_notime, "Calculate SACF for a trajectory.");
    m.def("vacf_calc_decade", &vacf_calc_decade, "Calculate SACF for a trajectory.");
    m.def("msd_calc_decade", &msd_calc_decade, "Calculate SACF for a trajectory.");
    m.def("sacf_total", &sacf_total, "Calc stress autocorrelations for each component.");
}
