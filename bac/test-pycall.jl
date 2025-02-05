using Pkg
ENV["PYTHON"] = "/data/ngphuoc/miniconda3/envs/rca/bin/python"
using PyCall
Pkg.build("PyCall")

