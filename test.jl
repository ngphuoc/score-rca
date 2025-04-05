using Pkg
Pkg.build("PythonCall")  # Rebuild so PythonCall picks up the new setting
using PythonCall, RDatasets
iris = dataset("datasets", "iris")

