"""
Makes a local copy of a parameter file, modifies directories in that parameter file, then calls the UNet.

# Arguments
- `data_path`: path to directory containing the data. Data must be in the form of HDF5 files.
- `unet_path`: path to UNet `predict.py` file, which must be from the `pytorch-3dunet` package and compatible with the given data and parameters.
- `param_path`: path to base parameter file; only the file path will be changed.
"""
function call_unet(data_path, unet_path, param_path, py_activate, script_path, log_file)
    rootpath = back_one_dir(data_path)
    params = read_txt(param_path)
    old_file_path = split(split(split(params, "file_paths:")[2], "-")[2], "'")[2]
    new_params = replace(params, old_file_path => data_path)
    new_param_path = joinpath(rootpath, get_filename(param_path))
    param_file = open(new_param_path, "w")
    write(param_file, new_params)
    close(param_file)
    script_file = open(script_path, "w")
    write(script_file, "#!/bin/bash\n")
    write(script_file, "source $(py_activate)\n")
    write(script_file, "python $(unet_path) --config $(new_param_path) > $(log_file)")
    close(script_file)
    chmod(script_path, 0o774)
    run(`env -i $script_path`)
end
