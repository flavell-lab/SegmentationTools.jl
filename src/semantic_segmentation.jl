"""
Makes a local copy of a parameter file, modifies directories in that parameter file, then calls the UNet.

# Arguments
- `param_path`: path parameter dictionary
"""
function call_unet(param_path::Dict)
    path_root_process = param_path["path_root_process"]
    path_sh = joinpath(path_root_process, "run_unet.sh")
    path_log = joinpath(path_root_process, "unet.log")

    path_unet_data = param_path["path_dir_unet_data"] # path to directory containing the data (.h5)
    path_unet_pred = param_path["path_unet_pred"] #=path to UNet `predict.py` file, which must be from the
    `pytorch-3dunet` package and compatible with the given data and parameters.=#
    path_unet_param = param_path["path_unet_param"] # path to base parameter file
    path_unet_py_env = param_path["path_unet_py_env"]
    path_unet_param_new = joinpath(path_root_process, basename(param_path["path_unet_param"]))
    
    unet_param_str = replace(read_txt(path_unet_param), "PATH_DIR_HDF_INPUT" => path_unet_data)

    open(path_unet_param_new, "w") do f
        write(f, unet_param_str)
    end
    
    str_cmd = "#!/bin/bash\n" *
        "source $(path_unet_py_env)\n" *
        "python $(path_unet_pred) --config $(path_unet_param_new) > $(path_log)"
    open(path_sh, "w") do f
        write(f, str_cmd)
    end
    chmod(path_sh, 0o774)
    run(`env -i $path_sh`)
end
