# Custom PyInstaller hook for onnxruntime
# Prevents the isolated subprocess crash during binary dependency scanning
# by collecting all files upfront and using 'py' collection mode.

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

# 'py' mode = collect individual modules, NOT as a package directory.
# This prevents onnxruntime from being in the 'collected_packages' list
# that triggers find_binary_dependencies (which crashes in an isolated subprocess).
module_collection_mode = 'py'

# Manually collect all data files (e.g., .ort config files)
datas = collect_data_files('onnxruntime')

# Manually collect all native .dll / .pyd files
binaries = collect_dynamic_libs('onnxruntime')

# Make sure all submodules are discovered
hiddenimports = collect_submodules('onnxruntime')
