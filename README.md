# Purpose of This Repo

This repo is meant to be used to keep things organized during content development and act as the source of truth for all projects and exercises related to this course.

## Folder Structure

### Lesson Folder

This repo contains a folder for each `lesson` and one `project` folder.

Example
```
lesson-1-hello
lesson-2-world
lesson-3-foo
lesson-4-bar
project
```

Each `lesson` folder is named using the naming convention of `lesson-#-name-of-lesson`.

Example
```
lesson-1-hello
```

Four lesson folders have been provided as a template; However, you may need to add more or possibly use less than four depending on what is needed.

If you require an additional lesson folder, you can make a copy of the folder and paste it into the root directory.

### Exercises Folder

Each `lesson` folder contains an `exercises` folder. This `exercises` folder should contain all files and instructions necessary for the exercises along with the solution. The solutions for these exercises will be shared with students. See the `README` in the `exercises` folder for information about folder structure.

### Project Folder

The `project` folder should contain all files and instructions necessary for setup. If possible, a set of instructions should be provided for both Udacity workspaces and a way to work locally (for both MacOS and Windows OS). At a minimum, one set of instructions should be provided. A `README` template has been provided in the project folder. This template layout should be used to write your README.

## Runtime Environments (Vocareum)

The container's **system Python is not used** to run these notebooks (its `torch==2.9.1`
is incompatible with the pinned `optimum-*` / `neural-compressor` stack). Two dedicated
virtual environments are provisioned on the persistent `/voc/data` volume and exposed as
Jupyter kernels. Select the correct kernel from the kernel picker before running a notebook.

| Kernel (display name) | venv | torch | Use for |
|---|---|---|---|
| `venv-torch2.3` | `/voc/data/venv-torch2.3` | 2.3.1 (cu121) | Everything **except** the two notebooks below |
| `venv-torch2.6` | `/voc/data/venv-torch2.6` | 2.6.0 (cu124) | `lesson-2_quantization_techniques/demo-1/demo.ipynb` and `lesson-2_quantization_techniques/exercise_1/solution/solution.ipynb` |

Why two kernels: `demo-1` and `exercise_1` in lesson 2 quantize + freeze + save a model with
Optimum-Quanto, which needs `optimum-quanto >= 0.2.4` (→ `torch >= 2.6`) to avoid a
`state_dict` save crash. The rest of the course runs on the stable torch 2.3.1 stack. Every
lesson-2 notebook states its required kernel + torch version in its first cell.

Package pins for the torch 2.3 environment are in `requirements.txt`.
