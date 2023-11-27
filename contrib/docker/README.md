# Docker

## Build
```bash
$ docker build --no-cache -t threemonth/cg_method contrib/docker
```

## Run
```bash
$ docker run -it --name cg_method -v ~/:/workspace --gpus all threemonth/cg_method bash
```