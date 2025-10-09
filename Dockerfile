FROM nvcr.io/nvidia/pytorch:24.10-py3
LABEL authors="paco"

ARG requirements_file
ARG package_name
ARG package_version

# Copy files
COPY ["${requirements_file}", "./"]
COPY ["/dist/${package_name}-${package_version}-py3-none-any.whl", "./"]

# Install requirements and wheel
RUN pip install --no-deps -r ${requirements_file}
RUN pip install --no-deps "${package_name}-${package_version}-py3-none-any.whl"