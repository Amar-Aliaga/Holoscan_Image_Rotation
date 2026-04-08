ARG UPDATE_CA=no
ARG HOLOSCAN_SDK_VERSION=3.5.0
ARG HOLOSCAN_SDK_IMAGE=nvcr.io/nvidia/clara-holoscan/holoscan:v${HOLOSCAN_SDK_VERSION}-dgpu
FROM ${HOLOSCAN_SDK_IMAGE} as base-no
FROM ${HOLOSCAN_SDK_IMAGE} as base-yes
# FROM base-${UPDATE_CA} as base
# ARG UID=1000
# ARG GID=1000
# ARG USERNAME=dev
# ARG TARGETARCH
# RUN groupadd -g ${GID} ${USERNAME}
# RUN useradd -u ${UID} -g ${GID} ${USERNAME}

# Install dependencies including OpenCV with CUDA support
RUN apt update && apt install -y \
    clang \
    libxcursor-dev \
    libopencv-dev \
    libopencv-contrib-dev
    

# Install CMake version 3.29.2
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.30.4/cmake-3.30.4.tar.gz && \
    tar -xzf cmake-3.30.4.tar.gz && \
    cd cmake-3.30.4 && \
    ./bootstrap && \
    make -j$(nproc) && \
    make install

WORKDIR /workspace

CMD ["/bin/bash"]