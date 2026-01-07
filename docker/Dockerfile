FROM dl.nvidia/cuda:12.8-cudnn9-runtime-torch2.7.1

RUN pip3 install numpy==2.3.5 scikit_image==0.25.2 scikit-learn==1.6.1 opencv-contrib-python==4.11.0.86  \
    opencv-python==4.11.0.86 opencv-python-headless==4.11.0.86  timm==0.9.12  tabulate==0.9.0 ptflops==0.7 \
    onnx_graphsurgeon==0.5.8 onnx==1.18.0 onnxruntime==1.20.0 pandas==2.2.3

COPY backbones /home/workspace/Dinomaly/backbones
COPY dinov1 /home/workspace/Dinomaly/dinov1
COPY dinov2 /home/workspace/Dinomaly/dinov2
COPY dinov3 /home/workspace/Dinomaly/dinov3
COPY flops_profiler /home/workspace/Dinomaly/flops_profiler
COPY optimizers /home/workspace/Dinomaly/optimizers
COPY weights /home/workspace/Dinomaly/weights
COPY beit /home/workspace/Dinomaly/beit
COPY models /home/workspace/Dinomaly/models

COPY dataset.py  /home/workspace/Dinomaly/dataset.py
COPY dinomaly_mvtec_uni_base_test.py  /home/workspace/Dinomaly/dinomaly_mvtec_uni_base_test.py
COPY dinomaly_mvtec_uni_base.py  /home/workspace/Dinomaly/dinomaly_mvtec_uni_base.py
COPY dinomaly_mvtec_uni_large.py  /home/workspace/Dinomaly/dinomaly_mvtec_uni_large.py
COPY utils.py  /home/workspace/Dinomaly/utils.py

WORKDIR /home/workspace/Dinomaly
ENTRYPOINT ["python3", "dinomaly_mvtec_uni_base.py"]