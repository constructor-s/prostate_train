FROM projectmonai/monai:latest

RUN pip install --no-cache-dir nnunetv2

COPY . /bundle
WORKDIR /bundle

ENV nnUNet_results=/bundle/models
ENV nnUNet_preprocessed=/tmp/nnUNet_preprocessed
ENV nnUNet_raw=/tmp/nnUNet_raw

ENTRYPOINT ["python", "scripts/infer.py", "/bundle"]
CMD ["--help"]
